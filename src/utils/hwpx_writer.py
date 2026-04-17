"""
HWPX (HWP XML format) 파일 생성 유틸리티.

hwpx 라이브러리의 Skeleton.hwpx 템플릿을 베이스로 section0.xml 만 교체하여
Python zipfile 로 직접 HWPX를 조립한다. 이 방식은 라이브러리 자체 ZIP 라이터의
manifest 누락 문제를 회피하고 안정적인 HWPX 파일을 생성한다.
"""

import io
import logging
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)


def _skeleton_path() -> Path:
    """hwpx 패키지에 포함된 Skeleton.hwpx 경로를 반환한다."""
    try:
        import hwpx as _hwpx_pkg
        skel = Path(_hwpx_pkg.__file__).parent / "data" / "Skeleton.hwpx"
        if skel.exists():
            return skel
    except Exception:
        pass
    raise FileNotFoundError("hwpx Skeleton.hwpx 를 찾을 수 없습니다.")


def _make_section_xml(lines: list[str]) -> bytes:
    """텍스트 줄 목록으로 section0.xml 을 생성한다.

    첫 번째 단락(섹션 속성)은 Skeleton 에서 그대로 가져오고,
    이후 단락을 텍스트 줄마다 추가한다.
    """
    # Skeleton 의 섹션 속성 단락(secPr 포함) 을 그대로 재사용
    skel = _skeleton_path()
    with zipfile.ZipFile(skel) as zf:
        raw_sec = zf.read("Contents/section0.xml").decode("utf-8")

    # <hs:sec ...> 태그만 추출 (첫 번째 단락 포함)
    # 텍스트 단락을 닫는 </hs:sec> 직전에 삽입한다.
    close_tag = "</hs:sec>"
    if close_tag not in raw_sec:
        raise ValueError("Skeleton section0.xml 구조가 예상과 다릅니다.")

    # 단락 XML 생성
    para_chunks: list[str] = []
    for idx, line in enumerate(lines):
        para_id = 1000000 + idx  # 간단한 고유 ID
        text_node = f"<hp:t>{escape(line)}</hp:t>" if line else "<hp:t/>"
        para_chunks.append(
            f'<hp:p id="{para_id}" paraPrIDRef="0" styleIDRef="0"'
            f' pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="0">{text_node}</hp:run>'
            f"</hp:p>"
        )

    new_sec = raw_sec.replace(close_tag, "".join(para_chunks) + close_tag)
    return new_sec.encode("utf-8")


def make_hwpx(text: str) -> bytes:
    """텍스트를 HWPX 바이너리로 변환하여 반환한다.

    Skeleton.hwpx 를 베이스로 section0.xml 만 교체한다.
    manifest 를 올바르게 채우고 Python zipfile 로 직접 조립한다.

    Args:
        text: 보고서 본문 텍스트 (개행 포함)

    Returns:
        HWPX ZIP 바이너리
    """
    lines = text.split("\n")
    new_section = _make_section_xml(lines)
    preview_text = text[:1000].encode("utf-8")  # PrvText 미리보기

    skel = _skeleton_path()
    buf = io.BytesIO()

    with zipfile.ZipFile(skel, "r") as src, zipfile.ZipFile(buf, "w") as dst:
        # 1. mimetype 은 STORED(비압축) 으로 가장 먼저 기록
        info_mime = zipfile.ZipInfo("mimetype")
        info_mime.compress_type = zipfile.ZIP_STORED
        dst.writestr(info_mime, "application/hwp+zip")

        # 2. 나머지 파일 복사 (section0.xml · PrvText.txt · manifest 제외)
        skip = {
            "mimetype",
            "Contents/section0.xml",
            "Preview/PrvText.txt",
            "META-INF/manifest.xml",
        }
        for item in src.infolist():
            if item.filename in skip:
                continue
            data = src.read(item.filename)
            info = zipfile.ZipInfo(item.filename)
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)

        # 3. 교체된 section0.xml
        info_sec = zipfile.ZipInfo("Contents/section0.xml")
        info_sec.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info_sec, new_section)

        # 4. 미리보기 텍스트 갱신
        info_prv = zipfile.ZipInfo("Preview/PrvText.txt")
        info_prv.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info_prv, preview_text)

        # 5. 올바른 manifest.xml (모든 파트 목록 포함)
        manifest_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<odf:manifest xmlns:odf="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
            '<odf:file-entry odf:full-path="Contents/content.hpf"'
            '  odf:media-type="application/hwpml-package+xml"/>'
            '<odf:file-entry odf:full-path="Contents/header.xml"'
            '  odf:media-type="application/xml"/>'
            '<odf:file-entry odf:full-path="Contents/section0.xml"'
            '  odf:media-type="application/xml"/>'
            '<odf:file-entry odf:full-path="settings.xml"'
            '  odf:media-type="application/xml"/>'
            '<odf:file-entry odf:full-path="version.xml"'
            '  odf:media-type="application/xml"/>'
            '<odf:file-entry odf:full-path="Preview/PrvText.txt"'
            '  odf:media-type="text/plain"/>'
            '<odf:file-entry odf:full-path="Preview/PrvImage.png"'
            '  odf:media-type="image/png"/>'
            "</odf:manifest>"
        )
        info_mf = zipfile.ZipInfo("META-INF/manifest.xml")
        info_mf.compress_type = zipfile.ZIP_DEFLATED
        dst.writestr(info_mf, manifest_xml.encode("utf-8"))

    result = buf.getvalue()
    logger.debug(f"[HWPX] 생성 완료: {len(result)} bytes, {len(lines)} 줄")
    return result
