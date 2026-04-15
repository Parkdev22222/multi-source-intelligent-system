"""
HWPX (HWP XML format) 파일 생성 유틸리티.

HWPX는 ZIP 아카이브 기반의 HWP 형식으로,
한글과컴퓨터 문서를 XML 기반으로 표현한다.
"""

import io
import zipfile

_NS = "urn:schemas-hwpml-com:office:hwpml:2011:hwp"
_NS_HSP = "http://www.w3.org/XML/1998/namespace"

_CONTAINER_XML = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="Contents/content.hpf" media-type="application/hwp+zip"/>
  </rootfiles>
</container>
"""

_CONTENT_HPF = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">
  <metadata/>
  <manifest>
    <item id="header"   href="header.xml"            media-type="application/xml"/>
    <item id="section0" href="bodytext/section0.xml" media-type="application/xml"/>
  </manifest>
  <spine>
    <itemref idref="section0"/>
  </spine>
</package>
"""

_HEADER_XML = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<hh:head xmlns:hh="urn:schemas-hwpml-com:office:hwpml:2011:hwp"
         version="5.1.0.0">
  <hh:refList>
    <hh:fontfaces>
      <hh:fontface lang="ko">
        <hh:font name="함초롬바탕" type="ttf"/>
      </hh:fontface>
      <hh:fontface lang="en">
        <hh:font name="함초롬바탕" type="ttf"/>
      </hh:fontface>
    </hh:fontfaces>
    <hh:charShapes>
      <hh:charShape id="0" height="1000" textColor="0" shadeColor="16777215"
                    bold="0" italic="0" underline="0" strikeout="0" outline="0"
                    shadow="0" emboss="0" engrave="0" superScript="0"
                    subScript="0" ratio="100" spacing="0" relSz="100"
                    offset="0" vertAlign="0" charShadow="0"
                    useFontSpace="0" useKerning="0" symMark="0"
                    borderFillIDRef="0" fontRef0="0" fontRef1="0"
                    fontRef2="0" fontRef3="0" fontRef4="0"
                    fontRef5="0" fontRef6="0"/>
    </hh:charShapes>
    <hh:paraShapes>
      <hh:paraShape id="0" marginLeft="0" marginRight="0"
                    indent="0" marginTop="0" marginBottom="0"
                    lineSpacing="160" lineSpacingType="percent"
                    align="justify" widowOrphan="0" keepWithNext="0"
                    keepLines="0" pageBreakBefore="0"
                    fontLineHeight="0" snapToGrid="0"
                    condense="0" linkP="0" tabStopPosition="0"
                    collapseMargin="0"/>
    </hh:paraShapes>
    <hh:styles>
      <hh:style id="0" name="Normal" engName="Normal" type="para"
                nextStyleIDRef="0" langIDRef="1042"
                lockForm="0" charShapeIDRef="0" paraShapeIDRef="0"/>
    </hh:styles>
  </hh:refList>
</hh:head>
"""


def _escape_xml(text: str) -> str:
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_section_xml(text: str) -> str:
    """텍스트를 단락별로 분리하여 HWPX section0.xml 을 생성한다."""
    lines = text.split("\n")

    para_parts = []
    for line in lines:
        escaped = _escape_xml(line)
        para_parts.append(
            f'  <hp:p>'
            f'<hp:pPr><hp:paraShape paraPrIDRef="0"/><hp:charShape charPrIDRef="0"/></hp:pPr>'
            f'<hp:run><hp:charPr charPrIDRef="0"/><hp:t>{escaped}</hp:t></hp:run>'
            f'</hp:p>'
        )

    paragraphs = "\n".join(para_parts)
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<hp:sec xmlns:hp="{_NS}">\n'
        f'{paragraphs}\n'
        f'</hp:sec>\n'
    )


def make_hwpx(text: str) -> bytes:
    """텍스트를 HWPX 바이너리로 변환하여 반환한다.

    Args:
        text: 보고서 본문 텍스트 (개행 포함)

    Returns:
        HWPX ZIP 바이너리
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # mimetype은 반드시 비압축(STORED)으로 첫 번째에 써야 한다
        zf.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/hwp+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        zf.writestr("META-INF/container.xml", _CONTAINER_XML.encode("utf-8"))
        zf.writestr("Contents/content.hpf", _CONTENT_HPF.encode("utf-8"))
        zf.writestr("Contents/header.xml", _HEADER_XML.encode("utf-8"))
        zf.writestr(
            "Contents/bodytext/section0.xml",
            _build_section_xml(text).encode("utf-8"),
        )
    return buf.getvalue()
