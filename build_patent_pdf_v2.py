"""Generate the revised patent PDF matching the uploaded template's visual style.

Focus: 변화탐지(시계열 객체 페어링) + GraphRAG = 핵심 차별성
- FOV 5차원 분류는 부수적 종속항으로만 유지
- 종래기술: US 12,488,225 B1 (Booz Allen Hamilton, 2025)
- 코드 베이스: GitHub multi-source-intelligent-system 실제 구현 기준
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.pdfgen import canvas

# ── 한글 폰트 등록 ─────────────────────────────────────────────
FONT_REG_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
]
FONT_BOLD_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
]
for p in FONT_REG_PATHS:
    try:
        pdfmetrics.registerFont(TTFont("Nanum", p))
        break
    except Exception:
        pass
for p in FONT_BOLD_PATHS:
    try:
        pdfmetrics.registerFont(TTFont("NanumBold", p))
        break
    except Exception:
        pass

KOR = "Nanum"
KORB = "NanumBold"

# ── 색상 팔레트 (업로드 PDF와 동일한 라이트블루 톤) ─────────
ACCENT_BLUE = HexColor("#3b82f6")
TITLE_BLUE = HexColor("#1e3a8a")
LIGHT_BG = HexColor("#eff6ff")
LIGHT_BORDER = HexColor("#bfdbfe")
TEXT_DARK = HexColor("#111827")
TEXT_BODY = HexColor("#1f2937")
MUTED = HexColor("#6b7280")

# ── 스타일 정의 ─────────────────────────────────────────────
styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    "Title", parent=styles["Title"],
    fontName=KORB, fontSize=17, textColor=TITLE_BLUE,
    alignment=TA_CENTER, spaceAfter=14, leading=22,
)
style_h1 = ParagraphStyle(
    "H1", fontName=KORB, fontSize=14, textColor=TITLE_BLUE,
    leftIndent=10, spaceBefore=16, spaceAfter=8, leading=18,
    borderPadding=0,
)
style_h2 = ParagraphStyle(
    "H2", fontName=KORB, fontSize=11.5, textColor=TEXT_DARK,
    spaceBefore=10, spaceAfter=5, leading=15,
)
style_h3 = ParagraphStyle(
    "H3", fontName=KORB, fontSize=10.5, textColor=TEXT_DARK,
    spaceBefore=8, spaceAfter=3, leading=14,
)
style_body = ParagraphStyle(
    "Body", fontName=KOR, fontSize=10, textColor=TEXT_BODY,
    leading=16, alignment=TA_JUSTIFY, spaceAfter=4,
    firstLineIndent=10,
)
style_body_noind = ParagraphStyle(
    "BodyNoInd", fontName=KOR, fontSize=10, textColor=TEXT_BODY,
    leading=16, alignment=TA_JUSTIFY, spaceAfter=4,
)
style_bullet = ParagraphStyle(
    "Bullet", fontName=KOR, fontSize=10, textColor=TEXT_BODY,
    leading=15.5, leftIndent=18, bulletIndent=8, spaceAfter=3,
    alignment=TA_JUSTIFY,
)
style_meta_label = ParagraphStyle(
    "MetaLabel", fontName=KORB, fontSize=10, textColor=TITLE_BLUE,
)
style_meta_value = ParagraphStyle(
    "MetaValue", fontName=KOR, fontSize=10, textColor=TEXT_BODY, leading=15,
)
style_claim_title = ParagraphStyle(
    "ClaimTitle", fontName=KORB, fontSize=10.5, textColor=TITLE_BLUE,
    spaceAfter=5,
)
style_claim_body = ParagraphStyle(
    "ClaimBody", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY,
    leading=15, spaceAfter=3, leftIndent=4, alignment=TA_JUSTIFY,
)
style_fig_title = ParagraphStyle(
    "FigTitle", fontName=KORB, fontSize=10.5, textColor=TITLE_BLUE,
    alignment=TA_CENTER, spaceAfter=4,
)
style_fig_body = ParagraphStyle(
    "FigBody", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY,
    alignment=TA_CENTER, leading=14.5,
)

# ── 헬퍼: 박스로 감싼 영역 ────────────────────────────────
def boxed(flowables, bg=LIGHT_BG, border=LIGHT_BORDER,
          pad_l=12, pad_r=12, pad_t=10, pad_b=10):
    t = Table([[flowables]], colWidths=[None])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 0.7, border),
        ("LEFTPADDING",   (0, 0), (-1, -1), pad_l),
        ("RIGHTPADDING",  (0, 0), (-1, -1), pad_r),
        ("TOPPADDING",    (0, 0), (-1, -1), pad_t),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad_b),
    ]))
    return t


def title_meta_box():
    """발명의 명칭 박스 (제목 + 국문/영문)"""
    inner = [
        Paragraph("특허출원 명세서", ParagraphStyle(
            "x", fontName=KORB, fontSize=15, textColor=TITLE_BLUE,
            alignment=TA_CENTER, spaceAfter=2)),
        Paragraph("(변화탐지 시계열 페어링 및 GraphRAG 누적 인덱싱 기반)", ParagraphStyle(
            "x2", fontName=KOR, fontSize=10.5, textColor=TEXT_BODY,
            alignment=TA_CENTER, spaceAfter=10)),
        meta_row("발명의 명칭 (국문)",
                 "시계열 항공 영상의 객체 페어링 변화 탐지 및 지식 그래프 누적 인덱싱 기반 "
                 "AI 에이전트 상황 판독 보고서 자율 생성 시스템 및 방법"),
        meta_row("발명의 명칭 (영문)",
                 "System and Method for Automated Situational Intelligence Report Generation "
                 "via AI Agent Combining Time-Series Object Pairing-Based Change Detection and "
                 "Knowledge-Graph Accumulative Indexing on Aerial Imagery"),
    ]
    return boxed(inner, bg=HexColor("#f0f9ff"), border=LIGHT_BORDER,
                 pad_l=18, pad_r=18, pad_t=14, pad_b=14)


def meta_row(label, value):
    t = Table(
        [[Paragraph(label, style_meta_label),
          Paragraph(value, style_meta_value)]],
        colWidths=[3.8 * cm, None],
    )
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def h1(text):
    """좌측에 파란 세로 막대가 있는 H1 (PDF 양식의 ▍ 스타일)"""
    bar = Table(
        [[" ", Paragraph(text, style_h1)]],
        colWidths=[5, None],
    )
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), TITLE_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return KeepTogether([Spacer(1, 6), bar, Spacer(1, 2)])


def h2(text):
    return Paragraph(f"<b>({text})</b>" if text.startswith(("1)", "2)")) else f"<b>{text}</b>",
                     style_h2)


def h3(text):
    return Paragraph(f"<b>{text}</b>", style_h3)


def para(text, indent=True):
    return Paragraph(text, style_body if indent else style_body_noind)


def bullet(text):
    return Paragraph(f"• {text}", style_bullet)


def claim_box(title, body_paragraphs):
    inner = [Paragraph(title, style_claim_title)]
    for b in body_paragraphs:
        inner.append(Paragraph(b, style_claim_body))
    return boxed(inner, bg=HexColor("#f8fafc"), border=LIGHT_BORDER,
                 pad_l=12, pad_r=12, pad_t=10, pad_b=10)


def fig_box(title, body):
    inner = [Paragraph(f"[{title}]", style_fig_title),
             Paragraph(body, style_fig_body)]
    return boxed(inner, bg=HexColor("#f8fafc"), border=LIGHT_BORDER,
                 pad_l=14, pad_r=14, pad_t=10, pad_b=10)


# ── 페이지 번호 ────────────────────────────────────────────
def draw_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont(KOR, 9)
    canvas.setFillColor(MUTED)
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"- {page_num} -")
    canvas.restoreState()


# ── 문서 빌드 ──────────────────────────────────────────────
def build():
    out_path = "/home/user/multi-source-intelligent-system/data/patent_change_detection_graphrag.pdf"
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.0 * cm, bottomMargin=2.2 * cm,
    )

    story = []

    # ╔═══ 제목 박스 ═══╗
    story.append(title_meta_box())
    story.append(Spacer(1, 8))

    # ╔═══ 1. 개요 ═══╗
    story.append(h1("1. 개요"))

    story.append(h2("(1) 직무발명 수탁과제 무관 근거 설명"))

    story.append(h3("1) 제안 발명 내용 요약"))
    story.append(para(
        "본 발명은 시계열 항공 영상에 대해 (i) 텍스트 프롬프트 기반 객체 탐지 모델로 자산을 식별하고, "
        "(ii) <b>객체 마스크 기반 시각 임베딩 코사인 유사도와 Gale-Shapley 안정 매칭으로 객체 단위 "
        "시계열 페어링 변화 탐지</b>를 수행하며, (iii) <b>탐지·페어링 결과를 LLM 호출 없이 결정론적으로 "
        "공간 격자 지식 그래프에 누적 인덱싱(GraphRAG)</b>하여 누적 작전 패턴을 마이닝한 후, "
        "(iv) 변화 객체와 그래프 역사적 컨텍스트를 단일 LLM 에이전트에 주입하여 표준 IMINT 8개 섹션의 "
        "구조화된 상황 판독 보고서를 자율 생성하는 시스템입니다."
    ))

    story.append(h3("2) 중복성 및 차별성 설명"))
    story.append(bullet(
        "<b>중복성 설명</b>: 본 발명의 카테고리(LLM 기반 다중 출처 정보 융합 자동 보고서)에 가장 직접적인 "
        "선행 등록 특허로 미국 등록 특허 US 12,488,225 B1 (Booz Allen Hamilton, 2025)가 존재하나, "
        "동 특허는 다중 출처 정보를 추상적 공통 인텔리전스 픽처(CIP)로 융합하고 분석관의 PIR "
        "질의응답에 응답하는 구조이며, 본 발명의 핵심인 객체 단위 시계열 페어링과 그래프 카운터 "
        "누적 메모리 구조와는 본질적으로 상이합니다."
    ))
    story.append(bullet(
        "<b>차별성 설명</b>: 본 발명은 (가) <b>SAM 마스크로 배경을 제거한 객체 전용 임베딩</b>과 "
        "<b>Gale-Shapley 안정 매칭</b>으로 객체 단위 시계열 변화(new/matched/moved/disappeared)를 "
        "정밀 분류하고, (나) <b>LLM 호출 없이 결정론적으로 (객체 클래스 × 위치 격자) 노드와 공출현 "
        "엣지를 누적 인덱싱</b>하여 GraphRAG 지식 그래프를 구축하며, (다) <b>Louvain 알고리즘으로 "
        "자산 doctrine 군집을 자동 발견</b>하고, (라) <b>Local + Global 이중 검색</b>으로 압축된 "
        "역사적 컨텍스트 블록을 LLM 프롬프트에 prepend하여 환각을 차단하는 점에서 차별됩니다."
    ))

    story.append(h2("(2) 발명 제안 배경"))
    story.append(para(
        "현대 위성·드론 정찰 체계에서는 다중 소스로부터 수집되는 방대한 영상 정보(IMINT)를 신속 분석하여 "
        "지휘관에게 표준 판독 보고서 형태로 제공하는 것이 핵심 임무이다. 그러나 종래의 변화 탐지 "
        "시스템들은 영상 내 자산의 단순 출현·소실 등 이진 매칭만을 도출하는 데 그쳐, 실제 작전 환경에서 "
        "요구되는 표준 IMINT 보고서를 작성하기 위해서는 분석관이 수작업으로 맥락을 분석해야 하므로 "
        "지휘 결심이 지연되는 문제가 있었다."
    ))
    story.append(para(
        "최근 거대언어모델(LLM)을 보고서 생성에 활용하려는 시도가 있으나, (i) 영상 내 객체의 낱개 "
        "관측 정보만을 LLM에 주입할 경우 모델이 시계열 누적 맥락을 파악하지 못해 \"이번 프레임만 보는\" "
        "단발 보고에 그치며, (ii) 동일 객체가 반복 관측될 때 미세 GPS 오차로 분산되어 누적 통계를 형성하지 "
        "못하고, (iii) LLM이 정형 정보를 자연어로 풀어 재해석하는 과정에서 환각(hallucination)이 발생하는 "
        "치명적 한계가 있었다. 본 발명은 이러한 한계를 극복하기 위해 객체 단위 시계열 페어링 변화 탐지와 "
        "결정론적 지식 그래프 누적 인덱싱을 결합한 AI 에이전트 시스템을 제안한다."
    ))

    story.append(h2("(3) 산업상 이용 분야 (용도)"))
    story.append(bullet(
        "<b>국방 및 방위산업</b>: 위성·정찰기 IMINT 자산 시계열 변화 판독 및 표준 8섹션 보고서 자율 생성 체계"
    ))
    story.append(bullet(
        "<b>우주안보 및 국경 감시</b>: 전장·접경 지역 적대 자산 누적 활동 패턴 탐지 및 doctrine 패턴 자동 식별"
    ))
    story.append(bullet(
        "<b>민간 재난/재해 통제</b>: 시계열 위성 관측 데이터 기반 국토 재해 시설물 파손 추이 자동 판독 보고서"
    ))
    story.append(bullet(
        "<b>스마트시티 및 도시 변화 모니터링</b>: 도시 인프라·교통·시설물의 시계열 변화 자동 보고"
    ))

    story.append(h2("(4) 종래 기술"))
    story.append(para(
        "본 발명의 직접적 종래 기술은 미국 특허청에 정식 등록된 선행 특허인 "
        "<b>US 12,488,225 B1 \"Modular open system architecture for common intelligence picture "
        "generation\" (양수인: Booz Allen Hamilton Inc., 등록 2025)</b>이다. 동 종래 기술은 위성·OSINT·"
        "CYBER·SIGINT 등 다중 출처 정보를 모듈형 개방 시스템 아키텍처(MOSA) 기반으로 통합하여 "
        "공통 인텔리전스 픽처(Common Intelligence Picture, CIP)를 생성하고, 그 위에 다중 정보 융합 "
        "LLM 엔진을 도구 사용 에이전트로 활용하여 분석관이 자연어로 입력한 우선 정보 요구사항"
        "(Priority Intelligence Requirements, PIR)에 응답해 위성 임무 부여(satellite tasking), 대응 방안"
        "(courses of action, COA)을 자동 생성하고, 컨테이너화된 분석 워크벤치와 연동해 PDF/PPT 형태의 "
        "맞춤형 정보 산출물을 자동 생성하는 아키텍처를 개시하고 있다."
    ))

    story.append(h2("(5) 종래 기술의 문제점 / 한계"))

    problems = [
        ("객체 단위 시계열 페어링 변화 탐지 메커니즘 부재",
         "종래 기술(US 12,488,225)은 다중 출처 정보를 추상적 CIP로 융합하는 데 초점이 있으며, "
         "영상 내 개별 객체(전차·항공기·시설 등) 단위로 두 시점 간 매칭하여 "
         "new/matched/moved/disappeared로 분류하는 시계열 페어링 알고리즘이 정의되어 있지 않다. "
         "결과적으로 \"어떤 자산이 어디로 이동했는가\"와 같은 객체 추적 단위 정보가 산출되지 못한다."),
        ("동일 자산 반복 관측의 누적 메모리 부재",
         "종래 기술의 CIP는 \"현재 시점의 스냅샷\" 구조로, 동일 자산이 다른 세션에 걸쳐 반복 관측될 때 "
         "이를 단일 노드의 누적 카운터(new_count, disappeared_count, total_confidence 등)로 압축 "
         "저장하는 결정론적 그래프 인덱싱이 부재하다. 따라서 \"이 지역에서 N번째 반복 출현\" 같은 "
         "시계열 통계가 단일 조회로 산출되지 못한다."),
        ("자산 간 공출현 패턴 자동 발견(Louvain) 부재",
         "종래 기술은 LLM이 PIR에 응답해 데이터를 조회·요약하는 구조이며, 관측 데이터 자체로부터 "
         "자산 클래스 간 공출현 패턴(예: 기갑+APC+포병 = 기갑 복합체, 레이더+지휘소 = C2 인프라)을 "
         "그래프 알고리즘으로 자동 군집화·발견하는 구성이 정의되어 있지 않다."),
        ("환각(Hallucination) 차단을 위한 사전 압축 컨텍스트 블록 부재",
         "종래 기술은 LLM 에이전트가 도구 호출을 통해 그때그때 데이터를 조회·요약하므로 LLM 호출 "
         "횟수·토큰 비용·응답 지연이 가변적이고, LLM이 비결정적으로 추론한 정보가 보고서에 혼입되어 "
         "환각이 발생할 수 있다. 본 발명은 그래프에서 사전 압축된 ~500토큰 historical context 블록을 "
         "프롬프트 최상단에 prepend하여 환각을 원천 차단한다."),
        ("영상 변화 탐지 도메인 의미 가드레일 부재",
         "종래 기술은 범용 정보 융합 LLM으로, 영상 변화 탐지 특유의 의미 제약 — \"영상에서 객체가 더 "
         "이상 관측되지 않음(DISAPPEARED)\"을 \"파괴 확인(destroyed)\"으로 표현해서는 안 됨 — 을 "
         "시스템 프롬프트 차원에서 강제하는 구성이 정의되어 있지 않아 군사적 오판 가능성이 있다."),
        ("표준 IMINT 8섹션 강제 및 동일 LLM 번역 파이프라인 부재",
         "종래 기술의 보고서는 분석관 지정 PDF/PPT 형식으로 유동적이다. 본 발명은 CLASSIFICATION / "
         "EXECUTIVE SUMMARY / SITUATION / CHANGE ANALYSIS / THREAT ASSESSMENT / INTELLIGENCE GAPS / "
         "RECOMMENDED ACTIONS / APPENDIX의 8개 표준 IMINT 섹션을 시스템 프롬프트 차원에서 강제하고, "
         "동일 LLM 재호출 기반 한국어 번역(좌표·신뢰도 토큰 보존)으로 다국어 분석관 표준 양식 일관성을 "
         "확보한다."),
    ]
    for i, (t, d) in enumerate(problems, 1):
        story.append(bullet(f"<b>{i}. {t}</b>: {d}"))

    # ╔═══ 2. 상세 설명 ═══╗
    story.append(h1("2. 상세 설명"))

    story.append(h2("(1) 발명의 핵심 구성 및 작동 원리"))
    story.append(para(
        "본 발명은 하드웨어 컴퓨터 장치 또는 온디바이스 엣지 AI 하드웨어 플랫폼 상에서 구동되는 "
        "시스템으로서, 파이프라인 흐름에 따라 [구성 A] 시계열 객체 페어링 기반 변화 탐지부, "
        "[구성 B] GraphRAG 결정론적 누적 인덱싱 및 Louvain 군집 탐지부, [구성 C] 변화 객체 중심 "
        "LLM 에이전트 보고서 생성부의 3개 기능 모듈로 유기적으로 결합된다."
    ))

    story.append(h3("[구성 A] 시계열 객체 페어링 변화 탐지부 (Step 1 — 변화 탐지 차별성 원리)"))
    story.append(para(
        "본 구성은 현재 프레임 탐지 결과와 동일 지역의 직전 프레임 탐지 결과를 객체 단위로 매칭하여 "
        "시계열 변화 상태를 분류한다. 본 발명의 변화 탐지 원리는 단순 좌표 오버랩 비교를 넘어 "
        "다음의 차별 알고리즘을 수행한다.", indent=False
    ))
    story.append(bullet(
        "<b>SAM 마스크 기반 객체 전용 시각 임베딩</b>: 각 탐지 객체에 대해 SAM3 등이 산출한 "
        "세그멘테이션 마스크 RLE를 디코딩하여 배경 픽셀을 영(0)으로 마스킹(zeroing)한 후 4픽셀 "
        "패딩의 바운딩 박스로 크롭하고, CLIP/ViT 등의 비전 인코더에 입력하여 L2 정규화된 D차원 "
        "임베딩 벡터를 산출한다. 배경 제거를 통해 동일 클래스 객체 간 외형 식별력을 종래 대비 향상시켜 "
        "오매칭 발생률을 낮춘다."
    ))
    story.append(bullet(
        "<b>동일 클래스 N×M 유사도 행렬 및 Gale-Shapley 안정 매칭</b>: 현재 N개 객체와 과거 M개 "
        "객체의 임베딩을 한 번의 행렬곱으로 N×M 코사인 유사도 행렬을 산출한 후, 동일 클래스 쌍에 "
        "대해 임베딩 유사도와 크기 유사도의 가중합 점수를 산출하고, 점수 임계값 이상의 후보 쌍에 "
        "대해 <b>Gale-Shapley 지연 승인(deferred acceptance) 알고리즘</b>으로 1:1 안정 매칭을 수행한다. "
        "이를 통해 그리디 매칭의 국소 최적해 한계를 극복하고 중복·교차 페어링을 원천 배제한다."
    ))
    story.append(bullet(
        "<b>변화 상태 분류</b>: 매칭된 쌍의 지리 좌표 거리가 이동 임계값(≈100m) 미만이면 정지"
        "(matched), 이상이면 이동(moved), 미매칭 현재 객체는 신규 출현(new), 미매칭 과거 객체는 "
        "소실(disappeared)로 분류한다. 부수적으로 영상의 시야각(FOV) 경계면 외부에 위치한 객체는 "
        "촬영 공백(past_not_included / current_not_included)으로 별도 분류하여 단순 촬영 구역 "
        "불일치로 인한 오판을 배제한다."
    ))
    story.append(bullet(
        "<b>듀얼 모드 지원</b>: 영상 간 시간 간격이 짧을 때는 SAM3 비디오 추적기 기반 ID 추적 모드를, "
        "간격이 길거나 카메라 각도가 크게 변할 때는 상기 임베딩 유사도 모드를 자동 선택하여 적용한다."
    ))

    story.append(h3("[구성 B] GraphRAG 결정론적 누적 인덱싱 및 군집 탐지부 (Step 2 — GraphRAG 차별성 원리)"))
    story.append(para(
        "본 구성은 거대언어모델 파라미터 외부에 시공간적 토폴로지 지식 기단을 형성하여 시계열 누적 "
        "패턴 마이닝과 LLM 환각 차단을 동시에 달성한다.", indent=False
    ))
    story.append(bullet(
        "<b>격자 양자화 기반 결정론적 노드 키 생성</b>: 객체의 위경도를 소수점 2자리(약 1km 격자)로 "
        "양자화하여 <b>loc:{lat:.2f},{lon:.2f}</b> 형식의 위치 노드(Location Node) 키를, "
        "(객체 클래스 × 위치 키)로 <b>asset:{class}:{loc_key}</b> 형식의 자산 노드(Asset Node) 키를 "
        "결정론적으로 생성한다. 이로써 GPS 측정 오차에 따른 미세 좌표 변동이 동일 자산으로 자동 "
        "통합되며, 동일 입력에 대해 매번 동일한 그래프가 산출된다."
    ))
    story.append(bullet(
        "<b>LLM 호출 없는 카운터 누적 인덱싱</b>: 페어링 레코드의 status 필드(new/matched/moved/"
        "disappeared)와 신뢰도를 자산 노드의 카운터 속성(new_count, matched_count, moved_count, "
        "disappeared_count, total_confidence)에 결정론적으로 가산하고, first_seen·last_seen·sessions 등 "
        "시간 속성을 갱신한다. asset → location의 found_at 엣지와 동일 세션·동일 격자 내 공출현 자산 "
        "쌍의 co_occurred_with 엣지의 가중치 count를 누적한다. <b>전 과정에서 LLM 호출이 일체 발생하지 "
        "않아 인덱싱 비용·지연·재현성 모두 우위이다.</b>"
    ))
    story.append(bullet(
        "<b>Louvain 알고리즘 기반 자산 군집 자동 탐지</b>: co_occurred_with 엣지 가중치에 대해 "
        "Louvain 계층적 군집화 알고리즘을 구동하여 별도 LLM 연산 없이 자산 doctrine 군집"
        "(예: 기갑+APC+포병 = 기갑 복합체, 레이더+지휘소 = C2 인프라, 항공기+활주로+연료고 = 항공 "
        "작전 거점)을 자동 발견하고, 각 군집의 구성원 자산 분포·위치 클러스터 수·관측·신규·소실 "
        "통계를 구조적 member_summary로 자동 생성한다."
    ))
    story.append(bullet(
        "<b>Local + Global 이중 검색 및 사전 압축 컨텍스트 블록 생성</b>: 보고서 대상 좌표 반경 R 내 "
        "자산 노드 이력 통계를 조회하는 Local Search와, 동일 반경과 중첩되는 군집의 member_summary를 "
        "조회하는 Global Search를 병합하여 사전 설정 토큰 예산(예: ~500 tokens) 이내의 결정론적 "
        "historical context 블록을 생성한다. 이 블록은 [구성 C]의 LLM 프롬프트에 prepend되어 "
        "LLM의 환각을 원천 차단한다."
    ))

    story.append(h3("[구성 C] 변화 객체 중심 AI 에이전트 보고서 생성부 (Step 3)"))
    story.append(bullet(
        "<b>변화 객체 한정 추출</b>: 페어링 결과 중 활동 지표인 'new'와 'disappeared' 상태 객체만을 "
        "신뢰도 내림차순으로 정렬하여 상위 N건 + 잔여 요약 형태로 추출한다. 'matched'(정지) 및 "
        "'moved'(이동) 객체는 LLM 프롬프트에서 제외되어 토큰 효율과 분석 집중도가 동시에 확보된다."
    ))
    story.append(bullet(
        "<b>군사 도메인 의미 가드레일</b>: 시스템 프롬프트에 <b>\"'DISAPPEARED'는 영상에서의 미관측을 "
        "의미하며 파괴 확인(destroyed)이 아니다\"</b>는 의미 제약을 명시적으로 강제 주입하여 LLM의 "
        "군사적 오해석을 차단한다."
    ))
    story.append(bullet(
        "<b>표준 IMINT 8섹션 강제</b>: (1) CLASSIFICATION (2) EXECUTIVE SUMMARY (3) SITUATION "
        "(4) CHANGE ANALYSIS (5) THREAT ASSESSMENT (6) INTELLIGENCE GAPS (7) RECOMMENDED ACTIONS "
        "(8) APPENDIX의 8개 표준 IMINT 섹션 구조를 LLM 프롬프트에 강제하여 분석관 친화 산출물을 보장한다."
    ))
    story.append(bullet(
        "<b>동일 LLM 인스턴스 재사용 무오염 번역</b>: 1차로 영문 보고서를 생성한 후 동일 모델 인스턴스를 "
        "인메모리 상태로 재사용하여 한국어 번역 세션을 발동하며, 섹션 헤더·좌표·타임스탬프·클래스명·"
        "신뢰도 토큰은 변형 없이 바이패스(bypass)하고 분석 서술문만을 한국어로 변환하여 별도의 번역 "
        "모델 없이 다국어 일관성을 확보한다."
    ))

    story.append(h2("(2) 발명의 효과"))
    effects = [
        ("객체 단위 시계열 변화 탐지의 정밀화",
         "SAM 마스크 기반 배경 제거 임베딩과 Gale-Shapley 안정 매칭의 결합으로 종래 그리디 매칭 대비 "
         "오매칭률을 현저히 감소시키며, 객체별 new/matched/moved/disappeared 상태를 정확하게 분류한다."),
        ("결정론·재현성·감사 가능성 확보",
         "GraphRAG 인덱싱이 LLM 비호출 결정론 방식으로 동작하여 동일 입력에 동일 그래프가 산출되고, "
         "엣지 1행 = 1근거의 명시적 구조로 군사·법적 감사 요구를 충족한다."),
        ("시계열 인텔리전스 격상",
         "단발 프레임 변화 보고가 아닌 \"N번째 반복 출현, K번 소실 후 재등장\" 수준의 시계열 패턴 보고와 "
         "Louvain doctrine 군집 패턴(기갑 복합체·C2 인프라 등) 자동 식별로 보고서 품질이 격상된다."),
        ("LLM 환각 원천 차단 및 토큰 효율 극대화",
         "사전 압축된 ~500토큰의 결정론적 historical context 블록과 변화 객체 한정 전달, "
         "DISAPPEARED 의미 가드레일의 결합으로 환각을 원천 차단하면서 LLM 토큰 비용을 절감한다."),
        ("운용 비용 절감 및 다국어 일관성",
         "별도의 벡터 DB·임베딩 GPU·번역 모델 없이 SQLite + 그래프 알고리즘 + 단일 LLM 재호출만으로 "
         "전체 파이프라인이 동작하여 운용 비용·복잡도를 절감하고, 좌표 토큰 보존 규칙으로 다국어 "
         "보고서의 형식 일관성을 확보한다."),
    ]
    for i, (t, d) in enumerate(effects, 1):
        story.append(bullet(f"<b>{i}. {t}</b>: {d}"))

    # ╔═══ 3. 특허 청구범위 ═══╗
    story.append(h1("3. 특허 청구범위"))

    # 청구항 1
    story.append(claim_box(
        "[청구항 1] (독립항)",
        [
            "임의의 대상 지역을 촬영한 시계열 항공 영상 데이터를 수집하는 단계;",
            "입력된 최신 영상 및 과거 영상 각각에 대하여 텍스트 프롬프트 기반 객체 탐지 알고리즘을 "
            "적용하여 자산 클래스, 위경도 좌표, 바운딩 박스, 세그멘테이션 마스크를 포함하는 탐지 "
            "레코드를 생성하는 단계;",
            "각 탐지 객체에 대해 상기 세그멘테이션 마스크로 배경 픽셀을 영(0)으로 처리한 후 바운딩 "
            "박스 크롭 이미지로부터 비전 인코더에 의해 L2 정규화된 시각 임베딩 벡터를 산출하고, "
            "최신 영상의 N개 임베딩과 과거 영상의 M개 임베딩의 N×M 코사인 유사도 행렬을 산출하여 "
            "동일 클래스 쌍에 대해 임베딩 유사도와 크기 유사도의 가중합 점수가 임계값 이상인 후보 "
            "쌍에 대하여 Gale-Shapley 지연 승인 알고리즘으로 1:1 안정 매칭을 수행한 후, 매칭된 쌍의 "
            "지리 좌표 거리에 따라 정지(matched) 또는 이동(moved)을, 미매칭 현재 객체에는 신규 "
            "출현(new)을, 미매칭 과거 객체에는 소실(disappeared)을 부여하는 객체 단위 시계열 변화 "
            "상태 분류 단계;",
            "각 자산의 위경도 좌표를 정수형 격자 단위로 양자화하여 위치 노드 키를, 객체 클래스와 "
            "위치 키의 결합으로 자산 노드 키를 결정론적으로 생성하고, 상기 변화 상태에 따라 자산 "
            "노드의 누적 카운터 속성을 거대언어모델(LLM) 호출 없이 갱신하며, 자산-위치 간 found_at "
            "엣지 및 동일 세션·동일 격자 내 공출현 자산 쌍의 co_occurred_with 엣지의 가중치를 누적 "
            "갱신하여 지식 그래프를 형성하는 단계;",
            "상기 co_occurred_with 엣지 가중치를 기반으로 Louvain 계층적 군집화 알고리즘을 수행하여 "
            "자산 복합체 커뮤니티 요약문을 LLM 호출 없이 구조적으로 생성하고, 보고서 대상 좌표 반경 "
            "내 자산 노드 이력을 조회하는 지역 검색(Local Search)과 동일 반경과 중첩되는 커뮤니티 "
            "요약을 조회하는 전역 검색(Global Search)을 병합하여 사전 설정 토큰 예산 이내의 역사적 "
            "맥락 컨텍스트 블록을 생성하는 단계; 및",
            "상기 변화 상태가 신규 출현(new) 또는 소실(disappeared)에 해당하는 객체만을 신뢰도 "
            "내림차순으로 추출하고, 상기 역사적 맥락 컨텍스트 블록을 LLM 프롬프트의 선두에 "
            "prepend하며, 시스템 프롬프트에 \"'DISAPPEARED'는 영상 미관측을 의미하며 파괴 확인이 "
            "아니다\"라는 의미 제약과 표준 IMINT 8개 섹션 구조를 강제 주입하여 LLM 에이전트가 "
            "구조화된 상황 판독 보고서를 자동 생성하는 단계;",
            "를 포함하는 시계열 항공 영상의 객체 페어링 변화 탐지 및 지식 그래프 누적 인덱싱 기반 "
            "AI 에이전트 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 2
    story.append(claim_box(
        "[청구항 2] (종속항) — 마스크 기반 배경 제거 임베딩",
        [
            "제1항에 있어서, 상기 시계열 변화 상태 분류 단계의 임베딩 산출은,",
            "탐지된 자산의 세그멘테이션 마스크 RLE 정보를 디코딩하여 마스크 활성 영역 외의 픽셀을 "
            "영(0)으로 마스킹하고 바운딩 박스 주위에 사전 설정된 패딩 폭을 부가하여 크롭하는 단계; "
            "마스크 RLE가 부재하는 경우 바운딩 박스 크롭으로 폴백하는 단계; 및 산출된 임베딩 벡터를 "
            "L2 정규화하여 코사인 유사도 행렬 산출에 사용하는 단계를 포함하는 것을 특징으로 하는 "
            "AI 에이전트 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 3
    story.append(claim_box(
        "[청구항 3] (종속항) — Gale-Shapley 안정 매칭 + 듀얼 모드",
        [
            "제1항에 있어서, 상기 시계열 변화 상태 분류 단계는,",
            "(a) 임베딩 모델이 가용한 경우 임베딩 코사인 유사도 가중치 α 및 크기 유사도 가중치 "
            "(1−α)의 가중합을 점수로 사용하고, (b) 임베딩 모델이 부재한 경우 지리 좌표 근접도 점수로 "
            "폴백하며, (c) 점수가 임계값을 초과하는 후보 쌍에 대해 Gale-Shapley 지연 승인 알고리즘으로 "
            "1:1 안정 매칭을 수행하고, (d) 영상 간 시간 간격이 사전 설정 임계값 이하일 때는 비디오 "
            "추적기 기반 ID 추적 모드를 선택적으로 사용하는 듀얼 모드 구조를 갖는 것을 특징으로 하는 "
            "AI 에이전트 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 4 (★ 핵심)
    story.append(claim_box(
        "[청구항 4] (종속항) — LLM 비호출 결정론적 그래프 인덱싱 ★",
        [
            "제1항에 있어서, 상기 지식 그래프 형성 단계는,",
            "각 자산의 위경도 좌표를 소수점 N자리의 정수형 격자 단위로 반올림하여 \"loc:{lat:.Nf},"
            "{lon:.Nf}\" 형식의 결정론적 위치 노드 키를 생성하고, 객체 클래스 및 위치 키를 결합하여 "
            "\"asset:{class}:{loc_key}\" 형식의 결정론적 자산 노드 키를 생성하는 단계;",
            "페어링 레코드의 상태 필드 및 신뢰도를 자산 노드의 카운터 속성(new_count, matched_count, "
            "moved_count, disappeared_count, total_confidence) 및 시간 속성(first_seen, last_seen, "
            "sessions)에 LLM 호출 없이 결정론적으로 누적 가산하는 upsert 단계; 및",
            "자산-위치 간 found_at 엣지의 count 속성, 동일 세션·동일 위치 키 내 공출현 자산 쌍의 "
            "co_occurred_with 엣지의 count 속성 및 locations 속성을 LLM 호출 없이 결정론적으로 "
            "누적 갱신하는 단계;",
            "를 포함하여, 동일 입력에 대해 매번 동일한 그래프가 산출되는 결정론적 재현성을 갖는 "
            "것을 특징으로 하는 AI 에이전트 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 5
    story.append(claim_box(
        "[청구항 5] (종속항) — Louvain 군집 + member_summary 자동 생성",
        [
            "제1항에 있어서, 상기 Louvain 군집화 수행 단계는,",
            "co_occurred_with 엣지의 count 가중치를 입력으로 하여 NetworkX 등의 그래프 라이브러리 "
            "상에서 Louvain 계층적 군집화 알고리즘을 수행하는 단계; 및 각 커뮤니티에 대해 구성원 "
            "자산 클래스 분포·위치 클러스터 수·총 관측 수·신규 배치 수·소실 수를 포함하는 "
            "member_summary를 LLM 호출 없이 결정론적 통계 산식만으로 생성하여 그래프 데이터베이스에 "
            "기록하는 단계를 포함하는 것을 특징으로 하는 AI 에이전트 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 6
    story.append(claim_box(
        "[청구항 6] (종속항) — Local + Global 이중 검색 및 토큰 예산",
        [
            "제1항에 있어서, 상기 역사적 맥락 컨텍스트 블록 생성 단계는,",
            "보고서 대상 좌표를 중심으로 사전 설정 반경 R 이내의 자산 노드들의 이력 통계(누적 관측 "
            "수, 신규/지속/이동/소실 횟수, 평균 신뢰도, 최초/최종 관측 시각)를 조회하는 지역 검색"
            "(Local Search) 단계;",
            "동일 반경과 중첩되는 커뮤니티들의 member_summary를 조회하는 전역 검색(Global Search) "
            "단계; 및",
            "양 검색 결과를 사전 설정 토큰 예산(예: 500 토큰) 이내로 절단·머지하여 LLM 프롬프트 "
            "선두에 prepend되는 단일 컨텍스트 블록으로 포맷화하는 단계;",
            "를 포함하는 것을 특징으로 하는 AI 에이전트 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 7
    story.append(claim_box(
        "[청구항 7] (종속항) — 변화 객체 한정 + IMINT 8섹션 + 동일 LLM 번역",
        [
            "제1항에 있어서, 상기 보고서 자동 생성 단계는,",
            "페어링 결과 중 'matched' 및 'moved' 상태 객체는 LLM 프롬프트에서 제외하고 'new' 및 "
            "'disappeared' 상태 객체만을 신뢰도 내림차순으로 정렬하여 상위 N건 및 잔여 요약 형태로 "
            "프롬프트에 주입하는 단계;",
            "시스템 프롬프트에 (1) CLASSIFICATION (2) EXECUTIVE SUMMARY (3) SITUATION (4) CHANGE "
            "ANALYSIS (5) THREAT ASSESSMENT (6) INTELLIGENCE GAPS (7) RECOMMENDED ACTIONS "
            "(8) APPENDIX의 8개 표준 IMINT 섹션 구조와 \"'DISAPPEARED'는 영상 미관측이며 파괴 확인이 "
            "아니다\"라는 의미 제약을 강제 주입하는 단계; 및",
            "동일 LLM 모델 인스턴스를 인메모리 상태로 재사용하여 영문 보고서 생성 후 한국어 번역 "
            "세션을 발동하되, 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 토큰은 변형 없이 바이패스"
            "(bypass)하고 분석 서술문만을 한국어로 변환하는 단계;",
            "를 포함하는 것을 특징으로 하는 AI 에이전트 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 8 (부수적 — FOV)
    story.append(claim_box(
        "[청구항 8] (종속항) — 부수적 FOV 검증 가드",
        [
            "제1항에 있어서, 상기 객체 단위 시계열 변화 상태 분류 단계는,",
            "최신 영상 및 과거 영상 각각의 메타데이터로부터 위경도 경계면(field-of-view) 정보를 "
            "추출하는 단계; 및 미매칭 현재 객체가 과거 영상의 경계면 외부에 위치하는 경우 'past_not_"
            "included' 상태를, 미매칭 과거 객체가 최신 영상의 경계면 외부에 위치하는 경우 'current_not_"
            "included' 상태를 부여하여 단순 촬영 범위 불일치에 따른 관측 공백을 실제 변화와 구분하는 "
            "단계를 부수적으로 더 포함하는 것을 특징으로 하는 AI 에이전트 상황 판독 보고서 자율 생성 "
            "방법.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 9
    story.append(claim_box(
        "[청구항 9] (독립항) — 시스템 청구",
        [
            "제1항 내지 제8항 중 어느 한 항의 방법을 수행하는 하나 이상의 하드웨어 프로세서와 메모리를 "
            "포함하는 AI 에이전트 기반 항공 영상 상황 판독 보고서 자율 생성 시스템.",
        ],
    ))
    story.append(Spacer(1, 6))

    # 청구항 10
    story.append(claim_box(
        "[청구항 10] (독립항) — 기록 매체",
        [
            "제1항 내지 제8항 중 어느 한 항의 방법을 컴퓨터에서 실행시키기 위한 프로그램이 기록된 "
            "컴퓨터 판독 가능 기록 매체.",
        ],
    ))

    # ╔═══ 4. 도면 설명 ═══╗
    story.append(h1("4. 도면 설명 명기"))

    story.append(fig_box(
        "도 1] 시스템 전체 구성 아키텍처",
        "위성·드론 영상 적재 → SAM3 등 텍스트 프롬프트 객체 탐지 → "
        "마스크 기반 배경 제거 임베딩 및 Gale-Shapley 안정 매칭 시계열 페어링 → "
        "결정론적 지식 그래프 누적 인덱싱 및 Louvain 커뮤니티 탐지 → "
        "Local + Global 이중 검색 컨텍스트 블록 생성 → "
        "LLM 에이전트 IMINT 8섹션 보고서 자율 생성 → "
        "동일 LLM 인메모리 무오염 한국어 번역 파이프라인 관계도"
    ))
    story.append(Spacer(1, 8))

    story.append(fig_box(
        "도 2] 객체 단위 시계열 페어링 변화 탐지 메커니즘 (변화 탐지 차별성)",
        "SAM 마스크 RLE 디코딩 → 배경 픽셀 영(0) 처리 → 순수 객체 bbox 크롭 → "
        "CLIP/ViT 비전 인코더 임베딩 및 L2 정규화 → N×M 코사인 유사도 행렬 단일 GEMM 산출 → "
        "동일 클래스 쌍 가중합 점수화 → Gale-Shapley 지연 승인 1:1 안정 매칭 → "
        "geo_dist 임계값 기반 matched/moved 구분 → 미매칭 객체 new/disappeared 분류 흐름도"
    ))
    story.append(Spacer(1, 8))

    story.append(fig_box(
        "도 3] GraphRAG 결정론적 누적 인덱싱 및 Louvain 군집 탐지 (GraphRAG 차별성)",
        "위경도 격자 양자화(소수점 2자리 ≈ 1km) → \"loc:lat,lon\" / \"asset:class:loc\" 결정론적 "
        "노드 키 생성 → LLM 호출 없이 카운터(new/matched/moved/disappeared) upsert → "
        "asset-location 간 found_at 엣지 누적 → 동일 세션·동일 격자 공출현 자산 쌍의 "
        "co_occurred_with 엣지 가중치 누적 → 가중치 기반 Louvain 군집화 → 자산 doctrine 군집 "
        "(기갑 복합체, C2 인프라 등) 자동 발견 및 member_summary 구조적 자동 생성"
    ))
    story.append(Spacer(1, 8))

    story.append(fig_box(
        "도 4] Local + Global 이중 검색 및 LLM 프롬프트 컨텍스트 주입 흐름도",
        "현재 보고서 대상 좌표 → 반경 R 자산 노드 이력 통계 조회(Local Search) → "
        "동일 반경 중첩 커뮤니티 member_summary 조회(Global Search) → "
        "사전 설정 토큰 예산(~500 tokens) 절단·머지 → \"=== HISTORICAL CONTEXT ===\" 블록 포맷화 → "
        "new/disappeared 변화 객체 추출 → 시스템 프롬프트 의미 가드레일 및 IMINT 8섹션 강제 → "
        "LLM 프롬프트 선두 prepend → LLM 1차 호출(영문) → 동일 LLM 인스턴스 2차 호출(한국어 번역) → "
        "메타 헤더 부착 → Reports DB 저장"
    ))
    story.append(Spacer(1, 8))

    story.append(fig_box(
        "도 5] 전체 파이프라인 구동 시퀀스 순서도",
        "시작 → 이미지 적재 및 가변 초해상도(Real-ESRGAN x2/x4) → 글로벌/로컬 슬라이딩 윈도우 SAM3 "
        "탐지 → 마스크 RLE 디코딩 및 배경 zeroing → CLIP/ViT 임베딩 → 동일 클래스 N×M 코사인 유사도 "
        "행렬 → Gale-Shapley 안정 매칭 → 객체 단위 변화 상태(new/matched/moved/disappeared) 분류 "
        "[부수적으로 FOV 가드 적용] → 격자 양자화 결정론적 그래프 키 생성 → LLM 비호출 카운터 upsert → "
        "found_at 및 co_occurred_with 엣지 갱신 → Louvain 커뮤니티 자동 탐지 → Local + Global 이중 "
        "검색 및 ~500 토큰 컨텍스트 블록 생성 → 변화 객체 한정 추출 → IMINT 8섹션 + DISAPPEARED 의미 "
        "가드레일 시스템 프롬프트 → LLM 영문 생성 → 동일 LLM 인메모리 한국어 무오염 번역 → Reports DB "
        "저장 → 종료"
    ))

    # 빌드
    doc.build(story, onFirstPage=draw_page_number, onLaterPages=draw_page_number)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    build()
