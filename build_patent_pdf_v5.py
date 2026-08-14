"""Patent PDF v5 — main-claude branch aligned + KR10-2966704 as primary prior art.

Key changes from v4:
  - Base: main-claude branch (5-state, no "moved", "changed" for static structures)
  - Title: two-axis differentiation (변화탐지 + GraphRAG 보고서)
  - Domain: 일반 위성영상 (군사 표현 제거)
  - Primary prior art: KR10-2966704 (거대모델 기반 패치단위 변화탐지방법)
    * 발명자: 박현선, 이여울
    * 출원 2025.11.18 / 등록 2026.05.14
    * Detailed 5-axis differentiation matrix
  - Secondary prior art: US 12,488,225 (Booz Allen) + US 10,922,578 (Google)
  - Emphasize "객체 인스턴스 단위" throughout claims (vs "패치 단위" 인용특허)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

# ── 한글 폰트 (reportlab 4.x 안전 등록: family 이름을 폰트 이름으로 통일) ─
_registered = []
try:
    pdfmetrics.registerFont(TTFont("Nanum", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
    _registered.append("Nanum")
except Exception as e:
    print("Font load failed (Nanum):", e)
try:
    pdfmetrics.registerFont(TTFont("Nanum-Bold", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"))
    _registered.append("Nanum-Bold")
except Exception as e:
    print("Font load failed (Nanum-Bold):", e)

# family 매핑 — <b> 태그 사용 가능하도록
registerFontFamily("Nanum",
                   normal="Nanum",
                   bold="Nanum-Bold",
                   italic="Nanum",
                   boldItalic="Nanum-Bold")

KOR, KORB = "Nanum", "Nanum-Bold"

# ── 색상 ────────────────────────────────────────────────
TITLE_BLUE  = HexColor("#1e3a8a")
LIGHT_BG    = HexColor("#eff6ff")
LIGHT_BORDER = HexColor("#bfdbfe")
SOFT_BG     = HexColor("#f8fafc")
HILITE_BG   = HexColor("#fef9c3")
HILITE_BD   = HexColor("#fde047")
STAR_BG     = HexColor("#fee2e2")
STAR_BD     = HexColor("#fca5a5")
TABLE_HD    = HexColor("#1e293b")
TEXT_BODY   = HexColor("#1f2937")
TEXT_DARK   = HexColor("#111827")
MUTED       = HexColor("#6b7280")

# ── 스타일 ───────────────────────────────────────────────
style_h1 = ParagraphStyle("H1", fontName=KORB, fontSize=14, textColor=TITLE_BLUE,
                          leftIndent=10, spaceBefore=14, spaceAfter=6, leading=18)
style_h2 = ParagraphStyle("H2", fontName=KORB, fontSize=11.5, textColor=TEXT_DARK,
                          spaceBefore=10, spaceAfter=4, leading=15)
style_h3 = ParagraphStyle("H3", fontName=KORB, fontSize=10.5, textColor=TEXT_DARK,
                          spaceBefore=7, spaceAfter=2, leading=14)
style_body = ParagraphStyle("Body", fontName=KOR, fontSize=10, textColor=TEXT_BODY,
                            leading=16.5, alignment=TA_JUSTIFY, spaceAfter=4,
                            firstLineIndent=10)
style_body_ni = ParagraphStyle("BodyNI", fontName=KOR, fontSize=10, textColor=TEXT_BODY,
                               leading=16.5, alignment=TA_JUSTIFY, spaceAfter=4)
style_bullet = ParagraphStyle("Bullet", fontName=KOR, fontSize=10, textColor=TEXT_BODY,
                              leading=15.5, leftIndent=18, bulletIndent=8,
                              spaceAfter=3, alignment=TA_JUSTIFY)
style_note   = ParagraphStyle("Note", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY,
                              leading=14.5, leftIndent=18, spaceAfter=2,
                              alignment=TA_JUSTIFY)
style_meta_l = ParagraphStyle("ML", fontName=KORB, fontSize=10, textColor=TITLE_BLUE)
style_meta_v = ParagraphStyle("MV", fontName=KOR,  fontSize=10, textColor=TEXT_BODY, leading=15)
style_claim_t = ParagraphStyle("CT", fontName=KORB, fontSize=10.5, textColor=TITLE_BLUE,
                                spaceAfter=4)
style_claim_lead = ParagraphStyle("CL", fontName=KOR, fontSize=9.5, textColor=HexColor("#4b5563"),
                                  leading=14.5, spaceAfter=4, leftIndent=4)
style_claim_b = ParagraphStyle("CB", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY,
                                leading=15, spaceAfter=3, leftIndent=4, alignment=TA_JUSTIFY)
style_fig_t = ParagraphStyle("FT", fontName=KORB, fontSize=10.5, textColor=TITLE_BLUE,
                              alignment=TA_CENTER, spaceAfter=3)
style_fig_b = ParagraphStyle("FB", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY,
                              alignment=TA_LEFT, leading=14.5)

# ── 헬퍼 ──────────────────────────────────────────────────
def boxed(flowables, bg=LIGHT_BG, border=LIGHT_BORDER, pl=12, pr=12, pt=10, pb=10):
    t = Table([[flowables]], colWidths=[None])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),bg),
        ("BOX",(0,0),(-1,-1),0.7,border),
        ("LEFTPADDING",(0,0),(-1,-1),pl),("RIGHTPADDING",(0,0),(-1,-1),pr),
        ("TOPPADDING",(0,0),(-1,-1),pt),("BOTTOMPADDING",(0,0),(-1,-1),pb),
    ]))
    return t

def meta_row(label, value):
    t = Table([[Paragraph(label, style_meta_l), Paragraph(value, style_meta_v)]],
              colWidths=[3.8*cm, None])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                            ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
                            ("BOTTOMPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),0)]))
    return t

def title_meta_box():
    inner = [
        Paragraph("특허출원 명세서 (v5.1)", ParagraphStyle("x",fontName=KORB,fontSize=15,
                  textColor=TITLE_BLUE,alignment=TA_CENTER,spaceAfter=2)),
        Paragraph("(객체 페어링 기반 변화탐지 + GraphRAG 시계열 인덱싱 · 중복성 없음 정리본)",
                  ParagraphStyle("x2",fontName=KOR,fontSize=10.5,textColor=TEXT_BODY,
                  alignment=TA_CENTER,spaceAfter=10)),
        meta_row("발명의 명칭 (국문)",
                 "위성영상 시계열 변화 탐지 및 GraphRAG 기반 AI Agent "
                 "판독보고서 자동 생성 시스템 및 방법"),
        meta_row("발명의 명칭 (영문)",
                 "System and Method for Time-Series Change Detection and "
                 "GraphRAG-Augmented AI Agent Interpretation Report Generation "
                 "from Satellite Imagery"),
    ]
    return boxed(inner, bg=HexColor("#f0f9ff"), border=LIGHT_BORDER, pl=18, pr=18, pt=14, pb=14)

def h1(text):
    bar = Table([[" ", Paragraph(text, style_h1)]], colWidths=[5, None])
    bar.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),TITLE_BLUE),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    return KeepTogether([Spacer(1,6), bar, Spacer(1,2)])

def h2(text): return Paragraph(f"<b>{text}</b>", style_h2)
def h3(text): return Paragraph(f"<b>{text}</b>", style_h3)
def para(text, indent=True):
    return Paragraph(text, style_body if indent else style_body_ni)
def bullet(text): return Paragraph(f"• {text}", style_bullet)
def note(text):   return Paragraph(f"※ {text}", style_note)

def callout(text):
    p = Paragraph(text, ParagraphStyle("CO", fontName=KOR, fontSize=9.5,
                  textColor=TEXT_DARK, leading=14.5, alignment=TA_JUSTIFY))
    return boxed([p], bg=HILITE_BG, border=HILITE_BD, pl=10, pr=10, pt=7, pb=7)

def star_box(text):
    p = Paragraph(text, ParagraphStyle("SB", fontName=KOR, fontSize=9.5,
                  textColor=TEXT_DARK, leading=14.5, alignment=TA_JUSTIFY))
    return boxed([p], bg=STAR_BG, border=STAR_BD, pl=10, pr=10, pt=7, pb=7)

def claim_box(title, lead, body_paragraphs):
    inner = [Paragraph(title, style_claim_t)]
    if lead:
        inner.append(Paragraph(f"<i>📌 쉽게 말하면: {lead}</i>", style_claim_lead))
    for b in body_paragraphs:
        inner.append(Paragraph(b, style_claim_b))
    return boxed(inner, bg=SOFT_BG, border=LIGHT_BORDER)

def fig_box(title, body_lines):
    inner = [Paragraph(f"[{title}]", style_fig_t)]
    for line in body_lines:
        inner.append(Paragraph(line, style_fig_b))
    return boxed(inner, bg=SOFT_BG, border=LIGHT_BORDER, pl=14, pr=14, pt=10, pb=10)

def status_table_5state():
    """main-claude 5상태 분류표 (moved 없음, changed 있음)."""
    data = [
        ["상태", "과거", "현재", "FOV 체크", "의미"],
        ["matched",              "O", "O", "공통 영역",  "동일 객체 확인 (변화 없음)"],
        ["changed",              "O", "O", "동일 위치",  "고정 시설물 구조 변화"],
        ["new",                  "X", "O", "과거 FOV 내", "신규 출현 (변화 O)"],
        ["disappeared",          "O", "X", "현재 FOV 내", "소실 (변화 O)"],
        ["past_not_included",    "X", "O", "과거 FOV 밖", "촬영 공백 (오판 방지)"],
        ["current_not_included", "O", "X", "현재 FOV 밖", "촬영 공백 (오판 방지)"],
    ]
    t = Table(data, colWidths=[3.8*cm, 1.0*cm, 1.0*cm, 2.6*cm, 6.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), TABLE_HD),
        ("TEXTCOLOR",(0,0),(-1,0), HexColor("#ffffff")),
        ("FONTNAME",(0,0),(-1,0), KORB),
        ("FONTNAME",(0,1),(-1,-1), KOR),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("ALIGN",(1,0),(3,-1), "CENTER"),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("LINEBELOW",(0,0),(-1,-1), 0.3, LIGHT_BORDER),
        ("LEFTPADDING",(0,0),(-1,-1), 4),
        ("RIGHTPADDING",(0,0),(-1,-1), 4),
        ("TOPPADDING",(0,0),(-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ("BACKGROUND",(0,3),(-1,4), HexColor("#ecfdf5")),   # new/disappeared 강조
        ("BACKGROUND",(0,5),(-1,6), HexColor("#fff7ed")),   # FOV 공백 강조
        ("BACKGROUND",(0,2),(-1,2), HexColor("#fef3c7")),   # changed 강조
    ]))
    return t

def differentiation_matrix():
    """일반 접근 방식 vs 본 발명 5축 차별화 매트릭스."""
    data = [
        ["구분", "일반 접근 방식", "본 발명"],
        ["탐지 단위", "정규 패치 (격자 분할)", "★ SAM3 객체 인스턴스 (개별)"],
        ["매칭 방식", "동일 위치 패치 대조", "★ CLIP + Gale-Shapley 안정 매칭"],
        ["상태 분류", "변화 O/X 이진", "★ 5상태 (matched/changed/new/\ndisappeared/FOV공백)"],
        ["시계열 누적", "단발 두 시점 비교", "★ GraphRAG 결정론 누적 + Louvain\n자산 doctrine 군집 자동 발견"],
        ["최종 산출물", "변화 영역/패치 위치 출력", "★ LLM AI Agent 판독보고서\n(IMINT 8섹션 자연어 자동 생성)"],
        ["객체 이동 처리", "불가 (격자 고정)", "가능 (임베딩 기반 안정 매칭)"],
        ["배경 노이즈", "패치 내 배경·객체 혼합", "SAM 마스크 zeroing 제거"],
    ]
    t = Table(data, colWidths=[3.0*cm, 5.5*cm, 7.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), TABLE_HD),
        ("TEXTCOLOR",(0,0),(-1,0), HexColor("#ffffff")),
        ("FONTNAME",(0,0),(-1,0), KORB),
        ("FONTNAME",(0,1),(-1,-1), KOR),
        ("FONTNAME",(0,1),(0,-1), KORB),
        ("FONTSIZE",(0,0),(-1,-1), 8.5),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("ALIGN",(0,0),(-1,0), "CENTER"),
        ("LINEBELOW",(0,0),(-1,-1), 0.3, LIGHT_BORDER),
        ("LEFTPADDING",(0,0),(-1,-1), 5),
        ("RIGHTPADDING",(0,0),(-1,-1), 5),
        ("TOPPADDING",(0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("BACKGROUND",(0,1),(-1,5), HexColor("#f8fafc")),   # 상위 5개 (★ 표시)
    ]))
    return t

def db_table():
    """4-DB 구조."""
    data = [
        ["DB", "파일", "주요 테이블", "역할"],
        ["Sensor DB", "sensor_detections.db",
         "image_records\ndetection_records", "영상 메타 + 객체 탐지 결과"],
        ["Pairing DB", "object_pairings.db",
         "pairing_records", "객체 페어링 결과 (5상태)"],
        ["Graph DB", "graph.db",
         "graph_entities\ngraph_relations\ngraph_communities", "GraphRAG 누적 지식 그래프"],
        ["Report DB", "reports.db",
         "report_records", "LLM 보고서 영속 저장"],
    ]
    t = Table(data, colWidths=[2.4*cm, 4.0*cm, 4.4*cm, 4.7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), TABLE_HD),
        ("TEXTCOLOR",(0,0),(-1,0), HexColor("#ffffff")),
        ("FONTNAME",(0,0),(-1,0), KORB),
        ("FONTNAME",(0,1),(-1,-1), KOR),
        ("FONTSIZE",(0,0),(-1,-1), 9),
        ("VALIGN",(0,0),(-1,-1), "MIDDLE"),
        ("LINEBELOW",(0,0),(-1,-1), 0.3, LIGHT_BORDER),
        ("LEFTPADDING",(0,0),(-1,-1), 5),
        ("RIGHTPADDING",(0,0),(-1,-1), 5),
        ("TOPPADDING",(0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    return t

def draw_page_num(canvas, doc):
    canvas.saveState()
    canvas.setFont(KOR, 9); canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0]/2, 1.0*cm, f"- {canvas.getPageNumber()} -")
    canvas.restoreState()


# ─────────────────────────────────────────────────────────
def build():
    out = "/home/user/multi-source-intelligent-system/data/patent_v5_1_no_overlap.pdf"
    doc = SimpleDocTemplate(out, pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.0*cm, bottomMargin=2.2*cm)
    s = []

    # ─── 표지 ───
    s.append(title_meta_box()); s.append(Spacer(1,8))

    # ═══════════════════════════════════════════════════
    # 1. 개요
    # ═══════════════════════════════════════════════════
    s.append(h1("1. 개요"))
    s.append(h2("(1) 직무발명 수탁과제 무관 근거 설명"))

    s.append(h3("1) 제안 발명 내용 요약"))
    s.append(para(
        "본 발명은 위성·드론 시계열 영상에 대해 두 축으로 구성된 통합 시스템으로서, "
        "<b>[축 A] 객체 페어링 기반 시계열 변화 탐지</b>(SAM3 객체 탐지 → 마스크 배경 제거 CLIP 임베딩 → "
        "Gale-Shapley 안정 매칭 → 5상태 정밀 분류)와, <b>[축 B] GraphRAG 시계열 인덱싱 기반 AI Agent "
        "판독보고서 자동 생성</b>(격자 양자화 결정론 키 그래프 누적 → Louvain 자산 군집 자동 발견 → "
        "Local+Global 이중 검색 → LLM 표준 IMINT 보고서 자율 생성)이 유기적으로 결합된 end-to-end "
        "시스템이다."
    ))
    s.append(callout(
        "<b>핵심 차별 2축</b>: <b>①</b> 정규 패치 단위가 아닌 <b>객체 인스턴스 단위</b>의 "
        "페어링 기반 변화 탐지, <b>②</b> LLM 호출 없이 결정론적으로 누적되는 <b>GraphRAG "
        "시계열 메모리</b>를 통한 판독보고서 자동 생성."
    ))

    s.append(h3("2) 유사 수탁과제 기술 설명"))
    s.append(bullet("<b>없음</b>"))

    s.append(h3("3) 중복성 설명"))
    s.append(bullet("<b>없음</b>"))

    s.append(h3("4) 차별성 설명"))
    s.append(para(
        "본 발명은 위성·드론 영상 인텔리전스 분야에서 다음 5가지 축의 독자적 기술 구성을 결합한다는 "
        "점에서 차별된다.", indent=False
    ))
    s.append(bullet(
        "<b>객체 인스턴스 단위 페어링</b>: 정규 패치(격자) 단위 접근이 아닌, SAM3 세그멘테이션 모델로 "
        "각 개별 객체를 인스턴스 단위로 식별하여 두 시점 간 1:1 매칭 수행 → 객체가 격자를 넘어 이동한 "
        "경우에도 동일 객체 안정 추적 가능."
    ))
    s.append(bullet(
        "<b>마스크 배경 제거 + Gale-Shapley 안정 매칭</b>: SAM 마스크로 객체 외부 픽셀을 영(0)으로 "
        "zeroing한 후 CLIP 임베딩을 산출하고, 노벨 경제학상 수상의 Gale-Shapley 지연 승인 알고리즘으로 "
        "1:1 안정 매칭을 수행 → 배경 노이즈 원천 제거 및 그리디 매칭의 국소 최적 함정 회피."
    ))
    s.append(bullet(
        "<b>5상태 정밀 분류</b>: 변화 여부의 이진 판정이 아닌 matched/changed/new/disappeared/"
        "past_not_included/current_not_included의 5개 이상 상태로 정밀 배정 → 특히 촬영 범위 차이를 "
        "자산 변화로 오인하는 치명적 오판을 원천 배제."
    ))
    s.append(bullet(
        "<b>LLM 비호출 결정론 GraphRAG 시계열 인덱싱</b>: 위경도 격자 양자화 결정론 키에 기반해 "
        "페어링 결과를 <b>거대언어모델 호출 없이</b> 지식 그래프 카운터에 누적 upsert → 동일 입력에 "
        "매번 동일한 그래프 산출로 완전 재현성·감사 가능성 확보. Louvain 알고리즘으로 자산 doctrine "
        "군집(예: 산업 시설군, 교통 인프라)을 자동 발견."
    ))
    s.append(bullet(
        "<b>AI Agent 판독보고서 자율 생성</b>: Local + Global 이중 검색으로 사전 압축된 시공간 "
        "컨텍스트(~500 토큰)를 LLM 프롬프트에 prepend하고, 표준 IMINT 8섹션 구조와 도메인 의미 "
        "가드레일을 강제하며, 동일 LLM 인스턴스를 재호출해 좌표·수치 토큰을 바이패스한 무오염 한국어 "
        "번역까지 자율 생성 → 환각 차단 + 다국어 일관성 확보."
    ))

    s.append(h2("(2) 발명 제안 배경"))
    s.append(para(
        "위성·드론 시계열 영상 분석에서 두 시점 사이의 변화를 자동 탐지하는 것은 도시 계획·환경 "
        "감시·재난 대응·인프라 모니터링 등 다양한 응용에서 핵심이다. 종래의 정규 패치 단위 변화 "
        "탐지는 (i) 격자 분할로 인해 객체 이동을 동일 객체로 추적하지 못하고, (ii) 변화 여부만 이진 "
        "판정하여 어떤 자산이 어떻게 변했는지의 정보를 제공하지 못하며, (iii) 두 프레임 단발 비교에 "
        "그쳐 시계열 누적 패턴을 인식하지 못하는 한계가 있었다."
    ))
    s.append(para(
        "한편 최근 대규모 언어 모델(LLM)을 판독보고서 생성에 활용하려는 시도가 있으나, LLM에 정형 "
        "탐지 결과를 그대로 주입하는 방식은 (i) 시계열 누적 컨텍스트 부재로 단발 보고에 그치고, "
        "(ii) LLM이 정형 정보를 자연어로 재해석하는 과정에서 환각(hallucination)이 발생하는 위험이 있다."
    ))
    s.append(para(
        "본 발명은 (i) 객체 인스턴스 단위 페어링 기반 변화 탐지로 객체 이동에 대응하고 5상태로 정밀 "
        "분류하며, (ii) LLM 호출 없이 결정론적 지식 그래프에 누적 인덱싱한 시계열 컨텍스트를 사전 "
        "압축하여 LLM 프롬프트에 주입함으로써 환각을 원천 차단한 판독보고서를 자율 생성하는 두 축의 "
        "통합 시스템을 제안한다."
    ))

    s.append(h2("(3) 산업상 이용 분야 (용도)"))
    s.append(bullet("<b>스마트시티 인프라 모니터링</b>: 도시 건물·시설물 시계열 변화 자동 감시·보고"))
    s.append(bullet("<b>재난·재해 대응</b>: 위성 관측 기반 시설물 파손·복구 추이 자동 판독"))
    s.append(bullet("<b>환경·농업 모니터링</b>: 산림·농지·해양 시계열 변화 자동 추적"))
    s.append(bullet("<b>부동산·건설 관리</b>: 대규모 건설 현장 진척도 자동 판독 및 정기 보고"))
    s.append(bullet("<b>국토·해양 감시</b>: 국경·해안·항만 자산 활동 패턴 자동 인텔리전스"))

    s.append(h2("(4) 종래 기술 (관련 기술 배경)"))
    s.append(para(
        "본 발명이 속한 위성·항공 영상 인텔리전스 분야에서 다음과 같은 관련 기술 흐름이 존재한다. "
        "다만 본 발명의 통합 구성(객체 인스턴스 단위 페어링 + GraphRAG 시계열 인덱싱 + LLM AI Agent "
        "판독보고서 자동 생성)과 직접적으로 중복되는 선행 기술은 확인되지 아니한다.", indent=False
    ))

    s.append(h3("가. 위성영상 변화 탐지 일반"))
    s.append(bullet(
        "정규 패치(격자) 단위로 영상을 분할하여 두 시점 영상의 각 패치 특징을 비교함으로써 변화 "
        "여부를 이진 판정하는 접근이 일반적으로 알려져 있음."
    ))
    s.append(bullet(
        "딥러닝 기반 세그멘테이션 모델을 활용한 시멘틱 변화 탐지 및 U-Net·Siamese 구조 기반 "
        "변화 탐지 알고리즘 등이 연구·개발되어 왔음."
    ))

    s.append(h3("나. RAG 및 LLM 기반 보고서 생성 일반"))
    s.append(bullet(
        "대규모 언어 모델(LLM)에 벡터 데이터베이스(FAISS 등) 기반 텍스트 검색 결과를 컨텍스트로 "
        "주입하는 Retrieval-Augmented Generation(RAG) 기법이 널리 알려져 있음."
    ))
    s.append(bullet(
        "지식 그래프 형태로 정보를 구조화하여 LLM 프롬프트 컨텍스트로 활용하는 GraphRAG 접근이 "
        "학술적으로 제안된 바 있음(Microsoft, 2024)."
    ))

    s.append(h3("다. 원격 탐사 파운데이션 모델"))
    s.append(bullet(
        "SAM(Segment Anything Model), CLIP 등 비전 파운데이션 모델이 위성·항공 영상 분석에 "
        "활용되기 시작하고 있음."
    ))

    s.append(h2("(5) 관련 기술의 한계"))
    s.append(para(
        "상기 관련 기술들은 개별적으로 존재하나, 위성·항공 영상 시계열 분석 도메인에 특화되어 통합된 "
        "형태로 다음과 같은 한계를 가지며, 본 발명이 이를 극복한다.", indent=False
    ))
    problems = [
        ("객체 단위 시계열 페어링 부재",
         "정규 패치 단위 접근은 격자 위치가 고정되어 있어 객체가 이동한 경우 동일 객체 추적이 원천적으로 "
         "불가능하며, 어떤 자산이 어떻게 변화했는지에 대한 객체 인스턴스 단위 서술 정보를 산출할 수 없다."),
        ("이진 판정의 정보 밀도 한계",
         "변화 여부만을 판정하는 방식은 촬영 범위 차이로 인한 관측 공백을 자산 변화로 오인할 위험이 있고, "
         "매칭 성공/실패에 따른 세밀한 상태 구분(정지/이동/신규/소실/구조변화/촬영공백)이 어렵다."),
        ("시계열 누적 메모리 부재",
         "두 시점 단발 비교에 그쳐 동일 자산의 반복 관측 이력을 결정론적으로 누적 저장하는 그래프 "
         "메모리 구조가 부재하며, 반복 실행 시 시계열 패턴이 축적되지 않는다."),
        ("자산 doctrine 패턴 자동 발견 부재",
         "자주 함께 등장하는 자산 클래스들의 결합 패턴(예: 산업 시설군, 교통 인프라)을 그래프 알고리즘 "
         "으로 자동 군집화·발견하는 구성이 정의되어 있지 않다."),
        ("환각 위험 및 도메인 안전성 부재",
         "LLM 직접 활용 시 시공간 컨텍스트 부재로 환각(hallucination)이 발생할 위험이 있고, 도메인 "
         "특유의 의미 제약(예: 미관측 ≠ 파괴)이나 표준 IMINT 8섹션 형식을 강제하는 구성이 없다."),
    ]
    for i,(t,d) in enumerate(problems,1):
        s.append(bullet(f"<b>{i}. {t}</b> — {d}"))

    # ═══════════════════════════════════════════════════
    # 2. 상세 설명
    # ═══════════════════════════════════════════════════
    s.append(h1("2. 상세 설명"))

    s.append(h2("(1) 시스템 전체 구성 — 2축 · 4-DB 아키텍처"))
    s.append(para(
        "본 발명은 [축 A] 객체 페어링 기반 변화 탐지부(구성 A~C)와 [축 B] GraphRAG 시계열 인덱싱 "
        "및 LLM 판독보고서 생성부(구성 D~F)의 두 축을 유기적으로 결합한 시스템이며, 각 단계의 결과는 "
        "물리적으로 분리된 4개의 데이터베이스에 영속화된다."
    ))
    s.append(Spacer(1,3))
    s.append(db_table())
    s.append(Spacer(1,3))
    s.append(note(
        "모든 레코드에 동일한 session_id가 부여되어 단일 파이프라인 실행 단위로 묶이며, 분석관이 "
        "탐지 결과를 수정하면 페어링·그래프·보고서가 자동 재계산되는 HITL(Human-In-The-Loop) 재처리를 "
        "지원한다."
    ))

    s.append(h2("(2) 축 A: 객체 페어링 기반 시계열 변화 탐지부"))

    s.append(h3("[구성 A] 영상 객체 탐지부 (SAM3 · Sensor DB)"))
    s.append(callout(
        "Real-ESRGAN 초해상도 → SAM3 텍스트 프롬프트 zero-shot 탐지 → 멀티스케일 슬라이딩 윈도우 "
        "+ NMS 통합 → 객체별 (클래스·bbox·<b>세그멘테이션 마스크 RLE</b>·위경도)를 Sensor DB에 저장."
    ))

    s.append(h3("[구성 B] 객체 인스턴스 단위 페어링부 (Pairing Module)"))
    s.append(callout(
        "인용 특허가 정규 패치 단위로 격자 분할·비교하는 것과 달리, 본 발명은 <b>SAM3가 실제로 인식한 "
        "개별 객체 인스턴스를 단위</b>로 두 시점 객체를 1:1로 짝짓는다."
    ))
    s.append(bullet(
        "<b>마스크 기반 배경 제거 임베딩</b>: 각 객체의 mask_rle로 배경 픽셀을 영(0)으로 마스킹한 후 "
        "bbox 크롭하여 CLIP/ViT 인코더로 L2 정규화된 임베딩 산출. 배경 노이즈가 완전히 제거되어 "
        "동일 클래스 객체 간 식별력 극대화."
    ))
    s.append(bullet(
        "<b>N×M 유사도 행렬 + 동일 클래스 필터</b>: 현재 N개 객체와 과거 M개 객체의 임베딩 행렬곱으로 "
        "코사인 유사도 행렬을 단일 GEMM으로 산출한 뒤 동일 클래스 쌍만 후보로 유지. 점수 = 0.8 × "
        "CLIP 유사도 + 0.2 × 크기 유사도."
    ))
    s.append(bullet(
        "<b>Gale-Shapley 안정 매칭</b>: 점수 임계값 이상 후보 쌍에 대해 지연 승인(deferred acceptance) "
        "알고리즘으로 1:1 안정 매칭 수행. 그리디 매칭의 국소 최적 함정 회피, 객체가 격자를 넘어 이동한 "
        "경우에도 동일 객체 안정 추적."
    ))
    s.append(bullet(
        "<b>고정형 객체 전용 분기</b>: 건물·기지 등 물리적 이동 불가 자산은 별도 처리 — 위경도 초근접 "
        "결합 → CLIP 외형 비교로 <b>matched(변화 없음) / changed(구조 변화)</b> 구분 → 한쪽 탐지 실패 "
        "시 상대 영상 동일 위경도 crop 후 CLIP 유사도 재검증하여 <b>가상 탐지 레코드 합성 주입</b>으로 "
        "탐지 누락 보정."
    ))
    s.append(bullet(
        "<b>5상태 정밀 분류</b>: 매칭 결과 + FOV 검사로 아래 5상태 중 하나로 정밀 배정."
    ))
    s.append(Spacer(1,3))
    s.append(status_table_5state())
    s.append(Spacer(1,5))

    s.append(h2("(3) 축 B: GraphRAG 시계열 인덱싱 및 AI Agent 판독보고서 생성부"))

    s.append(h3("[구성 C] GraphRAG 결정론적 누적 인덱싱부 (Graph Indexer · Graph DB)"))
    s.append(callout(
        "페어링 결과를 <b>LLM 호출 없이</b> 격자 양자화 결정론 키로 지식 그래프에 누적 인덱싱한 후, "
        "Louvain 알고리즘으로 자산 doctrine 군집을 자동 발견하여 Graph DB에 저장."
    ))
    s.append(bullet(
        "<b>격자 양자화 결정론 키</b>: 위경도 소수점 2자리(≈1km)로 양자화하여 "
        "\"loc:37.50,127.00\" 형식의 Location 노드 키와 \"asset:{class}:{loc_key}\" 형식의 Asset "
        "노드 키를 결정론적으로 생성. GPS 미세 오차 흡수, 동일 자산 자동 통합."
    ))
    s.append(bullet(
        "<b>LLM 비호출 카운터 upsert</b>: status 필드를 자산 노드 카운터(new_count, matched_count, "
        "changed_count, disappeared_count, total_confidence)에 결정론 가산. <b>동일 입력에 대해 매번 "
        "동일한 그래프 산출</b> — 완전 재현성·감사 가능성 확보."
    ))
    s.append(bullet(
        "<b>관계 엣지 누적</b>: asset→location의 found_at 엣지, 동일 세션·동일 격자 공출현 자산 쌍의 "
        "co_occurred_with 엣지 count 가중치 누적."
    ))
    s.append(bullet(
        "<b>Louvain 군집 자동 탐지</b>: co_occurred_with 엣지 가중치 기반 Louvain 알고리즘으로 자산 "
        "doctrine 군집(예: 기갑 복합체, C2 인프라, 항공 작전 거점) 자동 발견. member_summary는 LLM "
        "호출 없이 결정론 통계 산식으로 자동 생성."
    ))

    s.append(h3("[구성 D] Local + Global 이중 검색부 (Graph Retriever)"))
    s.append(bullet(
        "<b>Local Search</b>: 보고서 대상 좌표 반경 R(예: 0.05° ≈ 5.5km) 내 자산 노드 이력 통계 조회 "
        "(누적 관측, 신규/지속/이동/소실 횟수, 평균 신뢰도, 최초/최종 관측 시각)."
    ))
    s.append(bullet(
        "<b>Global Search</b>: 동일 반경과 중첩되는 Louvain 커뮤니티의 member_summary 조회."
    ))
    s.append(bullet(
        "<b>사전 압축 컨텍스트 블록</b>: 양 검색 결과를 사전 설정 토큰 예산(예: ~500 토큰) 이내로 "
        "절단·머지하여 \"=== GRAPHRAG HISTORICAL CONTEXT ===\" 단일 블록으로 포맷화."
    ))

    s.append(h3("[구성 E] AI Agent 판독보고서 생성부 (단일 LLM 2회 호출 · Report DB)"))
    s.append(bullet(
        "<b>변화 객체 한정 추출</b>: 'new'·'disappeared' 상태만 신뢰도 내림차순 추출. matched·changed·"
        "FOV 공백 상태는 LLM 프롬프트에서 제외하여 토큰 효율과 분석 집중도 확보."
    ))
    s.append(bullet(
        "<b>도메인 의미 가드레일</b>: 시스템 프롬프트에 <i>\"'DISAPPEARED'는 영상 미관측을 의미하며 "
        "파괴(destroyed)가 아니다\"</i> 의미 제약을 강제 주입."
    ))
    s.append(bullet(
        "<b>표준 IMINT 8섹션 강제</b>: CLASSIFICATION / EXECUTIVE SUMMARY / SITUATION / CHANGE ANALYSIS / "
        "THREAT ASSESSMENT / INTELLIGENCE GAPS / RECOMMENDED ACTIONS / APPENDIX의 8개 섹션 구조 "
        "프롬프트 강제."
    ))
    s.append(bullet(
        "<b>동일 LLM 인메모리 재호출 무오염 번역</b>: 1차 영문 생성 후 같은 LLM 인스턴스로 한국어 번역 "
        "세션 발동. 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 토큰은 변형 없이 바이패스하여 다국어 "
        "일관성 확보."
    ))

    s.append(h2("(4) 발명의 효과"))
    effects = [
        ("객체 단위 정밀 변화 탐지",
         "인용 특허의 패치 단위(격자 고정) 한계를 극복해 객체 인스턴스 단위 추적으로 \"어떤 자산이 "
         "어디서 어디로 이동했다\"까지 정확 판별. 객체가 격자를 넘어 이동해도 동일 객체 안정 추적."),
        ("5상태 정밀 분류로 오판 방지",
         "인용 특허의 이진 판정(변화 O/X) 대비 5상태로 세밀 분류하여, 특히 촬영 범위 차이로 인한 "
         "관측 공백(past/current_not_included)을 자산 변화로 오인하는 치명적 오판을 원천 배제."),
        ("LLM 비호출 결정론 GraphRAG로 시계열 인텔리전스 확보",
         "그래프 인덱싱에 AI 호출이 전무하여 운용 비용 근본적 절감, 동일 입력에 매번 동일한 그래프 "
         "산출로 감사 가능성 확보. 사용할수록 두꺼워지는 시계열 메모리로 반복 패턴·doctrine 자동 발견."),
        ("환각 차단 + 토큰 효율 극대화",
         "사전 압축된 ~500 토큰 컨텍스트 + 변화 객체 한정 전달 + 도메인 가드레일 결합으로 LLM 환각 "
         "차단하면서 호출 비용 최소화."),
        ("표준 판독보고서 자율 생성",
         "인용 특허가 변화 패치 위치만 출력하는 데 반해, 본 발명은 자연어 IMINT 8섹션 판독보고서를 "
         "완전 자동 생성하여 분석관의 후속 해석 작업 제거."),
    ]
    for i,(t,d) in enumerate(effects,1):
        s.append(bullet(f"<b>{i}. {t}</b> — {d}"))

    # ═══════════════════════════════════════════════════
    # 3. 청구범위
    # ═══════════════════════════════════════════════════
    s.append(h1("3. 특허 청구범위"))
    s.append(note(
        "각 청구항 상단에 비전문가용 '쉽게 말하면' 요약을 함께 표시했습니다. 실제 권리범위는 그 아래의 "
        "정형 문장에 의해 정의됩니다. 인용 특허 KR10-2966704(패치 단위)와의 명확한 구분을 위해 "
        "\"객체 인스턴스 단위\" 표현을 청구항 도입부에 명시하였습니다."
    ))

    s.append(claim_box(
        "[청구항 1] (독립항) — 전체 방법",
        "정규 패치가 아닌 실제 객체 하나하나를 짝지어 5상태로 분류하고, AI 없이 그래프에 누적한 뒤 "
        "AI Agent가 표준 판독보고서를 자율 생성하는 방법.",
        [
            "임의의 대상 지역을 촬영한 시계열 위성·항공 영상을 수집하여 텍스트 프롬프트 기반 <b>객체 "
            "세그멘테이션 모델(예: SAM3)</b>로 각 개별 객체를 인스턴스 단위로 식별하고, 자산 클래스·"
            "위경도·바운딩 박스·세그멘테이션 마스크를 포함하는 탐지 레코드를 센서 데이터베이스에 "
            "저장하는 단계;",
            "정규 패치가 아닌 <b>객체 인스턴스 단위</b>로, 각 탐지 객체의 세그멘테이션 마스크로 배경 "
            "픽셀을 영(0)으로 처리한 후 바운딩 박스 크롭 이미지로부터 비전 인코더에 의해 L2 정규화된 "
            "시각 임베딩 벡터를 산출하고, 최신 영상의 N개 임베딩과 과거 영상의 M개 임베딩의 N×M 코사인 "
            "유사도 행렬에서 동일 클래스 쌍에 대해 임베딩 유사도와 크기 유사도의 가중합 점수가 임계값 "
            "이상인 후보 쌍에 대하여 <b>Gale-Shapley 지연 승인 알고리즘</b>으로 1:1 안정 매칭을 수행한 "
            "후, matched·changed·new·disappeared·past_not_included·current_not_included의 5개 이상 "
            "상태로 정밀 분류하여 페어링 데이터베이스에 저장하는 단계;",
            "각 자산의 위경도를 정수형 격자 단위로 양자화하여 위치 노드 키를, 객체 클래스와 위치 키의 "
            "결합으로 자산 노드 키를 결정론적으로 생성하고, 상기 5상태에 따라 자산 노드의 누적 카운터 "
            "속성을 <b>거대언어모델(LLM) 호출 없이</b> 갱신하며, 자산-위치 간 found_at 엣지 및 동일 "
            "세션·동일 격자 내 공출현 자산 쌍의 co_occurred_with 엣지의 가중치를 누적 갱신하여 그래프 "
            "데이터베이스에 지식 그래프를 형성하는 단계;",
            "상기 co_occurred_with 엣지 가중치를 기반으로 Louvain 군집화를 수행하여 자산 복합체 "
            "커뮤니티 요약문을 LLM 호출 없이 구조적으로 생성하고, 보고서 대상 좌표 반경 내 자산 노드 "
            "이력을 조회하는 Local Search와 동일 반경과 중첩되는 커뮤니티 요약을 조회하는 Global Search를 "
            "병합하여 사전 설정 토큰 예산 이내의 역사적 맥락 컨텍스트 블록을 생성하는 단계; 및",
            "상기 5상태 중 new 또는 disappeared에 해당하는 객체만을 신뢰도 내림차순으로 추출하고, "
            "상기 역사적 맥락 컨텍스트 블록을 LLM 프롬프트 선두에 prepend하며, 시스템 프롬프트에 "
            "\"'DISAPPEARED'는 영상 미관측을 의미하며 파괴 확인이 아니다\"라는 의미 제약과 표준 IMINT "
            "8섹션 구조를 강제 주입하여 LLM 에이전트로 영문 보고서를 생성한 후 동일 LLM 인스턴스를 "
            "재호출해 한국어로 번역하여 보고서 데이터베이스에 저장하는 단계;",
            "를 포함하는 <b>객체 인스턴스 단위</b> 시계열 변화 탐지 및 GraphRAG 기반 AI 에이전트 "
            "판독 보고서 자동 생성 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 2] (종속) — 마스크 기반 배경 제거 임베딩",
        "객체 비교 전 배경을 지운 뒤 그 객체만 잘라서 임베딩하여 배경 노이즈로 인한 오인 차단.",
        [
            "제1항에 있어서, 탐지된 자산의 세그멘테이션 마스크 RLE 정보를 디코딩하여 마스크 활성 "
            "영역 외 픽셀을 영(0)으로 마스킹하고 바운딩 박스 주위에 사전 설정 패딩 폭을 부가하여 "
            "크롭하며, 마스크 RLE가 부재하는 경우 바운딩 박스 크롭으로 폴백하고, 산출된 임베딩 벡터를 "
            "L2 정규화하여 코사인 유사도 행렬 산출에 사용하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 3] (종속) ★ Gale-Shapley 안정 매칭 (인용 특허 대비 핵심 차별)",
        "노벨 경제학상 수상 안정 매칭 기법으로 객체가 이동해도 정확히 1:1 짝을 맞춤.",
        [
            "제1항에 있어서, (a) 임베딩 유사도와 크기 유사도의 가중합을 점수로 사용하고, (b) 점수가 "
            "임계값을 초과하는 후보 쌍에 대해 <b>Gale-Shapley 지연 승인 알고리즘</b>으로 1:1 안정 "
            "매칭을 수행하며, 이를 통해 인용 특허의 정규 패치 위치 기반 대조 방식이 객체 이동 시 동일 "
            "객체 추적이 불가능한 한계를 극복하여 격자 경계를 넘어 이동한 객체도 동일 객체로 안정 "
            "추적하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 4] (종속) ★ 5상태 정밀 분류 (인용 특허 이진 판정 대비)",
        "'변화 O/X' 이진 판정이 아닌 5상태로 세밀하게 나눠 촬영 공백을 진짜 변화와 구분.",
        [
            "제1항에 있어서, 상기 페어링 데이터베이스의 status 필드는 (i) 두 영상 모두 존재하는 "
            "matched, (ii) 고정형 자산의 구조 변화 changed, (iii) 현재만 존재하고 과거 영상 시야각 "
            "경계면 내부의 new, (iv) 과거만 존재하고 현재 영상 시야각 경계면 내부의 disappeared, "
            "(v) 현재만 존재하고 과거 영상 시야각 경계면 외부의 past_not_included, (vi) 과거만 존재하고 "
            "현재 영상 시야각 경계면 외부의 current_not_included의 <b>5상태 이상으로 정밀 분류</b>되며, "
            "(v)·(vi) 상태 객체는 LLM 보고서 생성 시 변화 객체 추출 단계에서 제외되어 단순 촬영 범위 "
            "차이가 자산 변화로 오인되지 않도록 하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 5] (종속) — 고정형 객체 전용 분기 + 가상 탐지 합성",
        "건물·시설 같이 못 움직이는 객체는 위경도 초근접 결합 + 외형 변화 판정 + 탐지 누락 보정으로 "
        "별도 처리.",
        [
            "제1항에 있어서, 사전 정의된 정적 고정 자산 클래스에 속하는 객체에 대해서는 (a) 위경도 "
            "거리가 정밀 임계값 이내인 동일 클래스 쌍을 최우선 그리디 결합하는 단계, (b) 결합된 쌍의 "
            "CLIP 유사도가 임계값 이상이면 matched, 미만이면 changed(구조 변화) 상태를 부여하는 단계, "
            "및 (c) 한쪽 영상에서만 탐지된 경우 상대 영상의 동일 위경도 영역을 강제 크롭하여 CLIP "
            "유사도를 재산출하고 임계값 이상이면 source_type=synthetic으로 표시된 가상 탐지 레코드를 "
            "상대 영상 DB에 합성 주입하여 페어링을 복원하는 단계를 더 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 6] (종속) ★ LLM 비호출 결정론 GraphRAG 인덱싱 (인용 특허 대비 핵심 차별)",
        "AI를 부르지 않고 카운터만 자동으로 +1 누적하여 동일 입력에 매번 동일한 결과 (완전 재현).",
        [
            "제1항에 있어서, 상기 지식 그래프 형성 단계는, 각 자산의 위경도를 소수점 N자리의 정수형 "
            "격자 단위로 반올림하여 \"loc:{lat:.Nf},{lon:.Nf}\" 형식의 결정론적 위치 노드 키와 "
            "\"asset:{class}:{loc_key}\" 형식의 결정론적 자산 노드 키를 생성하는 단계; 페어링 레코드의 "
            "status 필드 및 신뢰도를 자산 노드의 카운터 속성에 <b>거대언어모델(LLM) 호출 없이 "
            "결정론적으로 누적 가산</b>하는 upsert 단계; 및 자산-위치 간 found_at 엣지와 동일 세션·동일 "
            "위치 키 내 공출현 자산 쌍의 co_occurred_with 엣지의 count 속성을 LLM 호출 없이 결정론적으로 "
            "누적 갱신하는 단계를 포함하여, 동일 입력에 매번 동일한 그래프가 산출되는 결정론적 재현성을 "
            "갖는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 7] (종속) — Louvain 자산 doctrine 군집 자동 발견",
        "자주 함께 등장하는 자산들을 자동 그룹화하여 doctrine 패턴 자동 발견.",
        [
            "제1항에 있어서, co_occurred_with 엣지의 count 가중치를 입력으로 NetworkX 등의 그래프 "
            "라이브러리 상에서 Louvain 계층적 군집화 알고리즘을 수행하는 단계; 및 각 커뮤니티에 대해 "
            "구성원 자산 클래스 분포·위치 클러스터 수·총 관측 수·신규 배치 수·소실 수를 포함하는 "
            "member_summary를 LLM 호출 없이 결정론적 통계 산식만으로 생성하여 그래프 데이터베이스에 "
            "기록하는 단계를 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 8] (종속) — Local + Global 이중 검색 및 사전 압축 컨텍스트",
        "해당 지역 메모만 추려 ~500 토큰으로 압축해 AI 입력 맨 앞에 붙임.",
        [
            "제1항에 있어서, 보고서 대상 좌표 반경 R 이내 자산 노드 이력 통계를 조회하는 Local Search "
            "단계; 동일 반경과 중첩되는 커뮤니티들의 member_summary를 조회하는 Global Search 단계; "
            "및 양 검색 결과를 사전 설정 토큰 예산 이내로 절단·머지하여 LLM 프롬프트 선두에 prepend "
            "되는 단일 컨텍스트 블록으로 포맷화하는 단계를 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 9] (종속) — 변화 객체 한정 + IMINT 8섹션 + 동일 LLM 번역",
        "변화 객체만 AI에 전달, 표준 8섹션 강제, 같은 AI로 좌표 보존 한국어 변환.",
        [
            "제1항에 있어서, 페어링 결과 중 matched·changed·past_not_included·current_not_included "
            "상태 객체는 LLM 프롬프트에서 제외하고 new·disappeared 상태 객체만 신뢰도 내림차순으로 "
            "주입하는 단계; 시스템 프롬프트에 표준 IMINT 8개 섹션 구조 및 \"'DISAPPEARED'는 영상 "
            "미관측이며 파괴가 아니다\" 의미 제약을 강제 주입하는 단계; 및 동일 LLM 인스턴스를 "
            "인메모리 재사용하여 영문 보고서 생성 후 한국어 번역 세션을 발동하되 섹션 헤더·좌표·"
            "타임스탬프·클래스명·신뢰도 토큰은 변형 없이 바이패스하고 분석 서술문만 변환하는 단계를 "
            "포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 10] (종속) — 4-DB 분리 + 세션 추적성",
        "탐지·페어링·그래프·보고서를 각각 별도 DB에 저장하고 session_id로 묶어 자동 재계산 (HITL).",
        [
            "제1항에 있어서, 영상·탐지 결과의 센서 데이터베이스, 페어링 결과의 페어링 데이터베이스, "
            "지식 그래프 노드·엣지·커뮤니티의 그래프 데이터베이스, LLM 보고서의 보고서 데이터베이스로 "
            "물리적으로 분리된 4개의 데이터베이스를 포함하며, 모든 레코드에 동일한 세션 식별자가 "
            "부여되어 분석관이 센서 DB의 탐지 결과를 수정하면 페어링·그래프·보고서가 동일 세션 단위로 "
            "자동 재계산되는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 11] (독립항) — 시스템 / 기록매체",
        "위의 방법을 수행하는 컴퓨터 시스템 및 그 프로그램이 기록된 기록매체.",
        [
            "제1항 내지 제10항 중 어느 한 항의 방법을 수행하는 하나 이상의 하드웨어 프로세서와 메모리를 "
            "포함하는 <b>객체 인스턴스 단위</b> 시계열 변화 탐지 및 GraphRAG 기반 AI 에이전트 판독 "
            "보고서 자동 생성 시스템 및 상기 방법을 컴퓨터에서 실행시키기 위한 프로그램이 기록된 "
            "컴퓨터 판독 가능 기록 매체.",
        ],
    ))

    # ═══════════════════════════════════════════════════
    # 4. 도면
    # ═══════════════════════════════════════════════════
    s.append(h1("4. 도면 설명 명기"))

    s.append(fig_box(
        "도 1] 시스템 전체 구성도 — 2축 · 4-DB 아키텍처",
        [
            "[시점 1 위성영상] + [시점 2 위성영상]",
            "         │",
            "         ▼",
            "  ┌──────────────────────────────┐",
            "  │  [구성 A] SAM3 탐지부          │ ──► Sensor DB",
            "  │  Real-ESRGAN SR + 멀티스케일   │     (image_records,",
            "  │  타일 + 마스크 RLE 산출        │      detection_records)",
            "  └──────────────────────────────┘",
            "         │ 현재·과거 탐지 결과 조회",
            "         ▼",
            "  ┌──────────────────────────────┐",
            "  │ [구성 B] Pairing Module        │",
            "  │ ★ 객체 인스턴스 단위             │",
            "  │ CLIP+ViT 임베딩 (배경 제거)   │ ──► Pairing DB",
            "  │ N×M 코사인 행렬 + 동일 클래스 │     (5상태:",
            "  │ Gale-Shapley 안정 매칭          │      matched, changed, new,",
            "  │ + 고정형 객체 분기              │      disappeared,",
            "  │ + FOV 가드 (5상태 분류)         │      past_not_included,",
            "  │                                │      current_not_included)",
            "  └──────────────────────────────┘",
            "         │ 페어링 배치",
            "         ▼",
            "  ┌──────────────────────────────┐",
            "  │  [구성 C] Graph Indexer        │ ──► Graph DB",
            "  │  격자 양자화(loc/asset 키)    │     (graph_entities,",
            "  │  ★ LLM 비호출 카운터 upsert    │      graph_relations,",
            "  │  found_at / co_occurred_with  │      graph_communities)",
            "  │  Louvain 군집 자동 탐지        │",
            "  └──────────────────────────────┘",
            "         │",
            "         ▼",
            "  ┌──────────────────────────────┐",
            "  │  [구성 D] Graph Retriever      │",
            "  │  Local Search + Global Search │",
            "  │  ★ ~500 토큰 컨텍스트 블록     │",
            "  └──────────────────────────────┘",
            "         │ historical context",
            "         ▼",
            "  ┌──────────────────────────────┐",
            "  │  [구성 E] LLM AI Agent         │ ──► Report DB",
            "  │  ① 영문 IMINT 8섹션 생성        │     (판독보고서)",
            "  │  ② 동일 인스턴스 한국어 번역    │",
            "  │  (좌표·수치 토큰 바이패스)     │",
            "  └──────────────────────────────┘",
        ]
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 2] 접근 방식 비교 — 정규 패치 vs 본 발명 (객체 인스턴스)",
        [
            "  <일반 접근 방식 (패치 단위)>        <본 발명 (객체 인스턴스 단위)>",
            "  ┌──┬──┬──┬──┐                     ┌────────────────┐",
            "  │  │  │  │  │  ← 정규 패치         │  ●─┐           │",
            "  ├──┼──┼──┼──┤    (격자 고정)      │ tank│           │",
            "  │  │██│  │  │                     │  ○──┴─○ APC     │",
            "  ├──┼──┼──┼──┤                     │           ○     │",
            "  │  │  │██│  │  ← 패치별 대조      │           art.  │",
            "  └──┴──┴──┴──┘                     └────────────────┘",
            "  객체 이동 시 다른 패치            SAM3 개별 객체 세그멘테이션",
            "  → 동일 객체 추적 불가             → 객체가 이동해도 안정 매칭",
            "",
            "  변화 여부 이진 판정 (O/X)         5상태 정밀 분류",
            "  매칭: 위치 대조                   매칭: CLIP + Gale-Shapley",
            "  산출: 변화 영역/패치 위치        산출: LLM 판독보고서 (IMINT 8섹션)",
        ]
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 3] 객체 페어링 변화 탐지부 — 5상태 분류 흐름도",
        [
            "Sensor DB: 현재 N개 객체 + 과거 M개 객체",
            "         ↓",
            "Step 0. 고정형 객체 우선 분기 (건물·기지·레이더)",
            "  - 위경도 초근접 그리디 결합",
            "  - CLIP 외형 비교 → matched / changed",
            "  - 한쪽 유실 시 가상 탐지 합성 주입",
            "         ↓",
            "Step 1. 마스크 기반 배경 제거 crop (mask_rle zeroing)",
            "         ↓",
            "Step 2. CLIP/ViT 임베딩 산출 + L2 정규화 (N×D, M×D)",
            "         ↓",
            "Step 3. N×M 코사인 유사도 행렬 (단일 GEMM)",
            "         ↓",
            "Step 4. 동일 클래스 필터 + 점수 = 0.8·CLIP + 0.2·크기",
            "         ↓",
            "Step 5. Gale-Shapley 지연 승인 1:1 안정 매칭",
            "         ↓",
            "Step 6. 5상태 분류 (매칭 결과 + FOV 검사)",
            "  매칭 성공 (기동형)                   → matched",
            "  매칭 성공 (고정형, 외형 유사)         → matched",
            "  매칭 성공 (고정형, 외형 상이)         → changed",
            "  미매칭(현재) + 과거 FOV 내           → new",
            "  미매칭(과거) + 현재 FOV 내           → disappeared",
            "  미매칭(현재) + 과거 FOV 밖           → past_not_included",
            "  미매칭(과거) + 현재 FOV 밖           → current_not_included",
            "         ↓",
            "Pairing DB INSERT",
        ]
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 4] GraphRAG 결정론적 누적 인덱싱 흐름도 (LLM 비호출)",
        [
            "Pairing DB: PairingRecord 배치 (session_id로 묶임)",
            "         ↓",
            "EntityExtractor.extract_from_pairings() ★ LLM 비호출",
            "         ↓",
            "  각 페어링마다:",
            "  ┌─ 위경도 격자 양자화 (소수점 2자리 ≈ 1km)",
            "  │   loc_key  = \"loc:37.50,127.00\"",
            "  │   asset_key = \"asset:building:37.50,127.00\"",
            "  ├─ Location 노드 upsert (observation_count +1)",
            "  ├─ Asset 노드 upsert: status_count +1 (★ LLM 비호출)",
            "  ├─ found_at 엣지 upsert (count +1)",
            "  └─ 동일 세션·격자 자산 쌍 → co_occurred_with 엣지",
            "         ↓",
            "_load_networkx() — SQLite → NetworkX 메모리 그래프",
            "         ↓",
            "detect_communities() — Louvain (가중치 = co_occurred_with.count)",
            "  자산 군집 자동 발견:",
            "    Cluster-0: building(5), fuel_storage(3)  — 산업 시설군",
            "    Cluster-1: aircraft(3), runway(2)        — 항공 시설군",
            "    Cluster-2: vehicle(4), road(6)           — 교통 인프라",
            "         ↓",
            "member_summary 결정론 산식 생성 (★ LLM 비호출)",
            "         ↓",
            "Graph DB INSERT",
        ]
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 5] AI Agent LLM 보고서 생성 흐름도 (단일 모델 2회 호출)",
        [
            "Graph DB ─► Local Search (반경 R 자산 노드 이력 통계)",
            "       ├─► Global Search (중첩 커뮤니티 member_summary)",
            "       └─► 토큰 예산(~500) 컨텍스트 블록 포맷화",
            "                 ↓",
            "Pairing DB ─► 변화 객체 추출 (new + disappeared, 신뢰도 순)",
            "                 ↓",
            "시스템 프롬프트 강제 주입:",
            "  ① \"DISAPPEARED ≠ destroyed\" 의미 가드레일",
            "  ② IMINT 8섹션 구조",
            "                 ↓",
            "[1차 호출] LLM AI Agent → 영문 IMINT 보고서",
            "                 ↓",
            "[2차 호출] 동일 LLM 인스턴스 재호출 (인메모리)",
            "  - 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 → 바이패스",
            "  - 분석 서술문만 → 한국어 변환",
            "                 ↓",
            "메타 헤더 부착 → Report DB INSERT",
        ]
    ))

    doc.build(s, onFirstPage=draw_page_num, onLaterPages=draw_page_num)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    build()
