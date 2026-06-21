"""Generate the patent PDF v4 — aligned with the improved system architecture.

Improvements over v3:
  - Architecture diagrams now show every module's tech stack
  - Pairing DB → Graph Indexer → Graph DB → LLM data flow made explicit
  - 6-state status table (matched/moved/new/disappeared/past_not_included/current_not_included)
    with explicit "FOV check" column
  - Korean translation step explicitly shown as a second LLM call
  - Same easy-to-read style (callouts, plain-language summaries) from v3
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

# ── 한글 폰트 ────────────────────────────────────────────
for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try:
        pdfmetrics.registerFont(TTFont("Nanum", p)); break
    except Exception: pass
for p in ["/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]:
    try:
        pdfmetrics.registerFont(TTFont("NanumBold", p)); break
    except Exception: pass
KOR, KORB = "Nanum", "NanumBold"

# ── 색상 ────────────────────────────────────────────────
TITLE_BLUE  = HexColor("#1e3a8a")
LIGHT_BG    = HexColor("#eff6ff")
LIGHT_BORDER = HexColor("#bfdbfe")
SOFT_BG     = HexColor("#f8fafc")
HILITE_BG   = HexColor("#fef9c3")
HILITE_BD   = HexColor("#fde047")
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
style_code = ParagraphStyle("Code", fontName="Courier", fontSize=9, textColor=TEXT_DARK,
                            leading=11, alignment=TA_LEFT)

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
        Paragraph("특허출원 명세서", ParagraphStyle("x",fontName=KORB,fontSize=15,
                  textColor=TITLE_BLUE,alignment=TA_CENTER,spaceAfter=2)),
        Paragraph("(개선 아키텍처 반영 — 4-DB 분리, 6상태 분류, GraphRAG 누적 인덱싱)",
                  ParagraphStyle("x2",fontName=KOR,fontSize=10.5,textColor=TEXT_BODY,
                  alignment=TA_CENTER,spaceAfter=10)),
        meta_row("발명의 명칭 (국문)",
                 "GraphRAG 기반 위성영상 시계열 변화 탐지 AI Agent 판독보고서 자동 생성 시스템 및 방법"),
        meta_row("발명의 명칭 (영문)",
                 "GraphRAG-Augmented AI Agent System and Method for Time-Series Change Detection "
                 "and Automated IMINT Report Generation on Satellite Imagery"),
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

def claim_box(title, lead, body_paragraphs):
    inner = [Paragraph(title, style_claim_t)]
    if lead:
        inner.append(Paragraph(f"<i>📌 쉽게 말하면: {lead}</i>", style_claim_lead))
    for b in body_paragraphs:
        inner.append(Paragraph(b, style_claim_b))
    return boxed(inner, bg=SOFT_BG, border=LIGHT_BORDER)

def fig_box(title, body_lines):
    """body_lines is a list of strings, each a separate centered paragraph."""
    inner = [Paragraph(f"[{title}]", style_fig_t)]
    for line in body_lines:
        inner.append(Paragraph(line, style_fig_b))
    return boxed(inner, bg=SOFT_BG, border=LIGHT_BORDER, pl=14, pr=14, pt=10, pb=10)

def status_table():
    """6상태 분류표 + FOV 체크 컬럼 — 도 2-1에 들어갈 표."""
    data = [
        ["상태", "과거", "현재", "FOV 체크", "의미"],
        ["matched",              "O", "O", "공통 영역",  "정지 객체 (이동 < 100m)"],
        ["moved",                "O", "O", "공통 영역",  "이동 객체 (이동 ≥ 100m)"],
        ["new",                  "X", "O", "과거 FOV 내", "신규 출현 (분석 대상)"],
        ["disappeared",          "O", "X", "현재 FOV 내", "소실 (분석 대상)"],
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
    ]))
    return t

def db_table():
    """4-DB 구조 — Sensor / Pairing / Graph / Report."""
    data = [
        ["DB", "파일", "주요 테이블", "역할"],
        ["Sensor DB", "sensor_detections.db",
         "image_records\ndetection_records", "영상 메타 + 객체 탐지 결과"],
        ["Pairing DB", "object_pairings.db",
         "pairing_records", "객체 페어링 결과 (6상태)"],
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
    out = "/home/user/multi-source-intelligent-system/data/patent_v4_new_architecture.pdf"
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
        "본 발명은 같은 지역을 시간 차이를 두고 촬영한 위성·드론 영상을 입력받아, "
        "(i) <b>텍스트 프롬프트 객체 탐지(SAM3)</b>로 자산을 식별하고 <b>Sensor DB</b>에 저장하며, "
        "(ii) <b>객체 마스크 기반 CLIP 임베딩과 Gale-Shapley 안정 매칭</b>으로 객체 단위 시계열 변화를 "
        "<b>6상태</b>로 분류해 <b>Pairing DB</b>에 누적하고, "
        "(iii) 페어링 결과를 <b>LLM 호출 없이 결정론적으로</b> 격자 양자화 키 기반 지식 그래프 "
        "(<b>Graph DB</b>)에 누적 인덱싱하며, "
        "(iv) <b>Local + Global 이중 검색</b>으로 압축된 역사적 컨텍스트를 산출하고, "
        "(v) 변화 객체와 함께 <b>AI Agent LLM</b>에 주입해 표준 IMINT 8섹션 한국어 판독 보고서를 자동 생성하여 "
        "<b>Report DB</b>에 저장하는 시스템이다."
    ))
    s.append(callout(
        "<b>핵심 차별 키워드</b>: ① 6상태 분류(FOV 공백 분리), ② CLIP + Gale-Shapley 안정 매칭, "
        "③ LLM 비호출 결정론적 그래프 인덱싱, ④ Louvain 자동 군집 탐지, ⑤ 4-DB 분리 아키텍처."
    ))

    s.append(h3("2) 중복성 및 차별성 설명"))
    s.append(bullet(
        "<b>중복성</b>: 본 발명 카테고리에 가장 직접적인 외국 등록 특허로 미국 등록 특허 "
        "<b>US 12,488,225 B1 (Booz Allen Hamilton, 2025년 등록)</b>이 있으나, 이는 다중 출처 정보를 "
        "공통 인텔리전스 픽처(CIP)로 추상 융합하고 분석관의 PIR 질의에 응답하는 구조이며, "
        "객체 단위 시계열 페어링·결정론적 그래프 누적·6상태 분류 등 본 발명의 핵심 구성을 결여한다."
    ))
    s.append(bullet(
        "<b>차별성</b>: 본 발명은 (가) 객체 단위 6상태 분류로 단순 촬영 범위 차이를 변화로 오인하지 "
        "않으며, (나) AI(LLM) 호출 없이 그래프 카운터를 결정론적으로 누적해 동일 입력에 동일 그래프를 "
        "산출하고, (다) Louvain 알고리즘으로 자산 doctrine 군집(예: 기갑 복합체)을 자동 발견하며, "
        "(라) 사전 압축된 ~500토큰 컨텍스트 블록을 LLM 프롬프트에 prepend하여 환각을 원천 차단하는 점에서 "
        "차별된다."
    ))

    s.append(h2("(2) 발명 제안 배경"))
    s.append(para(
        "위성·드론 정찰 체계에서는 다중 시점에 촬영된 영상을 비교하여 자산의 시계열 변화를 보고하는 "
        "것이 분석관의 핵심 임무이나, 종래 시스템은 영상 내 객체의 출현·소실을 *이진 매칭*으로만 도출하여 "
        "(i) 단순 촬영 범위 차이를 자산 변화로 오인하고, (ii) 동일 자산의 반복 관측 이력을 누적하지 못해 "
        "단발성 보고에 그치며, (iii) LLM을 보고서 생성에 활용해도 정형 정보를 자연어로 풀어 재해석하는 "
        "과정에서 환각(hallucination)이 발생하는 한계가 있었다."
    ))
    s.append(para(
        "본 발명은 (i) 객체 단위 페어링과 <b>FOV 공백 상태</b>의 추가 분리로 진짜 변화와 촬영 누락을 "
        "구분하고, (ii) <b>LLM 호출 없는 결정론적 그래프 인덱싱</b>으로 시계열 누적 패턴을 자동 축적하며, "
        "(iii) <b>Local + Global 이중 검색</b>으로 사전 압축된 컨텍스트를 LLM 프롬프트에 prepend하여 "
        "환각을 차단함으로써, 군사적으로 신뢰 가능한 자동 판독 보고서 생성 파이프라인을 제안한다."
    ))

    s.append(h2("(3) 산업상 이용 분야 (용도)"))
    s.append(bullet("<b>국방·방위산업</b>: 위성·정찰기 IMINT 영상 자동 판독 및 표준 8섹션 보고서 자율 생성"))
    s.append(bullet("<b>국경·전장 감시</b>: 적군 자산 누적 활동 패턴 및 doctrine 패턴 자동 식별"))
    s.append(bullet("<b>재난·재해 모니터링</b>: 위성으로 본 시설물 파손·복구 추이 자동 보고"))
    s.append(bullet("<b>스마트시티</b>: 도시 인프라·시설물의 시간별 변화 자동 감시"))

    s.append(h2("(4) 종래 기술"))
    s.append(para(
        "본 발명과 가장 가까운 실제 등록 특허는 미국 특허 <b>US 12,488,225 B1 (\"Modular open system "
        "architecture for common intelligence picture generation\", 양수인: Booz Allen Hamilton Inc., "
        "등록 2025년)</b>이다. 동 특허는 위성·OSINT·CYBER·SIGINT 등 다중 출처 정보를 모듈형 개방 "
        "시스템 아키텍처(MOSA) 기반으로 통합해 공통 인텔리전스 픽처(CIP)를 생성하고, 다중 정보 융합 "
        "LLM이 분석관의 우선 정보 요구사항(PIR)에 응답해 위성 임무 부여, 대응 방안(COA), PDF/PPT 형태의 "
        "보고서를 자동 생성하는 시스템을 개시한다."
    ))

    s.append(h2("(5) 종래 기술의 문제점 / 한계"))
    problems = [
        ("객체 단위 시계열 페어링·6상태 분류 부재",
         "종래는 다중 출처 정보를 추상 CIP로 융합하는 데 초점이 있어, 영상 내 개별 객체를 두 시점에 "
         "1:1로 짝지어 추적하는 알고리즘과 촬영 범위 차이를 분리 보호하는 FOV 공백 상태(past_not_"
         "included/current_not_included) 분류가 부재하다."),
        ("동일 자산의 반복 관측 누적 메모리 부재",
         "CIP는 현재 시점의 스냅샷 구조이며, GPS 노이즈를 흡수하면서 동일 자산의 N번째 반복 출현을 "
         "단일 노드 누적 카운터로 압축 저장하는 결정론적 그래프 인덱싱이 정의되어 있지 않다."),
        ("LLM 의존 비결정적 추출 → 감사 가능성 저하",
         "종래의 그래프 RAG 계열은 노드·엣지 추출에 LLM 호출이 필요해 동일 입력에 매번 결과가 달라지고 "
         "비용이 가변적이며, 군사·법적 책임이 요구되는 환경에서 \"왜 이 정보가 검색되었는가\"를 "
         "행 단위로 추적하기 어렵다."),
        ("자산 doctrine 패턴 자동 발견 부재",
         "전차+APC+포병의 결합(\"기갑 복합체\")이나 레이더+지휘소의 결합(\"C2 인프라\") 같은 "
         "doctrine 패턴을 그래프 알고리즘으로 자동 군집화·발견하는 구성이 없어, 분석관이 일일이 "
         "패턴을 식별해야 한다."),
        ("환각 차단을 위한 사전 압축 컨텍스트 부재",
         "LLM이 도구 호출로 데이터를 자유 조회하므로 응답 가변성과 환각 위험이 크고, 변화 분석에 "
         "무관한 정지/이동 객체로 토큰이 낭비된다."),
        ("도메인 의미 가드레일·표준 형식 강제 부재",
         "\"DISAPPEARED ≠ destroyed\"와 같은 군사 도메인 의미 제약, 표준 IMINT 8섹션 구조 강제, "
         "좌표·수치 토큰 보존 번역 등의 구성이 정의되어 있지 않다."),
    ]
    for i,(t,d) in enumerate(problems,1):
        s.append(bullet(f"<b>{i}. {t}</b> — {d}"))

    # ═══════════════════════════════════════════════════
    # 2. 상세 설명
    # ═══════════════════════════════════════════════════
    s.append(h1("2. 상세 설명"))

    s.append(h2("(1) 시스템 전체 구성 — 4-DB 분리 아키텍처"))
    s.append(para(
        "본 발명은 (A) 객체 탐지부, (B) 객체 페어링 변화 탐지부, (C) GraphRAG 누적 인덱싱부, "
        "(D) Local + Global 검색부, (E) AI Agent 보고서 생성부의 5개 기능 모듈과, 각 모듈이 분리된 "
        "DB로 영속화되는 <b>4-DB 분리 아키텍처</b>로 구성된다."
    ))
    s.append(Spacer(1,3))
    s.append(db_table())
    s.append(Spacer(1,3))
    s.append(note(
        "모든 레코드에 동일한 session_id가 부여되어 단일 파이프라인 실행 단위로 묶이며, 분석관이 "
        "탐지 결과를 수정하면 페어링·그래프·보고서가 자동 재계산되는 HITL(Human-In-The-Loop) "
        "구조를 지원한다."
    ))

    s.append(h2("(2) 모듈별 핵심 구성 및 작동 원리"))

    # ── 구성 A ──
    s.append(h3("[구성 A] 영상 객체 탐지부 (SAM3)"))
    s.append(callout(
        "Real-ESRGAN 초해상도로 영상을 목표 해상도까지 업스케일한 후, <b>SAM3 텍스트 프롬프트 모델</b>로 "
        "20종 군사 자산 클래스(전차·APC·포병·항공기·시설 등)를 zero-shot 탐지하여 (클래스, 신뢰도, "
        "bbox, mask_rle, 위경도)를 Sensor DB의 detection_records에 저장합니다."
    ))
    s.append(bullet(
        "<b>멀티스케일 슬라이딩 윈도우</b>: 전체 영상 1회 추론(대형 객체) + 1008×1008 타일 추론(소형 "
        "객체)을 병렬 수행 후 IoU 0.3 NMS로 통합 → 활주로·건물 단지와 차량·인원을 동일 파이프라인에서 탐지."
    ))
    s.append(bullet(
        "<b>마스크 RLE 저장</b>: SAM3가 산출한 픽셀 단위 세그멘테이션 마스크를 RLE(run-length "
        "encoding) JSON 문자열로 압축 저장하여, 이후 페어링 단계에서 배경 제거 임베딩에 활용한다."
    ))

    # ── 구성 B ──
    s.append(h3("[구성 B] 객체 페어링 변화 탐지부 — 6상태 분류 (Pairing Module)"))
    s.append(callout(
        "Sensor DB에서 현재 영상의 객체 N개와 동일 지역 직전 영상의 객체 M개를 가져와, "
        "<b>객체별 1:1 짝짓기</b>를 수행하고 <b>6가지 상태</b> 중 하나로 분류해 Pairing DB에 저장합니다."
    ))
    s.append(bullet(
        "<b>마스크 기반 배경 제거 임베딩</b>: 각 객체의 mask_rle로 배경 픽셀을 영(0)으로 마스킹한 후 "
        "bbox crop하여 CLIP/ViT 인코더에 입력 → L2 정규화된 D차원 임베딩 벡터 산출. 배경 노이즈를 "
        "제거해 동일 클래스 객체 간 식별력을 향상시킨다."
    ))
    s.append(bullet(
        "<b>N×M 코사인 유사도 행렬 + 동일 클래스 필터</b>: 두 임베딩 행렬의 단일 행렬곱(GEMM)으로 "
        "모든 쌍의 코사인 유사도를 한 번에 산출한 뒤 동일 클래스 쌍만 후보로 유지 (전차↔전차, APC↔APC). "
        "점수 = 0.8 × 코사인 유사도 + 0.2 × 크기 유사도."
    ))
    s.append(bullet(
        "<b>Gale-Shapley 안정 매칭</b>: 점수 임계값(0.5) 이상의 후보 쌍에 대해 <b>지연 승인(deferred "
        "acceptance) 알고리즘</b>으로 1:1 안정 매칭 수행. 그리디 매칭의 국소 최적 함정을 회피하고 "
        "중복·교차 페어링을 원천 배제한다."
    ))
    s.append(bullet(
        "<b>고정형 객체 전용 분기</b>: 건물·기지·레이더 등 <b>물리적으로 이동 불가능</b>한 고정 자산은 "
        "Step 0에서 별도 처리 — (i) 위경도 11m 이내 그리디 결합 → (ii) 외형 변화 판정으로 "
        "<b>matched(변화 없음) / changed(구조 변화)</b> 구분 → (iii) 한쪽 탐지 실패 시 상대 영상의 동일 "
        "위경도 영역을 강제 crop 후 CLIP 비교하여 가상 탐지 레코드 합성 주입(detection 누락 보정)."
    ))
    s.append(bullet(
        "<b>6상태 분류 (FOV 공백 분리 포함)</b>: 매칭 결과와 영상 FOV 경계면 검사로 객체를 다음 "
        "6상태 중 하나로 정밀 배정한다."
    ))
    s.append(Spacer(1,3))
    s.append(status_table())
    s.append(Spacer(1,5))

    # ── 구성 C ──
    s.append(h3("[구성 C] GraphRAG 결정론적 누적 인덱싱부 (Graph Indexer · Graph DB)"))
    s.append(callout(
        "Pairing DB의 페어링 결과를 <b>AI(LLM)를 부르지 않고도</b> 격자 양자화 기반 결정론 키로 "
        "지식 그래프 노드·엣지에 누적 인덱싱한 후, Louvain 알고리즘으로 자산 doctrine 군집을 "
        "자동 발견하여 Graph DB에 저장합니다."
    ))
    s.append(bullet(
        "<b>격자 양자화 결정론 키 생성</b>: 위경도 소수점 2자리(약 1km × 1km 격자)로 양자화하여 "
        "<i>\"loc:37.50,127.00\"</i> 형식의 Location 노드 키와 <i>\"asset:military_tank:37.50,127.00\"</i> "
        "형식의 Asset 노드 키를 결정론적으로 생성. GPS 미세 오차에도 동일 자산이 같은 노드로 자동 통합된다."
    ))
    s.append(bullet(
        "<b>카운터 누적 upsert (LLM 호출 0회)</b>: 페어링의 status 필드를 자산 노드의 카운터 속성"
        "(new_count, matched_count, moved_count, disappeared_count, total_confidence)에 산술 가산. "
        "<b>동일 입력에 대해 매번 동일한 그래프가 산출되는 완전 재현성</b>을 보장한다."
    ))
    s.append(bullet(
        "<b>관계 엣지 누적</b>: asset → location의 <i>found_at</i> 엣지와 동일 세션·동일 격자 공출현 "
        "자산 쌍의 <i>co_occurred_with</i> 엣지의 count 가중치를 누적 갱신한다."
    ))
    s.append(bullet(
        "<b>Louvain 군집 자동 탐지</b>: co_occurred_with 엣지 가중치 기반 Louvain 알고리즘으로 자산 "
        "doctrine 군집(예: 기갑+APC+포병 = 기갑 복합체, 레이더+지휘소 = C2 인프라, 항공기+활주로 = "
        "항공 작전 거점)을 자동 발견하고, 각 군집의 member_summary를 LLM 호출 없이 결정론적 통계 "
        "산식만으로 자동 생성한다."
    ))

    # ── 구성 D ──
    s.append(h3("[구성 D] Local + Global 이중 검색부 (Graph Retriever)"))
    s.append(bullet(
        "<b>Local Search</b>: 보고서 대상 좌표 반경 R(예: 0.05° ≈ 5.5km) 내의 자산 노드 이력 통계 "
        "(누적 관측 수, 신규/지속/이동/소실 횟수, 평균 신뢰도, 최초/최종 관측 시각)를 조회한다."
    ))
    s.append(bullet(
        "<b>Global Search</b>: 동일 반경과 중첩되는 Louvain 커뮤니티의 member_summary를 조회한다."
    ))
    s.append(bullet(
        "<b>사전 압축 컨텍스트 블록</b>: 양 검색 결과를 사전 설정 토큰 예산(예: 500 토큰) 이내로 "
        "절단·머지하여 <i>\"=== GRAPHRAG HISTORICAL CONTEXT ===\"</i> 단일 블록으로 포맷화한다."
    ))

    # ── 구성 E ──
    s.append(h3("[구성 E] AI Agent 보고서 생성부 — 단일 LLM 2회 호출 (LLM · Report DB)"))
    s.append(callout(
        "변화 객체(new/disappeared)와 압축 컨텍스트를 단일 LLM 에이전트에 주입하여 영문 IMINT 보고서 "
        "1차 생성 → 같은 LLM 인스턴스를 재호출해 한국어 무오염 번역 → Report DB에 영속 저장."
    ))
    s.append(bullet(
        "<b>변화 객체 한정 추출</b>: 페어링 결과 중 'new'·'disappeared' 상태만 신뢰도 내림차순으로 "
        "추출. 'matched'·'moved'·FOV 공백 상태는 LLM 프롬프트에서 제외하여 토큰 효율과 분석 집중도를 "
        "동시에 확보한다."
    ))
    s.append(bullet(
        "<b>군사 도메인 의미 가드레일 강제</b>: 시스템 프롬프트에 <i>\"'DISAPPEARED'는 영상에서의 "
        "미관측을 의미하며 파괴(destroyed)가 아니다\"</i> 의미 제약을 명시적으로 강제 주입하여 LLM의 "
        "군사적 오판을 차단한다."
    ))
    s.append(bullet(
        "<b>표준 IMINT 8섹션 강제</b>: (1) CLASSIFICATION (2) EXECUTIVE SUMMARY (3) SITUATION "
        "(4) CHANGE ANALYSIS (5) THREAT ASSESSMENT (6) INTELLIGENCE GAPS (7) RECOMMENDED ACTIONS "
        "(8) APPENDIX의 8개 섹션 구조를 LLM 프롬프트에 강제하여 분석관 친화 산출물을 보장한다."
    ))
    s.append(bullet(
        "<b>동일 LLM 인스턴스 2차 호출 한국어 번역</b>: 1차 영문 생성 후 같은 LLM 인스턴스를 인메모리 "
        "상태로 재호출하여 한국어 번역 세션 발동. <b>섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 토큰은 "
        "변형 없이 바이패스</b>하고 분석 서술문만 변환하여 다국어 일관성을 확보한다."
    ))

    s.append(h2("(3) 발명의 효과"))
    effects = [
        ("객체 단위 6상태 정밀 분류",
         "CLIP + Gale-Shapley 안정 매칭과 FOV 공백 상태 분리로 \"몇 대 늘었다·줄었다\" 수준이 아닌 "
         "\"어떤 자산이 어디로 이동했고, 어떤 공백이 단순 촬영 누락인가\"까지 정확히 추적한다."),
        ("LLM 비호출 결정론 — 비용·재현성·감사 가능성 동시 확보",
         "그래프 인덱싱에 AI 호출이 전혀 없어 운용 비용이 거의 0이고, 동일 입력에 매번 동일한 그래프가 "
         "산출되어 군사·법적 감사 요구를 충족한다."),
        ("시계열 인텔리전스 격상",
         "사용할수록 그래프가 누적되어 \"N번째 반복 배치, K번 소실 후 재등장\" 같은 시계열 패턴과 "
         "Louvain doctrine 군집이 보고서에 자동 반영된다."),
        ("환각 차단 + 토큰 효율",
         "사전 압축 ~500토큰 컨텍스트 + 변화 객체 한정 전달 + DISAPPEARED 의미 가드레일의 결합으로 "
         "환각을 원천 차단하면서 LLM 호출 비용을 절감한다."),
        ("4-DB 분리로 HITL 재처리 지원",
         "분석관이 Sensor DB의 탐지 결과를 수정하면 Pairing → Graph → Report가 자동 재계산되며, "
         "각 DB에 부여된 session_id로 실행 단위 역추적이 가능하다."),
    ]
    for i,(t,d) in enumerate(effects,1):
        s.append(bullet(f"<b>{i}. {t}</b> — {d}"))

    # ═══════════════════════════════════════════════════
    # 3. 청구범위
    # ═══════════════════════════════════════════════════
    s.append(h1("3. 특허 청구범위"))
    s.append(note(
        "각 청구항 상단에 비전문가용 '쉬운 설명'을 표시했습니다. 실제 권리범위는 그 아래의 정형 문장에 "
        "의해 정의됩니다."
    ))

    s.append(claim_box(
        "[청구항 1] (독립항) — 전체 방법",
        "위성·드론 영상에서 객체별 변화를 6상태로 분류하고, AI 없이 그래프에 누적해 두었다가, "
        "보고서 생성 시 그 그래프 메모를 압축해 LLM에게 전달하여 표준 보고서를 자동 작성하는 방법.",
        [
            "임의의 대상 지역을 촬영한 시계열 항공 영상을 수집하여 텍스트 프롬프트 기반 객체 탐지 "
            "알고리즘으로 자산 클래스·위경도·바운딩 박스·세그멘테이션 마스크를 포함하는 탐지 레코드를 "
            "생성해 센서 데이터베이스에 저장하는 단계;",
            "각 탐지 객체에 대해 상기 세그멘테이션 마스크로 배경 픽셀을 영(0)으로 처리한 후 바운딩 "
            "박스 크롭 이미지로부터 비전 인코더에 의해 L2 정규화된 시각 임베딩 벡터를 산출하고, 최신 "
            "영상의 N개 임베딩과 과거 영상의 M개 임베딩의 N×M 코사인 유사도 행렬에서 동일 클래스 쌍에 "
            "대해 임베딩 유사도와 크기 유사도의 가중합 점수가 임계값 이상인 후보 쌍에 대하여 Gale-"
            "Shapley 지연 승인 알고리즘으로 1:1 안정 매칭을 수행한 후, 매칭된 쌍의 지리 좌표 거리에 "
            "따라 정지(matched) 또는 이동(moved)을, 미매칭 현재 객체에는 신규 출현(new)을, 미매칭 과거 "
            "객체에는 소실(disappeared)을 부여하며, 미매칭 객체가 상대 영상의 시야각 경계면 외부에 "
            "위치하는 경우 과거 미포함(past_not_included) 또는 현재 미포함(current_not_included) "
            "상태를 부여하여 6상태로 정밀 분류하고 페어링 데이터베이스에 저장하는 단계;",
            "각 자산의 위경도 좌표를 정수형 격자 단위로 양자화하여 위치 노드 키를, 객체 클래스와 위치 "
            "키의 결합으로 자산 노드 키를 결정론적으로 생성하고, 상기 6상태에 따라 자산 노드의 누적 "
            "카운터 속성을 거대언어모델(LLM) 호출 없이 갱신하며, 자산-위치 간 found_at 엣지 및 동일 "
            "세션·동일 격자 내 공출현 자산 쌍의 co_occurred_with 엣지의 가중치를 누적 갱신하여 그래프 "
            "데이터베이스에 지식 그래프를 형성하는 단계;",
            "상기 co_occurred_with 엣지 가중치를 기반으로 Louvain 군집화를 수행하여 자산 복합체 "
            "커뮤니티 요약문을 LLM 호출 없이 구조적으로 생성하고, 보고서 대상 좌표 반경 내 자산 노드 "
            "이력을 조회하는 Local Search와 동일 반경과 중첩되는 커뮤니티 요약을 조회하는 Global "
            "Search를 병합하여 사전 설정 토큰 예산 이내의 역사적 맥락 컨텍스트 블록을 생성하는 단계; 및",
            "상기 6상태 중 신규 출현 또는 소실에 해당하는 객체만을 신뢰도 내림차순으로 추출하고, 상기 "
            "역사적 맥락 컨텍스트 블록을 LLM 프롬프트 선두에 prepend하며, 시스템 프롬프트에 "
            "\"'DISAPPEARED'는 영상 미관측을 의미하며 파괴 확인이 아니다\"라는 의미 제약과 표준 IMINT "
            "8개 섹션 구조를 강제 주입하여 LLM 에이전트로 영문 보고서를 생성한 후 동일 LLM 인스턴스를 "
            "재호출해 한국어로 번역하여 보고서 데이터베이스에 저장하는 단계;",
            "를 포함하는 위성·드론 영상의 시계열 변화 탐지 및 GraphRAG 누적 인덱싱 기반 AI 에이전트 "
            "판독 보고서 자동 생성 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 2] (종속) — 마스크 기반 배경 제거 임베딩",
        "객체 비교 전에 배경을 까맣게 지운 뒤 그 객체만 잘라서 비교하여 배경 노이즈로 인한 오인을 차단.",
        [
            "제1항에 있어서, 탐지된 자산의 세그멘테이션 마스크 RLE 정보를 디코딩하여 마스크 활성 영역 "
            "외의 픽셀을 영(0)으로 마스킹하고 바운딩 박스 주위에 사전 설정된 패딩 폭을 부가하여 크롭하며, "
            "마스크 RLE가 부재하는 경우 바운딩 박스 크롭으로 폴백하고, 산출된 임베딩 벡터를 L2 정규화하여 "
            "코사인 유사도 행렬 산출에 사용하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 3] (종속) ★ 6상태 FOV 공백 분류",
        "촬영 범위가 다른 부분에 있는 객체를 진짜 변화와 구분하여 분석관 오판을 방지하는 핵심 안전장치.",
        [
            "제1항에 있어서, 상기 페어링 데이터베이스의 status 필드는 (i) 두 영상에 모두 존재하고 이동 "
            "거리가 임계값 미만인 matched, (ii) 두 영상에 모두 존재하고 이동 거리가 임계값 이상인 moved, "
            "(iii) 현재 영상에만 존재하고 과거 영상의 시야각 경계면 내부에 위치하는 new, (iv) 과거 영상에만 "
            "존재하고 현재 영상의 시야각 경계면 내부에 위치하는 disappeared, (v) 현재 영상에만 존재하고 "
            "과거 영상의 시야각 경계면 외부에 위치하는 past_not_included, (vi) 과거 영상에만 존재하고 "
            "현재 영상의 시야각 경계면 외부에 위치하는 current_not_included의 6상태 중 하나로 분류되며, "
            "(v)·(vi) 상태에 해당하는 객체는 LLM 보고서 생성 시 변화 객체 추출 단계에서 제외되어 단순 "
            "촬영 범위 차이가 자산 변화로 오인되지 않도록 하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 4] (종속) — 고정형 객체 전용 분기 + 가상 탐지 합성",
        "건물·시설 같이 못 움직이는 객체는 위경도 11m 이내 결합 + 외형 변화 판정 + 탐지 누락 보정으로 "
        "처리하는 별도 알고리즘.",
        [
            "제1항에 있어서, 사전 정의된 정적 고정 자산 클래스(건물·기지·연료고·레이더 등)에 속하는 "
            "객체에 대해서는, (a) 위경도 거리가 정밀 임계값 이내인 동일 클래스 쌍을 최우선 그리디 결합하는 "
            "단계, (b) 결합된 쌍의 마스크 기반 배경 제거 임베딩의 CLIP 유사도가 임계값 이상이면 matched, "
            "미만이면 changed(구조 변화) 상태를 부여하는 단계, 및 (c) 한쪽 영상에서만 탐지된 경우 상대 "
            "영상의 동일 위경도 영역을 강제 크롭하여 CLIP 유사도를 재산출하고 임계값 이상이면 "
            "source_type=synthetic으로 표시된 가상 탐지 레코드를 상대 영상 DB에 합성 주입하여 페어링을 "
            "복원하는 단계를 더 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 5] (종속) ★ LLM 비호출 결정론적 그래프 인덱싱",
        "AI를 부르지 않고도 그래프 노드의 카운터를 +1씩 늘려서 자동 누적. 같은 입력에 매번 같은 결과 "
        "(완전 재현). 본 발명의 가장 강한 차별점.",
        [
            "제1항에 있어서, 상기 지식 그래프 형성 단계는, 각 자산의 위경도를 소수점 N자리의 정수형 "
            "격자 단위로 반올림하여 \"loc:{lat:.Nf},{lon:.Nf}\" 형식의 결정론적 위치 노드 키와 \"asset:"
            "{class}:{loc_key}\" 형식의 결정론적 자산 노드 키를 생성하는 단계; 페어링 레코드의 status "
            "필드 및 신뢰도를 자산 노드의 카운터 속성(new_count, matched_count, moved_count, "
            "disappeared_count, total_confidence) 및 시간 속성에 LLM 호출 없이 결정론적으로 누적 가산 "
            "upsert하는 단계; 및 자산-위치 간 found_at 엣지 및 동일 세션·동일 위치 키 내 공출현 자산 "
            "쌍의 co_occurred_with 엣지의 count 속성을 LLM 호출 없이 결정론적으로 누적 갱신하는 단계를 "
            "포함하여, 동일 입력에 대해 매번 동일한 그래프가 산출되는 결정론적 재현성을 갖는 것을 특징으로 "
            "하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 6] (종속) — Louvain 군집 + 결정론적 요약 자동 생성",
        "전차·장갑차·포병이 자주 같이 등장하면 \"기갑 복합체\"로 자동 묶고, 군집 요약도 AI 없이 통계로 "
        "자동 생성.",
        [
            "제1항에 있어서, co_occurred_with 엣지의 count 가중치를 입력으로 NetworkX 등의 그래프 "
            "라이브러리 상에서 Louvain 계층적 군집화 알고리즘을 수행하는 단계; 및 각 커뮤니티에 대해 "
            "구성원 자산 클래스 분포·위치 클러스터 수·총 관측 수·신규 배치 수·소실 수를 포함하는 "
            "member_summary를 LLM 호출 없이 결정론적 통계 산식만으로 생성하여 그래프 데이터베이스의 "
            "graph_communities 테이블에 기록하는 단계를 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 7] (종속) — Local + Global 이중 검색 + 토큰 예산 컨텍스트",
        "해당 지역과 관련된 메모만 추려 ~500자로 압축해 AI 입력 맨 앞에 붙여 주는 방식.",
        [
            "제1항에 있어서, 보고서 대상 좌표 반경 R 이내의 자산 노드 이력 통계를 조회하는 Local Search "
            "단계; 동일 반경과 중첩되는 커뮤니티들의 member_summary를 조회하는 Global Search 단계; 및 "
            "양 검색 결과를 사전 설정 토큰 예산(예: 500 토큰) 이내로 절단·머지하여 LLM 프롬프트 선두에 "
            "prepend되는 단일 컨텍스트 블록으로 포맷화하는 단계를 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 8] (종속) — 변화 객체 한정 + IMINT 8섹션 + 동일 LLM 2회 호출",
        "정지·이동·FOV 공백은 빼고 신규·소실만 AI에 전달, 표준 8개 항목과 안전 문구 강제, 같은 AI 인스턴스로 "
        "한국어 변환 시 좌표·수치 보존.",
        [
            "제1항에 있어서, 페어링 결과 중 matched·moved·past_not_included·current_not_included 상태 "
            "객체는 LLM 프롬프트에서 제외하고 new·disappeared 상태 객체만 신뢰도 내림차순으로 주입하는 "
            "단계; 시스템 프롬프트에 표준 IMINT 8개 섹션 구조 및 \"'DISAPPEARED'는 영상 미관측이며 파괴가 "
            "아니다\" 의미 제약을 강제 주입하는 단계; 및 동일 LLM 인스턴스를 인메모리 상태로 재사용하여 "
            "영문 보고서 생성 후 한국어 번역 세션을 발동하되 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 "
            "토큰은 변형 없이 바이패스하고 분석 서술문만 변환하는 단계를 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 9] (종속) — 4-DB 분리 + 세션 추적성",
        "탐지·페어링·그래프·보고서를 각각 별도 DB로 저장하고 session_id로 묶어, 결과 수정 시 자동 재계산 "
        "(HITL).",
        [
            "제1항에 있어서, 영상 메타데이터 및 탐지 결과를 저장하는 센서 데이터베이스, 페어링 결과를 "
            "저장하는 페어링 데이터베이스, 지식 그래프 노드·엣지·커뮤니티를 저장하는 그래프 데이터베이스, "
            "및 LLM 보고서를 저장하는 보고서 데이터베이스로 물리적으로 분리된 4개의 데이터베이스를 "
            "포함하며, 모든 레코드에 동일한 세션 식별자(session_id)가 부여되어 분석관이 센서 DB의 탐지 "
            "결과를 수정하면 페어링·그래프·보고서가 동일 세션 단위로 자동 재계산되는 것을 특징으로 하는 "
            "방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 10] (독립항) — 시스템 / 기록매체",
        "위의 방법을 수행하는 컴퓨터 시스템 및 그 프로그램이 기록된 기록매체.",
        [
            "제1항 내지 제9항 중 어느 한 항의 방법을 수행하는 하나 이상의 하드웨어 프로세서와 메모리를 "
            "포함하는 GraphRAG 기반 AI 에이전트 판독 보고서 자동 생성 시스템 및 상기 방법을 컴퓨터에서 "
            "실행시키기 위한 프로그램이 기록된 컴퓨터 판독 가능 기록 매체.",
        ],
    ))

    # ═══════════════════════════════════════════════════
    # 4. 도면
    # ═══════════════════════════════════════════════════
    s.append(h1("4. 도면 설명 명기"))

    s.append(fig_box(
        "도 1] 시스템 전체 구성도 (4-DB 분리 아키텍처)",
        [
            "[3/17 영상] + [3/18 영상]",
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
            "  │  [구성 B] Pairing Module       │ ──► Pairing DB",
            "  │  CLIP+ViT 임베딩 (배경 제거)   │     (6상태:",
            "  │  N×M 코사인 행렬 + 동일 클래스 │      matched, moved,",
            "  │  Gale-Shapley 안정 매칭         │      new, disappeared,",
            "  │  + 고정형 객체 분기            │      past_not_included,",
            "  │  + FOV 가드 (6상태 분류)       │      current_not_included)",
            "  └──────────────────────────────┘",
            "         │ 페어링 레코드 배치",
            "         ▼",
            "  ┌──────────────────────────────┐",
            "  │  [구성 C] Graph Indexer        │ ──► Graph DB",
            "  │  격자 양자화(loc/asset 키)    │     (graph_entities,",
            "  │  LLM 비호출 카운터 upsert      │      graph_relations,",
            "  │  found_at / co_occurred_with  │      graph_communities)",
            "  │  Louvain 군집 자동 탐지        │",
            "  └──────────────────────────────┘",
            "         │",
            "         ▼",
            "  ┌──────────────────────────────┐",
            "  │  [구성 D] Graph Retriever      │",
            "  │  Local Search + Global Search │",
            "  │  ~500 토큰 컨텍스트 블록       │",
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
        "도 2] 객체 페어링 변화 탐지부 — 6상태 분류 흐름도",
        [
            "Sensor DB: 현재 영상 N개 객체 + 과거 영상 M개 객체",
            "         ↓",
            "Step 0. 고정형 객체 우선 분기 (건물·기지·레이더)",
            "  - 위경도 11m 이내 그리디 결합",
            "  - CLIP 외형 비교 → matched / changed",
            "  - 한쪽 유실 시 가상 탐지 합성 주입",
            "         ↓",
            "Step 1. 마스크 기반 배경 제거 crop (mask_rle 디코딩 후 zeroing)",
            "         ↓",
            "Step 2. CLIP/ViT 임베딩 산출 + L2 정규화 (N×D, M×D)",
            "         ↓",
            "Step 3. N×M 코사인 유사도 행렬 (단일 GEMM)",
            "         ↓",
            "Step 4. 동일 클래스 필터 + 점수 = 0.8·CLIP + 0.2·크기",
            "         ↓",
            "Step 5. Gale-Shapley 지연 승인 1:1 안정 매칭 (점수 > 0.5)",
            "         ↓",
            "Step 6. 6상태 분류 (위경도 거리 + FOV 검사)",
            "  매칭 + dist < 100m → matched",
            "  매칭 + dist ≥ 100m → moved",
            "  미매칭(현재) + 과거 FOV 내 → new",
            "  미매칭(과거) + 현재 FOV 내 → disappeared",
            "  미매칭(현재) + 과거 FOV 밖 → past_not_included (촬영 공백)",
            "  미매칭(과거) + 현재 FOV 밖 → current_not_included (촬영 공백)",
            "         ↓",
            "Pairing DB INSERT (insert_pairings_bulk)",
        ]
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 3] GraphRAG 결정론적 누적 인덱싱 흐름도",
        [
            "Pairing DB: PairingRecord 배치 (session_id로 묶임)",
            "         ↓",
            "EntityExtractor.extract_from_pairings() — LLM 비호출",
            "         ↓",
            "  각 페어링 마다:",
            "  ┌─ 위경도 격자 양자화 (소수점 2자리 ≈ 1km)",
            "  │   loc_key  = \"loc:37.50,127.00\"",
            "  │   asset_key = \"asset:military_tank:37.50,127.00\"",
            "  ├─ Location 노드 upsert (observation_count +1)",
            "  ├─ Asset 노드 upsert: status_count +1 (LLM 호출 0회)",
            "  ├─ found_at 엣지 upsert (count +1)",
            "  └─ 동일 세션·격자 자산 쌍 → co_occurred_with 엣지 (양방향)",
            "         ↓",
            "_load_networkx() — SQLite → NetworkX 메모리 그래프",
            "         ↓",
            "detect_communities() — Louvain (가중치 = co_occurred_with.count)",
            "  자산 군집 자동 발견:",
            "    Cluster-0: tank(5), APC(3), artillery(2) — 기갑 복합체",
            "    Cluster-1: radar(4), command_post(2)     — C2 인프라",
            "    Cluster-2: aircraft(3), runway(2)        — 항공 작전 거점",
            "         ↓",
            "각 군집에 대해 member_summary 결정론 산식 생성 (LLM 비호출)",
            "         ↓",
            "Graph DB INSERT (graph_communities 전체 재구축)",
        ]
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 4] AI Agent LLM 보고서 생성 흐름도 (단일 모델 2회 호출)",
        [
            "Graph DB ─► Local Search (반경 R 자산 노드 이력 통계)",
            "       │",
            "       ├─► Global Search (중첩 커뮤니티 member_summary)",
            "       │",
            "       └─► 토큰 예산(~500) 컨텍스트 블록 포맷화:",
            "             === GRAPHRAG HISTORICAL CONTEXT ===",
            "             ...",
            "             === END HISTORICAL CONTEXT ===",
            "                 ↓",
            "Pairing DB ─► 변화 객체 추출 (new + disappeared만, 신뢰도 순)",
            "                 ↓",
            "시스템 프롬프트 강제 주입:",
            "  ① \"DISAPPEARED ≠ destroyed\" 의미 가드레일",
            "  ② IMINT 8섹션 구조 (CLASSIFICATION ~ APPENDIX)",
            "                 ↓",
            "최종 프롬프트 조립 (컨텍스트 prepend + 변화 객체)",
            "                 ↓",
            "[1차 호출] LLM AI Agent → 영문 IMINT 보고서",
            "                 ↓",
            "[2차 호출] 동일 LLM 인스턴스 재호출 (인메모리)",
            "  - 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 → 바이패스",
            "  - 분석 서술문만 → 한국어 변환",
            "                 ↓",
            "메타 헤더 부착 (모델명, 세션ID, 관측 시각, 페어링 통계)",
            "                 ↓",
            "Report DB INSERT (report_records)",
        ]
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 5] 전체 파이프라인 구동 시퀀스",
        [
            "시작",
            " │",
            " ▼ ① 영상 적재 + Real-ESRGAN 초해상도",
            " ▼ ② SAM3 멀티스케일 탐지 → Sensor DB",
            " ▼ ③ 마스크 RLE 배경 제거 crop",
            " ▼ ④ CLIP/ViT 임베딩 산출",
            " ▼ ⑤ N×M 코사인 행렬 + 동일 클래스 필터",
            " ▼ ⑥ Gale-Shapley 안정 매칭",
            " ▼ ⑦ FOV 검사 + 6상태 분류 → Pairing DB",
            " ▼ ⑧ 격자 양자화 결정론 키 생성",
            " ▼ ⑨ LLM 비호출 그래프 카운터 upsert → Graph DB",
            " ▼ ⑩ Louvain 군집 자동 탐지",
            " ▼ ⑪ Local + Global 검색 + ~500 토큰 컨텍스트",
            " ▼ ⑫ 변화 객체 추출 + IMINT 8섹션 + 가드레일",
            " ▼ ⑬ LLM 1차 호출 (영문 보고서)",
            " ▼ ⑭ 동일 LLM 2차 호출 (한국어 번역, 좌표 보존)",
            " ▼ ⑮ Report DB 저장 + 메타 헤더 부착",
            "종료",
        ]
    ))

    doc.build(s, onFirstPage=draw_page_num, onLaterPages=draw_page_num)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    build()
