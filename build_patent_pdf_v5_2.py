"""Patent PDF v5.2 — main-claude aligned, 차별성 설명 간단화."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)

pdfmetrics.registerFont(TTFont("Nanum", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"))
pdfmetrics.registerFont(TTFont("Nanum-Bold", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"))
registerFontFamily("Nanum", normal="Nanum", bold="Nanum-Bold", italic="Nanum", boldItalic="Nanum-Bold")
KOR, KORB = "Nanum", "Nanum-Bold"

TITLE_BLUE, LIGHT_BG, LIGHT_BORDER = HexColor("#1e3a8a"), HexColor("#eff6ff"), HexColor("#bfdbfe")
SOFT_BG, HILITE_BG, HILITE_BD = HexColor("#f8fafc"), HexColor("#fef9c3"), HexColor("#fde047")
TABLE_HD, TEXT_BODY, TEXT_DARK, MUTED = HexColor("#1e293b"), HexColor("#1f2937"), HexColor("#111827"), HexColor("#6b7280")

sH1 = ParagraphStyle("H1", fontName=KORB, fontSize=14, textColor=TITLE_BLUE, leftIndent=10, spaceBefore=14, spaceAfter=6, leading=18)
sH2 = ParagraphStyle("H2", fontName=KORB, fontSize=11.5, textColor=TEXT_DARK, spaceBefore=10, spaceAfter=4, leading=15)
sH3 = ParagraphStyle("H3", fontName=KORB, fontSize=10.5, textColor=TEXT_DARK, spaceBefore=7, spaceAfter=2, leading=14)
sBody = ParagraphStyle("Body", fontName=KOR, fontSize=10, textColor=TEXT_BODY, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=4, firstLineIndent=10)
sBodyN = ParagraphStyle("BodyN", fontName=KOR, fontSize=10, textColor=TEXT_BODY, leading=16.5, alignment=TA_JUSTIFY, spaceAfter=4)
sBul = ParagraphStyle("Bul", fontName=KOR, fontSize=10, textColor=TEXT_BODY, leading=15.5, leftIndent=18, bulletIndent=8, spaceAfter=3, alignment=TA_JUSTIFY)
sNote = ParagraphStyle("Note", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY, leading=14.5, leftIndent=18, spaceAfter=2, alignment=TA_JUSTIFY)
sML = ParagraphStyle("ML", fontName=KORB, fontSize=10, textColor=TITLE_BLUE)
sMV = ParagraphStyle("MV", fontName=KOR, fontSize=10, textColor=TEXT_BODY, leading=15)
sCT = ParagraphStyle("CT", fontName=KORB, fontSize=10.5, textColor=TITLE_BLUE, spaceAfter=4)
sCL = ParagraphStyle("CL", fontName=KOR, fontSize=9.5, textColor=HexColor("#4b5563"), leading=14.5, spaceAfter=4, leftIndent=4)
sCB = ParagraphStyle("CB", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY, leading=15, spaceAfter=3, leftIndent=4, alignment=TA_JUSTIFY)
sFT = ParagraphStyle("FT", fontName=KORB, fontSize=10.5, textColor=TITLE_BLUE, alignment=TA_CENTER, spaceAfter=3)
sFB = ParagraphStyle("FB", fontName=KOR, fontSize=9.5, textColor=TEXT_BODY, alignment=TA_LEFT, leading=14.5)


def boxed(flow, bg=LIGHT_BG, bd=LIGHT_BORDER, pl=12, pr=12, pt=10, pb=10):
    t = Table([[flow]], colWidths=[None])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),bg),("BOX",(0,0),(-1,-1),0.7,bd),
        ("LEFTPADDING",(0,0),(-1,-1),pl),("RIGHTPADDING",(0,0),(-1,-1),pr),
        ("TOPPADDING",(0,0),(-1,-1),pt),("BOTTOMPADDING",(0,0),(-1,-1),pb)]))
    return t

def meta(l, v):
    t = Table([[Paragraph(l, sML), Paragraph(v, sMV)]], colWidths=[3.8*cm, None])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),0)]))
    return t

def h1(t):
    bar = Table([[" ", Paragraph(t, sH1)]], colWidths=[5, None])
    bar.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),TITLE_BLUE),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    return KeepTogether([Spacer(1,6), bar, Spacer(1,2)])

def h2(t): return Paragraph(f"<b>{t}</b>", sH2)
def h3(t): return Paragraph(f"<b>{t}</b>", sH3)
def p(t, i=True): return Paragraph(t, sBody if i else sBodyN)
def b(t): return Paragraph(f"• {t}", sBul)
def note(t): return Paragraph(f"※ {t}", sNote)

def callout(t):
    return boxed([Paragraph(t, ParagraphStyle("CO", fontName=KOR, fontSize=9.5, textColor=TEXT_DARK, leading=14.5, alignment=TA_JUSTIFY))],
                 bg=HILITE_BG, bd=HILITE_BD, pl=10, pr=10, pt=7, pb=7)

def claim(title, lead, body):
    inner = [Paragraph(title, sCT)]
    if lead: inner.append(Paragraph(f"<i>📌 쉽게 말하면: {lead}</i>", sCL))
    for x in body: inner.append(Paragraph(x, sCB))
    return boxed(inner, bg=SOFT_BG, bd=LIGHT_BORDER)

def fig(title, lines):
    inner = [Paragraph(f"[{title}]", sFT)] + [Paragraph(l, sFB) for l in lines]
    return boxed(inner, bg=SOFT_BG, bd=LIGHT_BORDER, pl=14, pr=14, pt=10, pb=10)

def status_table():
    data = [["상태","과거","현재","FOV 체크","의미"],
        ["matched","O","O","공통 영역","동일 객체 확인 (변화 없음)"],
        ["changed","O","O","동일 위치","고정 시설물 구조 변화"],
        ["new","X","O","과거 FOV 내","신규 출현 (변화 O)"],
        ["disappeared","O","X","현재 FOV 내","소실 (변화 O)"],
        ["past_not_included","X","O","과거 FOV 밖","촬영 공백 (오판 방지)"],
        ["current_not_included","O","X","현재 FOV 밖","촬영 공백 (오판 방지)"]]
    t = Table(data, colWidths=[3.8*cm,1.0*cm,1.0*cm,2.6*cm,6.0*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),TABLE_HD),("TEXTCOLOR",(0,0),(-1,0),HexColor("#ffffff")),
        ("FONTNAME",(0,0),(-1,0),KORB),("FONTNAME",(0,1),(-1,-1),KOR),("FONTSIZE",(0,0),(-1,-1),9),
        ("ALIGN",(1,0),(3,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LINEBELOW",(0,0),(-1,-1),0.3,LIGHT_BORDER),("LEFTPADDING",(0,0),(-1,-1),4),
        ("RIGHTPADDING",(0,0),(-1,-1),4),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("BACKGROUND",(0,3),(-1,4),HexColor("#ecfdf5")),("BACKGROUND",(0,5),(-1,6),HexColor("#fff7ed")),
        ("BACKGROUND",(0,2),(-1,2),HexColor("#fef3c7"))]))
    return t

def db_table():
    data = [["DB","파일","주요 테이블","역할"],
        ["Sensor DB","sensor_detections.db","image_records\ndetection_records","영상 메타 + 객체 탐지 결과"],
        ["Pairing DB","object_pairings.db","pairing_records","객체 페어링 결과 (5상태)"],
        ["Graph DB","graph.db","graph_entities\ngraph_relations\ngraph_communities","GraphRAG 누적 지식 그래프"],
        ["Report DB","reports.db","report_records","LLM 보고서 영속 저장"]]
    t = Table(data, colWidths=[2.4*cm,4.0*cm,4.4*cm,4.7*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),TABLE_HD),("TEXTCOLOR",(0,0),(-1,0),HexColor("#ffffff")),
        ("FONTNAME",(0,0),(-1,0),KORB),("FONTNAME",(0,1),(-1,-1),KOR),("FONTSIZE",(0,0),(-1,-1),9),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LINEBELOW",(0,0),(-1,-1),0.3,LIGHT_BORDER),
        ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    return t

def draw_pg(canvas, doc):
    canvas.saveState(); canvas.setFont(KOR, 9); canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0]/2, 1.0*cm, f"- {canvas.getPageNumber()} -")
    canvas.restoreState()


def build():
    out = "/home/user/multi-source-intelligent-system/data/patent_v5_2_simple.pdf"
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=2.2*cm, rightMargin=2.2*cm, topMargin=2.0*cm, bottomMargin=2.2*cm)
    s = []

    # ─── 표지 ───
    title_box = boxed([
        Paragraph("특허출원 명세서 (v5.2)", ParagraphStyle("x",fontName=KORB,fontSize=15,textColor=TITLE_BLUE,alignment=TA_CENTER,spaceAfter=2)),
        Paragraph("(중복성 없음 · 차별성 간단화)", ParagraphStyle("x2",fontName=KOR,fontSize=10.5,textColor=TEXT_BODY,alignment=TA_CENTER,spaceAfter=10)),
        meta("발명의 명칭 (국문)", "위성영상 시계열 변화 탐지 및 GraphRAG 기반 AI Agent 판독보고서 자동 생성 시스템 및 방법"),
        meta("발명의 명칭 (영문)", "System and Method for Time-Series Change Detection and GraphRAG-Augmented AI Agent Interpretation Report Generation from Satellite Imagery"),
    ], bg=HexColor("#f0f9ff"), bd=LIGHT_BORDER, pl=18, pr=18, pt=14, pb=14)
    s.append(title_box); s.append(Spacer(1,8))

    # ═══ 1. 개요 ═══
    s.append(h1("1. 개요"))
    s.append(h2("(1) 직무발명 수탁과제 무관 근거 설명"))

    s.append(h3("1) 제안 발명 내용 요약"))
    s.append(p(
        "본 발명은 위성·드론 시계열 영상에 대해 두 축으로 구성된 통합 시스템으로서, "
        "<b>[축 A] 객체 페어링 기반 시계열 변화 탐지</b>(SAM3 객체 탐지 → 마스크 배경 제거 CLIP 임베딩 → "
        "Gale-Shapley 안정 매칭 → 5상태 정밀 분류)와, <b>[축 B] GraphRAG 시계열 인덱싱 기반 AI Agent "
        "판독보고서 자동 생성</b>(격자 양자화 결정론 키 그래프 누적 → Louvain 자산 군집 자동 발견 → "
        "Local+Global 이중 검색 → LLM 표준 IMINT 보고서 자율 생성)이 유기적으로 결합된 end-to-end 시스템이다."
    ))
    s.append(callout(
        "<b>핵심 차별 2축</b>: <b>①</b> 정규 패치 단위가 아닌 <b>객체 인스턴스 단위</b>의 페어링 기반 변화 탐지, "
        "<b>②</b> LLM 호출 없이 결정론적으로 누적되는 <b>GraphRAG 시계열 메모리</b>를 통한 판독보고서 자동 생성."
    ))

    s.append(h3("2) 유사 수탁과제 기술 설명"))
    s.append(b("<b>없음</b>"))

    s.append(h3("3) 중복성 설명"))
    s.append(b("<b>없음</b>"))

    s.append(h3("4) 차별성 설명"))
    s.append(p(
        "본 발명은 위성·드론 영상에서 격자 단위로 변화 여부만 판정하는 일반적 방식과 달리, "
        "<b>SAM3로 개별 객체를 인스턴스 단위로 식별</b>하고 <b>마스크 배경 제거 CLIP 임베딩과 "
        "Gale-Shapley 안정 매칭</b>으로 두 시점의 객체를 1:1로 짝지어 <b>정지·구조변화·신규·"
        "소실·촬영공백의 5상태로 정밀 분류</b>하며, 그 결과를 격자 양자화 키에 기반해 <b>LLM "
        "호출 없이 지식 그래프(GraphRAG)에 결정론적으로 누적</b>하고 <b>Louvain 알고리즘으로 "
        "자산 doctrine 패턴을 자동 발견</b>한 뒤, 이 누적된 시공간 컨텍스트를 압축하여 LLM "
        "프롬프트에 주입함으로써 <b>표준 IMINT 8섹션 판독보고서를 자율 생성</b>한다는 점에서 "
        "차별된다."
    ))

    s.append(h2("(2) 발명 제안 배경"))
    s.append(p(
        "위성·드론 시계열 영상 분석에서 두 시점 사이 변화를 자동 탐지하는 것은 도시 계획·환경 감시·재난 대응·"
        "인프라 모니터링 등 다양한 응용에서 핵심이다. 종래의 정규 패치 단위 변화 탐지는 (i) 격자 분할로 객체 "
        "이동을 동일 객체로 추적하지 못하고, (ii) 변화 여부만 이진 판정하여 어떤 자산이 어떻게 변했는지 정보를 "
        "제공하지 못하며, (iii) 두 프레임 단발 비교에 그쳐 시계열 누적 패턴을 인식하지 못하는 한계가 있었다."
    ))
    s.append(p(
        "또한 LLM에 정형 탐지 결과를 그대로 주입하는 방식은 시계열 누적 컨텍스트 부재로 단발 보고에 그치고, "
        "LLM 재해석 과정에서 환각(hallucination) 위험이 발생한다. 본 발명은 객체 인스턴스 단위 페어링 기반 "
        "변화 탐지와 LLM 비호출 결정론적 지식 그래프에 누적 인덱싱된 시계열 컨텍스트를 결합함으로써 이러한 "
        "한계를 극복한 통합 시스템을 제안한다."
    ))

    s.append(h2("(3) 산업상 이용 분야 (용도)"))
    for x in ["<b>스마트시티 인프라 모니터링</b>: 도시 건물·시설물 시계열 변화 자동 감시·보고",
              "<b>재난·재해 대응</b>: 위성 관측 기반 시설물 파손·복구 추이 자동 판독",
              "<b>환경·농업 모니터링</b>: 산림·농지·해양 시계열 변화 자동 추적",
              "<b>부동산·건설 관리</b>: 대규모 건설 현장 진척도 자동 판독 및 정기 보고",
              "<b>국토·해양 감시</b>: 국경·해안·항만 자산 활동 패턴 자동 인텔리전스"]:
        s.append(b(x))

    s.append(h2("(4) 종래 기술 (관련 기술 배경)"))
    s.append(p(
        "본 발명이 속한 위성·항공 영상 인텔리전스 분야에서 다음과 같은 관련 기술 흐름이 존재한다. 다만 "
        "본 발명의 통합 구성(객체 인스턴스 단위 페어링 + GraphRAG 시계열 인덱싱 + LLM AI Agent 판독보고서 "
        "자동 생성)과 직접적으로 중복되는 선행 기술은 확인되지 아니한다.", i=False
    ))
    s.append(h3("가. 위성영상 변화 탐지 일반"))
    s.append(b("정규 패치(격자) 단위로 영상을 분할하여 두 시점 영상의 각 패치 특징을 비교함으로써 변화 여부를 이진 판정하는 접근이 일반적."))
    s.append(b("딥러닝 기반 세그멘테이션 모델을 활용한 시멘틱 변화 탐지 및 U-Net·Siamese 구조 기반 변화 탐지 알고리즘 등이 연구되어 옴."))
    s.append(h3("나. RAG 및 LLM 기반 보고서 생성 일반"))
    s.append(b("대규모 언어 모델(LLM)에 벡터 데이터베이스(FAISS 등) 기반 텍스트 검색 결과를 컨텍스트로 주입하는 RAG 기법이 널리 알려져 있음."))
    s.append(b("지식 그래프 형태로 정보를 구조화하여 LLM 프롬프트 컨텍스트로 활용하는 GraphRAG 접근이 학술적으로 제안된 바 있음."))
    s.append(h3("다. 원격 탐사 파운데이션 모델"))
    s.append(b("SAM(Segment Anything Model), CLIP 등 비전 파운데이션 모델이 위성·항공 영상 분석에 활용되기 시작하고 있음."))

    s.append(h2("(5) 관련 기술의 한계"))
    for i,(t,d) in enumerate([
        ("객체 단위 시계열 페어링 부재", "정규 패치 단위 접근은 격자 위치 고정으로 객체 이동 시 동일 객체 추적 불가."),
        ("이진 판정의 정보 밀도 한계", "변화 여부만 판정으로 촬영 범위 차이를 자산 변화로 오인 위험."),
        ("시계열 누적 메모리 부재", "두 시점 단발 비교로 반복 관측 이력 누적 저장 불가."),
        ("자산 doctrine 패턴 자동 발견 부재", "공출현 패턴을 그래프 알고리즘으로 자동 군집화하는 구성 없음."),
        ("환각 위험 및 도메인 안전성 부재", "LLM 직접 활용 시 환각 발생, 도메인 의미 제약·IMINT 형식 강제 부재."),
    ], 1):
        s.append(b(f"<b>{i}. {t}</b> — {d}"))

    # ═══ 2. 상세 설명 ═══
    s.append(h1("2. 상세 설명"))
    s.append(h2("(1) 시스템 전체 구성 — 4-DB 분리 아키텍처"))
    s.append(p("본 발명은 축 A(구성 A~B)와 축 B(구성 C~E)를 유기적으로 결합한 시스템이며, 각 단계 결과는 4개 DB에 영속화된다."))
    s.append(Spacer(1,3)); s.append(db_table()); s.append(Spacer(1,3))
    s.append(note("모든 레코드에 session_id가 부여되어 파이프라인 실행 단위로 묶이며, 분석관이 탐지 결과 수정 시 페어링·그래프·보고서가 자동 재계산되는 HITL 재처리를 지원한다."))

    s.append(h2("(2) 축 A: 객체 페어링 기반 시계열 변화 탐지부"))
    s.append(h3("[구성 A] 영상 객체 탐지부 (SAM3)"))
    s.append(b("Real-ESRGAN 초해상도 → SAM3 텍스트 프롬프트 zero-shot 탐지 → 멀티스케일 슬라이딩 윈도우 + NMS → (클래스·bbox·mask_rle·위경도) Sensor DB 저장."))
    s.append(h3("[구성 B] 객체 인스턴스 단위 페어링부"))
    s.append(callout("정규 패치가 아닌 <b>SAM3가 실제로 인식한 개별 객체 인스턴스</b>를 단위로 두 시점 객체를 1:1로 짝짓는다."))
    s.append(b("<b>마스크 배경 제거 임베딩</b>: mask_rle로 배경 픽셀 zeroing → bbox crop → CLIP/ViT L2 정규화 임베딩."))
    s.append(b("<b>N×M 유사도 행렬 + 동일 클래스 필터</b>: 단일 GEMM으로 코사인 유사도 산출, 동일 클래스 쌍만 후보 유지. 점수 = 0.8×CLIP + 0.2×크기."))
    s.append(b("<b>Gale-Shapley 안정 매칭</b>: 지연 승인 알고리즘으로 1:1 안정 매칭. 객체가 격자를 넘어 이동해도 동일 객체 추적."))
    s.append(b("<b>고정형 객체 전용 분기</b>: 위경도 초근접 결합 → CLIP 외형 비교로 matched/changed 구분 → 탐지 실패 시 가상 탐지 합성 주입."))
    s.append(b("<b>5상태 정밀 분류</b>: 매칭 결과 + FOV 검사로 아래 5상태 중 하나로 배정."))
    s.append(Spacer(1,3)); s.append(status_table()); s.append(Spacer(1,5))

    s.append(h2("(3) 축 B: GraphRAG 시계열 인덱싱 및 AI Agent 판독보고서 생성부"))
    s.append(h3("[구성 C] GraphRAG 결정론적 누적 인덱싱부"))
    s.append(callout("페어링 결과를 <b>LLM 호출 없이</b> 격자 양자화 결정론 키로 지식 그래프에 누적 인덱싱한 후 Louvain으로 자산 doctrine 군집 자동 발견."))
    s.append(b("<b>격자 양자화 결정론 키</b>: 위경도 소수점 2자리(≈1km) 양자화, \"loc:...\" / \"asset:...\" 키. GPS 오차 흡수·동일 자산 자동 통합."))
    s.append(b("<b>LLM 비호출 카운터 upsert</b>: status 필드를 자산 노드 카운터에 결정론 가산. <b>동일 입력에 매번 동일 그래프</b> — 완전 재현성·감사 가능성."))
    s.append(b("<b>Louvain 군집 자동 탐지</b>: co_occurred_with 엣지 가중치 기반 자산 doctrine 군집 자동 발견, member_summary 결정론 산식 생성."))
    s.append(h3("[구성 D] Local + Global 이중 검색부"))
    s.append(b("<b>Local Search</b>: 반경 R 내 자산 노드 이력 통계 조회."))
    s.append(b("<b>Global Search</b>: 중첩 커뮤니티 member_summary 조회."))
    s.append(b("<b>사전 압축 컨텍스트</b>: ~500 토큰 이내 단일 블록 포맷화."))
    s.append(h3("[구성 E] AI Agent 판독보고서 생성부 (단일 LLM 2회 호출)"))
    s.append(b("<b>변화 객체 한정 추출</b>: new·disappeared만 신뢰도 순 추출, matched/changed/FOV 공백 제외."))
    s.append(b("<b>도메인 의미 가드레일</b>: \"'DISAPPEARED' ≠ destroyed\" 시스템 프롬프트 강제."))
    s.append(b("<b>IMINT 8섹션 강제</b>: CLASSIFICATION~APPENDIX 8섹션 구조 강제."))
    s.append(b("<b>동일 LLM 재호출 무오염 번역</b>: 영문 생성 후 같은 인스턴스로 한국어 변환, 좌표·수치 토큰 바이패스."))

    s.append(h2("(4) 발명의 효과"))
    for i,(t,d) in enumerate([
        ("객체 단위 정밀 변화 탐지", "패치 단위 한계 극복, 객체 이동에도 동일 객체 안정 추적."),
        ("5상태 정밀 분류로 오판 방지", "촬영 범위 차이를 자산 변화로 오인하는 오판 원천 배제."),
        ("LLM 비호출 결정론 GraphRAG", "인덱싱 비용 근본 절감 + 동일 입력에 동일 그래프 = 감사 가능성 확보."),
        ("환각 차단 + 토큰 효율", "사전 압축 ~500토큰 컨텍스트 + 도메인 가드레일로 환각 원천 차단."),
        ("표준 판독보고서 자율 생성", "자연어 IMINT 8섹션 자동 생성으로 분석관 후속 작업 제거."),
    ], 1):
        s.append(b(f"<b>{i}. {t}</b> — {d}"))

    # ═══ 3. 청구범위 ═══
    s.append(h1("3. 특허 청구범위"))
    s.append(note("각 청구항 상단에 '쉽게 말하면' 요약을 함께 표시. 실제 권리범위는 정형 문장에 의해 정의됨."))

    s.append(claim("[청구항 1] (독립항) — 전체 방법",
        "정규 패치가 아닌 실제 객체 하나하나를 짝지어 5상태로 분류하고, AI 없이 그래프에 누적한 뒤 AI Agent가 표준 판독보고서를 자율 생성하는 방법.",
        [
            "임의의 대상 지역을 촬영한 시계열 위성·항공 영상을 수집하여 텍스트 프롬프트 기반 <b>객체 세그멘테이션 모델</b>로 각 개별 객체를 인스턴스 단위로 식별하고, 자산 클래스·위경도·바운딩 박스·세그멘테이션 마스크를 포함하는 탐지 레코드를 저장하는 단계;",
            "정규 패치가 아닌 <b>객체 인스턴스 단위</b>로, 세그멘테이션 마스크로 배경 픽셀을 영(0)으로 처리한 후 바운딩 박스 크롭 이미지로부터 L2 정규화 시각 임베딩을 산출하고, N×M 코사인 유사도 행렬에서 동일 클래스 쌍에 대해 임베딩·크기 유사도 가중합 점수가 임계값 이상인 후보 쌍에 대해 <b>Gale-Shapley 지연 승인 알고리즘</b>으로 1:1 안정 매칭 후 matched·changed·new·disappeared·past_not_included·current_not_included의 5개 이상 상태로 정밀 분류하는 단계;",
            "각 자산 위경도를 정수형 격자 단위로 양자화한 위치 노드 키와 (클래스, 위치) 결합의 자산 노드 키를 결정론적으로 생성하고, 상기 5상태에 따라 자산 노드 카운터를 <b>LLM 호출 없이</b> 갱신하며, found_at 및 co_occurred_with 엣지 가중치를 누적 갱신하는 단계;",
            "co_occurred_with 가중치 기반 Louvain 군집화로 자산 복합체 커뮤니티 요약문을 LLM 호출 없이 생성하고, Local Search와 Global Search를 병합하여 토큰 예산 이내 역사적 맥락 컨텍스트 블록을 생성하는 단계; 및",
            "new 또는 disappeared 객체만 신뢰도 순 추출하여 상기 컨텍스트 블록을 LLM 프롬프트 선두에 prepend하고, 표준 IMINT 8섹션 구조 및 의미 가드레일을 강제 주입하여 영문 보고서 생성 후 동일 LLM 인스턴스로 한국어 번역하는 단계;",
            "를 포함하는 <b>객체 인스턴스 단위</b> 시계열 변화 탐지 및 GraphRAG 기반 AI 에이전트 판독보고서 자동 생성 방법.",
        ]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 2] (종속) — 마스크 기반 배경 제거 임베딩",
        "객체 비교 전 배경을 지운 뒤 그 객체만 잘라서 임베딩하여 배경 노이즈로 인한 오인 차단.",
        ["제1항에 있어서, 세그멘테이션 마스크 RLE 정보를 디코딩하여 활성 영역 외 픽셀을 영(0)으로 마스킹하고 사전 설정 패딩 폭을 부가하여 크롭하며, 산출된 임베딩 벡터를 L2 정규화하여 유사도 행렬 산출에 사용하는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 3] (종속) ★ Gale-Shapley 안정 매칭",
        "노벨 경제학상 수상 안정 매칭 기법으로 객체가 이동해도 정확히 1:1 짝을 맞춤.",
        ["제1항에 있어서, 임베딩 유사도와 크기 유사도의 가중합 점수가 임계값을 초과하는 후보 쌍에 대해 <b>Gale-Shapley 지연 승인 알고리즘</b>으로 1:1 안정 매칭을 수행하여, 격자 경계를 넘어 이동한 객체도 동일 객체로 안정 추적하는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 4] (종속) ★ 5상태 정밀 분류",
        "'변화 O/X' 이진 판정이 아닌 5상태로 세밀하게 나눠 촬영 공백을 진짜 변화와 구분.",
        ["제1항에 있어서, status 필드는 (i) matched, (ii) changed, (iii) new, (iv) disappeared, (v) past_not_included, (vi) current_not_included의 <b>5상태 이상으로 정밀 분류</b>되며, (v)·(vi) 상태 객체는 LLM 보고서 생성 시 변화 객체 추출에서 제외되어 단순 촬영 범위 차이가 자산 변화로 오인되지 않도록 하는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 5] (종속) — 고정형 객체 전용 분기 + 가상 탐지 합성",
        "건물·시설 같이 못 움직이는 객체는 초근접 결합 + 외형 변화 판정 + 탐지 누락 보정으로 처리.",
        ["제1항에 있어서, 사전 정의된 정적 고정 자산에 대해 (a) 위경도 초근접 그리디 결합, (b) CLIP 유사도 기준 matched/changed 판정, (c) 한쪽 탐지 실패 시 상대 영상 동일 위경도 강제 crop 후 CLIP 재산출로 가상 탐지 레코드 합성 주입하여 페어링 복원하는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 6] (종속) ★ LLM 비호출 결정론 GraphRAG 인덱싱",
        "AI를 부르지 않고 카운터만 자동으로 +1 누적하여 완전 재현 · 감사 가능.",
        ["제1항에 있어서, 각 자산 위경도를 소수점 N자리 격자로 반올림한 결정론 키를 생성하고, status 필드 및 신뢰도를 자산 노드 카운터 속성에 <b>LLM 호출 없이 결정론 누적 upsert</b>하며, found_at·co_occurred_with 엣지 count도 LLM 호출 없이 누적 갱신하여 동일 입력에 매번 동일한 그래프가 산출되는 결정론적 재현성을 갖는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 7] (종속) — Louvain 자산 doctrine 군집 자동 발견",
        "자주 함께 등장하는 자산들을 자동 그룹화하여 doctrine 패턴 자동 발견.",
        ["제1항에 있어서, co_occurred_with 엣지 count 가중치 기반 Louvain 군집화로 자산 복합체 커뮤니티를 발견하고, 각 커뮤니티에 대해 구성원 자산·위치·관측 통계 member_summary를 LLM 호출 없이 결정론 산식으로 생성하는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 8] (종속) — Local + Global 이중 검색 및 사전 압축 컨텍스트",
        "해당 지역 메모만 추려 ~500 토큰으로 압축해 AI 입력 맨 앞에 붙임.",
        ["제1항에 있어서, 반경 R 내 자산 이력 조회 Local Search와 중첩 커뮤니티 member_summary 조회 Global Search를 병합하여 사전 설정 토큰 예산 이내로 절단·머지, LLM 프롬프트 선두 prepend 단일 컨텍스트 블록으로 포맷화하는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 9] (종속) — 변화 객체 한정 + IMINT 8섹션 + 동일 LLM 번역",
        "변화 객체만 AI에 전달, 표준 8섹션 강제, 같은 AI로 좌표 보존 한국어 변환.",
        ["제1항에 있어서, matched·changed·past/current_not_included는 제외하고 new·disappeared만 신뢰도 순 주입하는 단계; IMINT 8섹션 구조 및 \"'DISAPPEARED' ≠ destroyed\" 의미 제약 강제 주입 단계; 동일 LLM 인스턴스 재사용해 영문 생성 후 한국어 번역, 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 토큰 바이패스로 분석 서술문만 변환하는 단계를 포함하는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 10] (종속) — 4-DB 분리 + 세션 추적성",
        "탐지·페어링·그래프·보고서를 별도 DB에 저장하고 session_id로 묶어 자동 재계산 (HITL).",
        ["제1항에 있어서, 센서·페어링·그래프·보고서의 물리적으로 분리된 4개 DB를 포함하며, 모든 레코드에 동일 session_id 부여로 분석관이 탐지 결과 수정 시 페어링·그래프·보고서가 동일 세션 단위로 자동 재계산되는 것을 특징으로 하는 방법."]))
    s.append(Spacer(1,5))

    s.append(claim("[청구항 11] (독립항) — 시스템 / 기록매체",
        "위 방법을 수행하는 컴퓨터 시스템 및 그 프로그램이 기록된 기록매체.",
        ["제1항 내지 제10항 중 어느 한 항의 방법을 수행하는 하나 이상의 하드웨어 프로세서와 메모리를 포함하는 시스템 및 상기 방법을 컴퓨터에서 실행시키기 위한 프로그램이 기록된 컴퓨터 판독 가능 기록 매체."]))

    # ═══ 4. 도면 ═══
    s.append(h1("4. 도면 설명 명기"))

    s.append(fig("도 1] 시스템 전체 구성도 — 4-DB 아키텍처", [
        "[시점 1 위성영상] + [시점 2 위성영상]",
        "         │",
        "         ▼  SAM3 탐지 (Real-ESRGAN SR + 멀티스케일)",
        "     Sensor DB (image_records, detection_records)",
        "         │  현재·과거 탐지 조회",
        "         ▼  Pairing Module (★ 객체 인스턴스 단위)",
        "         │    · CLIP+ViT 임베딩 (배경 제거)",
        "         │    · N×M 코사인 + 동일 클래스 필터",
        "         │    · Gale-Shapley 안정 매칭",
        "         │    · 고정형 객체 전용 분기 + FOV 가드",
        "     Pairing DB (5상태: matched/changed/new/",
        "                  disappeared/past_not_included/",
        "                  current_not_included)",
        "         │",
        "         ▼  Graph Indexer (★ LLM 비호출)",
        "         │    · 격자 양자화 loc/asset 키",
        "         │    · found_at / co_occurred_with 누적",
        "         │    · Louvain 군집 자동 탐지",
        "     Graph DB (graph_entities/relations/communities)",
        "         │",
        "         ▼  Graph Retriever",
        "         │    · Local + Global Search",
        "         │    · ~500 토큰 컨텍스트 블록",
        "         ▼  LLM AI Agent",
        "         │    ① 영문 IMINT 8섹션 생성",
        "         │    ② 동일 인스턴스 한국어 번역",
        "     Report DB (판독보고서)",
    ]))
    s.append(Spacer(1,8))

    s.append(fig("도 2] 접근 방식 비교 — 정규 패치 vs 객체 인스턴스", [
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
    ]))
    s.append(Spacer(1,8))

    s.append(fig("도 3] 객체 페어링 변화 탐지부 5상태 분류 흐름도", [
        "Sensor DB (현재 N + 과거 M) → Step 0. 고정형 객체 우선 분기 →",
        "Step 1. mask_rle zeroing crop → Step 2. CLIP/ViT 임베딩 →",
        "Step 3. N×M 코사인 유사도 행렬 → Step 4. 동일 클래스 필터 →",
        "Step 5. Gale-Shapley 안정 매칭 → Step 6. 5상태 분류 → Pairing DB",
    ]))
    s.append(Spacer(1,8))

    s.append(fig("도 4] GraphRAG 결정론적 누적 인덱싱 (★ LLM 비호출)", [
        "Pairing DB → EntityExtractor →",
        "  · 격자 양자화 loc_key/asset_key 생성",
        "  · Location·Asset 노드 upsert (카운터 +1)",
        "  · found_at/co_occurred_with 엣지 갱신",
        "→ NetworkX 메모리 그래프 → Louvain 군집 자동 탐지",
        "→ member_summary 결정론 생성 → Graph DB",
    ]))
    s.append(Spacer(1,8))

    s.append(fig("도 5] AI Agent LLM 2회 호출 판독보고서 생성 흐름", [
        "Graph DB → Local + Global Search → ~500 토큰 컨텍스트 블록",
        "Pairing DB → 변화 객체 추출 (new+disappeared) → 프롬프트 조립",
        "  · 시스템: \"DISAPPEARED ≠ destroyed\" + IMINT 8섹션 강제",
        "[1차] LLM → 영문 IMINT 보고서",
        "[2차] 동일 LLM → 한국어 번역 (좌표·수치 바이패스) → Report DB",
    ]))

    doc.build(s, onFirstPage=draw_pg, onLaterPages=draw_pg)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    build()
