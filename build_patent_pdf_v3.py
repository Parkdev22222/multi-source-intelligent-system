"""Generate the easy-to-read patent PDF (v3) — same template, plain language."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
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
                              alignment=TA_CENTER, leading=14.5)

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
        Paragraph("(변화탐지 시계열 페어링 및 GraphRAG 누적 인덱싱 기반 · 비전문가 친화본)",
                  ParagraphStyle("x2",fontName=KOR,fontSize=10.5,textColor=TEXT_BODY,
                  alignment=TA_CENTER,spaceAfter=10)),
        meta_row("발명의 명칭 (국문)",
                 "위성·드론 영상의 시간적 변화 탐지와 누적 지식 그래프를 활용한 "
                 "AI 보조 상황 보고서 자동 작성 시스템 및 방법"),
        meta_row("발명의 명칭 (영문)",
                 "AI-Assisted Situational Report Generation System Combining "
                 "Time-Series Change Detection and Accumulative Knowledge Graph on Aerial Imagery"),
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
    """노란 음영의 '쉬운 설명' 콜아웃 박스"""
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

def fig_box(title, body):
    return boxed([Paragraph(f"[{title}]", style_fig_t),
                  Paragraph(body, style_fig_b)],
                  bg=SOFT_BG, border=LIGHT_BORDER, pl=14, pr=14, pt=10, pb=10)

def draw_page_num(canvas, doc):
    canvas.saveState()
    canvas.setFont(KOR, 9); canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0]/2, 1.0*cm, f"- {canvas.getPageNumber()} -")
    canvas.restoreState()


# ─────────────────────────────────────────────────────────
def build():
    out = "/home/user/multi-source-intelligent-system/data/patent_change_detection_graphrag_easy.pdf"
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
        "본 발명은 같은 지역을 시간 차이를 두고 촬영한 위성·드론 사진들을 컴퓨터가 자동으로 비교하여 "
        "'무엇이 새로 나타났고, 무엇이 사라졌으며, 무엇이 움직였는지'를 찾아내고, 이 결과를 "
        "그래프 형태의 '기억 노트'에 차곡차곡 쌓아둔 뒤, AI(거대언어모델)에게 그 기억을 참고시켜 "
        "정해진 8개 항목의 군사 판독 보고서를 자동으로 작성해 주는 시스템입니다."
    ))
    s.append(callout(
        "<b>한 줄 요약</b>: 두 장의 위성 사진을 자동으로 비교 → 변화한 객체만 골라내기 → "
        "지역별·종류별로 누적 기록 → AI가 사람 분석관처럼 보고서를 써 주는 시스템입니다."
    ))

    s.append(h3("2) 중복성 및 차별성 설명"))
    s.append(bullet(
        "<b>중복성</b>: 본 발명과 분야가 가장 가까운 외국 등록 특허로 <b>미국 등록 특허 "
        "US 12,488,225 B1 (Booz Allen Hamilton, 2025년 등록)</b>이 있습니다. 이 특허는 여러 정보원"
        "(위성·인터넷 공개 정보·사이버 정보 등)을 한 화면에 모아 보여주고, 분석관이 \"이런 게 궁금하다\"라고 "
        "질문하면 AI가 답을 정리해 주는 식의 시스템입니다. <b>본 발명과 큰 그림은 비슷하지만, 작동 방식과 "
        "기억 구조가 본질적으로 다릅니다.</b>"
    ))
    s.append(bullet(
        "<b>차별성</b>: 본 발명은 (가) <b>객체 하나하나를 두 사진에서 짝지어</b> 어떤 객체가 어떻게 "
        "변했는지를 직접 비교하고, (나) <b>AI를 부르지 않고도 시간에 따라 쌓이는 '누적 기억 노트'</b>"
        "(지식 그래프)를 만들어 같은 객체의 반복 출현·소실 기록을 자동으로 모아두며, "
        "(다) AI에게 보고서를 시킬 때는 <b>이 기억 노트에서 필요한 부분만 압축해서 미리 넘겨주어</b> "
        "AI가 엉뚱한 말(환각)을 하지 못하게 막는다는 점에서 차별됩니다."
    ))

    s.append(h2("(2) 발명 제안 배경"))
    s.append(para(
        "위성·드론으로 같은 지역을 여러 번 촬영하면 그 안에 무엇이 새로 생기고 사라졌는지를 분석관이 "
        "직접 눈으로 비교해야 했습니다. 사진이 늘어날수록 분석관의 업무가 폭발적으로 늘어나고 "
        "지휘관이 결심을 내리는 데 시간이 더 걸리는 문제가 있었습니다."
    ))
    s.append(para(
        "최근에는 ChatGPT 같은 거대 AI(LLM)에게 보고서를 시키는 시도가 늘어났지만, 다음과 같은 "
        "한계가 있었습니다: ① AI에게 매번 \"이 사진에서 본 것\"만 알려주면 같은 지역에서 비슷한 일이 "
        "예전에도 있었는지를 AI가 알지 못해 매번 단발성 보고만 생성됩니다. ② 같은 전차라도 "
        "촬영 위치가 1m만 달라져도 AI는 다른 객체로 착각합니다. ③ AI가 정해지지 않은 부분을 "
        "스스로 상상해서 채워 넣는 '환각(Hallucination)' 현상이 발생합니다. 본 발명은 이러한 문제들을 "
        "모두 해결하기 위해 제안되었습니다."
    ))

    s.append(h2("(3) 산업상 이용 분야 (용도)"))
    s.append(bullet("<b>국방·방위산업</b>: 위성·정찰기 영상 자동 판독 및 표준 보고서 자동 작성"))
    s.append(bullet("<b>국경·전장 감시</b>: 적군 자산의 누적 활동 패턴 자동 추적 및 doctrine(작전 형태) 자동 식별"))
    s.append(bullet("<b>재난·재해 모니터링</b>: 위성으로 본 시설물 파손·복구 추이 자동 보고"))
    s.append(bullet("<b>스마트시티</b>: 도시 인프라·시설물의 시간별 변화 자동 감시 보고"))

    s.append(h2("(4) 종래 기술"))
    s.append(para(
        "본 발명과 가장 비슷한 실제 등록 특허는 <b>미국 특허 US 12,488,225 B1 "
        "(\"Modular open system architecture for common intelligence picture generation\", "
        "양수인: Booz Allen Hamilton Inc., 등록 2025년)</b>입니다."
    ))
    s.append(callout(
        "<b>이 특허를 한 마디로</b>: 위성·인터넷·통신 등 여러 곳에서 모은 정보를 <b>한 화면(공통 "
        "인텔리전스 픽처, CIP)</b>으로 합쳐서 보여 주고, 분석관이 \"북쪽 부대 동향이 궁금하다\" 같은 "
        "<b>질문을 자연어로 던지면 AI가 정보를 찾아 답변·보고서를 만들어 주는</b> 시스템. "
        "쉽게 말해 \"군사 정보용 ChatGPT + 통합 대시보드\" 같은 시스템입니다."
    ))

    s.append(h2("(5) 종래 기술의 문제점 / 한계"))
    s.append(para(
        "위 인용 발명은 \"AI에게 묻고 답 듣기\"라는 큰 틀에서는 비슷하나, 본 발명이 다루는 \"시간이 "
        "지나면서 어떤 객체가 어떻게 변했는지를 자동 추적·기록·보고\"라는 문제에 적용하면 다음과 "
        "같은 한계가 있습니다.", indent=False
    ))
    problems = [
        ("객체별로 누가 어떻게 변했는지 알 수 없음",
         "인용 발명은 모든 정보를 큰 '한 장의 그림(CIP)'으로 합쳐 보여 줍니다. 하지만 \"3번 전차가 "
         "어제는 A 위치에 있었는데 오늘은 B 위치로 이동했다\"처럼 <b>객체 하나하나의 변화</b>를 "
         "자동으로 짝지어 추적하는 기능이 없습니다."),
        ("'같은 객체의 누적 기록'을 만들지 않음",
         "인용 발명은 매번 '지금 이 순간의 스냅샷'만 보여 줍니다. 그래서 \"이 지역에 전차가 10일간 "
         "5번째 반복 출현 중\" 같은 시간에 걸친 패턴을 자동으로 만들어 두지 않아, AI에게 \"이번 게 "
         "몇 번째냐\"라고 물어보면 매번 새로 세야 합니다."),
        ("자주 함께 등장하는 자산들의 묶음 패턴 자동 발견 부재",
         "전차+장갑차+포병이 자주 함께 등장하면 \"기갑 복합체\"라는 작전 형태를 의미합니다. 인용 "
         "발명은 이런 묶음을 <b>자동으로 찾아내는 기능</b>이 없어 분석관이 일일이 발견해야 합니다."),
        ("AI가 환각(없는 정보를 상상해서 채움)을 일으킬 수 있음",
         "인용 발명은 AI에게 정보를 자유롭게 찾도록 맡깁니다. AI가 그 과정에서 추측한 내용이 "
         "보고서에 섞여 들어갈 위험이 있습니다."),
        ("'미관측'을 '파괴'로 잘못 해석할 위험",
         "위성에 안 보인다고 '파괴됨'은 아닙니다. 그런데 일반 AI는 이를 자주 혼동하여 군사적으로 "
         "치명적인 오판을 일으킬 수 있는데, 인용 발명에는 이를 막는 안전장치가 없습니다."),
        ("표준 군사 보고서 양식 강제 부재",
         "본 발명이 채택한 정형 보고서(8개 표준 IMINT 섹션)를 자동으로 따르도록 강제하는 구성과 "
         "고유 좌표·시각 정보를 변형 없이 보존하는 한국어 번역 기능이 인용 발명에는 없습니다."),
    ]
    for i,(t,d) in enumerate(problems,1):
        s.append(bullet(f"<b>{i}. {t}</b> — {d}"))

    # ═══════════════════════════════════════════════════
    # 2. 상세 설명
    # ═══════════════════════════════════════════════════
    s.append(h1("2. 상세 설명"))
    s.append(h2("(1) 발명의 핵심 구성 및 작동 원리"))
    s.append(para(
        "본 발명은 (A) <b>두 사진 비교 — 변화 탐지부</b>, (B) <b>누적 기억 노트 — 지식 그래프부</b>, "
        "(C) <b>AI 보고서 작성부</b>의 세 부분으로 나뉩니다. 셋이 하나의 파이프라인으로 이어져 "
        "동작하지만, 각 부분이 독립적인 기능을 합니다."
    ))

    # ── 구성 A ──
    s.append(h3("[구성 A] 두 사진 비교 — 객체별 변화 탐지부 (Step 1)"))
    s.append(callout(
        "<b>비유</b>: 사람 분석관이 어제 사진과 오늘 사진을 양손에 들고 \"이 전차는 어제도 있었고\", "
        "\"저 전차는 어제는 없던 것\"처럼 <b>객체 하나하나를 짝지어 확인</b>하는 과정을 컴퓨터가 자동으로 "
        "수행하는 부분입니다."
    ))
    s.append(bullet(
        "<b>① 객체만 깔끔하게 잘라내기</b> — 각 객체 주변의 배경(나무·도로 등)을 검은색(0)으로 지워서, "
        "객체 자체만 남긴 작은 이미지 조각을 만듭니다. 배경이 섞이면 '같은 전차'가 다르게 보일 수 있어서 "
        "배경을 미리 제거하는 것입니다. "
        "<i>(전문 용어: SAM 마스크로 zeroing 후 bbox crop)</i>"
    ))
    s.append(bullet(
        "<b>② 객체 사진을 숫자 벡터로 변환</b> — 각 객체 조각을 비전 AI(CLIP/ViT)에 넣어 객체의 모양·"
        "특징을 숫자 묶음(임베딩)으로 바꿉니다. 두 객체가 얼마나 비슷한지는 이 숫자 묶음의 '방향이 얼마나 "
        "비슷한가'(코사인 유사도)로 계산합니다. <i>0이면 전혀 안 닮음, 1이면 거의 같음</i>."
    ))
    s.append(bullet(
        "<b>③ 가장 좋은 짝을 안정적으로 찾기</b> — 어제 사진의 객체 M개와 오늘 사진의 객체 N개를 "
        "<b>모두 짝지어 비슷한 정도를 표로 만들고</b>, 같은 종류끼리만 후보를 골라 \"누가 누구의 가장 좋은 "
        "짝인지\"를 <b>Gale-Shapley 알고리즘</b>(노벨 경제학상 수상 기법으로, 한 명이 여러 명에게 동시에 "
        "선택되지 않도록 자리를 자동 조정해 주는 방법)으로 정확하게 1:1 매칭합니다."
    ))
    s.append(bullet(
        "<b>④ 4가지 상태로 분류</b> — 짝을 찾은 객체 중 거의 같은 자리면 <b>matched(정지)</b>, "
        "조금 이동했으면 <b>moved(이동)</b>, 짝을 못 찾고 오늘만 있으면 <b>new(신규)</b>, "
        "어제만 있으면 <b>disappeared(소실)</b>로 분류합니다. "
        "<i>(부수적으로 두 사진의 촬영 범위가 다른 경우 '관측 공백' 상태를 추가로 부여하여 단순 촬영 "
        "범위 차이를 변화로 오인하지 않도록 보호합니다.)</i>"
    ))

    # ── 구성 B ──
    s.append(h3("[구성 B] 누적 기억 노트 — 지식 그래프부 (Step 2 · GraphRAG)"))
    s.append(callout(
        "<b>비유</b>: 분석관 사무실 벽에 붙어 있는 <b>커다란 메모판</b>을 상상하세요. \"강남 일대\" 칸에는 "
        "그동안 그 지역에서 관측된 전차·장갑차·포병의 출현·소실 횟수가 정자로 쌓여 가고, 색 줄로 "
        "\"전차+포병이 같이 자주 등장하더라\" 같은 패턴이 표시되어 있는 메모판입니다. 이 메모판을 "
        "<b>컴퓨터가 자동으로 매일 갱신</b>해 주는 것이 이 부분의 역할입니다."
    ))
    s.append(bullet(
        "<b>① 1km 격자로 위치 정리</b> — 위경도 소수점 2자리까지 반올림하여 약 1km × 1km 격자 단위로 "
        "지역을 묶습니다. 그러면 GPS 오차로 좌표가 살짝 어긋난 동일 객체도 <b>같은 칸</b>으로 자동 "
        "통합되어 누적 통계가 흩어지지 않습니다."
    ))
    s.append(bullet(
        "<b>② AI를 부르지 않고 카운터만 증가</b> — \"강남에서 전차가 새로 등장(new)\"이면 <b>강남 칸의 "
        "전차 'new 횟수'에 1을 더할 뿐</b>입니다. AI(LLM)를 부르지 않으므로 비용이 거의 없고, 같은 입력에 "
        "대해 매번 같은 결과가 나오는 <b>완전 재현 가능</b>한 구조입니다. 이는 인용 발명의 \"AI가 매번 "
        "정리해 줌\" 방식과 가장 큰 차별점입니다."
    ))
    s.append(bullet(
        "<b>③ 자주 함께 등장하는 자산들의 묶음 자동 발견</b> — '강남 칸'에서 전차와 포병이 같이 자주 "
        "나타나면 그 둘 사이에 굵은 연결선을 그어 두고, <b>Louvain 알고리즘</b>(친한 노드들을 자동 그룹화 "
        "하는 기법)으로 \"전차+장갑차+포병 = 기갑 복합체\", \"레이더+지휘소 = C2 인프라\" 같은 작전 "
        "형태를 자동 발견합니다. 분석관이 일일이 찾을 필요가 없습니다."
    ))
    s.append(bullet(
        "<b>④ 필요한 부분만 골라 AI에게 압축 전달</b> — 보고서를 작성할 시점이 되면, <b>해당 지역에 "
        "관련된 메모만 추려 500자 정도로 압축한 작은 메모지</b>를 만들어 AI에게 전달합니다 "
        "(Local Search + Global Search). 메모지 안에는 \"이 지역에서 전차가 8번째 누적 출현 중, 평균 "
        "신뢰도 0.86, 같은 지역의 doctrine 패턴: 기갑 복합체\" 같은 정보가 들어 있습니다."
    ))

    # ── 구성 C ──
    s.append(h3("[구성 C] AI 보고서 작성부 (Step 3)"))
    s.append(callout(
        "<b>비유</b>: 마지막 단계는 \"AI 분석관에게 정해진 양식의 보고서를 시키기\"입니다. "
        "AI에게 자유롭게 쓰게 두면 엉뚱한 말을 할 수 있으므로, <b>꼭 써야 할 항목·금지 표현·압축된 "
        "참고 자료</b>를 미리 정해서 함께 넘겨줍니다."
    ))
    s.append(bullet(
        "<b>① 변화한 것만 골라 전달</b> — '정지(matched)'와 '이동(moved)'은 보고서에서 빼고, "
        "'신규(new)'와 '소실(disappeared)'만 신뢰도 순으로 정렬해 AI에게 줍니다. 변화 분석에 무관한 "
        "객체로 AI 입력을 채우지 않습니다."
    ))
    s.append(bullet(
        "<b>② 군사 오판 방지 안전장치 강제</b> — \"DISAPPEARED는 영상에서 더 이상 관측되지 않는다는 "
        "뜻이며 파괴(destroyed)가 아니다\"라는 의미 제약을 시스템 프롬프트에 강제로 넣어, AI가 미관측을 "
        "파괴로 잘못 해석하지 않도록 막습니다."
    ))
    s.append(bullet(
        "<b>③ 군사 보고서 표준 8개 섹션 강제</b> — (1) 분류 (2) 핵심 요약 (3) 상황 (4) 변화 분석 "
        "(5) 위협 평가 (6) 정보 공백 (7) 권고 조치 (8) 부록의 8개 항목을 반드시 채우도록 AI에게 양식을 "
        "강제합니다."
    ))
    s.append(bullet(
        "<b>④ 같은 AI로 한국어 번역(무오염)</b> — 영문으로 1차 보고서를 만든 뒤 같은 AI를 한 번 더 "
        "호출하여 한국어로 옮기는데, <b>좌표·시각·신뢰도·클래스명 같은 정확한 정보는 그대로 두고 "
        "설명 문장만 한국어로 변환</b>합니다. 별도 번역기 없이 다국어 일관성을 확보합니다."
    ))

    s.append(h2("(2) 발명의 효과"))
    effects = [
        ("객체 단위로 정확한 변화 탐지",
         "두 사진의 객체 하나하나를 짝지어 비교하므로 \"몇 대 늘었다·줄었다\" 수준이 아니라 "
         "\"어떤 자산이 어디로 움직였다\"까지 정확히 추적합니다."),
        ("AI 호출 없이 누적되는 기억 — 비용·재현성 우수",
         "그래프 메모판이 매번 같은 입력에 같은 결과를 산출하고, AI 호출 비용이 전혀 들지 않아 "
         "운용 비용이 매우 낮고 군사 감사가 가능합니다."),
        ("시간을 거치며 더 똑똑해지는 보고서",
         "사용할수록 메모판이 두꺼워져 \"5회째 반복 배치\", \"평소엔 없던 이상 군집 출현\" 같은 "
         "시계열 인텔리전스가 자연스럽게 보고됩니다."),
        ("AI의 환각 차단",
         "메모판에서 미리 추린 정확한 정보만을 AI에게 압축 전달하므로 AI가 상상력으로 빈칸을 채우는 "
         "위험이 최소화됩니다."),
        ("군사 오판 방지",
         "'DISAPPEARED ≠ destroyed' 안전장치 + 8개 섹션 강제로 분석관·지휘관이 신뢰할 수 있는 "
         "표준 형식 보고서를 자동 산출합니다."),
    ]
    for i,(t,d) in enumerate(effects,1):
        s.append(bullet(f"<b>{i}. {t}</b> — {d}"))

    # ═══════════════════════════════════════════════════
    # 3. 청구범위
    # ═══════════════════════════════════════════════════
    s.append(h1("3. 특허 청구범위"))
    s.append(note(
        "각 청구항 상단에는 비전문가용 '쉬운 설명'을 함께 표시했습니다. 실제 권리범위는 그 아래의 "
        "정형 문장에 의해 정의됩니다."
    ))

    s.append(claim_box(
        "[청구항 1] (독립항) — 전체 방법",
        "같은 지역을 두 시점에 촬영한 위성·드론 사진에서 객체별 변화를 자동으로 짝지어 찾고, 그 결과를 "
        "지역별·종류별로 누적 그래프에 쌓아두며, AI에게 그 누적 기록을 압축해 전달해 표준 보고서를 "
        "자동 작성하는 방법.",
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
            "지리 좌표 거리에 따라 정지 또는 이동을, 미매칭 현재 객체에는 신규 출현을, 미매칭 과거 "
            "객체에는 소실을 부여하는 단계;",
            "각 자산의 위경도 좌표를 정수형 격자 단위로 양자화하여 위치 노드 키를, 객체 클래스와 "
            "위치 키의 결합으로 자산 노드 키를 결정론적으로 생성하고, 상기 변화 상태에 따라 자산 "
            "노드의 누적 카운터 속성을 거대언어모델(LLM) 호출 없이 갱신하며, 자산-위치 간 found_at "
            "엣지 및 동일 세션·동일 격자 내 공출현 자산 쌍의 co_occurred_with 엣지의 가중치를 누적 "
            "갱신하여 지식 그래프를 형성하는 단계;",
            "상기 co_occurred_with 엣지 가중치를 기반으로 Louvain 군집화를 수행하여 자산 복합체 "
            "커뮤니티 요약문을 LLM 호출 없이 구조적으로 생성하고, 보고서 대상 좌표 반경 내 자산 노드 "
            "이력을 조회하는 Local Search와 동일 반경과 중첩되는 커뮤니티 요약을 조회하는 Global "
            "Search를 병합하여 사전 설정 토큰 예산 이내의 역사적 맥락 컨텍스트 블록을 생성하는 단계; 및",
            "상기 변화 상태가 신규 출현 또는 소실에 해당하는 객체만을 신뢰도 내림차순으로 추출하고, "
            "상기 역사적 맥락 컨텍스트 블록을 LLM 프롬프트 선두에 prepend하며, 시스템 프롬프트에 "
            "\"'DISAPPEARED'는 영상 미관측을 의미하며 파괴 확인이 아니다\"라는 의미 제약과 표준 IMINT "
            "8개 섹션 구조를 강제 주입하여 LLM 에이전트가 구조화된 상황 판독 보고서를 자동 생성하는 "
            "단계;",
            "를 포함하는 AI 보조 상황 판독 보고서 자율 생성 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 2] (종속) — 객체만 깔끔하게 잘라내기",
        "객체 비교 전에 배경을 까맣게 지운 뒤 그 객체만 잘라서 비교 — 배경 때문에 다른 객체로 오인하지 "
        "않도록 함.",
        [
            "제1항에 있어서, 탐지된 자산의 세그멘테이션 마스크 RLE 정보를 디코딩하여 마스크 활성 영역 "
            "외의 픽셀을 영(0)으로 마스킹하고 바운딩 박스 주위에 사전 설정된 패딩 폭을 부가하여 "
            "크롭하며, 마스크 RLE가 부재하는 경우 바운딩 박스 크롭으로 폴백하고, 산출된 임베딩 벡터를 "
            "L2 정규화하여 코사인 유사도 행렬 산출에 사용하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 3] (종속) — 안정적인 1:1 짝짓기",
        "두 사진 객체들의 비슷한 정도를 점수표로 만들고, 누구도 손해 보지 않는 짝을 자동으로 찾는 방법 "
        "(Gale-Shapley) + 영상 간격이 짧으면 비디오 추적기를 대신 쓰는 듀얼 모드.",
        [
            "제1항에 있어서, (a) 임베딩 모델이 가용한 경우 임베딩 유사도와 크기 유사도의 가중합을 "
            "점수로 사용하고, (b) 임베딩 모델이 부재한 경우 지리 좌표 근접도 점수로 폴백하며, "
            "(c) 점수가 임계값을 초과하는 후보 쌍에 대해 Gale-Shapley 지연 승인 알고리즘으로 1:1 안정 "
            "매칭을 수행하고, (d) 영상 간 시간 간격이 사전 설정 임계값 이하일 때는 비디오 추적기 기반 "
            "ID 추적 모드를 선택적으로 사용하는 듀얼 모드 구조를 갖는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 4] (종속) ★ AI 없이 자동으로 쌓이는 기억 노트",
        "AI를 부르지 않고도 그래프 노드의 카운터를 +1씩 늘려서 자동 누적 — 같은 입력에 매번 같은 결과 "
        "(완전 재현). 본 발명의 가장 강한 차별점.",
        [
            "제1항에 있어서, 상기 지식 그래프 형성 단계는, 각 자산의 위경도 좌표를 소수점 N자리의 "
            "정수형 격자 단위로 반올림하여 \"loc:{lat:.Nf},{lon:.Nf}\" 형식의 결정론적 위치 노드 키를 "
            "생성하고, 객체 클래스 및 위치 키를 결합하여 \"asset:{class}:{loc_key}\" 형식의 결정론적 "
            "자산 노드 키를 생성하는 단계; 페어링 레코드의 상태 필드 및 신뢰도를 자산 노드의 카운터 "
            "속성(new_count, matched_count, moved_count, disappeared_count, total_confidence) 및 "
            "시간 속성에 LLM 호출 없이 결정론적으로 누적 가산하는 upsert 단계; 및 자산-위치 간 "
            "found_at 엣지와 동일 세션·동일 위치 키 내 공출현 자산 쌍의 co_occurred_with 엣지의 "
            "count 속성을 LLM 호출 없이 결정론적으로 누적 갱신하는 단계를 포함하여, 동일 입력에 대해 "
            "매번 동일한 그래프가 산출되는 결정론적 재현성을 갖는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 5] (종속) — '함께 자주 등장하는' 묶음 자동 발견",
        "전차·장갑차·포병이 자주 같이 등장하면 \"기갑 복합체\"라고 자동 묶어 주는 알고리즘 적용.",
        [
            "제1항에 있어서, co_occurred_with 엣지의 count 가중치를 입력으로 Louvain 계층적 군집화 "
            "알고리즘을 수행하는 단계; 및 각 커뮤니티에 대해 구성원 자산 클래스 분포·위치 클러스터 "
            "수·총 관측 수·신규 배치 수·소실 수를 포함하는 member_summary를 LLM 호출 없이 결정론적 "
            "통계 산식만으로 생성하여 그래프 데이터베이스에 기록하는 단계를 포함하는 것을 특징으로 "
            "하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 6] (종속) — 필요한 부분만 압축해 AI에 전달",
        "보고서 대상 지역의 메모만 추려 500자 정도로 압축해 AI 입력 맨 앞에 붙여 줌.",
        [
            "제1항에 있어서, 보고서 대상 좌표를 중심으로 사전 설정 반경 R 이내의 자산 노드 이력 통계"
            "(누적 관측 수, 신규/지속/이동/소실 횟수, 평균 신뢰도, 최초/최종 관측 시각)를 조회하는 "
            "Local Search 단계; 동일 반경과 중첩되는 커뮤니티들의 member_summary를 조회하는 Global "
            "Search 단계; 및 양 검색 결과를 사전 설정 토큰 예산(예: 500 토큰) 이내로 절단·머지하여 "
            "LLM 프롬프트 선두에 prepend되는 단일 컨텍스트 블록으로 포맷화하는 단계를 포함하는 것을 "
            "특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 7] (종속) — 변화 객체만, 표준 양식 강제, 같은 AI로 번역",
        "정지·이동 객체는 빼고 신규·소실만 AI에 전달, 표준 8개 항목과 안전 문구 강제, 같은 AI로 한국어 "
        "변환 시 좌표·수치는 그대로 보존.",
        [
            "제1항에 있어서, 페어링 결과 중 'matched' 및 'moved' 상태 객체는 LLM 프롬프트에서 제외하고 "
            "'new' 및 'disappeared' 상태 객체만을 신뢰도 내림차순으로 주입하는 단계; 시스템 프롬프트에 "
            "표준 IMINT 8개 섹션 구조 및 \"'DISAPPEARED'는 영상 미관측이며 파괴가 아니다\"라는 의미 "
            "제약을 강제 주입하는 단계; 및 동일 LLM 인스턴스를 인메모리 상태로 재사용하여 영문 보고서 "
            "생성 후 한국어 번역 세션을 발동하되, 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 토큰은 "
            "바이패스하고 분석 서술문만을 변환하는 단계를 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 8] (종속) — 부수적 촬영 범위 가드",
        "두 사진의 촬영 범위가 다른 부분에 있는 객체를 '진짜 변화'와 구분하여 오인을 방지하는 부수 "
        "안전장치.",
        [
            "제1항에 있어서, 최신 영상 및 과거 영상 각각의 메타데이터로부터 위경도 경계면 정보를 "
            "추출하고, 미매칭 객체가 상대 영상의 경계면 외부에 위치하는 경우 'past_not_included' 또는 "
            "'current_not_included' 상태를 부여하여 단순 촬영 범위 불일치에 따른 관측 공백을 실제 "
            "변화와 구분하는 단계를 부수적으로 더 포함하는 것을 특징으로 하는 방법.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 9] (독립항) — 시스템 청구",
        "위의 방법을 수행하는 컴퓨터 시스템.",
        [
            "제1항 내지 제8항 중 어느 한 항의 방법을 수행하는 하나 이상의 하드웨어 프로세서와 "
            "메모리를 포함하는 AI 보조 상황 판독 보고서 자율 생성 시스템.",
        ],
    ))
    s.append(Spacer(1,5))

    s.append(claim_box(
        "[청구항 10] (독립항) — 기록매체",
        "위 방법을 실행시키는 프로그램이 들어 있는 USB·SSD 등 기록매체.",
        [
            "제1항 내지 제8항 중 어느 한 항의 방법을 컴퓨터에서 실행시키기 위한 프로그램이 기록된 "
            "컴퓨터 판독 가능 기록 매체.",
        ],
    ))

    # ═══════════════════════════════════════════════════
    # 4. 도면
    # ═══════════════════════════════════════════════════
    s.append(h1("4. 도면 설명 명기"))

    s.append(fig_box(
        "도 1] 시스템 전체 구성도",
        "위성·드론 사진 수집 → 객체 탐지(SAM3) → 객체만 잘라 비교(변화 탐지부) → "
        "지역별 누적 메모판 갱신(GraphRAG) → 압축 메모 추출 → AI에게 표준 보고서 작성 시키기 → "
        "같은 AI로 한국어 번역 → 보고서 DB 저장"
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 2] 객체별 변화 탐지 메커니즘 (변화 탐지 차별성)",
        "각 객체 배경 검게 지우기 → 객체 조각만 잘라 비전 AI에 통과 → 객체끼리 비슷한 정도(코사인 "
        "유사도) 표 만들기 → 같은 종류끼리만 Gale-Shapley로 1:1 매칭 → 거리에 따라 정지/이동, "
        "짝 못 찾으면 신규/소실로 분류"
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 3] 누적 기억 노트 자동 갱신 (GraphRAG 차별성)",
        "1km 격자 칸으로 위치 정리 → 칸별·종류별 카운터(new/matched/moved/disappeared) +1 갱신 "
        "(AI 호출 0회) → 같이 자주 등장한 자산 쌍 연결선 굵게 → Louvain으로 \"기갑 복합체\", "
        "\"C2 인프라\" 같은 묶음 자동 발견"
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 4] AI에게 보고서 시키기 흐름",
        "해당 지역 메모 추리기 → ~500자로 압축 → 변화 객체(new·disappeared)만 골라 신뢰도 순 정렬 → "
        "표준 8개 항목·안전 문구 강제 → AI 1차 영문 보고서 → 같은 AI 2차 한국어 변환 "
        "(좌표·수치는 보존) → 보고서 DB 저장"
    ))
    s.append(Spacer(1,8))

    s.append(fig_box(
        "도 5] 전체 파이프라인 시퀀스 (시간 순)",
        "시작 → 사진 적재 + 초해상도 → SAM3 탐지 → 배경 제거 + 임베딩 → 코사인 유사도 표 → "
        "Gale-Shapley 매칭 → 객체별 4상태 분류 [+부수적 FOV 가드] → 1km 격자 양자화 → AI 없이 "
        "카운터 갱신 → Louvain 군집 → 압축 메모 생성 → AI 영문 보고서 → 같은 AI 한국어 번역 → "
        "보고서 DB 저장 → 종료"
    ))

    doc.build(s, onFirstPage=draw_page_num, onLaterPages=draw_page_num)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    build()
