"""Generate the patent application as a .docx file."""

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def set_korean_font(run, font_name="맑은 고딕", size=11, bold=False):
    run.font.name = font_name
    run.font.size = Pt(size)
    run.font.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), font_name)
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)


def add_title(doc, text, size=18):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_korean_font(run, size=size, bold=True)
    p.paragraph_format.space_after = Pt(12)


def add_h1(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_korean_font(run, size=15, bold=True)
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)


def add_h2(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_korean_font(run, size=13, bold=True)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)


def add_h3(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_korean_font(run, size=11.5, bold=True)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)


def add_para(doc, text, size=11, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_korean_font(run, size=size, bold=bold)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.5


def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.75 + 0.75 * level)
    run = p.add_run(text)
    set_korean_font(run, size=11)
    p.paragraph_format.line_spacing = 1.4


def add_code_block(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "Consolas")
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.line_spacing = 1.1
    # 음영 추가
    pPr = p._element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)


def add_table(doc, header, rows, widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Light Grid Accent 1"
    for j, h in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_korean_font(run, size=10.5, bold=True)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            set_korean_font(run, size=10)
    if widths:
        for j, w in enumerate(widths):
            for row in table.rows:
                row.cells[j].width = Cm(w)
    doc.add_paragraph()


def build():
    doc = Document()

    # 페이지 여백
    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # ───────── 제목 ─────────
    add_title(doc, "특 허 출 원 명 세 서", size=20)
    add_para(doc, "")

    # ───────── 발명의 명칭 ─────────
    add_h1(doc, "【발명의 명칭】")
    add_para(
        doc,
        "지식 그래프 기반 시계열 컨텍스트 주입을 이용한 AI 에이전트 군사 판독보고서 자동 생성 시스템 및 그 방법",
        bold=True,
    )
    add_para(
        doc,
        "(영문: AI Agent-Based Military Intelligence Report Generation System and Method "
        "Using Knowledge-Graph-Augmented Temporal Context Injection)",
    )

    # ───────── 1. 개요 ─────────
    add_h1(doc, "1. 개요")

    add_h2(doc, "(1) 수탁과제 무관 근거 설명")

    add_h3(doc, "1) 제안 발명 내용 요약")
    add_para(
        doc,
        "영상 객체 탐지·페어링 결과를 지식 그래프로 누적하고, 그 그래프에서 추출한 시계열 컨텍스트를 "
        "LLM에 주입하여 표준 IMINT 8개 섹션의 군사 판독보고서를 자동 생성하는 AI 에이전트 시스템.",
    )

    add_h3(doc, "2) 중복성·차별성")
    add_bullet(doc, "중복성: 없음")
    add_bullet(
        doc,
        "차별성: 기존 LLM 보고서 자동 생성 특허는 (i) 텍스트 코퍼스 기반 RAG 또는 "
        "(ii) 단발 프레임 탐지만을 LLM 입력으로 사용함. 본 발명은 영상 탐지의 정형 출력을 "
        "LLM 호출 없이 결정론적으로 지식 그래프에 누적하고, Local + Global 이중 검색으로 산출한 "
        "~500토큰 시계열 컨텍스트를 LLM 프롬프트에 prepend하여 시계열 패턴 인텔리전스 보고로 "
        "격상시키는 점에서 차별됨.",
    )

    add_h2(doc, "(2) 발명 제안 배경")
    add_para(
        doc,
        "위성·드론 영상 인텔리전스 분야는 SAM, YOLO-World 등 zero-shot 텍스트 프롬프트 기반 "
        "객체 탐지 모델이 보급되면서 탐지 자동화 수준이 크게 향상되었다. 그러나 군사 분석관이 "
        "요구하는 최종 산출물은 탐지 결과의 나열이 아니라 표준 IMINT 형식의 정형 판독보고서이며, "
        "이는 LLM에 탐지 결과를 단순 입력하는 방식만으로는 정확도·일관성·도메인 적합성을 확보하기 "
        "어렵다.",
    )
    add_para(
        doc,
        "또한 단일 프레임의 탐지 결과만 LLM에 제공하는 경우 '이 지역에서 5회째 반복되는 배치 패턴', "
        "'직전 3회 모두 소실 후 재출현' 같은 시계열 인텔리전스 패턴이 보고서에 반영되지 못해 분석관의 "
        "후속 판단을 보조하지 못한다. 본 발명은 (i) 영상 탐지·페어링 결과를 LLM 호출 없이 그래프로 "
        "결정론적 누적하고, (ii) 그래프 기반 지역·전역 이중 검색으로 시계열 컨텍스트를 추출하며, "
        "(iii) 변화 객체(신규·소실)만을 표준 8섹션 프롬프트와 결합하여 LLM에게 전달하는 AI 에이전트 "
        "시스템을 제안한다.",
    )

    add_h2(doc, "(3) 산업상 이용 분야 (용도)")
    for s in [
        "국방·정보 분야 위성/드론 영상 판독 자동화 시스템",
        "국가정보·해양감시·국경감시 등 ISR(Intelligence, Surveillance, Reconnaissance) 자동 보고 시스템",
        "재난·산림·해상사고 모니터링용 정형 보고서 자동 생성 시스템",
        "민간 위성영상 서비스(농업·도시 변화·공급망)의 시계열 변화 자동 보고서",
        "자율감시 로봇·드론 군집의 임무 결과 보고서 자동화",
    ]:
        add_bullet(doc, s)

    add_h2(doc, "(4) 종래 기술")

    add_h3(doc, "Microsoft Research, GraphRAG (2024)")
    add_para(
        doc,
        "본 발명과 가장 유사한 단일 종래 기술은 Microsoft Research가 2024년 4월 24일 공개한 "
        "GraphRAG 시스템이다. 해당 기술은 다음 문헌 및 공개 구현에 의해 명확히 기록되어 있어 "
        "특허 심사 단계에서 인용 가능한 공지기술에 해당한다.",
    )
    add_bullet(
        doc,
        "[비특허문헌 1] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, "
        "Apurva Mody, Steven Truitt, Jonathan Larson, \"From Local to Global: A Graph RAG "
        "Approach to Query-Focused Summarization\", arXiv:2404.16130, Microsoft Research, "
        "2024.04.24.",
    )
    add_bullet(
        doc,
        "[비특허문헌 2] Microsoft, \"GraphRAG: Unlocking LLM discovery on narrative private "
        "data\", Microsoft Research Blog, 2024.02.13.",
    )
    add_bullet(
        doc,
        "[비특허문헌 3] Microsoft, GraphRAG 공식 오픈소스 구현 — "
        "https://github.com/microsoft/graphrag (Apache-2.0 License, 2024 공개).",
    )

    add_para(doc, "Microsoft GraphRAG의 핵심 구성은 다음과 같다.")
    add_bullet(
        doc,
        "(가) 입력 문서 코퍼스(자연어 텍스트)를 청크 단위로 분할한 후, 대규모 언어 모델(LLM)을 "
        "이용해 각 청크에서 엔티티(entity)와 관계(relation)를 추출하여 지식 그래프를 구축한다.",
    )
    add_bullet(
        doc,
        "(나) 구축된 지식 그래프에 대해 Leiden 등의 그래프 군집 알고리즘을 적용해 계층적 커뮤니티를 "
        "탐지하고, 각 커뮤니티의 요약문을 LLM이 자연어로 생성하여 보고서 형태로 저장한다.",
    )
    add_bullet(
        doc,
        "(다) 사용자 질의에 대해 (i) 노드 이웃 단위로 검색하는 Local Search 모드와 "
        "(ii) 커뮤니티 요약을 검색하는 Global Search 모드의 두 가지 검색을 제공한다.",
    )
    add_bullet(
        doc,
        "(라) 검색 결과를 LLM 프롬프트에 컨텍스트로 주입하여 질의응답 또는 요약을 생성한다.",
    )

    add_h2(doc, "(5) 종래 기술의 문제점 / 한계")
    add_para(
        doc,
        "Microsoft GraphRAG는 자연어 문서 코퍼스에 대한 일반 목적의 질의응답·요약 시스템으로 "
        "설계되어, 본 발명의 적용 도메인인 군사 위성·드론 영상 인텔리전스 보고서 자동 생성에 그대로 "
        "적용할 경우 아래와 같은 본질적 한계를 갖는다.",
    )
    problems = [
        ("입력 도메인 불일치 — 자연어 vs 정형 영상 탐지 튜플",
         "Microsoft GraphRAG는 입력을 '자연어 텍스트 청크'로 전제한다. 그러나 본 발명의 입력은 영상 "
         "객체 탐지·페어링 결과인 정형 튜플(class, lat, lon, status, confidence, time)이다. "
         "이를 GraphRAG에 적용하려면 (i) 정형 튜플을 자연어로 풀어서 → (ii) 청크화 → (iii) 임베딩 → "
         "(iv) LLM 엔티티 추출의 4단계 우회를 거쳐야 하며 그 과정에서 정확한 수치(좌표·신뢰도)가 "
         "토큰화 손실을 입는다."),
        ("LLM 의존 비결정적 엔티티 추출",
         "Microsoft GraphRAG는 그래프 노드·엣지 추출을 LLM 호출로 수행한다. 동일 입력에 대해 매번 "
         "다른 엔티티가 추출될 수 있어 군사 시스템에 요구되는 결정론·재현성·감사가능성을 만족하지 "
         "못한다. 또한 노드 수가 증가할수록 LLM 호출 비용이 선형으로 증가한다."),
        ("시계열 누적 메커니즘 부재",
         "Microsoft GraphRAG는 정적 코퍼스에 대한 단일 인덱싱을 전제로 한다. 동일 자산이 다른 "
         "시점에 동일 지역에서 반복 관측될 때 이를 동일 엔티티로 누적 카운팅하는 결정론적 키 "
         "메커니즘이 정의되어 있지 않다. 결과적으로 '이 지역에서 N번째 반복 출현' 같은 시계열 패턴이 "
         "보고서에 반영되지 못한다."),
        ("GPS 노이즈 흡수 메커니즘 부재",
         "동일 객체의 두 관측이 GPS 측정 오차로 미세하게 다른 좌표(예: 37.5765 ↔ 37.5766)일 때 "
         "Microsoft GraphRAG는 두 좌표를 서로 다른 엔티티로 임베딩한다. 이를 동일 자산으로 통합하기 "
         "위한 격자 양자화(quantization) 단계가 부재하다."),
        ("도메인 안전성 제약 부재",
         "Microsoft GraphRAG는 일반 목적 프롬프트를 사용하므로 군사 도메인의 의미론적 가드레일 "
         "(예: 'DISAPPEARED'를 'destroyed'로 오해석하면 안 됨)을 시스템 프롬프트에 강제하는 구성이 "
         "없다. 일반 LLM이 '미관측'을 '확정 파괴'로 표현하여 군사적 오판을 유발할 수 있다."),
        ("변화 객체 한정 메커니즘 부재",
         "Microsoft GraphRAG는 검색된 모든 컨텍스트를 LLM에 전달한다. 군사 변화 분석에 필요한 "
         "신규 출현(new)·소실(disappeared) 객체만을 추출해 토큰을 절약하는 도메인 특화 필터링 "
         "구성이 없다."),
        ("정형 IMINT 보고서 형식 강제 부재",
         "Microsoft GraphRAG는 일반 요약문을 산출한다. CLASSIFICATION/EXECUTIVE SUMMARY/SITUATION/"
         "CHANGE ANALYSIS/THREAT ASSESSMENT/INTELLIGENCE GAPS/RECOMMENDED ACTIONS/APPENDIX의 "
         "표준 IMINT 8개 섹션을 강제하여 분석관 친화 산출물을 보장하는 구성이 없다."),
    ]
    add_table(doc, ["#", "Microsoft GraphRAG의 한계", "본 발명 적용 시 문제점"],
              [[str(i + 1), t, d] for i, (t, d) in enumerate(problems)],
              widths=[0.8, 4.5, 10.7])

    add_para(
        doc,
        "따라서 Microsoft GraphRAG는 본 발명의 직접적 종래 기술이나, 군사 영상 인텔리전스 도메인에서 "
        "요구되는 (i) 정형 튜플 직접 인덱싱, (ii) 결정론적 LLM-free 엔티티 추출, (iii) 시계열 누적 "
        "카운터, (iv) 격자 양자화, (v) 도메인 의미론 제약, (vi) 변화 객체 한정 전달, (vii) 표준 "
        "IMINT 형식 강제의 7가지 차별 구성을 모두 결여하고 있으므로 본 발명의 진보성이 인정된다.",
        bold=False,
    )

    # ───────── 2. 상세 설명 ─────────
    add_h1(doc, "2. 상세 설명")

    add_h2(doc, "(1) 발명의 내용 (구성)")
    add_para(doc, "본 발명은 다음 5개 기능 모듈을 포함한다.")

    add_h3(doc, "A. 결정론적 그래프 인덱서 (Graph Indexer) [200]")
    for s in [
        "위경도를 0.01°(약 1km) 격자로 양자화하여 위치(location) 노드 키 생성",
        "(객체 클래스 × 위치) 고유 쌍으로 자산(asset) 노드 키 생성",
        "status 필드(new/matched/moved/disappeared)를 자산 노드 카운터 속성에 LLM 호출 없이 결정론적 누적",
        "asset → location 의 found_at 엣지, asset ↔ asset 의 co_occurred_with 엣지를 누적 가중치로 갱신",
    ]:
        add_bullet(doc, s)

    add_h3(doc, "B. 커뮤니티 탐지부 (Community Detector) [300]")
    for s in [
        "co_occurred_with 엣지 가중치 기반 Louvain 알고리즘으로 자산 군집 자동 발견",
        "각 군집의 member_summary(자산 분포·위치 클러스터·관측/신규/소실 통계)를 LLM 비호출로 구조 생성",
    ]:
        add_bullet(doc, s)

    add_h3(doc, "C. 이중 모드 그래프 검색부 (Local + Global Retriever) [500]")
    for s in [
        "Local Search: 현재 좌표 반경 R 내 자산 노드 이력 통계 조회",
        "Global Search: 동일 반경과 중첩되는 커뮤니티의 member_summary 조회",
        "두 결과를 토큰 예산(~500 tokens) 이내로 절단·머지하여 단일 컨텍스트 블록 생성",
    ]:
        add_bullet(doc, s)

    add_h3(doc, "D. 변화 객체 중심 프롬프트 조립부 (Prompt Composer) [600]")
    for s in [
        "페어링 결과 중 'new'·'disappeared' 상태만 추출, 신뢰도 내림차순 정렬, 상위 20건 + 잔여 요약",
        "시스템 프롬프트에 군사 의미 제약 강제: \"'DISAPPEARED'는 미관측을 의미하며 파괴 확인이 아님\"",
        "표준 IMINT 8개 섹션 강제: CLASSIFICATION / EXECUTIVE SUMMARY / SITUATION / CHANGE ANALYSIS / "
        "THREAT ASSESSMENT / INTELLIGENCE GAPS / RECOMMENDED ACTIONS / APPENDIX",
        "C 모듈의 컨텍스트 블록을 사용자 프롬프트 선두에 prepend",
    ]:
        add_bullet(doc, s)

    add_h3(doc, "E. 동일 LLM 재호출 기반 번역·헤더 부착부 [700]")
    for s in [
        "동일 LLM 모델을 1차(영문 보고서) → 2차(한국어 번역)로 재호출",
        "번역 규칙: 섹션 헤더·좌표·타임스탬프·클래스명·신뢰도 토큰 보존, 서술 텍스트만 번역",
        "메타 헤더(모델명·세션ID·관측 시각·페어링 통계) 부착하여 최종 산출",
    ]:
        add_bullet(doc, s)

    add_h2(doc, "(2) 발명의 효과")
    effects = [
        ("시계열 인텔리전스 격상",
         "단발 프레임 변화 보고가 아닌 'N번째 반복 출현, K번 소실 후 재등장' 수준의 시계열 패턴 보고 자동 생성."),
        ("결정론·재현성",
         "그래프 인덱싱이 LLM 비호출 결정론 방식 → 동일 입력에 동일 그래프, 군사 감사 가능성 확보."),
        ("운용 비용 절감",
         "임베딩 모델·벡터 DB·임베딩 GPU 추론 불필요. SQLite + 그래프 알고리즘만으로 RAG 컨텍스트 산출."),
        ("토큰 효율 극대",
         "정지·이동 객체 제외 + ~500 토큰 그래프 컨텍스트만 추가 → 보고서 깊이 강화 + 토큰 절감."),
        ("군사 오판 방지",
         "'DISAPPEARED ≠ destroyed' 의미 제약으로 LLM 군사적 오해석 위험 억제."),
        ("다국어 일관성",
         "동일 LLM 재호출 번역 + 토큰 보존 규칙으로 별도 번역 모델 없이 형식 일관성 확보."),
        ("그래프 패턴 자동 발견",
         "Louvain 군집화로 '기갑·포병 복합체', 'C2 인프라' 같은 doctrine 패턴 자동 식별."),
    ]
    add_table(doc, ["#", "효과", "내용"],
              [[str(i + 1), t, d] for i, (t, d) in enumerate(effects)],
              widths=[1.0, 4.0, 11.0])

    # ───────── 3. 도면 ─────────
    add_h1(doc, "3. 도면")

    add_h2(doc, "(1) 개략적인 전체 그림 — 도 1")
    add_code_block(
        doc,
        """┌──────────────────────────────────────────────────────────────────────────┐
│        AI 에이전트 기반 판독보고서 자동 생성 시스템 [1000]                │
│                                                                          │
│  [입력 측 — 종래기술 활용]                                                │
│  영상 객체 탐지부 [110] → 시계열 페어링부 [120]                          │
│                  │                                                       │
│                  ▼  PairingRecord 배치                                   │
│  [본 발명의 핵심부]                                                       │
│  A. 결정론적 그래프 인덱서 [200]                                          │
│     ─ 격자 양자화부 [210]                                                │
│     ─ 자산 노드 누적부 [220]   ─ 엣지 누적부 [230]                       │
│                  │                                                       │
│                  ▼                                                       │
│  B. 커뮤니티 탐지부 [300]                                                │
│     ─ Louvain 군집화 [310]    ─ member_summary 생성부 [320]              │
│                  │                                                       │
│                  ▼                                                       │
│  지식 그래프 저장소 [400]                                                │
│  (graph_entities / graph_relations / graph_communities)                  │
│                  │                                                       │
│                  ▼                                                       │
│  C. 그래프 검색부 [500]                                                  │
│     ─ Local Search [510]      ─ Global Search [520]                      │
│     ─ 토큰 예산 컨텍스트 생성부 [530]                                     │
│                  │                                                       │
│                  ▼                                                       │
│  D. 변화 객체 중심 프롬프트 조립부 [600]                                 │
│     ─ new/disappeared 추출 [610]  ─ 시스템 프롬프트 제약 [620]            │
│     ─ 8섹션 강제부 [630]          ─ 컨텍스트 prepend [640]                │
│                  │                                                       │
│                  ▼                                                       │
│  E. AI 에이전트 LLM 호출부 [700]                                          │
│     ─ 영문 보고서 생성 [710]    ─ 한국어 번역 [720]                      │
│     ─ 메타 헤더 조립부 [730]                                             │
│                  │                                                       │
│                  ▼                                                       │
│        최종 IMINT 판독보고서 [800] → 보고서 DB                            │
└──────────────────────────────────────────────────────────────────────────┘""",
    )

    add_h2(doc, "(2) 발명의 특징부 세부 도면")

    add_h3(doc, "도 2 — A. 결정론적 그래프 인덱서 [200] 내부 동작")
    add_code_block(
        doc,
        """PairingRecord 입력
(current_class, current_lat/lon, past_class, past_lat/lon,
 status, confidence, session_id)
        │
        ▼
┌─────────────────────────────────────┐
│ [210] 격자 양자화부                  │
│  loc_key = "loc:{lat:.2f},{lon:.2f}" │
│  (소수점 2자리 = 약 1km 격자)         │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [220] 자산 노드 upsert (LLM 비호출)  │
│  asset_id = "asset:{class}:{loc_key}"│
│                                     │
│  status에 따라 카운터 증가:           │
│    new → new_count++                │
│    matched → matched_count++        │
│    moved → moved_count++            │
│    disappeared → disappeared_count++│
│                                     │
│  total_confidence += conf           │
│  sessions.append(session_id)        │
│  first/last_seen 갱신               │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [230] 엣지 누적부                    │
│  found_at: asset → location          │
│            properties.count++         │
│  co_occurred_with: asset ↔ asset      │
│   (같은 session_id + 같은 loc_key)    │
│   properties.count++                  │
└─────────────────┬───────────────────┘
                  ▼
          지식 그래프 저장소 [400]""",
    )

    add_h3(doc, "도 3 — C. 그래프 검색부 [500] 내부 동작")
    add_code_block(
        doc,
        """현재 페어링 중심 좌표 (lat_c, lon_c)
        │
        ▼
┌─────────────────────────────────────┐
│ [510] Local Search                   │
│   - 반경 R(예: 0.05°) 내 asset 노드   │
│   - 자산별 누적 통계 조회             │
│     (new/matched/moved/disappeared,  │
│      avg_conf, first/last_seen)      │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [520] Global Search                  │
│   - 반경 R과 중첩되는 community 조회 │
│   - member_summary 추출              │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [530] 컨텍스트 블록 생성부            │
│   - 토큰 예산(예: 500 tokens) 절단    │
│   - "=== GRAPHRAG HISTORICAL         │
│      CONTEXT ===" 포맷화              │
└─────────────────┬───────────────────┘
                  ▼
        D. 프롬프트 조립부 [600] 로 전달""",
    )

    add_h3(doc, "도 4 — D + E. 프롬프트 조립부 및 AI 에이전트 LLM 호출부")
    add_code_block(
        doc,
        """페어링 배치 입력
        │
        ▼
┌─────────────────────────────────────┐
│ [610] 변화 객체 추출                 │
│   - status ∈ {new, disappeared}     │
│   - confidence 내림차순 정렬          │
│   - 상위 20건 + "외 N건" 요약          │
└─────────────────┬───────────────────┘
                  │  +  [500]의 컨텍스트 블록
                  ▼
┌─────────────────────────────────────┐
│ [620] 시스템 프롬프트 제약            │
│  "You are a military IMINT analyst.  │
│   DISAPPEARED ≠ destroyed …"         │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [630] 8섹션 구조 강제부               │
│  1. CLASSIFICATION                   │
│  2. EXECUTIVE SUMMARY                │
│  3. SITUATION                        │
│  4. CHANGE ANALYSIS                  │
│  5. THREAT ASSESSMENT                │
│  6. INTELLIGENCE GAPS                │
│  7. RECOMMENDED ACTIONS              │
│  8. APPENDIX                         │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [640] 컨텍스트 prepend → 최종 prompt │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [710] LLM 1차 호출 → 영문 보고서      │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [720] 동일 LLM 2차 호출 → 한국어 번역 │
│  섹션헤더/좌표/클래스명 보존          │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ [730] 메타 헤더 부착                 │
│  (모델명, 세션ID, 관측시각, 통계)    │
└─────────────────┬───────────────────┘
                  ▼
        최종 IMINT 보고서 [800]""",
    )

    add_h2(doc, "(3) 발명의 설명 (동작 순서도) — 도 5")
    add_code_block(
        doc,
        """              ( 시작 )
                 │
                 ▼
  S100 : 페어링 결과 배치 수신
                 │
                 ▼
  S200 : 격자 양자화 → loc_key 생성
                 │
                 ▼
  S300 : 자산 노드 upsert (카운터 +1, LLM 비호출)
                 │
                 ▼
  S400 : found_at / co_occurred_with 엣지 갱신
                 │
                 ▼
  S500 : Louvain 커뮤니티 탐지
                 │
                 ▼
  S600 : Local Search (반경 R 자산 이력)
                 │
                 ▼
  S700 : Global Search (중첩 커뮤니티 요약)
                 │
                 ▼
  S800 : 토큰 예산 컨텍스트 블록 생성
                 │
                 ▼
  S900 : new/disappeared 추출 + 신뢰도 정렬
                 │
                 ▼
  S1000: 시스템 프롬프트 + 8섹션 강제 + 컨텍스트 prepend
                 │
                 ▼
  S1100: LLM 1차 호출 → 영문 IMINT 보고서
                 │
                 ▼
  S1200: 동일 LLM 2차 호출 → 한국어 번역
                 │
                 ▼
  S1300: 메타 헤더 부착 → 보고서 DB 저장
                 │
                 ▼
              ( 종료 )""",
    )

    add_h2(doc, "(4) 도면에 대한 상세 설명")

    add_h3(doc, "도 1 (전체 시스템)")
    add_para(
        doc,
        "본 시스템 [1000]은 입력 측 종래 기술인 영상 객체 탐지부 [110] 및 시계열 페어링부 [120]로부터 "
        "페어링 레코드(PairingRecord)를 입력받아, 본 발명의 핵심 5개 모듈(A~E)을 순차 진행해 최종 "
        "IMINT 판독보고서 [800]을 산출한다. 입력 측 탐지기는 SAM3, YOLO-World 등으로 대체 가능하며 "
        "본 청구범위에 종속되지 않는다. 본 발명의 권리범위는 모듈 [200]~[700]이 형성하는 결합 구조에 있다.",
    )

    add_h3(doc, "도 2 (그래프 인덱서)")
    add_para(
        doc,
        "격자 양자화부 [210]은 GPS 측정 오차 및 미세 좌표 변동을 단일 위치 노드로 흡수하여 동일 자산의 "
        "반복 관측을 누적 카운터로 압축한다. 자산 노드 upsert부 [220]은 LLM 호출 없이 페어링 상태 필드를 "
        "결정론적으로 가산하므로 동일 입력에 매번 동일한 그래프를 산출한다. 엣지 누적부 [230]은 자산-위치 "
        "found_at 관계와 동일 세션·동일 격자 공출현 자산 쌍의 co_occurred_with 관계를 누적하여 Louvain "
        "알고리즘이 가중치 기반 군집화를 수행할 수 있도록 한다.",
    )

    add_h3(doc, "도 3 (그래프 검색부)")
    add_para(
        doc,
        "Local Search [510]은 보고서 대상 좌표 반경 내 자산 노드 이력을 직접 조회하여 \"이 지역에서 "
        "이 자산이 어떤 빈도·패턴으로 출현·소실되었는가\"를 LLM이 참고할 형태로 추출한다. "
        "Global Search [520]은 동일 반경과 중첩되는 군집의 member_summary를 조회하여 \"이 지역이 어느 "
        "doctrine 패턴(예: 기갑·포병 복합체)에 속하는가\"라는 메타 수준 컨텍스트를 제공한다. "
        "컨텍스트 블록 생성부 [530]은 두 결과를 사전 설정 토큰 예산 이내로 압축한다.",
    )

    add_h3(doc, "도 4 (프롬프트 조립 + LLM 호출)")
    add_para(
        doc,
        "변화 객체 추출부 [610]은 활동 지표인 신규·소실 객체만 LLM에 전달하여 토큰 효율과 분석 집중도를 "
        "동시에 확보한다. 시스템 프롬프트 제약부 [620]은 \"'DISAPPEARED' ≠ 파괴 확인\"이라는 의미론적 "
        "가드레일을 명시하여 LLM의 군사적 오해석을 억제한다. 8섹션 강제부 [630]은 표준 IMINT 형식을 "
        "강제해 분석관 친화 산출물을 보장한다. LLM 호출부 [710][720]은 동일 모델을 1차(영문) → 2차(한국어 "
        "번역)로 재호출하는 에이전트 패턴을 형성하며, 별도 번역 모델 없이 GPU·메모리 자원을 절감한다. "
        "메타 헤더 부착부 [730]은 모델명·세션ID·관측 시각 등 추적 가능한 메타데이터를 산출물 선두에 "
        "삽입해 감사 가능성을 확보한다.",
    )

    add_h3(doc, "도 5 (동작 순서도)")
    add_para(
        doc,
        "본 발명의 방법은 페어링 결과 입력(S100) → 그래프 누적(S200~S400) → 커뮤니티 탐지(S500) → "
        "이중 검색(S600~S800) → 프롬프트 조립(S900~S1000) → LLM 에이전트(S1100~S1200) → 최종 산출(S1300)의 "
        "6단계로 진행된다. S300의 카운터 누적은 LLM 비호출 결정론 단계, S500은 그래프 알고리즘 단계, "
        "S1100/S1200은 동일 LLM의 다단계 호출 단계로서, 본 발명은 이 세 종류의 이질적 연산을 단일 "
        "파이프라인 내에서 결합하는 구성을 그 진보적 특징으로 한다.",
    )

    # ───────── 권장 청구항 골격 ─────────
    add_h1(doc, "[참고] 권장 청구항 골격")
    add_table(
        doc,
        ["청구항", "핵심"],
        [
            ["독립항 1 (시스템)", "A + B + C + D + E 결합 시스템 — 본 발명의 가장 넓은 권리범위"],
            ["독립항 2 (방법)", "S100~S1300 순차 방법 청구"],
            ["독립항 3 (기록매체)", "컴퓨터 판독 가능 기록매체 청구"],
            ["종속항 a", "0.01° 격자 양자화"],
            ["종속항 b", "LLM-free 결정론 인덱싱 ★ (Microsoft GraphRAG 대비 진보성)"],
            ["종속항 c", "Louvain + co_occurred_with 가중치"],
            ["종속항 d", "Local + Global 이중 검색"],
            ["종속항 e", "~500 토큰 예산"],
            ["종속항 f", "new/disappeared 한정 LLM 전달"],
            ["종속항 g", "DISAPPEARED 의미 제약 ★ (도메인 안전성)"],
            ["종속항 h", "표준 IMINT 8섹션 강제"],
            ["종속항 i", "동일 LLM 번역 재호출"],
            ["종속항 j", "메타 헤더 부착으로 감사 가능성"],
        ],
        widths=[4.0, 12.0],
    )

    # ───────── 변리사 검증 추천 ─────────
    add_h1(doc, "[참고] 변리사 검증 추천 작업 (출원 전)")
    for s in [
        "KIPRIS 키워드 검색: 'LLM 보고서', 'RAG 보고서', '지식그래프 보고서', '위성영상 변화탐지 보고서', "
        "'IMINT 자동 생성'",
        "Google Patents 검색: 'intelligence report generation LLM', 'graph rag report', "
        "'change detection report automation'",
        "Microsoft GraphRAG 관련 실제 출원 여부 확인 (있다면 청구항 7로 회피)",
        "Palantir 정보 분석 자동화 특허군 검토",
        "SAM/SAM2/SAM3 관련 Meta 출원 검토",
    ]:
        add_bullet(doc, s)

    out_path = "/home/user/multi-source-intelligent-system/data/AI에이전트_판독보고서_특허명세서.docx"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc.save(out_path)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    build()
