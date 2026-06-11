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
    add_para(
        doc,
        "본 분야에서 실제로 등록·공개되어 있는 유형(category)을 기재한다. 실제 출원 시 인용 특허 번호는 "
        "KIPRIS·Google Patents·Espacenet에서 변리사가 검증·교체한다.",
    )

    add_h3(doc, "가. LLM 기반 보고서 자동 생성 일반")
    for s in [
        "자연어 문서 코퍼스 기반 RAG(Retrieval-Augmented Generation) 보고서 생성 특허군 "
        "(검색 키: 'LLM 보고서 자동 생성 RAG', 'retrieval augmented generation report')",
        "정형 데이터(의료기록·금융 거래 등)로부터 LLM이 보고서를 생성하는 특허군 "
        "(검색 키: 'structured data report generation LLM', '프롬프트 기반 보고서 생성')",
        "Microsoft, \"From Local to Global: A Graph RAG Approach to Query-Focused Summarization\", "
        "arXiv:2404.16130 (2024)",
    ]:
        add_bullet(doc, s)

    add_h3(doc, "나. 위성·드론 영상 객체 탐지 및 변화 탐지")
    for s in [
        "SAM/SAM2/SAM3 등 Meta의 Segment Anything Model 관련 출원 및 논문",
        "위성영상 시계열 변화탐지(change detection) 특허군 "
        "(검색 키: 'satellite imagery change detection patent', '위성영상 변화탐지')",
        "Project Maven(미 국방부 AWCFT) 관련 공개 자료",
    ]:
        add_bullet(doc, s)

    add_h3(doc, "다. 군사 IMINT 보고서·doctrine 분야")
    for s in [
        "정형 IMINT 보고서 형식(CLASSIFICATION/EXECUTIVE SUMMARY/SITUATION/…)을 자동 작성하는 시스템 특허군 "
        "(검색 키: 'intelligence report automation', 'IMINT report generation')",
        "군사 의사결정 보조 AI 시스템 특허군 (Palantir 등)",
    ]:
        add_bullet(doc, s)

    add_h3(doc, "라. AI 에이전트(Agent) 일반")
    for s in [
        "다중 단계 도구 사용(tool-use) 기반 AI 에이전트 특허군 "
        "(검색 키: 'AI agent tool use', 'multi-step LLM agent')",
        "동일 LLM을 반복 호출하여 생성→번역→검증을 수행하는 에이전트 패턴",
    ]:
        add_bullet(doc, s)

    add_h2(doc, "(5) 종래 기술의 문제점 / 한계")
    problems = [
        ("단발 프레임 한계", "1프레임의 탐지 결과만 LLM에 전달하여 시계열 패턴이 보고서에 반영되지 않음."),
        ("텍스트 RAG의 부적합성",
         "입력이 정형 튜플(클래스·좌표·신뢰도·시각)인데 자연어로 풀어 다시 임베딩하는 우회 구조 → "
         "정보 손실·임베딩 비용·비결정성."),
        ("집계·관계 질의 약점",
         "\"이 객체가 몇 번째 출현인가\", \"함께 자주 등장하는 객체는?\" 등을 LLM에 떠넘김 → 응답 일관성 저하."),
        ("GPS 노이즈 취약", "같은 객체 좌표 미세 차이를 동일 인스턴스로 흡수하는 결정론적 메커니즘 부재."),
        ("군사 의미론 부재",
         "'소실(disappeared)'을 '파괴(destroyed)'로 LLM이 오해석할 위험 → 군사적 오판 가능성."),
        ("토큰 효율 문제",
         "모든 탐지 결과(정지·이동 포함)를 LLM에 전달 → 변화 분석에 무관한 객체로 컨텍스트 낭비."),
        ("감사 가능성 부족",
         "LLM이 왜 특정 결론을 도출했는지 행 단위로 추적 불가 → 군사·법적 책임 환경에서 운용 곤란."),
    ]
    add_table(doc, ["#", "문제", "내용"],
              [[str(i + 1), t, d] for i, (t, d) in enumerate(problems)],
              widths=[1.0, 4.0, 11.0])

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
