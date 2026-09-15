"""MSIS 프로젝트 전체 flow + 기술 스택 도면.

각 파이프라인 스테이지에 사용된 모델/라이브러리를 함께 표시.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
import matplotlib.font_manager as fm

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 색상
INPUT_C   = "#0284c7"
DETECT_C  = "#0891b2"
PAIR_C    = "#ea580c"
GRAPH_C   = "#7c3aed"
LLM_C     = "#dc2626"
FRONT_C   = "#16a34a"
DB_C      = "#475569"
TECH_BG   = "#fef3c7"
TECH_BD   = "#f59e0b"
STAGE_BG  = "#f8fafc"
STAGE_BD  = "#e2e8f0"
TEXT      = "#111827"
MUTED     = "#64748b"
ARROW     = "#334155"

fig, ax = plt.subplots(figsize=(15, 19))
ax.set_xlim(0, 15); ax.set_ylim(0, 22)
ax.axis("off")

# 제목
ax.text(7.5, 21.4, "MSIS 프로젝트 전체 Flow · 기술 스택",
        ha="center", fontsize=17, fontweight="bold", color=TEXT)
ax.text(7.5, 20.95,
        "위성·드론 시계열 영상 → 객체 탐지 → 페어링 변화 탐지 → GraphRAG 이력 누적 → LLM 판독보고서",
        ha="center", fontsize=10.5, color=MUTED, style="italic")


def stage_box(x, y, w, h, color, label, subtitle):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                 facecolor=color, edgecolor="none"))
    ax.text(x+w/2, y+h/2+0.15, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(x+w/2, y+h/2-0.25, subtitle, ha="center", va="center",
            fontsize=9, color="white", style="italic")

def tech_tag(x, y, w, h, items, title="사용 기술"):
    """기술 스택 태그 박스."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                 facecolor=TECH_BG, edgecolor=TECH_BD, lw=1.2))
    ax.text(x+w/2, y+h-0.2, title, ha="center", fontsize=8.5,
            fontweight="bold", color="#78350f")
    for i, item in enumerate(items):
        ax.text(x+0.15, y+h-0.5-i*0.24, f"· {item}",
                fontsize=7.8, color="#78350f")

def db_cyl(x, y, w, h, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=DB_C,
                           edgecolor="#1e293b", lw=1))
    ax.add_patch(Ellipse((x+w/2, y+h), w, h*0.35, facecolor=DB_C,
                          edgecolor="#1e293b", lw=1))
    ax.add_patch(Ellipse((x+w/2, y), w, h*0.35, facecolor=DB_C,
                          edgecolor="#1e293b", lw=1))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white")

def arrow_down(x, y1, y2, color=ARROW, lw=2.2):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2),
                                  arrowstyle="->,head_length=11,head_width=8",
                                  color=color, lw=lw))

def arrow_right(x1, y, x2, color=ARROW, lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y),
                                  arrowstyle="->,head_length=8,head_width=6",
                                  color=color, lw=lw))


# ═══════════════════════════════════════════════════════════════
# [1] 입력
# ═══════════════════════════════════════════════════════════════
stage_box(2.5, 19.3, 5.0, 1.2, INPUT_C, "두 시점 위성·드론 영상 입력",
          "metadata.json (촬영 시각·지리 bbox) 포함")

tech_tag(9.0, 19.15, 5.5, 1.5,
         ["metadata.json (lat_min/max, lon_min/max, capture_time)",
          "GeoTIFF / PNG + JSON 사이드카",
          "위성 (satellite) · 드론 (drone) 양대 소스 지원"])

arrow_down(5.0, 19.3, 18.4)


# ═══════════════════════════════════════════════════════════════
# [2] 객체 탐지 (SAM3)
# ═══════════════════════════════════════════════════════════════
stage_box(2.5, 17.1, 5.0, 1.3, DETECT_C, "객체 탐지 · 좌표 변환",
          "사용자 지정 클래스의 인스턴스 검출")

tech_tag(9.0, 16.9, 5.5, 1.6,
         ["SAM3 (Segment Anything Model 3) — zero-shot",
          "출력: bbox + 픽셀 마스크 + confidence",
          "pixel_to_geo(): 지리 bbox 선형 보간 → 위경도",
          "이미지 초해상 (Real-ESRGAN 계열)"])

# → Sensor DB
db_cyl(8.5, 15.4, 2.5, 0.55, "Sensor DB")
arrow_right(7.5, 16.9, 9.75, color=DETECT_C)
ax.text(8.6, 17.05, "저장", ha="center", fontsize=7.5, color=DETECT_C, style="italic")

arrow_down(5.0, 17.1, 15.9)


# ═══════════════════════════════════════════════════════════════
# [3] 공통 전처리 (마스크 배경 제거)
# ═══════════════════════════════════════════════════════════════
stage_box(2.5, 14.6, 5.0, 1.3, "#0d9488", "공통 전처리 · 배경 제거",
          "SAM3 마스크로 순수 객체 crop 생성")

tech_tag(9.0, 14.4, 5.5, 1.4,
         ["_mask_crop() — 배경 픽셀 zeroing",
          "CLIP (ViT-based) 임베딩 준비",
          "고정형·이동형 파이프라인이 공용 사용"])

arrow_down(5.0, 14.6, 13.4)


# ═══════════════════════════════════════════════════════════════
# [4] 이원 페어링
# ═══════════════════════════════════════════════════════════════
stage_box(2.5, 11.9, 5.0, 1.5, PAIR_C, "이원 페어링 변화 탐지",
          "고정형/이동형 분리 처리 → 다상태 분류")

tech_tag(9.0, 11.6, 5.5, 2.1,
         ["고정형: 위경도 근접 (~11m) + CLIP 비교",
          "  → matched / changed",
          "이동형: N×M 코사인 유사도 행렬 (GEMM)",
          "  + Gale-Shapley 안정 매칭 알고리즘",
          "상태: matched · changed · new · disappeared",
          "     · past_not_included · current_not_included"])

# → Pairing DB
db_cyl(8.5, 10.4, 2.5, 0.55, "Pairing DB")
arrow_right(7.5, 11.6, 9.75, color=PAIR_C)
ax.text(8.6, 11.75, "저장", ha="center", fontsize=7.5, color=PAIR_C, style="italic")

arrow_down(5.0, 11.9, 9.4)


# ═══════════════════════════════════════════════════════════════
# [5] GraphRAG 이력 누적
# ═══════════════════════════════════════════════════════════════
stage_box(2.5, 7.9, 5.0, 1.5, GRAPH_C, "GraphRAG 이력 누적",
          "결정론적 지식 그래프 · AI 호출 없음")

tech_tag(9.0, 7.4, 5.5, 2.3,
         ["위경도 소수점 2자리 반올림 (≈ 1 km 격자)",
          "결정론적 노드 키: loc:37.58,126.97 · asset:tank:...",
          "upsert 방식 (관측 횟수·엣지 가중치 +1)",
          "Louvain 커뮤니티 탐지 → 반복 관측 패턴",
          "Local + Global 검색 → ~500 토큰 압축 컨텍스트"])

# → Graph DB
db_cyl(8.5, 6.5, 2.5, 0.55, "Graph DB")
arrow_right(7.5, 7.7, 9.75, color=GRAPH_C)
ax.text(8.6, 7.85, "저장", ha="center", fontsize=7.5, color=GRAPH_C, style="italic")

arrow_down(5.0, 7.9, 5.9)


# ═══════════════════════════════════════════════════════════════
# [6] 판독보고서 자율 생성
# ═══════════════════════════════════════════════════════════════
stage_box(2.5, 4.4, 5.0, 1.5, LLM_C, "판독보고서 자율 생성",
          "LLM에 압축 컨텍스트 주입 → 영문 → 한국어")

tech_tag(9.0, 3.9, 5.5, 2.3,
         ["LLM: EXAONE (한·영 대형 언어 모델)",
          "서빙: vLLM (FastAPI와 동일 프로세스)",
          "정형 9 섹션 (분류등급·핵심요약·상황·변화분석·",
          "  촬영공백·위협평가·정보공백·권고조치·부록)",
          "영문 생성 → 동일 인스턴스로 한국어 번역",
          "  (좌표·수치·타임스탬프 원본 보존)"])

# → Report DB
db_cyl(8.5, 3.0, 2.5, 0.55, "Report DB")
arrow_right(7.5, 4.2, 9.75, color=LLM_C)
ax.text(8.6, 4.35, "저장", ha="center", fontsize=7.5, color=LLM_C, style="italic")

arrow_down(5.0, 4.4, 2.4)


# ═══════════════════════════════════════════════════════════════
# [7] 프론트엔드 전달 (사용자)
# ═══════════════════════════════════════════════════════════════
stage_box(2.5, 0.9, 5.0, 1.5, FRONT_C, "웹 UI · HITL 재처리",
          "지도 시각화 + 재처리 요청")

tech_tag(9.0, 0.5, 5.5, 1.9,
         ["Backend: FastAPI + Uvicorn (Python)",
          "Frontend: Vanilla JavaScript + HTML",
          "지도: Leaflet.js + OpenStreetMap 타일",
          "HITL: session_id 기반 특정 단계 재실행",
          "  (탐지 수정 → 페어링·그래프·보고서 재계산)"])


# ═══════════════════════════════════════════════════════════════
# 좌측 라벨 (파이프라인 단계 번호)
# ═══════════════════════════════════════════════════════════════
stages_meta = [
    (19.9, "①"),
    (17.75, "②"),
    (15.25, "③"),
    (12.65, "④"),
    (8.65, "⑤"),
    (5.15, "⑥"),
    (1.65, "⑦"),
]
for y, num in stages_meta:
    ax.add_patch(Circle((1.5, y), 0.32, facecolor="white",
                        edgecolor=ARROW, lw=1.5))
    ax.text(1.5, y, num, ha="center", va="center", fontsize=13,
            fontweight="bold", color=ARROW)

# 하단 범례: 4대 DB 요약
ax.text(11.75, 0.25, "※ 4개 DB(Sensor / Pairing / Graph / Report)는 SQLite 기반, "
        "session_id로 분리 저장",
        ha="center", fontsize=7.5, color=MUTED, style="italic")

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/tech_stack_flow.png"
plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
