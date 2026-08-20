"""전체 시스템 아키텍처 도면 — SAM3 → Sensor DB → Pairing → Pairing DB
→ GraphRAG (Graph DB 참조 포함) → EXAONE → 판독보고서 → Report DB."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import matplotlib.font_manager as fm

# 한글 폰트
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try:
        fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 색상
IMG_BLUE      = "#93c5fd"
SAM_TEAL      = "#0891b2"
PAIR_ORANGE   = "#ea580c"
GRAPH_PURPLE  = "#7c3aed"
LLM_RED       = "#dc2626"
REPORT_GREEN  = "#16a34a"
DB_GRAY       = "#64748b"
ARROW_DARK    = "#1e293b"
ARROW_DASH    = "#94a3b8"
TEXT_DARK     = "#111827"
BG_CREAM      = "#fef9c3"

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")

# 제목
ax.text(10, 11.5,
        "GraphRAG 기반 위성영상 시계열 변화 탐지 AI Agent 판독보고서 자동 생성 시스템 — 전체 아키텍처",
        ha="center", fontsize=15, fontweight="bold", color=TEXT_DARK)

# ── 헬퍼 ──
def box(x, y, w, h, label, sub=None, fc=SAM_TEAL, tc="white",
        fontsize=11, subsize=8.5):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           facecolor=fc, edgecolor="none")
    ax.add_patch(patch)
    if sub:
        ax.text(x + w/2, y + h/2 + 0.15, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=tc)
        ax.text(x + w/2, y + h/2 - 0.25, sub, ha="center", va="center",
                fontsize=subsize, color=tc, style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=tc)

def db_cylinder(x, y, w, h, label, sub=None, fc=DB_GRAY):
    # 원기둥 모양(DB 표기)
    e_top = Ellipse((x + w/2, y + h), w, h*0.3, facecolor=fc,
                    edgecolor="#334155", lw=1.2, zorder=2)
    rect  = patches.Rectangle((x, y), w, h, facecolor=fc,
                              edgecolor="#334155", lw=1.2, zorder=1)
    e_bot = Ellipse((x + w/2, y), w, h*0.3, facecolor=fc,
                    edgecolor="#334155", lw=1.2, zorder=3)
    ax.add_patch(rect); ax.add_patch(e_top); ax.add_patch(e_bot)
    if sub:
        ax.text(x + w/2, y + h/2 + 0.15, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white")
        ax.text(x + w/2, y + h/2 - 0.25, sub, ha="center", va="center",
                fontsize=8, color="white", style="italic")
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white")

def arrow(x1, y1, x2, y2, color=ARROW_DARK, lw=2.5, style="->",
          mut=18, dashed=False, label=None, lx=None, ly=None):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=f"{style},head_length={mut*0.55},head_width={mut*0.4}",
                        color=color, lw=lw,
                        linestyle="dashed" if dashed else "solid")
    ax.add_patch(a)
    if label:
        ax.text(lx if lx is not None else (x1+x2)/2,
                ly if ly is not None else (y1+y2)/2 + 0.15,
                label, ha="center", fontsize=8.5, color=color,
                fontweight="bold" if not dashed else "normal")

# ═══════════════════════════════════════════════════
# 1) 입력 이미지 2개 (좌측)
# ═══════════════════════════════════════════════════
box(0.5, 8.5, 2.5, 1.2, "이미지 1", "(과거 시점 t₁)", fc=IMG_BLUE, tc=TEXT_DARK)
box(0.5, 6.5, 2.5, 1.2, "이미지 2", "(현재 시점 t₂)", fc=IMG_BLUE, tc=TEXT_DARK)

# ═══════════════════════════════════════════════════
# 2) SAM3
# ═══════════════════════════════════════════════════
box(4, 7.5, 2.5, 1.3, "SAM3",
    "Zero-shot 객체 탐지\n(mask + bbox + class)", fc=SAM_TEAL,
    fontsize=13, subsize=8)

# 이미지 → SAM3 화살표
arrow(3.0, 9.1, 4.0, 8.5, mut=15)
arrow(3.0, 7.1, 4.0, 8.1, mut=15)

# ═══════════════════════════════════════════════════
# 3) Sensor DB
# ═══════════════════════════════════════════════════
db_cylinder(7.5, 9.0, 1.8, 1.5, "Sensor DB",
            "탐지 결과 저장", fc=DB_GRAY)

# SAM3 → Sensor DB (INSERT)
arrow(6.5, 8.5, 7.5, 9.5, color=SAM_TEAL, mut=15, label="INSERT",
      lx=7.0, ly=9.15)

# ═══════════════════════════════════════════════════
# 4) Pairing Module
# ═══════════════════════════════════════════════════
box(10, 7.5, 2.8, 1.5,
    "Pairing Module",
    "CLIP 임베딩 +\nGale-Shapley 매칭",
    fc=PAIR_ORANGE, fontsize=12, subsize=8)

# SAM3 → Pairing (현재 탐지 직접 전달)
arrow(6.5, 8.0, 10.0, 8.1, color=SAM_TEAL, mut=15,
      label="현재 탐지", lx=8.3, ly=8.3)

# Sensor DB → Pairing (과거 탐지 조회)
arrow(8.4, 9.0, 10.5, 8.9, color=DB_GRAY, mut=15,
      dashed=True, label="과거 탐지 조회",
      lx=9.5, ly=9.05)

# ═══════════════════════════════════════════════════
# 5) Pairing DB
# ═══════════════════════════════════════════════════
db_cylinder(13.5, 9.0, 1.8, 1.5, "Pairing DB",
            "5상태 페어링", fc=DB_GRAY)

# Pairing → Pairing DB
arrow(12.8, 8.5, 13.5, 9.5, color=PAIR_ORANGE, mut=15,
      label="INSERT", lx=13.15, ly=9.15)

# 5상태 표시 (Pairing DB 아래)
ax.text(14.4, 8.65,
        "matched / changed /\nnew / disappeared /\nFOV 공백",
        ha="center", fontsize=7.5, color=TEXT_DARK, style="italic",
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="#f1f5f9", edgecolor="#cbd5e1"))

# ═══════════════════════════════════════════════════
# 6) GraphRAG (2단계: 인덱싱 + 검색)
# ═══════════════════════════════════════════════════
# GraphRAG 컨테이너 박스
grbox = FancyBboxPatch((6.5, 3.5), 9.5, 3.0, boxstyle="round,pad=0.1",
                       facecolor="#f5f3ff", edgecolor=GRAPH_PURPLE,
                       lw=2, linestyle="dashed")
ax.add_patch(grbox)
ax.text(11.25, 6.3, "GraphRAG", ha="center", fontsize=12,
        fontweight="bold", color=GRAPH_PURPLE)

# GraphRAG Indexer
box(7, 4.8, 2.5, 1.2,
    "Graph Indexer",
    "격자 양자화 + upsert\n(LLM 호출 없음)",
    fc=GRAPH_PURPLE, fontsize=10.5, subsize=7.5)

# Graph DB (중앙)
db_cylinder(10.5, 4.3, 1.8, 1.7, "Graph DB",
            "지식 그래프\n(누적)", fc="#5b21b6")

# Retriever (Local+Global)
box(13.3, 4.8, 2.5, 1.2,
    "Retriever",
    "Local + Global\n(~500 토큰 압축)",
    fc=GRAPH_PURPLE, fontsize=10.5, subsize=7.5)

# Pairing DB → Graph Indexer (그래프 갱신 흐름)
arrow(14.4, 9.0, 12.5, 6.5, color=PAIR_ORANGE, mut=15,
      label="페어링 배치")
arrow(12.5, 6.5, 8.5, 6.0, color=PAIR_ORANGE, mut=15)

# Graph Indexer → Graph DB (INSERT/UPSERT)
arrow(9.5, 5.4, 10.5, 5.15, color=GRAPH_PURPLE, mut=15,
      label="UPSERT", lx=10.0, ly=5.5)

# Graph DB → Retriever (READ)
arrow(12.3, 5.15, 13.3, 5.4, color=GRAPH_PURPLE, mut=15,
      label="READ", lx=12.8, ly=5.5)

# GraphRAG 설명 텍스트 (아래쪽)
ax.text(11.25, 3.85,
        "① 페어링 결과를 격자 인덱스 노드 키로 그래프에 누적 저장  "
        "  ② 보고서 생성 시 관련 지역·군집 요약을 검색해 압축 컨텍스트 생성",
        ha="center", fontsize=8.5, color=TEXT_DARK, style="italic")

# ═══════════════════════════════════════════════════
# 7) EXAONE (LLM)
# ═══════════════════════════════════════════════════
box(6.5, 1.3, 3.0, 1.5,
    "EXAONE (LLM)",
    "① 영문 보고서 생성\n② 동일 인스턴스 한국어 번역",
    fc=LLM_RED, fontsize=12, subsize=8)

# Retriever → LLM (컨텍스트 주입)
arrow(14.5, 4.8, 9.5, 2.5, color=GRAPH_PURPLE, mut=15,
      label="Historical Context\n(~500 토큰) prepend",
      lx=13.5, ly=3.15)

# Pairing DB → LLM (변화 객체 전달, new + disappeared만)
arrow(14.4, 9.0, 8.0, 2.8, color=PAIR_ORANGE, mut=15,
      dashed=True,
      label="변화 객체\n(new + disappeared)",
      lx=11.5, ly=6.9)

# ═══════════════════════════════════════════════════
# 8) 판독보고서
# ═══════════════════════════════════════════════════
box(10.5, 1.3, 2.5, 1.5,
    "판독보고서",
    "정형 8섹션 자연어 보고서",
    fc=REPORT_GREEN, fontsize=11.5, subsize=8)

arrow(9.5, 2.05, 10.5, 2.05, color=LLM_RED, mut=15)

# ═══════════════════════════════════════════════════
# 9) Report DB
# ═══════════════════════════════════════════════════
db_cylinder(14.5, 1.1, 1.8, 1.7, "Report DB",
            "판독보고서 저장", fc=DB_GRAY)

arrow(13.0, 2.05, 14.5, 2.05, color=REPORT_GREEN, mut=15,
      label="INSERT", lx=13.75, ly=2.3)

# ═══════════════════════════════════════════════════
# 범례
# ═══════════════════════════════════════════════════
legend_x, legend_y = 0.5, 0.5
ax.add_patch(patches.Rectangle((legend_x, legend_y), 5.5, 3.5,
                                facecolor="#f8fafc",
                                edgecolor="#cbd5e1", lw=1))
ax.text(legend_x + 0.2, legend_y + 3.1, "범례", fontsize=10.5,
        fontweight="bold", color=TEXT_DARK)

# 화살표 범례
def leg_arrow(y, color, dashed, text):
    a = FancyArrowPatch((legend_x + 0.3, y), (legend_x + 1.3, y),
                        arrowstyle="->,head_length=6,head_width=4",
                        color=color, lw=2,
                        linestyle="dashed" if dashed else "solid")
    ax.add_patch(a)
    ax.text(legend_x + 1.5, y, text, fontsize=8.5, va="center",
            color=TEXT_DARK)

leg_arrow(legend_y + 2.5, SAM_TEAL, False,   "SAM3 탐지 결과 흐름")
leg_arrow(legend_y + 2.1, PAIR_ORANGE, False, "페어링 결과 흐름")
leg_arrow(legend_y + 1.7, PAIR_ORANGE, True,  "변화 객체 (new + disappeared)")
leg_arrow(legend_y + 1.3, GRAPH_PURPLE, False, "GraphRAG 인덱싱/검색")
leg_arrow(legend_y + 0.9, DB_GRAY, True,      "DB 조회 (읽기)")
leg_arrow(legend_y + 0.5, LLM_RED, False,     "LLM 출력")

# 데이터 흐름 요약 (하단 우측)
summary_x, summary_y = 17.0, 0.5
ax.text(summary_x, summary_y + 3.5,
        "데이터 흐름 순서", fontsize=10, fontweight="bold",
        color=TEXT_DARK)
flow_text = (
    "① 이미지 2개 → SAM3 탐지\n"
    "② 탐지 결과 → Sensor DB\n"
    "③ Pairing Module\n"
    "   (현재 탐지 + 과거 조회)\n"
    "④ 5상태 분류 → Pairing DB\n"
    "⑤ GraphRAG Indexer\n"
    "   → Graph DB 누적\n"
    "⑥ Retriever가 Graph DB\n"
    "   조회 → 압축 컨텍스트\n"
    "⑦ EXAONE + 컨텍스트\n"
    "   + 변화 객체 → 보고서\n"
    "⑧ 판독보고서 → Report DB"
)
ax.text(summary_x, summary_y + 0.2, flow_text, fontsize=7.5,
        color=TEXT_DARK, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#f8fafc",
                  edgecolor="#cbd5e1"))

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/full_system_architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
