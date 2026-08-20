"""전체 시스템 아키텍처 도면 v2 — 단순화·명확화.

레이아웃 원칙:
- 상단→하단 단일 방향 세로 흐름
- 각 단계는 [모듈] + [DB] 한 쌍 (좌우 배치)
- 모듈 ↔ DB 간 짧은 양방향 화살표 (write/read)
- 단계 간에는 굵은 세로 화살표 하나만
- 범례·요약박스 제거 (그림 자체가 설명이 되도록)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import matplotlib.font_manager as fm

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 색상
IMG        = "#93c5fd"
SAM3       = "#0891b2"
PAIR       = "#ea580c"
GRAPH      = "#7c3aed"
LLM        = "#dc2626"
REPORT     = "#16a34a"
DB         = "#475569"
ARROW      = "#1e293b"
BG_STAGE   = "#f8fafc"
TEXT       = "#111827"

fig, ax = plt.subplots(figsize=(12, 15))
ax.set_xlim(0, 12)
ax.set_ylim(0, 18)
ax.axis("off")

# 제목
ax.text(6, 17.3, "전체 시스템 아키텍처",
        ha="center", fontsize=17, fontweight="bold", color=TEXT)
ax.text(6, 16.85,
        "위성영상 시계열 변화 탐지 및 GraphRAG 기반 판독보고서 자동 생성",
        ha="center", fontsize=11, color="#475569", style="italic")

# ── 헬퍼 ──
def module(x, y, w, h, label, sub=None, fc=SAM3, fs=13, ss=9.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor="none")
    ax.add_patch(p)
    if sub:
        ax.text(x+w/2, y+h/2+0.22, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white")
        ax.text(x+w/2, y+h/2-0.32, sub, ha="center", va="center",
                fontsize=ss, color="white")
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color="white")

def db(x, y, w, h, label, sub=None, fc=DB):
    # 원기둥
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc,
                                    edgecolor="#1e293b", lw=1.2, zorder=1))
    ax.add_patch(Ellipse((x+w/2, y+h), w, h*0.28, facecolor=fc,
                          edgecolor="#1e293b", lw=1.2, zorder=2))
    ax.add_patch(Ellipse((x+w/2, y), w, h*0.28, facecolor=fc,
                          edgecolor="#1e293b", lw=1.2, zorder=3))
    if sub:
        ax.text(x+w/2, y+h/2+0.18, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(x+w/2, y+h/2-0.25, sub, ha="center", va="center",
                fontsize=8.5, color="white")
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

def flow_down(x, y1, y2, label=None, color=ARROW, lw=3):
    a = FancyArrowPatch((x, y1), (x, y2),
                        arrowstyle="->,head_length=14,head_width=10",
                        color=color, lw=lw)
    ax.add_patch(a)
    if label:
        ax.text(x+0.3, (y1+y2)/2, label, fontsize=9.5, va="center",
                color=color, fontweight="bold")

def db_link(x1, y1, x2, y2, write_lbl="저장", read_lbl=None, color=DB):
    """모듈 ↔ DB 양방향 화살표."""
    # 저장 화살표 (모듈 → DB)
    a1 = FancyArrowPatch((x1, y1+0.15), (x2, y2+0.15),
                         arrowstyle="->,head_length=8,head_width=6",
                         color=color, lw=2)
    ax.add_patch(a1)
    ax.text((x1+x2)/2, y1+0.45, write_lbl, ha="center", fontsize=8.5,
            color=color, fontweight="bold")
    # 조회 화살표 (DB → 모듈) — read_lbl이 있을 때만
    if read_lbl:
        a2 = FancyArrowPatch((x2, y2-0.15), (x1, y1-0.15),
                             arrowstyle="->,head_length=8,head_width=6",
                             color=color, lw=2, linestyle="dashed")
        ax.add_patch(a2)
        ax.text((x1+x2)/2, y1-0.5, read_lbl, ha="center", fontsize=8.5,
                color=color, style="italic")

# 각 단계 배경 (연한 회색으로 구획)
def stage_bg(y_bot, h, label):
    ax.add_patch(patches.Rectangle((0.3, y_bot), 11.4, h,
                                    facecolor=BG_STAGE,
                                    edgecolor="#e2e8f0", lw=1))
    ax.text(0.55, y_bot+h-0.3, label, fontsize=9, color="#64748b",
            fontweight="bold")

# ═══════════════════════════════════════════════════
# 1단계: 입력
# ═══════════════════════════════════════════════════
stage_bg(14.8, 1.6, "① 입력")
module(2.0, 15.1, 3.2, 1.1, "이미지 1", "(과거 시점 t₁)", fc=IMG, fs=12, ss=9)
# text color adjustment (IMG is light blue - use dark text)
# regenerate text
ax.text(3.6, 15.1+1.1/2+0.15, "이미지 1", ha="center", va="center",
        fontsize=12, fontweight="bold", color=TEXT)
ax.text(3.6, 15.1+1.1/2-0.25, "(과거 시점 t₁)", ha="center", va="center",
        fontsize=9, color=TEXT)

module(6.8, 15.1, 3.2, 1.1, "이미지 2", "(현재 시점 t₂)", fc=IMG, fs=12, ss=9)
ax.text(8.4, 15.1+1.1/2+0.15, "이미지 2", ha="center", va="center",
        fontsize=12, fontweight="bold", color=TEXT)
ax.text(8.4, 15.1+1.1/2-0.25, "(현재 시점 t₂)", ha="center", va="center",
        fontsize=9, color=TEXT)

# 세로 화살표 (입력 → 탐지)
flow_down(6, 14.9, 13.7)

# ═══════════════════════════════════════════════════
# 2단계: 객체 탐지
# ═══════════════════════════════════════════════════
stage_bg(12.3, 1.4, "② 객체 탐지")
module(2.0, 12.6, 4.0, 1.0, "SAM3", "Zero-shot 객체 탐지", fc=SAM3, fs=13, ss=9)
db(8.0, 12.5, 2.5, 1.2, "Sensor DB", "탐지 결과", fc=DB)
db_link(6.0, 13.1, 8.0, 13.1, write_lbl="저장", color=SAM3)

flow_down(6, 12.3, 11.0)

# ═══════════════════════════════════════════════════
# 3단계: 페어링 (Sensor DB에서 과거 조회)
# ═══════════════════════════════════════════════════
stage_bg(9.6, 1.4, "③ 시계열 페어링 (과거 탐지 조회 → 5상태 분류)")
module(2.0, 9.9, 4.0, 1.0, "Pairing Module",
       "CLIP + Gale-Shapley 매칭", fc=PAIR, fs=13, ss=9)
db(8.0, 9.8, 2.5, 1.2, "Pairing DB", "5상태 페어링", fc=DB)
db_link(6.0, 10.4, 8.0, 10.4, write_lbl="저장", color=PAIR)

# 과거 탐지 조회 표시 (Sensor DB → Pairing Module) — 별도 곡선 화살표
arr_past = FancyArrowPatch((9.25, 12.5), (4.0, 10.9),
                            connectionstyle="arc3,rad=0.3",
                            arrowstyle="->,head_length=10,head_width=7",
                            color=SAM3, lw=1.8, linestyle="dashed")
ax.add_patch(arr_past)
ax.text(7.5, 11.7, "과거 탐지 조회",
        ha="center", fontsize=9, color=SAM3, style="italic",
        bbox=dict(boxstyle="round,pad=0.2",
                  facecolor="white",
                  edgecolor=SAM3, lw=0.8))

flow_down(6, 9.6, 8.3)

# ═══════════════════════════════════════════════════
# 4단계: GraphRAG
# ═══════════════════════════════════════════════════
stage_bg(6.9, 1.4, "④ GraphRAG (지식 그래프 누적 + 컨텍스트 검색)")
module(2.0, 7.2, 4.0, 1.0, "GraphRAG",
       "Indexer (upsert) + Retriever (Local+Global)",
       fc=GRAPH, fs=13, ss=8)
db(8.0, 7.1, 2.5, 1.2, "Graph DB", "지식 그래프 누적", fc=GRAPH)
db_link(6.0, 7.7, 8.0, 7.7, write_lbl="누적 저장",
        read_lbl="컨텍스트 조회", color=GRAPH)

flow_down(6, 6.9, 5.6)

# ═══════════════════════════════════════════════════
# 5단계: EXAONE (LLM)
# ═══════════════════════════════════════════════════
stage_bg(4.2, 1.4, "⑤ LLM 판독보고서 생성 (컨텍스트 + 변화 객체 주입)")
module(2.0, 4.5, 4.0, 1.0, "EXAONE (LLM)",
       "영문 생성 + 한국어 번역", fc=LLM, fs=13, ss=9)

# 변화 객체 전달 (Pairing DB → LLM)
arr_ch = FancyArrowPatch((9.25, 9.8), (5.0, 5.5),
                          connectionstyle="arc3,rad=-0.3",
                          arrowstyle="->,head_length=10,head_width=7",
                          color=PAIR, lw=1.8, linestyle="dashed")
ax.add_patch(arr_ch)
ax.text(8.4, 7.5, "변화 객체\n(new + disappeared)",
        ha="center", fontsize=9, color=PAIR, style="italic",
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="white", edgecolor=PAIR, lw=0.8))

flow_down(6, 4.2, 2.9)

# ═══════════════════════════════════════════════════
# 6단계: 최종 출력
# ═══════════════════════════════════════════════════
stage_bg(1.5, 1.4, "⑥ 판독보고서 산출")
module(2.0, 1.8, 4.0, 1.0, "판독보고서",
       "정형 8섹션 자연어 보고서", fc=REPORT, fs=13, ss=9)
db(8.0, 1.7, 2.5, 1.2, "Report DB", "보고서 영속 저장", fc=DB)
db_link(6.0, 2.3, 8.0, 2.3, write_lbl="저장", color=REPORT)

# 최하단 요약 한 줄
ax.text(6, 0.7,
        "각 DB는 세션 식별자(session_id)로 묶여 HITL 재처리 시 자동 재계산됩니다.",
        ha="center", fontsize=9.5, color="#475569", style="italic")

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/full_system_architecture_v2.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
