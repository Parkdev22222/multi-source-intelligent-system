"""도 2 v4 통합본 v2 — 공통 전처리(마스크 배경 제거) 후 이원 처리.

레이아웃:
  [1] 두 시점 SAM3 탐지 입력 (마스크 + bbox)
  [2] 공통 전처리: 마스크 기반 배경 제거로 순수 객체 crop 준비
       (고정형·이동형 모두 동일한 로직 사용)
  [3] 클래스 판정 분기
    좌 (고정형·노란톤)                우 (이동형·주황톤)
    ① 위경도 초근접 그리디 결합       ① 준비된 crop → 배치 CLIP 임베딩
    ② 결합된 쌍의 CLIP 외형 비교       ② N×M 유사도 행렬
    ③ 한쪽 유실 시 가상 탐지 합성      ③ Gale-Shapley 매칭
  [4] 좌·우 결과 통합 → 5상태 분류
  [5] Pairing DB
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                 Circle, Ellipse, Polygon)
import matplotlib.font_manager as fm
import numpy as np

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 색상
FIX_MAIN   = "#f59e0b"
FIX_LIGHT  = "#fef3c7"
FIX_BG     = "#fffbeb"
MOV_MAIN   = "#ea580c"
MOV_LIGHT  = "#fed7aa"
MOV_BG     = "#fff7ed"
EMB_PURPLE = "#7c3aed"
NEUTRAL    = "#334155"
TEXT       = "#111827"
MUTED      = "#64748b"
DB         = "#475569"
COMMON     = "#3b82f6"
COMMON_BG  = "#dbeafe"
STAGE_BG   = "#f8fafc"
STAGE_BD   = "#e2e8f0"
GREEN      = "#16a34a"
RED        = "#dc2626"
GREY       = "#94a3b8"
GRID       = "#cbd5e1"

fig, ax = plt.subplots(figsize=(15, 23))
ax.set_xlim(0, 15); ax.set_ylim(0, 26)
ax.axis("off")

# 제목
ax.text(7.5, 25.4, "도 2. 객체 페어링 변화 탐지 상세",
        ha="center", fontsize=16, fontweight="bold", color=TEXT)
ax.text(7.5, 24.95,
        "공통 전처리(마스크 배경 제거) 후 클래스에 따라 이원 파이프라인 → 5상태 통합 분류",
        ha="center", fontsize=10.5, color=MUTED, style="italic")


def stage_bg(y_bot, h, num, title, color=STAGE_BG):
    ax.add_patch(Rectangle((0.3, y_bot), 14.4, h, facecolor=color,
                           edgecolor=STAGE_BD, lw=1))
    ax.text(0.55, y_bot+h-0.28, f"[{num}] {title}",
            fontsize=10.5, color=MUTED, fontweight="bold")

def arrow_down(x, y1, y2, color=NEUTRAL, lw=2.5, label=None):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2),
                                  arrowstyle="->,head_length=13,head_width=9",
                                  color=color, lw=lw))
    if label:
        ax.text(x+0.3, (y1+y2)/2, label, fontsize=9, va="center",
                color=color, style="italic")

def arrow(x1, y1, x2, y2, color=NEUTRAL, lw=2, mut=10):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle=f"->,head_length={mut*0.55},head_width={mut*0.42}",
                                  color=color, lw=lw))


# ═══════════════════════════════════════════════════════════════
# [1단계] 입력 (SAM3 탐지 결과)
# ═══════════════════════════════════════════════════════════════
stage_bg(23.2, 1.5, "1단계", "두 시점 위성·드론 영상의 SAM3 탐지 결과 (bbox + 마스크)")

# 과거 프레임
ax.add_patch(Rectangle((2.0, 23.3), 4.0, 1.2, facecolor="#e0f2fe",
                       edgecolor="#0369a1", lw=1.5))
ax.text(4.0, 24.35, "과거 프레임 (t₁)", ha="center", fontsize=10.5,
        fontweight="bold", color="#0c4a6e")
ax.add_patch(Rectangle((2.5, 23.55), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((3.3, 23.6), 0.35, 0.25, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((4.3, 23.5), 0.4, 0.28, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((5.1, 23.65), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.text(4.0, 23.4, "고정형(건물) + 이동형(전차) 혼합", ha="center",
        fontsize=8, color=MUTED)

# 현재 프레임
ax.add_patch(Rectangle((9.0, 23.3), 4.0, 1.2, facecolor="#e0f2fe",
                       edgecolor="#0369a1", lw=1.5))
ax.text(11.0, 24.35, "현재 프레임 (t₂)", ha="center", fontsize=10.5,
        fontweight="bold", color="#0c4a6e")
ax.add_patch(Rectangle((9.5, 23.55), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((11.5, 23.6), 0.4, 0.28, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((12.1, 23.5), 0.35, 0.25, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((10.3, 23.65), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.text(11.0, 23.4, "고정형(건물) + 이동형(전차) 혼합", ha="center",
        fontsize=8, color=MUTED)

arrow_down(7.5, 23.2, 22.7)


# ═══════════════════════════════════════════════════════════════
# [2단계] 공통 전처리 (마스크 배경 제거)
# ═══════════════════════════════════════════════════════════════
stage_bg(19.9, 2.8, "2단계 · 공통 전처리",
         "마스크 기반 배경 제거로 순수 객체 crop 준비 (고정형·이동형 모두 동일 로직)",
         color=COMMON_BG)

ax.text(7.5, 22.15,
        "SAM3 마스크로 배경(도로·초지·기타 픽셀)을 zeroing 후 bbox crop → 두 파이프라인 공용 입력",
        ha="center", fontsize=8.5, color="#1e3a8a", style="italic")

# 공통 전처리 시각화: 3단계 (원본 → 마스크 → 배경제거)
def bg_remove_row(x0, y0, obj_color, obj_label, is_fixed=True):
    """마스크 기반 배경 제거 3단계 시각화. is_fixed에 따라 아이콘 모양 달라짐."""
    # 원본 crop (배경 + 객체)
    ax.add_patch(Rectangle((x0, y0), 0.85, 0.7, facecolor="#f5f5f4",
                           edgecolor=GREY, lw=0.8))
    # 배경 요소들
    ax.add_patch(Circle((x0+0.2, y0+0.55), 0.09, facecolor="#84cc16", alpha=0.6))
    ax.add_patch(Circle((x0+0.7, y0+0.15), 0.06, facecolor="#78716c", alpha=0.6))
    # 객체
    if is_fixed:
        ax.add_patch(Rectangle((x0+0.28, y0+0.2), 0.35, 0.3,
                               facecolor=obj_color, edgecolor="white", lw=0.5))
    else:
        ax.add_patch(Rectangle((x0+0.32, y0+0.22), 0.25, 0.2,
                               facecolor=obj_color, edgecolor="white", lw=0.5))
    ax.text(x0+0.425, y0-0.15, "원본 crop", ha="center", fontsize=7, color=MUTED)

    arrow(x0+0.9, y0+0.35, x0+1.2, y0+0.35, color="#334155", lw=1, mut=5)

    # SAM3 마스크
    ax.add_patch(Rectangle((x0+1.2, y0), 0.85, 0.7, facecolor="black",
                           edgecolor=GREY, lw=0.8))
    if is_fixed:
        ax.add_patch(Rectangle((x0+1.48, y0+0.2), 0.35, 0.3,
                               facecolor="white", edgecolor="none"))
    else:
        ax.add_patch(Rectangle((x0+1.52, y0+0.22), 0.25, 0.2,
                               facecolor="white", edgecolor="none"))
    ax.text(x0+1.625, y0-0.15, "SAM3 마스크", ha="center", fontsize=7, color=MUTED)

    arrow(x0+2.1, y0+0.35, x0+2.4, y0+0.35, color="#334155", lw=1, mut=5)

    # 배경 zeroing 후 crop
    ax.add_patch(Rectangle((x0+2.4, y0), 0.85, 0.7, facecolor="black",
                           edgecolor=GREY, lw=0.8))
    if is_fixed:
        ax.add_patch(Rectangle((x0+2.68, y0+0.2), 0.35, 0.3,
                               facecolor=obj_color, edgecolor="white", lw=0.5))
    else:
        ax.add_patch(Rectangle((x0+2.72, y0+0.22), 0.25, 0.2,
                               facecolor=obj_color, edgecolor="white", lw=0.5))
    ax.text(x0+2.825, y0-0.15, "순수 객체 crop", ha="center", fontsize=7,
            color="#166534", fontweight="bold")

    # 라벨 (행 좌측)
    ax.text(x0-0.15, y0+0.35, obj_label, ha="right", va="center",
            fontsize=8.5, fontweight="bold",
            color="#78350f" if is_fixed else "#7c2d12")

# 상단 행: 고정형 (건물)
bg_remove_row(1.55, 20.85, FIX_MAIN, "고정형", is_fixed=True)
# 하단 행: 이동형 (전차)
bg_remove_row(1.55, 20.05, MOV_MAIN, "이동형", is_fixed=False)

# 우측: 두 타입 모두 같은 로직임을 강조하는 브라켓 + 문구
ax.plot([5.5, 5.9, 5.9, 5.5], [20.05, 20.05, 21.55, 21.55],
        color=COMMON, lw=1.5)
ax.plot([5.9, 6.15], [20.8, 20.8], color=COMMON, lw=1.5)
ax.text(6.25, 21.0, "동일한 로직", va="center",
        fontsize=8, color=COMMON, fontweight="bold")
ax.text(6.25, 20.75, "_mask_crop()", va="center",
        fontsize=7.5, color=COMMON, family="monospace")
ax.text(6.25, 20.55, "_clip_embedder", va="center",
        fontsize=7.5, color=COMMON, family="monospace")

# 최종 산출물 강조
ax.add_patch(FancyBboxPatch((8.5, 20.15), 5.6, 1.4,
                             boxstyle="round,pad=0.05",
                             facecolor="white", edgecolor=COMMON, lw=1.5))
ax.text(11.3, 21.35, "공용 산출물: 배경 제거된 객체 crop 집합",
        ha="center", fontsize=9.5, fontweight="bold", color=COMMON)
ax.text(11.3, 20.98,
        "· 두 시점 모두에 대해 생성됨",
        ha="center", fontsize=8, color=TEXT)
ax.text(11.3, 20.75,
        "· 고정형·이동형 각 파이프라인이 이 crop을 그대로 CLIP에 넣음",
        ha="center", fontsize=8, color=TEXT)
ax.text(11.3, 20.4,
        "→ 하류에서 배경 제거 재수행 불필요",
        ha="center", fontsize=8, color="#166534", style="italic",
        fontweight="bold")

arrow_down(7.5, 19.9, 19.4)


# ═══════════════════════════════════════════════════════════════
# [3단계] 클래스 판정 분기
# ═══════════════════════════════════════════════════════════════
stage_bg(18.4, 1.0, "3단계", "각 객체의 클래스 판정 → 두 파이프라인으로 분기")

diamond = Polygon([(7.5, 19.2), (9.2, 18.8), (7.5, 18.5), (5.8, 18.8)],
                   facecolor=COMMON, edgecolor="none")
ax.add_patch(diamond)
ax.text(7.5, 18.85, "고정형 클래스?",
        ha="center", va="center", fontsize=10, fontweight="bold", color="white")

arrow(5.8, 18.75, 3.7, 17.8, color=FIX_MAIN, lw=2.8, mut=12)
ax.text(4.5, 18.35, "예 (건물·시설)", fontsize=9, color=FIX_MAIN, fontweight="bold")

arrow(9.2, 18.75, 11.3, 17.8, color=MOV_MAIN, lw=2.8, mut=12)
ax.text(10.5, 18.35, "아니오 (전차·차량)", fontsize=9, color=MOV_MAIN, fontweight="bold")


# ═══════════════════════════════════════════════════════════════
# 좌측 파이프라인 배경 (고정형)
# ═══════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((0.3, 3.5), 7.0, 14.2, facecolor=FIX_BG,
                       edgecolor=FIX_MAIN, lw=1.5, linestyle="dashed"))
ax.text(0.55, 17.45, "좌측 파이프라인 — 고정형 (건물·시설 · 이동 불가)",
        fontsize=10.5, color="#78350f", fontweight="bold")


# ─── 고정형 ① 위경도 초근접 그리디 결합 ───
ax.text(3.65, 17.05, "① 위경도 초근접 (~11m) 그리디 결합",
        ha="center", fontsize=10, fontweight="bold", color=FIX_MAIN)

mx, my = 1.2, 14.7
ms = 5.0
ax.add_patch(Rectangle((mx, my), ms, 2.15, facecolor="#f0fdf4",
                       edgecolor=GRID, lw=1))
for i in range(6):
    ax.plot([mx + i*ms/5, mx + i*ms/5], [my, my+2.15], color=GRID, lw=0.5)
for j in range(4):
    ax.plot([mx, mx+ms], [my + j*2.15/3, my + j*2.15/3], color=GRID, lw=0.5)

ax.add_patch(Rectangle((mx+1.6, my+1.15), 0.35, 0.28, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.text(mx+1.75, my+1.5, "과거 건물 A", fontsize=7.5, color="#78350f",
        ha="center", fontweight="bold")

ax.add_patch(Rectangle((mx+1.7, my+1.05), 0.35, 0.28, facecolor=FIX_MAIN,
                       edgecolor="#7c2d12", lw=1))
ax.text(mx+1.9, my+0.75, "현재 건물 A'", fontsize=7.5, color="#78350f",
        ha="center", fontweight="bold")

ax.add_patch(Circle((mx+1.85, my+1.2), 0.45, facecolor="none",
                    edgecolor=FIX_MAIN, lw=1.5, linestyle="--"))
ax.annotate("≤ 11m", xy=(mx+2.3, my+1.65), fontsize=8,
            color=FIX_MAIN, fontweight="bold")

ax.add_patch(Rectangle((mx+4.0, my+0.5), 0.35, 0.28, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.text(mx+4.2, my+0.2, "건물 B (다른 위치)", fontsize=7.5, color=MUTED,
        ha="center")

ax.text(mx+ms/2, my-0.2,
        "가까운 두 건물 쌍이 자동 결합됨",
        ha="center", fontsize=8, color="#78350f", style="italic")

arrow_down(3.65, 14.45, 13.3, color=FIX_MAIN, lw=2, label="결합됨")


# ─── 고정형 ② 결합된 쌍의 CLIP 외형 비교 ───
ax.text(3.65, 13.05, "② 결합된 쌍의 CLIP 외형 비교",
        ha="center", fontsize=10, fontweight="bold", color=FIX_MAIN)
ax.text(3.65, 12.75,
        "(2단계에서 준비된 배경 제거 crop 그대로 사용)",
        ha="center", fontsize=7.8, color=COMMON, style="italic")

def crop_img(x, y, obj_color, label):
    # 배경 제거된 crop을 시각화 (검은 배경 + 객체)
    ax.add_patch(Rectangle((x, y), 0.75, 0.6, facecolor="black",
                           edgecolor=GREY, lw=1))
    ax.add_patch(Rectangle((x+0.2, y+0.15), 0.35, 0.3,
                           facecolor=obj_color, edgecolor="white", lw=0.5))
    ax.text(x-0.05, y+0.3, label, ha="right", va="center",
            fontsize=8, fontweight="bold", color=TEXT)

crop_img(1.15, 12.05, FIX_MAIN, "과거 crop")
crop_img(1.15, 11.25, FIX_MAIN, "현재 crop")

ax.add_patch(FancyBboxPatch((3.5, 11.45), 1.7, 1.0, boxstyle="round,pad=0.04",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=1.2))
ax.text(4.35, 12.12, "CLIP",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#78350f")
ax.text(4.35, 11.82, "유사도 계산",
        ha="center", va="center", fontsize=8.5, color="#78350f", style="italic")

arrow(1.9, 12.35, 3.5, 12.15, color="#334155", lw=1.4, mut=7)
arrow(1.9, 11.55, 3.5, 11.75, color="#334155", lw=1.4, mut=7)
arrow(5.2, 11.95, 5.7, 11.95, color="#334155", lw=1.4, mut=7)

ax.add_patch(FancyBboxPatch((5.7, 12.1), 1.35, 0.35, boxstyle="round,pad=0.02",
                             facecolor=MOV_MAIN, edgecolor="none"))
ax.text(6.35, 12.27, "matched", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.add_patch(FancyBboxPatch((5.7, 11.45), 1.35, 0.35, boxstyle="round,pad=0.02",
                             facecolor=FIX_MAIN, edgecolor="none"))
ax.text(6.35, 11.62, "changed", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.text(7.1, 12.27, "≥ 임계값", ha="left", fontsize=7, color=MUTED)
ax.text(7.1, 11.62, "< 임계값", ha="left", fontsize=7, color=MUTED)

arrow_down(3.65, 11.25, 10.35, color=FIX_MAIN, lw=2)


# ─── 고정형 ③ 한쪽 유실 시 가상 탐지 합성 ───
ax.text(3.65, 10.05, "③ 한쪽 유실 시 가상 탐지 합성",
        ha="center", fontsize=10, fontweight="bold", color=FIX_MAIN)

ax.add_patch(Rectangle((0.85, 9.25), 0.8, 0.6, facecolor="#f5f5f4",
                       edgecolor=GREY, lw=1))
ax.text(1.25, 9.55, "?", ha="center", va="center", fontsize=22, color=RED,
        fontweight="bold")
ax.text(1.25, 9.1, "현재 (탐지 실패)", ha="center", fontsize=7, color=RED)

ax.annotate("", xy=(1.25, 8.65), xytext=(1.25, 9.05),
            arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=1.2))
ax.text(2.0, 8.85, "같은 위경도\n강제 crop", fontsize=7, color=MUTED, style="italic")

crop_img(1.15, 7.85, FIX_MAIN, "과거 crop")
crop_img(1.15, 7.05, FIX_MAIN, "강제 crop")

ax.add_patch(FancyBboxPatch((3.5, 7.25), 1.7, 1.0, boxstyle="round,pad=0.04",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=1.2))
ax.text(4.35, 7.92, "CLIP",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#78350f")
ax.text(4.35, 7.62, "재검증",
        ha="center", va="center", fontsize=8.5, color="#78350f", style="italic")

arrow(1.9, 8.15, 3.5, 7.95, color="#334155", lw=1.4, mut=7)
arrow(1.9, 7.35, 3.5, 7.55, color="#334155", lw=1.4, mut=7)
arrow(5.2, 7.75, 5.7, 7.55, color="#334155", lw=1.2, mut=6)
arrow(5.2, 7.75, 5.7, 7.05, color="#334155", lw=1.2, mut=6)

ax.add_patch(FancyBboxPatch((5.7, 7.35), 1.35, 0.4,
                             boxstyle="round,pad=0.03",
                             facecolor=GREEN, edgecolor="none"))
ax.text(6.35, 7.55, "synthetic\n주입", ha="center", va="center", fontsize=8,
        fontweight="bold", color="white")
ax.add_patch(FancyBboxPatch((5.7, 6.85), 1.35, 0.4,
                             boxstyle="round,pad=0.03",
                             facecolor=RED, edgecolor="none"))
ax.text(6.35, 7.05, "disappeared\n확정", ha="center", va="center", fontsize=8,
        fontweight="bold", color="white")

ax.text(3.65, 6.65,
        "→ 매칭 복원으로 SAM3 탐지 누락 자동 보정",
        ha="center", fontsize=8.5, color="#78350f", style="italic")

arrow_down(3.65, 6.4, 3.2, color=FIX_MAIN, lw=2.5)


# ═══════════════════════════════════════════════════════════════
# 우측 파이프라인 배경 (이동형)
# ═══════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((7.8, 3.5), 6.9, 14.2, facecolor=MOV_BG,
                       edgecolor=MOV_MAIN, lw=1.5, linestyle="dashed"))
ax.text(8.05, 17.45, "우측 파이프라인 — 이동형 (전차·차량 · 이동 가능)",
        fontsize=10.5, color="#7c2d12", fontweight="bold")


# ─── 이동형 ① 배치 CLIP 임베딩 ───
ax.text(11.25, 17.05, "① 준비된 crop → 배치 CLIP 임베딩",
        ha="center", fontsize=10, fontweight="bold", color=MOV_MAIN)
ax.text(11.25, 16.75,
        "(2단계에서 준비된 배경 제거 crop 그대로 사용)",
        ha="center", fontsize=7.8, color=COMMON, style="italic")

np.random.seed(42)
_BASE = [[0.14, 0.08, 0.13, 0.05, 0.10],
         [0.06, 0.15, 0.09, 0.13, 0.07],
         [0.11, 0.06, 0.08, 0.14, 0.13]]

def vec_row(x_obj, y_row, x_vec, label, noise_seed):
    ax.text(x_obj + 0.2, y_row + 0.65, label, ha="center", fontsize=8,
            fontweight="bold", color=TEXT)
    # 배경 제거된 crop 아이콘 3개
    for i in range(3):
        yy = y_row + 0.1 + i * 0.17
        ax.add_patch(Rectangle((x_obj + 0.02, yy), 0.3, 0.14,
                                facecolor="black", edgecolor=GREY, lw=0.4))
        ax.add_patch(Rectangle((x_obj + 0.09, yy+0.03), 0.16, 0.08,
                                facecolor=MOV_MAIN, edgecolor="none"))
    arrow(x_obj + 0.35, y_row + 0.35, x_obj + 0.85, y_row + 0.35,
          color="#334155", lw=1, mut=5)
    ax.text(x_vec + 0.4, y_row + 0.65, "벡터", ha="center", fontsize=8,
            color=TEXT, fontweight="bold")
    rng = np.random.default_rng(noise_seed)
    for i in range(3):
        yy = y_row + 0.1 + i * 0.17
        base = _BASE[i]
        for j, bv in enumerate(base):
            h = max(0.03, bv + rng.uniform(-0.012, 0.012))
            ax.add_patch(Rectangle((x_vec + 0.05 + j*0.12, yy),
                                    0.10, h*0.7,
                                    facecolor=EMB_PURPLE, edgecolor="none"))

vec_row(x_obj=8.2, y_row=15.55, x_vec=12.6, label="과거", noise_seed=1)
vec_row(x_obj=8.2, y_row=14.7,  x_vec=12.6, label="현재", noise_seed=2)

# 중앙 인코더 박스 (두 행 공유)
ax.add_patch(FancyBboxPatch((9.55, 14.65), 1.4, 1.6, boxstyle="round,pad=0.06",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=1.5))
ax.text(10.25, 15.8, "비전 AI\n인코더",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#78350f")
ax.text(10.25, 15.2, "(ViT/CLIP)",
        ha="center", va="center", fontsize=8, color="#78350f", style="italic")
arrow(8.6, 15.9, 9.55, 15.9, color="#334155", lw=1, mut=5)
arrow(8.6, 15.05, 9.55, 15.05, color="#334155", lw=1, mut=5)
arrow(10.95, 15.9, 12.6, 15.9, color="#334155", lw=1, mut=5)
arrow(10.95, 15.05, 12.6, 15.05, color="#334155", lw=1, mut=5)

ax.text(11.25, 14.45, "같은 객체 → 거의 같은 벡터",
        ha="center", fontsize=8, color="#7c2d12", style="italic")

arrow_down(11.25, 14.35, 13.65, color=MOV_MAIN, lw=2)


# ─── 이동형 ② N×M 유사도 행렬 ───
ax.text(11.25, 13.35, "② N×M 코사인 유사도 행렬",
        ha="center", fontsize=10, fontweight="bold", color=MOV_MAIN)

mat_x, mat_y = 9.4, 11.5
cell_size = 0.55
for j, name in enumerate(["현재1", "현재2", "현재3"]):
    ax.text(mat_x + 0.5 + j*cell_size + cell_size/2,
            mat_y + 3*cell_size + 0.18, name,
            ha="center", fontsize=8, fontweight="bold", color=TEXT)
for i, name in enumerate(["과거1", "과거2", "과거3"]):
    ax.text(mat_x + 0.35, mat_y + (2-i)*cell_size + cell_size/2,
            name, ha="right", va="center", fontsize=8,
            fontweight="bold", color=TEXT)

vals = [[0.89, 0.42, 0.15],
        [0.35, 0.87, 0.22],
        [0.18, 0.28, 0.81]]
for i in range(3):
    for j in range(3):
        v = vals[i][j]
        c = plt.cm.YlOrRd(v)
        ax.add_patch(Rectangle((mat_x+0.5+j*cell_size, mat_y+(2-i)*cell_size),
                               cell_size, cell_size, facecolor=c,
                               edgecolor="white", lw=0.8))
        ax.text(mat_x+0.5+j*cell_size+cell_size/2,
                mat_y+(2-i)*cell_size+cell_size/2,
                f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="black" if v < 0.5 else "white", fontweight="bold")

ax.text(13.6, 12.55, "값이 높을수록\n외형이 닮음",
        ha="left", fontsize=8, color=MUTED, style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=MUTED, lw=0.6))

arrow_down(11.25, 11.35, 8.7, color=MOV_MAIN, lw=2)


# ─── 이동형 ③ Gale-Shapley 매칭 ───
ax.text(11.25, 8.4, "③ Gale-Shapley 안정 매칭 (1:1)",
        ha="center", fontsize=10, fontweight="bold", color=MOV_MAIN)

for i in range(3):
    cy = 6.9 + i*0.4
    ax.add_patch(Circle((8.7, cy), 0.16, facecolor=MOV_MAIN,
                        edgecolor="white", lw=1))
    ax.text(8.35, cy, f"과거{i+1}", ha="right", va="center", fontsize=8, color=TEXT)

for i in range(3):
    cy = 6.9 + i*0.4
    ax.add_patch(Circle((13.8, cy), 0.16, facecolor=MOV_MAIN,
                        edgecolor="white", lw=1))
    ax.text(14.15, cy, f"현재{i+1}", ha="left", va="center", fontsize=8, color=TEXT)

ax.plot([8.86, 13.64], [6.9, 6.9], color=GREEN, lw=2.5, alpha=0.9)
ax.plot([8.86, 13.64], [7.3, 7.3], color=GREEN, lw=2.5, alpha=0.9)
ax.plot([8.86, 13.64], [7.7, 7.7], color=GREEN, lw=2.5, alpha=0.9)

ax.text(11.25, 8.1, "각자에게 최선의 짝을 안정적으로 배정",
        ha="center", fontsize=8.5, color=GREEN, fontweight="bold")

ax.add_patch(FancyBboxPatch((9.0, 5.9), 4.5, 0.65,
                             boxstyle="round,pad=0.04",
                             facecolor=MOV_MAIN, edgecolor="none"))
ax.text(11.25, 6.22, "매칭 성공 → matched (위경도 거리 무관)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
arrow_down(11.25, 6.7, 6.55, color=MOV_MAIN, lw=1.5)

arrow_down(11.25, 5.9, 3.2, color=MOV_MAIN, lw=2.5)


# ═══════════════════════════════════════════════════════════════
# [4단계] 5상태 통합 분류
# ═══════════════════════════════════════════════════════════════
stage_bg(1.85, 1.4, "4단계", "좌·우 파이프라인 결과 통합 + FOV 검증 → 5상태 분류")

states = [
    ("matched", "동일 객체", MOV_MAIN),
    ("changed", "구조 변화", FIX_MAIN),
    ("new", "신규 출현", "#16a34a"),
    ("disappeared", "소실", "#dc2626"),
    ("FOV 공백", "촬영 범위 밖", "#94a3b8"),
]
xs = [0.55, 3.4, 6.25, 9.1, 11.95]
for (name, desc, c), xx in zip(states, xs):
    ax.add_patch(FancyBboxPatch((xx, 2.15), 2.5, 0.95,
                                 boxstyle="round,pad=0.05",
                                 facecolor=c, edgecolor="none"))
    ax.text(xx+1.25, 2.85, name, ha="center", fontsize=10.5,
            fontweight="bold", color="white")
    ax.text(xx+1.25, 2.45, desc, ha="center", fontsize=8.5, color="white")

arrow_down(7.5, 1.85, 1.55, color=NEUTRAL, lw=2.5)
ax.add_patch(Rectangle((5.5, 0.6), 4.0, 0.55, facecolor=DB,
                       edgecolor="#1e293b", lw=1))
ax.add_patch(Ellipse((7.5, 1.15), 4.0, 0.16, facecolor=DB,
                     edgecolor="#1e293b", lw=1))
ax.add_patch(Ellipse((7.5, 0.6), 4.0, 0.16, facecolor=DB,
                     edgecolor="#1e293b", lw=1))
ax.text(7.5, 0.85, "Pairing DB (저장)", ha="center", fontsize=10.5,
        fontweight="bold", color="white")

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/fig2_v4_unified_visual.png"
plt.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
