"""도 2 v4 통합본 (시각화 강화) — 고정형·이동형 이원 처리를 v2 스타일 도식화로.

레이아웃:
  [1] 두 시점 영상 입력 (탐지 객체 아이콘)
  [2] 클래스 판정 분기 (다이아몬드)
    좌 (고정형·노란톤)                우 (이동형·주황톤)
    ① 지도에 두 건물 위경도 근접 시각화   ① 배경 제거 (원본→마스크→객체만)
    ② 두 crop CLIP 비교 → matched/changed  ② 객체 → 인코더 → 벡터
    ③ 가상 탐지 합성 시각화               ③ N×M 유사도 히트맵
                                          ④ Gale-Shapley 매칭 선
  [3] 5상태 통합 분류
  [4] Pairing DB
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
STAGE_BG   = "#f8fafc"
STAGE_BD   = "#e2e8f0"
GREEN      = "#16a34a"
RED        = "#dc2626"
GREY       = "#94a3b8"
GRID       = "#cbd5e1"

fig, ax = plt.subplots(figsize=(15, 21))
ax.set_xlim(0, 15); ax.set_ylim(0, 23)
ax.axis("off")

# 제목
ax.text(7.5, 22.5, "도 2. 객체 페어링 변화 탐지 상세 (고정형·이동형 이원 처리)",
        ha="center", fontsize=16, fontweight="bold", color=TEXT)
ax.text(7.5, 22.1,
        "객체 클래스에 따라 두 파이프라인이 병렬 실행되어 5상태로 통합 분류",
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
# [1단계] 입력
# ═══════════════════════════════════════════════════════════════
stage_bg(20.6, 1.6, "1단계", "두 시점의 위성 영상 탐지 결과 입력")

# 과거 프레임
ax.add_patch(Rectangle((2.0, 20.7), 4.0, 1.3, facecolor="#e0f2fe",
                       edgecolor="#0369a1", lw=1.5))
ax.text(4.0, 21.85, "과거 프레임 (t₁)", ha="center", fontsize=10.5,
        fontweight="bold", color="#0c4a6e")
# 건물(고정형) + 전차(이동형) 섞임
ax.add_patch(Rectangle((2.5, 21.0), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))   # 건물
ax.add_patch(Rectangle((3.3, 21.05), 0.35, 0.25, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))   # 전차
ax.add_patch(Rectangle((4.3, 20.95), 0.4, 0.28, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))   # 전차
ax.add_patch(Rectangle((5.1, 21.1), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))   # 건물
ax.text(4.0, 20.85, "건물 2 + 전차 2 탐지", ha="center", fontsize=8, color=MUTED)

# 현재 프레임
ax.add_patch(Rectangle((9.0, 20.7), 4.0, 1.3, facecolor="#e0f2fe",
                       edgecolor="#0369a1", lw=1.5))
ax.text(11.0, 21.85, "현재 프레임 (t₂)", ha="center", fontsize=10.5,
        fontweight="bold", color="#0c4a6e")
ax.add_patch(Rectangle((9.5, 21.0), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((11.5, 21.1), 0.4, 0.28, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((12.1, 21.0), 0.35, 0.25, facecolor=MOV_MAIN,
                       edgecolor="white", lw=1))
ax.add_patch(Rectangle((10.3, 21.1), 0.55, 0.35, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.text(11.0, 20.85, "건물 2 + 전차 2 탐지", ha="center", fontsize=8, color=MUTED)

arrow_down(7.5, 20.6, 19.6)


# ═══════════════════════════════════════════════════════════════
# [2단계] 클래스 판정 분기
# ═══════════════════════════════════════════════════════════════
stage_bg(18.6, 1.0, "2단계", "각 객체의 클래스 판정 → 두 파이프라인으로 분기")

# 다이아몬드
diamond = Polygon([(7.5, 19.4), (9.2, 19.0), (7.5, 18.7), (5.8, 19.0)],
                   facecolor=COMMON, edgecolor="none")
ax.add_patch(diamond)
ax.text(7.5, 19.05, "고정형 클래스?",
        ha="center", va="center", fontsize=10, fontweight="bold", color="white")

# 좌 분기 (고정형)
arrow(5.8, 18.95, 3.7, 18.0, color=FIX_MAIN, lw=2.8, mut=12)
ax.text(4.5, 18.55, "예 (건물·시설)", fontsize=9, color=FIX_MAIN, fontweight="bold")

# 우 분기 (이동형)
arrow(9.2, 18.95, 11.3, 18.0, color=MOV_MAIN, lw=2.8, mut=12)
ax.text(10.5, 18.55, "아니오 (전차·차량)", fontsize=9, color=MOV_MAIN, fontweight="bold")


# ═══════════════════════════════════════════════════════════════
# 좌측 파이프라인 배경 (고정형)
# ═══════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((0.3, 3.5), 7.0, 14.4, facecolor=FIX_BG,
                       edgecolor=FIX_MAIN, lw=1.5, linestyle="dashed"))
ax.text(0.55, 17.65, "좌측 파이프라인 — 고정형 (건물·시설 · 이동 불가)",
        fontsize=10.5, color="#78350f", fontweight="bold")


# ─── 고정형 ① 위경도 초근접 그리디 결합 ───
ax.text(3.65, 17.25, "① 위경도 초근접 (~11m) 그리디 결합",
        ha="center", fontsize=10, fontweight="bold", color=FIX_MAIN)

# 지도 mini
mx, my = 1.2, 14.9
ms = 5.0
ax.add_patch(Rectangle((mx, my), ms, 2.15, facecolor="#f0fdf4",
                       edgecolor=GRID, lw=1))
# 격자
for i in range(6):
    ax.plot([mx + i*ms/5, mx + i*ms/5], [my, my+2.15], color=GRID, lw=0.5)
for j in range(4):
    ax.plot([mx, mx+ms], [my + j*2.15/3, my + j*2.15/3], color=GRID, lw=0.5)

# 과거 건물 위치
ax.add_patch(Rectangle((mx+1.6, my+1.15), 0.35, 0.28, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.text(mx+1.75, my+1.5, "과거 건물 A", fontsize=7.5, color="#78350f",
        ha="center", fontweight="bold")

# 현재 건물 위치 (거의 같은 위치)
ax.add_patch(Rectangle((mx+1.7, my+1.05), 0.35, 0.28, facecolor=FIX_MAIN,
                       edgecolor="#7c2d12", lw=1))
ax.text(mx+1.9, my+0.75, "현재 건물 A'", fontsize=7.5, color="#78350f",
        ha="center", fontweight="bold")

# 근접 원 (~11m)
ax.add_patch(Circle((mx+1.85, my+1.2), 0.45, facecolor="none",
                    edgecolor=FIX_MAIN, lw=1.5, linestyle="--"))
ax.annotate("≤ 11m", xy=(mx+2.3, my+1.65), fontsize=8,
            color=FIX_MAIN, fontweight="bold")

# 다른 건물 (멀리 있음 → 다른 쌍)
ax.add_patch(Rectangle((mx+4.0, my+0.5), 0.35, 0.28, facecolor=FIX_MAIN,
                       edgecolor="white", lw=1))
ax.text(mx+4.2, my+0.2, "건물 B (다른 위치)", fontsize=7.5, color=MUTED,
        ha="center")

ax.text(mx+ms/2, my-0.2,
        "가까운 두 건물 쌍이 자동 결합됨",
        ha="center", fontsize=8, color="#78350f", style="italic")

arrow_down(3.65, 14.65, 13.5, color=FIX_MAIN, lw=2, label="결합됨")


# ─── 고정형 ② CLIP 외형 비교 → matched/changed ───
ax.text(3.65, 13.25, "② 결합된 쌍의 CLIP 외형 비교",
        ha="center", fontsize=10, fontweight="bold", color=FIX_MAIN)

# 두 crop 나란히
def crop_img(x, y, obj_color, label):
    ax.add_patch(Rectangle((x, y), 0.8, 0.8, facecolor="#f5f5f4",
                           edgecolor=GREY, lw=1))
    ax.add_patch(Rectangle((x+0.15, y+0.15), 0.5, 0.5,
                           facecolor=obj_color, edgecolor="white", lw=0.5))
    ax.text(x+0.4, y-0.2, label, ha="center", fontsize=8, color=MUTED)

crop_img(0.8, 12.15, FIX_MAIN, "과거 crop")
crop_img(2.35, 12.15, FIX_MAIN, "현재 crop")

# CLIP 박스
ax.add_patch(FancyBboxPatch((3.65, 12.15), 1.5, 0.8, boxstyle="round,pad=0.04",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=1.2))
ax.text(4.4, 12.55, "CLIP\n유사도",
        ha="center", va="center", fontsize=9, fontweight="bold", color="#78350f")

# 화살표
arrow(1.6, 12.55, 2.35, 12.55, color="#334155", lw=1.2, mut=6)
arrow(3.15, 12.55, 3.65, 12.55, color="#334155", lw=1.2, mut=6)
arrow(5.15, 12.55, 5.7, 12.55, color="#334155", lw=1.2, mut=6)

# 결과 (matched or changed)
ax.add_patch(FancyBboxPatch((5.7, 12.6), 1.35, 0.35, boxstyle="round,pad=0.02",
                             facecolor=MOV_MAIN, edgecolor="none"))
ax.text(6.35, 12.77, "matched", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.add_patch(FancyBboxPatch((5.7, 12.15), 1.35, 0.35, boxstyle="round,pad=0.02",
                             facecolor=FIX_MAIN, edgecolor="none"))
ax.text(6.35, 12.32, "changed", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.text(7.1, 12.77, "≥ 0.5", ha="left", fontsize=7, color=MUTED)
ax.text(7.1, 12.32, "< 0.5", ha="left", fontsize=7, color=MUTED)

arrow_down(3.65, 11.85, 10.9, color=FIX_MAIN, lw=2)


# ─── 고정형 ③ 한쪽 유실 시 가상 탐지 합성 ───
ax.text(3.65, 10.55, "③ 한쪽 유실 시 가상 탐지 합성",
        ha="center", fontsize=10, fontweight="bold", color=FIX_MAIN)

# 과거만 있고 현재는 없음
crop_img(0.8, 9.3, FIX_MAIN, "과거 crop")
# 현재는 미탐지 (X 표시)
ax.add_patch(Rectangle((2.35, 9.3), 0.8, 0.8, facecolor="#f5f5f4",
                       edgecolor=GREY, lw=1))
ax.text(2.75, 9.7, "?", ha="center", va="center", fontsize=30, color=RED,
        fontweight="bold")
ax.text(2.75, 9.1, "현재 (탐지 실패)", ha="center", fontsize=7, color=RED)

# 화살표 아래로 (강제 crop)
ax.annotate("", xy=(2.75, 8.3), xytext=(2.75, 8.9),
            arrowprops=dict(arrowstyle="->", color=NEUTRAL, lw=1.2))
ax.text(3.4, 8.6, "같은 위경도\n강제 crop", fontsize=7.5, color=MUTED, style="italic")

# 강제 crop 결과
crop_img(2.35, 7.5, FIX_MAIN, "강제 crop")

# 다시 CLIP 비교
ax.add_patch(FancyBboxPatch((3.65, 7.5), 1.5, 0.8, boxstyle="round,pad=0.04",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=1.2))
ax.text(4.4, 7.9, "CLIP\n재검증", ha="center", va="center", fontsize=9,
        fontweight="bold", color="#78350f")
arrow(3.15, 7.9, 3.65, 7.9, color="#334155", lw=1.2, mut=6)
arrow(5.15, 7.9, 5.7, 7.9, color="#334155", lw=1.2, mut=6)

# 결과
ax.add_patch(FancyBboxPatch((5.7, 7.85), 1.35, 0.4,
                             boxstyle="round,pad=0.03",
                             facecolor=GREEN, edgecolor="none"))
ax.text(6.35, 8.05, "synthetic\n주입", ha="center", va="center", fontsize=8,
        fontweight="bold", color="white")
ax.add_patch(FancyBboxPatch((5.7, 7.35), 1.35, 0.4,
                             boxstyle="round,pad=0.03",
                             facecolor=RED, edgecolor="none"))
ax.text(6.35, 7.55, "disappeared\n확정", ha="center", va="center", fontsize=8,
        fontweight="bold", color="white")

ax.text(3.65, 7.15,
        "→ 매칭 복원으로 SAM3 탐지 누락 자동 보정",
        ha="center", fontsize=8.5, color="#78350f", style="italic")

# 화살표: 좌측 파이프라인 → 5상태 통합
arrow_down(3.65, 6.9, 3.2, color=FIX_MAIN, lw=2.5, label=None)


# ═══════════════════════════════════════════════════════════════
# 우측 파이프라인 배경 (이동형)
# ═══════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((7.8, 3.5), 6.9, 14.4, facecolor=MOV_BG,
                       edgecolor=MOV_MAIN, lw=1.5, linestyle="dashed"))
ax.text(8.05, 17.65, "우측 파이프라인 — 이동형 (전차·차량 · 이동 가능)",
        fontsize=10.5, color="#7c2d12", fontweight="bold")


# ─── 이동형 ① 배경 제거 ───
ax.text(11.25, 17.25, "① 마스크로 배경 제거",
        ha="center", fontsize=10, fontweight="bold", color=MOV_MAIN)

def bg_remove_mini(x0, y0, obj_color):
    # 원본
    ax.add_patch(Rectangle((x0, y0), 0.7, 0.7, facecolor="#f5f5f4",
                           edgecolor=GREY, lw=0.8))
    ax.add_patch(Circle((x0+0.2, y0+0.5), 0.09, facecolor="#84cc16", alpha=0.6))
    ax.add_patch(Rectangle((x0+0.28, y0+0.15), 0.2, 0.16,
                           facecolor=obj_color, edgecolor="white", lw=0.4))
    ax.text(x0+0.35, y0-0.15, "원본", ha="center", fontsize=7, color=MUTED)

    arrow(x0+0.75, y0+0.35, x0+1.0, y0+0.35, color="#334155", lw=1, mut=4)

    # 마스크
    ax.add_patch(Rectangle((x0+1.0, y0), 0.7, 0.7, facecolor="black",
                           edgecolor=GREY, lw=0.8))
    ax.add_patch(Rectangle((x0+1.28, y0+0.15), 0.2, 0.16,
                           facecolor="white", edgecolor="none"))
    ax.text(x0+1.35, y0-0.15, "마스크", ha="center", fontsize=7, color=MUTED)

    arrow(x0+1.75, y0+0.35, x0+2.0, y0+0.35, color="#334155", lw=1, mut=4)

    # 배경 제거
    ax.add_patch(Rectangle((x0+2.0, y0), 0.7, 0.7, facecolor="black",
                           edgecolor=GREY, lw=0.8))
    ax.add_patch(Rectangle((x0+2.28, y0+0.15), 0.2, 0.16,
                           facecolor=obj_color, edgecolor="white", lw=0.4))
    ax.text(x0+2.35, y0-0.15, "객체만", ha="center", fontsize=7, color=MUTED,
            fontweight="bold")

bg_remove_mini(8.4, 16.15, MOV_MAIN)
bg_remove_mini(11.5, 16.15, MOV_MAIN)

ax.text(11.25, 15.75,
        "배경(도로·나무 등) 제거 → 순수 객체 외형만 남김",
        ha="center", fontsize=8.5, color="#7c2d12", style="italic")

arrow_down(11.25, 15.55, 14.85, color=MOV_MAIN, lw=2)


# ─── 이동형 ② 인코더 → 벡터 ───
ax.text(11.25, 14.55, "② 비전 AI 인코더 → 특징 벡터",
        ha="center", fontsize=10, fontweight="bold", color=MOV_MAIN)

np.random.seed(42)
_BASE = [[0.14, 0.08, 0.13, 0.05, 0.10],
         [0.06, 0.15, 0.09, 0.13, 0.07],
         [0.11, 0.06, 0.08, 0.14, 0.13]]

def vec_row(x_obj, y_row, x_vec, label, noise_seed):
    ax.text(x_obj + 0.2, y_row + 0.65, label, ha="center", fontsize=8,
            fontweight="bold", color=TEXT)
    # 3개 객체 아이콘 (세로 배치)
    for i in range(3):
        yy = y_row + 0.1 + i * 0.17
        ax.add_patch(Rectangle((x_obj + 0.05, yy), 0.24, 0.14,
                                facecolor=MOV_MAIN, edgecolor="white", lw=0.5))
    # 화살표 → 인코더
    arrow(x_obj + 0.35, y_row + 0.35, x_obj + 0.85, y_row + 0.35,
          color="#334155", lw=1, mut=5)
    # 인코더 (좁게)
    if noise_seed == 1:  # 상단 행에만 인코더 그리기 (두 행 공유)
        pass
    # 화살표 인코더 → 벡터
    # (인코더 자체는 중앙에 하나만 그림 - 아래서 처리)
    # 벡터 3개
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

# 두 행: 과거, 현재
vec_row(x_obj=8.2, y_row=13.75, x_vec=12.6, label="과거", noise_seed=1)
vec_row(x_obj=8.2, y_row=12.9,  x_vec=12.6, label="현재", noise_seed=2)

# 중앙 인코더 박스 (두 행 공유)
ax.add_patch(FancyBboxPatch((9.55, 12.85), 1.4, 1.6, boxstyle="round,pad=0.06",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=1.5))
ax.text(10.25, 14.0, "비전 AI\n인코더",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#78350f")
ax.text(10.25, 13.4, "(ViT/CLIP)",
        ha="center", va="center", fontsize=8, color="#78350f", style="italic")
# 화살표: 각 행에서 인코더로
arrow(8.6, 14.1, 9.55, 14.1, color="#334155", lw=1, mut=5)
arrow(8.6, 13.25, 9.55, 13.25, color="#334155", lw=1, mut=5)
arrow(10.95, 14.1, 12.6, 14.1, color="#334155", lw=1, mut=5)
arrow(10.95, 13.25, 12.6, 13.25, color="#334155", lw=1, mut=5)

ax.text(11.25, 12.65, "같은 객체 → 거의 같은 벡터",
        ha="center", fontsize=8, color="#7c2d12", style="italic")

arrow_down(11.25, 12.55, 11.85, color=MOV_MAIN, lw=2)


# ─── 이동형 ③ N×M 유사도 히트맵 ───
ax.text(11.25, 11.55, "③ N×M 코사인 유사도 행렬",
        ha="center", fontsize=10, fontweight="bold", color=MOV_MAIN)

mat_x, mat_y = 9.4, 9.7
cell_size = 0.55
# 헤더
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

ax.text(13.6, 10.75, "값이 높을수록\n외형이 닮음",
        ha="left", fontsize=8, color=MUTED, style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor=MUTED, lw=0.6))

arrow_down(11.25, 9.55, 8.5, color=MOV_MAIN, lw=2)


# ─── 이동형 ④ Gale-Shapley 매칭 ───
ax.text(11.25, 8.2, "④ Gale-Shapley 안정 매칭 (1:1)",
        ha="center", fontsize=10, fontweight="bold", color=MOV_MAIN)

# 좌 과거 3개 원
for i in range(3):
    cy = 6.7 + i*0.4
    ax.add_patch(Circle((8.7, cy), 0.16, facecolor=MOV_MAIN,
                        edgecolor="white", lw=1))
    ax.text(8.35, cy, f"과거{i+1}", ha="right", va="center", fontsize=8, color=TEXT)

# 우 현재 3개 원
for i in range(3):
    cy = 6.7 + i*0.4
    ax.add_patch(Circle((13.8, cy), 0.16, facecolor=MOV_MAIN,
                        edgecolor="white", lw=1))
    ax.text(14.15, cy, f"현재{i+1}", ha="left", va="center", fontsize=8, color=TEXT)

# 매칭 선 (직선)
ax.plot([8.86, 13.64], [6.7, 6.7], color=GREEN, lw=2.5, alpha=0.9)
ax.plot([8.86, 13.64], [7.1, 7.1], color=GREEN, lw=2.5, alpha=0.9)
ax.plot([8.86, 13.64], [7.5, 7.5], color=GREEN, lw=2.5, alpha=0.9)

ax.text(11.25, 7.9, "각자에게 최선의 짝을 안정적으로 배정",
        ha="center", fontsize=8.5, color=GREEN, fontweight="bold")

# 최종 결과 (matched)
ax.add_patch(FancyBboxPatch((9.0, 5.8), 4.5, 0.65,
                             boxstyle="round,pad=0.04",
                             facecolor=MOV_MAIN, edgecolor="none"))
ax.text(11.25, 6.12, "매칭 성공 → matched (위경도 거리 무관)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")
arrow_down(11.25, 6.6, 6.45, color=MOV_MAIN, lw=1.5)

# 화살표: 우측 파이프라인 → 5상태 통합
arrow_down(11.25, 5.8, 3.2, color=MOV_MAIN, lw=2.5)


# ═══════════════════════════════════════════════════════════════
# [3단계] 5상태 통합 분류
# ═══════════════════════════════════════════════════════════════
stage_bg(1.85, 1.4, "3단계", "좌·우 파이프라인 결과 통합 + FOV 검증 → 5상태 분류")

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

# 4단계: Pairing DB
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
