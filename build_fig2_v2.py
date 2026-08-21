"""도 2 v2 — Pairing Module 상세 (시각화 강화, 일반 표현).

이해하기 쉽도록:
- "mask_rle → zeroing" 같은 기술 용어 대신 실제 그림으로 보여줌
- 각 단계마다 축소 이미지 아이콘 사용
- 개념 설명을 화살표 아래에 넣어 자연스러운 흐름
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle
import matplotlib.font_manager as fm
import numpy as np

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PAIR    = "#ea580c"
STAGE   = "#f8fafc"
STAGE_BD = "#e2e8f0"
TEXT    = "#111827"
MUTED   = "#64748b"
SKY     = "#93c5fd"
GREEN   = "#16a34a"
RED     = "#dc2626"
YELLOW  = "#f59e0b"
GREY    = "#94a3b8"
DB      = "#475569"
EMB_PURPLE = "#7c3aed"

fig, ax = plt.subplots(figsize=(13, 15))
ax.set_xlim(0, 13); ax.set_ylim(0, 17)
ax.axis("off")

# 제목
ax.text(6.5, 16.4, "도 2. 객체 페어링 변화 탐지 상세",
        ha="center", fontsize=16, fontweight="bold", color=TEXT)
ax.text(6.5, 16.0,
        "두 시점 위성 영상의 객체를 짝지어 어떤 것이 어떻게 변했는지 판단",
        ha="center", fontsize=10.5, color=MUTED, style="italic")


def stage_bg(y_bot, h, num, title, color=STAGE):
    ax.add_patch(Rectangle((0.3, y_bot), 12.4, h,
                           facecolor=color, edgecolor=STAGE_BD, lw=1))
    ax.text(0.55, y_bot+h-0.28, f"[{num}] {title}",
            fontsize=10.5, color=MUTED, fontweight="bold")


def arrow_down(x, y1, y2, lw=2.5, label=None, color="#334155"):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2),
                                  arrowstyle="->,head_length=13,head_width=9",
                                  color=color, lw=lw))
    if label:
        ax.text(x + 0.35, (y1+y2)/2, label, fontsize=9.5,
                va="center", color=color, style="italic")


# ═══════════════════════════════════════════════════════════════
# [1단계] 입력: 두 시점 위성 영상
# ═══════════════════════════════════════════════════════════════
stage_bg(14.3, 1.6, "1단계", "두 시점의 위성 영상 입력")

# 과거 영상
ax.add_patch(Rectangle((1.5, 14.4), 3.5, 1.4, facecolor="#e0f2fe",
                        edgecolor="#0369a1", lw=1.5))
ax.text(3.25, 15.65, "과거 영상 (t₁)", ha="center", fontsize=10.5,
        fontweight="bold", color="#0c4a6e")
# 객체 아이콘 (사각형 3개 = 전차·전차·건물)
for cx, cy, cl in [(2.2, 14.8, PAIR), (3.0, 14.9, PAIR), (4.2, 14.7, YELLOW)]:
    ax.add_patch(Rectangle((cx, cy), 0.35, 0.25, facecolor=cl, edgecolor="white", lw=1))
ax.text(3.25, 14.5, "탐지된 객체 3개", ha="center", fontsize=8, color=MUTED)

# 현재 영상
ax.add_patch(Rectangle((8.0, 14.4), 3.5, 1.4, facecolor="#e0f2fe",
                        edgecolor="#0369a1", lw=1.5))
ax.text(9.75, 15.65, "현재 영상 (t₂)", ha="center", fontsize=10.5,
        fontweight="bold", color="#0c4a6e")
for cx, cy, cl in [(8.7, 14.85, PAIR), (10.0, 14.8, PAIR), (10.7, 14.7, YELLOW)]:
    ax.add_patch(Rectangle((cx, cy), 0.35, 0.25, facecolor=cl, edgecolor="white", lw=1))
ax.text(9.75, 14.5, "탐지된 객체 3개", ha="center", fontsize=8, color=MUTED)

arrow_down(3.25, 14.3, 13.4)
arrow_down(9.75, 14.3, 13.4)

# ═══════════════════════════════════════════════════════════════
# [2단계] 객체별 배경 제거 (마스킹)
# ═══════════════════════════════════════════════════════════════
stage_bg(11.6, 1.8, "2단계", "객체별 배경 제거 (같은 객체 판단에 방해되는 도로·나무 등 제거)")

# 3단계 시각화: 원본 → 마스크 → 결과
def bg_remove_vis(x0, y0, obj_color):
    # 원본 crop (배경+객체 혼합)
    ax.add_patch(Rectangle((x0, y0), 0.9, 0.9, facecolor="#f5f5f4", edgecolor=GREY, lw=1))
    ax.add_patch(Circle((x0+0.35, y0+0.35), 0.13, facecolor="#84cc16", alpha=0.6))  # 나무
    ax.add_patch(Rectangle((x0+0.35, y0+0.15), 0.25, 0.2, facecolor=obj_color, edgecolor="white", lw=0.5))
    ax.text(x0+0.45, y0-0.15, "원본 crop", ha="center", fontsize=7, color=MUTED)
    ax.text(x0+0.45, y0-0.32, "(배경 섞임)", ha="center", fontsize=6.5, color=MUTED, style="italic")

    # 화살표
    ax.add_patch(FancyArrowPatch((x0+1.0, y0+0.45), (x0+1.3, y0+0.45),
                                  arrowstyle="->,head_length=5,head_width=4",
                                  color="#334155", lw=1.2))

    # 마스크 (객체 영역만 흰색, 나머지 검정)
    ax.add_patch(Rectangle((x0+1.3, y0), 0.9, 0.9, facecolor="black", edgecolor=GREY, lw=1))
    ax.add_patch(Rectangle((x0+1.65, y0+0.15), 0.25, 0.2, facecolor="white", edgecolor="none"))
    ax.text(x0+1.75, y0-0.15, "마스크", ha="center", fontsize=7, color=MUTED)
    ax.text(x0+1.75, y0-0.32, "(객체=흰색)", ha="center", fontsize=6.5, color=MUTED, style="italic")

    ax.add_patch(FancyArrowPatch((x0+2.3, y0+0.45), (x0+2.6, y0+0.45),
                                  arrowstyle="->,head_length=5,head_width=4",
                                  color="#334155", lw=1.2))

    # 결과 (배경 제거된 객체만)
    ax.add_patch(Rectangle((x0+2.6, y0), 0.9, 0.9, facecolor="black", edgecolor=GREY, lw=1))
    ax.add_patch(Rectangle((x0+2.95, y0+0.15), 0.25, 0.2, facecolor=obj_color, edgecolor="white", lw=0.5))
    ax.text(x0+3.05, y0-0.15, "객체만 남김", ha="center", fontsize=7, color=MUTED, fontweight="bold")
    ax.text(x0+3.05, y0-0.32, "(배경 검정)", ha="center", fontsize=6.5, color=MUTED, style="italic")

bg_remove_vis(0.7, 12.15, PAIR)
bg_remove_vis(7.5, 12.15, PAIR)

# 중앙 설명
ax.text(6.5, 12.85, "각 객체의\nmask 정보를 이용",
        ha="center", fontsize=8.5, color=MUTED, style="italic")

arrow_down(3.25, 11.6, 10.7)
arrow_down(9.75, 11.6, 10.7)

# ═══════════════════════════════════════════════════════════════
# [3단계] 시각 특징 벡터로 변환 (임베딩)
# ═══════════════════════════════════════════════════════════════
stage_bg(8.9, 1.8, "3단계", "각 객체의 외형을 숫자 벡터로 변환 (ViT/CLIP 비전 인코더)")

# 좌: 과거 3개 객체 → 벡터
def emb_group(x0, y0, label):
    ax.text(x0+1.4, y0+1.4, label, ha="center", fontsize=10, fontweight="bold", color=TEXT)
    for i in range(3):
        # 객체 아이콘
        ax.add_patch(Rectangle((x0, y0+i*0.35), 0.25, 0.2, facecolor=PAIR, edgecolor="white", lw=0.5))
        # 화살표
        ax.add_patch(FancyArrowPatch((x0+0.3, y0+0.1+i*0.35), (x0+0.65, y0+0.1+i*0.35),
                                      arrowstyle="->,head_length=4,head_width=3",
                                      color="#334155", lw=1))
        # 벡터 (막대들)
        for j in range(6):
            h = np.random.uniform(0.05, 0.18)
            ax.add_patch(Rectangle((x0+0.7+j*0.13, y0+i*0.35), 0.11, h, facecolor=EMB_PURPLE, edgecolor="none"))
    ax.text(x0+1.4, y0-0.25, "특징 벡터", ha="center", fontsize=8, color=MUTED, style="italic")

np.random.seed(42)
emb_group(1.3, 9.2, "과거 객체 → 벡터")
emb_group(8.3, 9.2, "현재 객체 → 벡터")

# 중앙 설명 박스
ax.add_patch(FancyBboxPatch((5.0, 9.4), 3.0, 1.0, boxstyle="round,pad=0.05",
                            facecolor="#fef3c7", edgecolor="#f59e0b", lw=1))
ax.text(6.5, 10.0, "비전 AI 인코더", ha="center", fontsize=10,
        fontweight="bold", color="#78350f")
ax.text(6.5, 9.65, "(ViT / CLIP)", ha="center", fontsize=9,
        color="#78350f", style="italic")

arrow_down(6.5, 8.9, 7.85, label=None)

# ═══════════════════════════════════════════════════════════════
# [4단계] 모든 쌍의 유사도 계산 (N×M 행렬)
# ═══════════════════════════════════════════════════════════════
stage_bg(6.6, 1.35, "4단계", "모든 과거-현재 쌍의 유사도 계산 (외형이 얼마나 닮았는가)")

# 행렬 시각화 (3×3)
mat_x, mat_y = 4.5, 6.9
cell_size = 0.6

# 헤더
for j, name in enumerate(["현재1", "현재2", "현재3"]):
    ax.text(mat_x + 0.5 + j*cell_size + cell_size/2, mat_y + 3*cell_size + 0.2,
            name, ha="center", fontsize=8, fontweight="bold", color=TEXT)
for i, name in enumerate(["과거1", "과거2", "과거3"]):
    ax.text(mat_x + 0.35, mat_y + (2-i)*cell_size + cell_size/2,
            name, ha="right", va="center", fontsize=8, fontweight="bold", color=TEXT)

# 값 매트릭스
values = [[0.89, 0.42, 0.15],
          [0.35, 0.87, 0.22],
          [0.18, 0.28, 0.81]]
for i in range(3):
    for j in range(3):
        v = values[i][j]
        # 진한 색이 높은 값
        c = plt.cm.YlOrRd(v)
        ax.add_patch(Rectangle((mat_x + 0.5 + j*cell_size,
                                 mat_y + (2-i)*cell_size),
                                cell_size, cell_size, facecolor=c, edgecolor="white", lw=1))
        ax.text(mat_x + 0.5 + j*cell_size + cell_size/2,
                mat_y + (2-i)*cell_size + cell_size/2,
                f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="black" if v < 0.5 else "white",
                fontweight="bold")

# 설명
ax.text(9.5, 7.6, "값이 높을수록\n외형이 닮음",
        ha="left", fontsize=9, color=MUTED, style="italic",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=MUTED, lw=0.8))

arrow_down(6.5, 6.6, 5.5)

# ═══════════════════════════════════════════════════════════════
# [5단계] 안정 매칭 (Gale-Shapley)
# ═══════════════════════════════════════════════════════════════
stage_bg(4.2, 1.3, "5단계", "안정 매칭 알고리즘으로 최적의 1:1 짝짓기 (Gale-Shapley)")

# 좌: 과거 객체 3개
for i in range(3):
    cy = 4.4 + i*0.3
    ax.add_patch(Circle((2.5, cy), 0.15, facecolor=PAIR, edgecolor="white", lw=1))
    ax.text(2.2, cy, f"과거{i+1}", ha="right", va="center", fontsize=8.5, color=TEXT)

# 우: 현재 객체 3개
for i in range(3):
    cy = 4.4 + i*0.3
    ax.add_patch(Circle((10.5, cy), 0.15, facecolor=PAIR, edgecolor="white", lw=1))
    ax.text(10.8, cy, f"현재{i+1}", ha="left", va="center", fontsize=8.5, color=TEXT)

# 매칭 선 (유사도 행렬 기준: 과거i ↔ 현재i 로 straight matching)
ax.plot([2.65, 10.35], [4.4, 4.4], color=GREEN, lw=2.5, alpha=0.9)  # 과거1↔현재1 (0.89)
ax.plot([2.65, 10.35], [4.7, 4.7], color=GREEN, lw=2.5, alpha=0.9)  # 과거2↔현재2 (0.87)
ax.plot([2.65, 10.35], [5.0, 5.0], color=GREEN, lw=2.5, alpha=0.9)  # 과거3↔현재3 (0.81)

# 매칭 이름 표시
ax.text(6.5, 5.25, "각자에게 최선의 짝을 안정적으로 배정",
        ha="center", fontsize=9, color=GREEN, fontweight="bold")

arrow_down(6.5, 4.2, 3.0)

# ═══════════════════════════════════════════════════════════════
# [6단계] 5가지 상태로 분류 + Pairing DB 저장
# ═══════════════════════════════════════════════════════════════
stage_bg(0.8, 2.1, "6단계", "매칭 결과 + 촬영범위 검사로 5가지 상태 분류 → Pairing DB 저장")

# 5개 상태 박스
states = [
    ("matched", "동일 객체\n(변화 없음)", PAIR),
    ("changed", "구조 변화\n(고정 시설물)", YELLOW),
    ("new", "신규 출현\n(현재만)", GREEN),
    ("disappeared", "소실\n(과거만)", RED),
    ("촬영 공백", "촬영 범위 밖\n(비교 불가)", GREY),
]
xs = [0.7, 3.0, 5.3, 7.6, 9.9]
for (name, desc, c), xx in zip(states, xs):
    ax.add_patch(FancyBboxPatch((xx, 1.5), 2.0, 1.1, boxstyle="round,pad=0.05",
                                 facecolor=c, edgecolor="none"))
    ax.text(xx+1.0, 2.25, name, ha="center", fontsize=10,
            fontweight="bold", color="white")
    ax.text(xx+1.0, 1.85, desc, ha="center", fontsize=8, color="white")

# 하단: Pairing DB
ax.add_patch(Rectangle((4.0, 0.85), 5.0, 0.5, facecolor=DB,
                        edgecolor="#1e293b", lw=1))
ax.add_patch(Ellipse((6.5, 1.35), 5.0, 0.15, facecolor=DB,
                      edgecolor="#1e293b", lw=1))
ax.add_patch(Ellipse((6.5, 0.85), 5.0, 0.15, facecolor=DB,
                      edgecolor="#1e293b", lw=1))
ax.text(6.5, 1.1, "Pairing DB (저장)", ha="center", fontsize=10.5,
        fontweight="bold", color="white")

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/fig2_pairing_module_v2.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
