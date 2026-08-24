"""도 3 v2 — GraphRAG 상세 (이해하기 쉬운 시각화)

제목: 'GraphRAG 기반 시공간 이력 누적 및 압축 컨텍스트 생성'

각 단계를 실제 그림으로 도식화:
1) Pairing DB 입력 (실제 표 형태로)
2) 격자 양자화 (지도 위 격자 그리드로 시각화)
3) 노드/엣지 upsert (실제 노드 원, 엣지 선, 카운터 숫자로 표현)
4) Louvain 군집화 (컬러 그룹으로 노드 묶기)
5) Graph DB 저장 (원기둥)
6) Local + Global 검색 → 압축 컨텍스트 블록
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
import matplotlib.font_manager as fm
import numpy as np

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GRAPH   = "#7c3aed"
STAGE   = "#f8fafc"
STAGE_BD = "#e2e8f0"
TEXT    = "#111827"
MUTED   = "#64748b"
DB      = "#475569"
PAIR    = "#ea580c"
GREEN   = "#16a34a"
BLUE    = "#3b82f6"
YELLOW  = "#f59e0b"
RED     = "#dc2626"
GRID    = "#cbd5e1"
NODE_A  = "#a78bfa"   # asset 노드 (연보라)
NODE_L  = "#f59e0b"   # location 노드 (주황)

fig, ax = plt.subplots(figsize=(13, 16))
ax.set_xlim(0, 13); ax.set_ylim(0, 18)
ax.axis("off")

# 제목
ax.text(6.5, 17.4, "도 3. GraphRAG 기반 시공간 이력 누적 및 압축 컨텍스트 생성",
        ha="center", fontsize=15, fontweight="bold", color=TEXT)
ax.text(6.5, 17.0,
        "페어링 결과를 AI 호출 없이 지식 그래프에 누적하고, 관련 이력을 압축해 LLM에 전달",
        ha="center", fontsize=10, color=MUTED, style="italic")


def stage_bg(y_bot, h, num, title):
    ax.add_patch(Rectangle((0.3, y_bot), 12.4, h,
                           facecolor=STAGE, edgecolor=STAGE_BD, lw=1))
    ax.text(0.55, y_bot+h-0.28, f"[{num}] {title}",
            fontsize=10.5, color=MUTED, fontweight="bold")

def arrow_down(x, y1, y2, lw=2.5, color="#334155"):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2),
                                  arrowstyle="->,head_length=13,head_width=9",
                                  color=color, lw=lw))

def arrow_right(x1, y, x2, color="#334155", lw=2):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y),
                                  arrowstyle="->,head_length=9,head_width=7",
                                  color=color, lw=lw))


# ═══════════════════════════════════════════════════════════════
# [1단계] 페어링 결과 입력
# ═══════════════════════════════════════════════════════════════
stage_bg(15.5, 1.4, "1단계", "페어링 결과 입력 (Pairing DB로부터 배치 조회)")

# 표 형태로 페어링 레코드 3~4개 표시
tx, ty = 2.0, 15.7
ax.add_patch(Rectangle((tx, ty), 9.0, 0.9, facecolor="white", edgecolor=DB, lw=1))
# 헤더
headers = ["객체 클래스", "위경도", "상태", "신뢰도"]
xs = [0.6, 3.0, 5.5, 7.5]
for h, xx in zip(headers, xs):
    ax.text(tx + xx, ty + 0.68, h, fontsize=8.5, fontweight="bold", color=TEXT)
ax.plot([tx, tx+9.0], [ty+0.55, ty+0.55], color=DB, lw=0.5)

rows = [("tank",     "(37.5765, 126.9680)", "new",         "0.87"),
        ("APC",      "(37.5771, 126.9698)", "new",         "0.79"),
        ("building", "(37.5750, 126.9660)", "matched",     "0.91")]
for i, row in enumerate(rows):
    yy = ty + 0.35 - i * 0.15
    for val, xx in zip(row, xs):
        ax.text(tx + xx, yy, val, fontsize=7.5, color=TEXT, family="monospace")

arrow_down(6.5, 15.4, 14.5)

# ═══════════════════════════════════════════════════════════════
# [2단계] 격자 양자화
# ═══════════════════════════════════════════════════════════════
stage_bg(12.6, 1.9, "2단계", "위경도를 1km 격자로 반올림 → 결정론적 노드 키 생성")

# 좌: 지도에 격자 그리드
gx, gy = 1.0, 12.8
gs = 0.35  # cell size
grid_w, grid_h = 5, 4
for i in range(grid_w+1):
    ax.plot([gx + i*gs, gx + i*gs], [gy, gy + grid_h*gs], color=GRID, lw=0.7)
for j in range(grid_h+1):
    ax.plot([gx, gx + grid_w*gs], [gy + j*gs, gy + j*gs], color=GRID, lw=0.7)
# 배경 (지도 느낌)
ax.add_patch(Rectangle((gx, gy), grid_w*gs, grid_h*gs, facecolor="#f0fdf4", zorder=-1))

# 객체 위치 (약간 세밀한 좌표) → 특정 격자 셀 안에 표시
# 3개 객체를 같은 셀 근처에 배치
objects = [
    (gx + 2.15*gs, gy + 2.65*gs, PAIR),   # tank
    (gx + 2.35*gs, gy + 2.45*gs, PAIR),   # APC
    (gx + 2.55*gs, gy + 2.75*gs, YELLOW), # building
]
for cx, cy, col in objects:
    ax.add_patch(Circle((cx, cy), 0.08, facecolor=col, edgecolor="white", lw=1, zorder=3))

# 해당 격자 셀 강조
ax.add_patch(Rectangle((gx + 2*gs, gy + 2*gs), gs, gs,
                       facecolor="none", edgecolor=GRAPH, lw=2))
ax.text(gx + 2.5*gs, gy - 0.2, "같은 격자 셀 (약 1km²)",
        ha="center", fontsize=8, color=GRAPH, fontweight="bold")

ax.text(gx + grid_w*gs/2, gy + grid_h*gs + 0.15, "지도 + 1km 격자",
        ha="center", fontsize=9, color=TEXT, fontweight="bold")

# 우: 결정론 키 생성
kx, ky = 4.5, 13.0
ax.add_patch(FancyArrowPatch((gx + grid_w*gs + 0.1, gy + grid_h*gs/2),
                              (kx - 0.1, ky + 0.8),
                              arrowstyle="->,head_length=8,head_width=6",
                              color="#334155", lw=1.8))
ax.text((gx + grid_w*gs + kx)/2, gy + grid_h*gs/2 + 0.35,
        "격자화\n(소수점 2자리)", ha="center", fontsize=8,
        color=MUTED, style="italic")

# 키 예시 박스
key_box_w = 7.5
ax.add_patch(FancyBboxPatch((kx, ky), key_box_w, 1.3, boxstyle="round,pad=0.08",
                             facecolor="#f3e8ff", edgecolor=GRAPH, lw=1.2))
ax.text(kx + key_box_w/2, ky + 1.05, "결정론적 노드 키 (같은 격자 → 같은 키)",
        ha="center", fontsize=9.5, fontweight="bold", color=GRAPH)
ax.text(kx + 0.2, ky + 0.7,
        "loc:37.58,126.97", fontsize=8.5, family="monospace", color=TEXT)
ax.text(kx + 3.0, ky + 0.7,
        "← 위치 노드", fontsize=8.5, color=TEXT)
ax.text(kx + 0.2, ky + 0.45,
        "asset:tank:37.58,126.97", fontsize=8.5, family="monospace", color=TEXT)
ax.text(kx + 3.0, ky + 0.45,
        "← (전차 × 격자) 노드", fontsize=8.5, color=TEXT)
ax.text(kx + 0.2, ky + 0.2,
        "asset:APC:37.58,126.97", fontsize=8.5, family="monospace", color=TEXT)
ax.text(kx + 3.0, ky + 0.2,
        "← (APC × 격자) 노드", fontsize=8.5, color=TEXT)

arrow_down(6.5, 12.6, 11.4)

# ═══════════════════════════════════════════════════════════════
# [3단계] 노드·엣지 자동 누적 (AI 호출 없음)
# ═══════════════════════════════════════════════════════════════
stage_bg(9.2, 2.2, "3단계", "노드·엣지 자동 누적 upsert (AI 호출 없이 카운터 +1)")

# 좌: Before (기존 그래프 상태 - 카운터 낮음)
bx, by = 0.8, 9.5
ax.text(bx + 2.0, by + 1.7, "Before (기존)", ha="center", fontsize=9.5,
        fontweight="bold", color=TEXT)
# location 노드
ax.add_patch(Circle((bx + 2.0, by + 1.2), 0.38, facecolor=NODE_L, edgecolor="white", lw=1.5))
ax.text(bx + 2.0, by + 1.2, "위치", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white")
# asset 노드 2개
ax.add_patch(Circle((bx + 0.8, by + 0.35), 0.38, facecolor=NODE_A, edgecolor="white", lw=1.5))
ax.text(bx + 0.8, by + 0.35, "전차", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.text(bx + 0.8, by - 0.2, "count=2", ha="center", fontsize=7.5, color=MUTED)

ax.add_patch(Circle((bx + 3.2, by + 0.35), 0.38, facecolor=NODE_A, edgecolor="white", lw=1.5))
ax.text(bx + 3.2, by + 0.35, "APC", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.text(bx + 3.2, by - 0.2, "count=1", ha="center", fontsize=7.5, color=MUTED)

# 엣지
ax.plot([bx + 0.8, bx + 2.0], [by + 0.73, by + 0.82], color=MUTED, lw=1.2)
ax.plot([bx + 3.2, bx + 2.0], [by + 0.73, by + 0.82], color=MUTED, lw=1.2)

# 중앙 화살표 (upsert 처리)
mx = 5.0
ax.add_patch(FancyArrowPatch((mx, by + 0.8), (mx + 1.5, by + 0.8),
                              arrowstyle="->,head_length=12,head_width=9",
                              color=GRAPH, lw=2.5))
ax.text(mx + 0.75, by + 1.25, "upsert\n(AI 호출 없음,\n카운터만 +1)",
        ha="center", fontsize=8.5, color=GRAPH, fontweight="bold")

# 우: After (카운터 증가)
ax_x, ay = 7.5, 9.5
ax.text(ax_x + 2.0, ay + 1.7, "After (누적 갱신)", ha="center",
        fontsize=9.5, fontweight="bold", color=TEXT)
# location 노드
ax.add_patch(Circle((ax_x + 2.0, ay + 1.2), 0.38, facecolor=NODE_L, edgecolor="white", lw=1.5))
ax.text(ax_x + 2.0, ay + 1.2, "위치", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white")
# asset 노드
ax.add_patch(Circle((ax_x + 0.8, ay + 0.35), 0.38, facecolor=NODE_A, edgecolor="white", lw=1.5))
ax.text(ax_x + 0.8, ay + 0.35, "전차", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.text(ax_x + 0.8, ay - 0.2, "count=3", ha="center", fontsize=7.5,
        color=GREEN, fontweight="bold")

ax.add_patch(Circle((ax_x + 3.2, ay + 0.35), 0.38, facecolor=NODE_A, edgecolor="white", lw=1.5))
ax.text(ax_x + 3.2, ay + 0.35, "APC", ha="center", va="center", fontsize=9,
        fontweight="bold", color="white")
ax.text(ax_x + 3.2, ay - 0.2, "count=2", ha="center", fontsize=7.5,
        color=GREEN, fontweight="bold")

# 엣지 - 굵어짐 (공출현 강화)
ax.plot([ax_x + 0.8, ax_x + 2.0], [ay + 0.73, ay + 0.82], color=GRAPH, lw=2.0)
ax.plot([ax_x + 3.2, ax_x + 2.0], [ay + 0.73, ay + 0.82], color=GRAPH, lw=2.0)
# co_occurred_with 엣지 (asset 간)
ax.plot([ax_x + 1.18, ax_x + 2.82], [ay + 0.35, ay + 0.35],
        color=GRAPH, lw=2.5, linestyle="--")
ax.text(ax_x + 2.0, ay + 0.1, "공출현 관계", ha="center",
        fontsize=7.5, color=GRAPH, style="italic")

# 하단 설명
ax.text(6.5, 9.35,
        "노드 색상: 주황=위치, 보라=자산  ·  실선=관측 관계(자산-위치),  점선=공출현 관계(자산-자산)",
        ha="center", fontsize=8, color=MUTED, style="italic")

arrow_down(6.5, 9.2, 7.5)

# ═══════════════════════════════════════════════════════════════
# [4단계] Louvain 군집화 (자산 doctrine 패턴 자동 발견)
# ═══════════════════════════════════════════════════════════════
stage_bg(5.1, 2.4, "4단계", "Louvain 군집화 — 자주 함께 등장하는 자산들을 자동 그룹화")

# 여러 노드 배치 + 컬러 군집
lx, ly = 1.0, 5.4
# Cluster 1: tank + APC + artillery (연보라 배경)
c1_bg = patches.Ellipse((lx + 1.5, ly + 1.2), 2.7, 1.4,
                         facecolor=NODE_A, alpha=0.2, edgecolor=GRAPH, lw=1.5, linestyle="--")
ax.add_patch(c1_bg)
ax.text(lx + 1.5, ly + 2.15, "군집 1: 기갑 복합체", ha="center",
        fontsize=8.5, fontweight="bold", color=GRAPH)
# 노드
for (nx, ny, lbl) in [(lx+0.6, ly+1.4, "tank"), (lx+2.4, ly+1.4, "APC"), (lx+1.5, ly+0.7, "artillery")]:
    ax.add_patch(Circle((nx, ny), 0.25, facecolor=NODE_A, edgecolor="white", lw=1.5))
    ax.text(nx, ny-0.42, lbl, ha="center", fontsize=7.5, color=TEXT)
# 엣지
ax.plot([lx+0.85, lx+2.15], [ly+1.4, ly+1.4], color=GRAPH, lw=1.5)
ax.plot([lx+0.7, lx+1.4], [ly+1.2, ly+0.8], color=GRAPH, lw=1.5)
ax.plot([lx+2.3, lx+1.6], [ly+1.2, ly+0.8], color=GRAPH, lw=1.5)

# Cluster 2: radar + command post (파랑 배경)
lx2 = 5.0
c2_bg = patches.Ellipse((lx2 + 1.2, ly + 1.2), 2.2, 1.4,
                         facecolor=BLUE, alpha=0.2, edgecolor=BLUE, lw=1.5, linestyle="--")
ax.add_patch(c2_bg)
ax.text(lx2 + 1.2, ly + 2.15, "군집 2: C2 인프라", ha="center",
        fontsize=8.5, fontweight="bold", color=BLUE)
for (nx, ny, lbl) in [(lx2+0.5, ly+1.2, "radar"), (lx2+1.9, ly+1.2, "command\npost")]:
    ax.add_patch(Circle((nx, ny), 0.25, facecolor=BLUE, edgecolor="white", lw=1.5))
    ax.text(nx, ny-0.55, lbl, ha="center", fontsize=7.5, color=TEXT)
ax.plot([lx2+0.75, lx2+1.65], [ly+1.2, ly+1.2], color=BLUE, lw=1.5)

# Cluster 3: aircraft + runway (녹색)
lx3 = 8.6
c3_bg = patches.Ellipse((lx3 + 1.2, ly + 1.2), 2.4, 1.4,
                         facecolor=GREEN, alpha=0.2, edgecolor=GREEN, lw=1.5, linestyle="--")
ax.add_patch(c3_bg)
ax.text(lx3 + 1.2, ly + 2.15, "군집 3: 항공 작전 거점", ha="center",
        fontsize=8.5, fontweight="bold", color=GREEN)
for (nx, ny, lbl) in [(lx3+0.5, ly+1.2, "aircraft"), (lx3+1.9, ly+1.2, "runway")]:
    ax.add_patch(Circle((nx, ny), 0.25, facecolor=GREEN, edgecolor="white", lw=1.5))
    ax.text(nx, ny-0.55, lbl, ha="center", fontsize=7.5, color=TEXT)
ax.plot([lx3+0.75, lx3+1.65], [ly+1.2, ly+1.2], color=GREEN, lw=1.5)

# 하단 설명
ax.text(6.5, 5.25,
        "→ 자주 함께 등장한 자산들을 그래프 알고리즘이 자동으로 그룹화 (AI 호출 없음)",
        ha="center", fontsize=8.5, color=MUTED, style="italic")

arrow_down(6.5, 5.1, 4.2)

# ═══════════════════════════════════════════════════════════════
# [5단계] Graph DB 저장
# ═══════════════════════════════════════════════════════════════
stage_bg(3.2, 1.0, "5단계", "지식 그래프 영속 저장")

# Graph DB (원기둥)
ax.add_patch(Rectangle((4.5, 3.4), 4.0, 0.55, facecolor=GRAPH, edgecolor="#1e293b", lw=1))
ax.add_patch(Ellipse((6.5, 3.95), 4.0, 0.16, facecolor=GRAPH, edgecolor="#1e293b", lw=1))
ax.add_patch(Ellipse((6.5, 3.4), 4.0, 0.16, facecolor=GRAPH, edgecolor="#1e293b", lw=1))
ax.text(6.5, 3.65, "Graph DB (노드 + 엣지 + 군집)", ha="center",
        fontsize=10.5, fontweight="bold", color="white")

arrow_down(6.5, 3.2, 2.4)

# ═══════════════════════════════════════════════════════════════
# [6단계] Local + Global 검색 → 압축 컨텍스트
# ═══════════════════════════════════════════════════════════════
stage_bg(0.5, 1.9, "6단계", "보고서 생성 시 관련 이력 검색 → 압축 컨텍스트")

# Local Search
ax.add_patch(FancyBboxPatch((0.7, 1.1), 3.4, 1.1, boxstyle="round,pad=0.05",
                             facecolor="#f3e8ff", edgecolor=GRAPH, lw=1.2))
ax.text(2.4, 1.85, "Local Search", ha="center", fontsize=10.5,
        fontweight="bold", color=GRAPH)
ax.text(2.4, 1.55, "대상 지역 반경 R 내", ha="center", fontsize=8.5, color=TEXT)
ax.text(2.4, 1.35, "자산 이력·카운터 조회", ha="center", fontsize=8.5, color=TEXT)

# Global Search
ax.add_patch(FancyBboxPatch((4.4, 1.1), 3.4, 1.1, boxstyle="round,pad=0.05",
                             facecolor="#f3e8ff", edgecolor=GRAPH, lw=1.2))
ax.text(6.1, 1.85, "Global Search", ha="center", fontsize=10.5,
        fontweight="bold", color=GRAPH)
ax.text(6.1, 1.55, "관련 군집 요약", ha="center", fontsize=8.5, color=TEXT)
ax.text(6.1, 1.35, "(doctrine 패턴)", ha="center", fontsize=8.5, color=TEXT)

arrow_right(4.1, 1.65, 4.4, color=GRAPH, lw=2)

# 압축 컨텍스트 블록
ax.add_patch(FancyBboxPatch((8.4, 1.05), 4.2, 1.2, boxstyle="round,pad=0.05",
                             facecolor="#dbeafe", edgecolor=BLUE, lw=1.5))
ax.text(10.5, 1.95, "~500 토큰 압축 컨텍스트", ha="center", fontsize=10.5,
        fontweight="bold", color=BLUE)
ax.text(10.5, 1.6, "\"기갑 복합체 8회 반복 관측,", ha="center",
        fontsize=8, color=TEXT, style="italic")
ax.text(10.5, 1.4, " 지난 30일간 3회 재배치...\"", ha="center",
        fontsize=8, color=TEXT, style="italic")
ax.text(10.5, 1.15, "→ LLM 프롬프트에 prepend", ha="center",
        fontsize=8, color=BLUE)

arrow_right(7.85, 1.65, 8.4, color=BLUE, lw=2)

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/fig3_graphrag_v2.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
