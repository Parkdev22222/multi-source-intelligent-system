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
stage_bg(15.0, 1.9, "1단계",
         "Pairing DB로부터 최근 회차 비교의 결과 배치를 조회 (한 회차 = 두 시점 영상 1쌍의 변화 탐지 결과)")

# 부제: 3단계 타임라인과의 연결 명시
ax.text(6.5, 16.32,
        "예: 3회차 비교(2025.04.10 ↔ 08.20)에서 나온 페어링 레코드 배치",
        ha="center", fontsize=8.3, color=GRAPH, style="italic",
        fontweight="bold")

# 표
tx, ty = 0.9, 15.05
tw, th = 11.2, 1.10
ax.add_patch(Rectangle((tx, ty), tw, th, facecolor="white",
                       edgecolor=DB, lw=1))

# 헤더
headers = ["회차", "클래스", "상태", "과거 좌표", "현재 좌표", "신뢰도"]
xs = [0.15, 0.95, 2.20, 3.85, 6.85, 9.95]
for h, xx in zip(headers, xs):
    ax.text(tx + xx, ty + th - 0.18, h, fontsize=8.5,
            fontweight="bold", color=TEXT)
ax.plot([tx, tx+tw], [ty + th - 0.30, ty + th - 0.30], color=DB, lw=0.5)

# 실제 데이터 행 (matched / new / disappeared 각 1건)
rows = [
    ("3회차", "tank",     "matched",     "(37.5765,126.9680)", "(37.5765,126.9680)", "0.91"),
    ("3회차", "APC",      "new",         "—",                   "(37.5771,126.9698)", "0.79"),
    ("3회차", "building", "disappeared", "(37.5750,126.9660)", "—",                   "0.87"),
]
status_colors = {"matched": GREEN, "new": BLUE, "disappeared": RED}
for i, row in enumerate(rows):
    yy = ty + 0.60 - i * 0.20
    for j, (val, xx) in enumerate(zip(row, xs)):
        if j == 2:  # 상태 열 강조
            ax.text(tx + xx, yy, val, fontsize=7.5,
                    color=status_colors.get(val, TEXT),
                    fontweight="bold")
        elif j in (3, 4):  # 좌표 열은 monospace
            ax.text(tx + xx, yy, val, fontsize=7.2,
                    color=TEXT if val != "—" else MUTED,
                    family="monospace")
        else:
            ax.text(tx + xx, yy, val, fontsize=7.5, color=TEXT)

arrow_down(6.5, 15.0, 14.5)

# ═══════════════════════════════════════════════════════════════
# [2단계] 격자 양자화
# ═══════════════════════════════════════════════════════════════
stage_bg(12.6, 1.9, "2단계", "위경도를 소수점 2자리로 반올림 (≈ 1 km 격자 단위) → 결정론적 노드 키 생성")

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

# 객체 위치 (약간 세밀한 좌표) → 같은 격자 셀 안에 표시
# 2개 자산(tank, APC)이 같은 셀에 떨어짐 → 같은 위치 노드 키로 양자화됨
objects = [
    (gx + 2.25*gs, gy + 2.60*gs, PAIR, "전차"),
    (gx + 2.60*gs, gy + 2.35*gs, PAIR, "APC"),
]
for cx, cy, col, lbl in objects:
    ax.add_patch(Circle((cx, cy), 0.10, facecolor=col, edgecolor="white", lw=1.2, zorder=3))
    ax.text(cx + 0.15, cy + 0.02, lbl, fontsize=7, color=TEXT, zorder=4)

# 해당 격자 셀 강조
ax.add_patch(Rectangle((gx + 2*gs, gy + 2*gs), gs, gs,
                       facecolor="none", edgecolor=GRAPH, lw=2))
ax.text(gx + 2.5*gs, gy - 0.2, "같은 격자 셀 (약 1 km × 1 km)",
        ha="center", fontsize=8, color=GRAPH, fontweight="bold")

ax.text(gx + grid_w*gs/2, gy + grid_h*gs + 0.15, "지도 + 약 1km 격자",
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
stage_bg(8.8, 2.6, "3단계",
         "새 영상 쌍으로 비교를 반복할수록 같은 (자산, 격자) 조합의 관측 횟수가 +1씩 쌓임 (AI 호출 없이 upsert만)")

# 시간축 라벨
ax.text(0.55, 11.0, "시간 →", fontsize=8.5, color=MUTED,
        style="italic", fontweight="bold")

def draw_graph_snapshot(cx, cy, tank_count, apc_count, coocc_weight,
                        title, subtitle, highlight=False):
    """(자산 × 격자) 그래프 스냅샷을 그린다.
    tank_count / apc_count : 관측 횟수
    coocc_weight : 공출현 엣지 가중치 (0이면 엣지 생략)
    """
    # 스냅샷 제목
    tcolor = GRAPH if highlight else TEXT
    ax.text(cx, cy + 1.55, title, ha="center", fontsize=10,
            fontweight="bold", color=tcolor)
    ax.text(cx, cy + 1.3, subtitle, ha="center", fontsize=7.5,
            color=MUTED, style="italic")

    # 좌표
    loc_y = cy + 0.85
    tank_x = cx - 1.0
    apc_x  = cx + 1.0
    asset_y = cy - 0.15
    edge_color = GRAPH if highlight else MUTED

    # ─── 엣지 먼저 그림 (zorder=2, 노드가 나중에 덮음) ───
    # found_at 실선 (자산 중심 → 위치 중심)
    edge_lw_base = 1.5 + coocc_weight * 0.4
    ax.plot([tank_x, cx], [asset_y, loc_y],
            color=edge_color, lw=edge_lw_base, zorder=2)
    ax.plot([apc_x, cx], [asset_y, loc_y],
            color=edge_color, lw=edge_lw_base, zorder=2)

    # co_occurred_with 점선 (자산 ↔ 자산)
    if coocc_weight > 0:
        coocc_lw = 1.3 + coocc_weight * 0.8
        ax.plot([tank_x, apc_x], [asset_y, asset_y],
                color=edge_color, lw=coocc_lw,
                linestyle="--", zorder=2)
        # 공출현 라벨 (엣지 하단)
        ax.text(cx, asset_y - 0.24,
                f"공출현 {coocc_weight}회",
                ha="center", fontsize=7, zorder=6,
                color=edge_color, style="italic", fontweight="bold")

    # ─── 노드 (엣지 위에 덮이도록 zorder=4) ───
    # 위치 노드
    ax.add_patch(Circle((cx, loc_y), 0.32, facecolor=NODE_L,
                        edgecolor="white", lw=1.5, zorder=4))
    ax.text(cx, loc_y, "위치", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=5)

    # 전차 노드
    ax.add_patch(Circle((tank_x, asset_y), 0.32, facecolor=NODE_A,
                        edgecolor="white", lw=1.5, zorder=4))
    ax.text(tank_x, asset_y, "전차", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=5)

    # APC 노드
    ax.add_patch(Circle((apc_x, asset_y), 0.32, facecolor=NODE_A,
                        edgecolor="white", lw=1.5, zorder=4))
    ax.text(apc_x, asset_y, "APC", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=5)

    # ─── 관측 횟수 뱃지 (노드 아래) ───
    badge_c = "#dcfce7" if highlight else "white"
    badge_ec = GREEN if highlight else MUTED
    txt_c = GREEN if highlight else MUTED
    fw = "bold" if highlight else "normal"
    for asset_x, cnt in [(tank_x, tank_count), (apc_x, apc_count)]:
        ax.add_patch(FancyBboxPatch((asset_x-0.65, asset_y-0.78), 1.3, 0.3,
                                     boxstyle="round,pad=0.02",
                                     facecolor=badge_c, edgecolor=badge_ec,
                                     lw=0.9, zorder=4))
        ax.text(asset_x, asset_y-0.63, f"관측 횟수 {cnt}회",
                ha="center", va="center", fontsize=7.8, color=txt_c,
                fontweight=fw, zorder=5)


# ─── 좌: Before (1~2회차 누적) ───
draw_graph_snapshot(cx=2.0, cy=9.6,
                    tank_count=2, apc_count=1, coocc_weight=1,
                    title="Before (1~2회차 누적)",
                    subtitle="전차·APC 이미 같은 격자 관측 이력 존재",
                    highlight=False)

# ─── 중앙: 3회차 upsert 이벤트 ───
mid_x = 6.5
# 이벤트 카드
ax.add_patch(FancyBboxPatch((mid_x - 1.35, 9.2), 2.7, 1.15,
                             boxstyle="round,pad=0.05",
                             facecolor="#faf5ff", edgecolor=GRAPH, lw=1.5))
ax.text(mid_x, 10.15, "3회차 upsert (지금)",
        ha="center", fontsize=9.5, fontweight="bold", color=GRAPH)
ax.text(mid_x, 9.88, "2025.04.10 ↔ 08.20",
        ha="center", fontsize=7.5, color=TEXT)
ax.text(mid_x, 9.63,
        "전차 · APC 같은 격자 재관측",
        ha="center", fontsize=8, color=TEXT)
ax.text(mid_x, 9.35,
        "→ 관측 횟수 +1, 공출현 엣지 +1",
        ha="center", fontsize=8, color=GRAPH,
        fontweight="bold", style="italic")

# 좌→중앙 화살표
ax.add_patch(FancyArrowPatch((3.4, 9.6), (5.15, 9.75),
                              arrowstyle="->,head_length=8,head_width=6",
                              color="#334155", lw=1.4))
# 중앙→우 화살표
ax.add_patch(FancyArrowPatch((7.85, 9.75), (9.6, 9.6),
                              arrowstyle="->,head_length=8,head_width=6",
                              color=GRAPH, lw=1.8))
ax.text(8.72, 9.98, "upsert", ha="center", fontsize=7.5,
        color=GRAPH, fontweight="bold", style="italic")

# ─── 우: After (3회차 반영) ───
draw_graph_snapshot(cx=11.0, cy=9.6,
                    tank_count=3, apc_count=2, coocc_weight=2,
                    title="After (3회차 반영)",
                    subtitle="관측 횟수와 공출현 엣지 가중치 모두 +1",
                    highlight=True)

# 하단 범례
ax.text(6.5, 8.94,
        "실선 = found_at (자산 → 위치)  ·  점선 = co_occurred_with (자산 ↔ 자산, 두께 = 공출현 누적 횟수)",
        ha="center", fontsize=7.3, color=MUTED, style="italic")

arrow_down(6.5, 8.8, 7.5)

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
