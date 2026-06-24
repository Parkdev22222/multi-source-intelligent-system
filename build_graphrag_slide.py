"""Generate the GraphRAG architecture slide diagram in the same style as the user's change-detection slide."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.font_manager as fm

# 한글 폰트
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 사용자 슬라이드와 같은 톤
BLUE_DARK   = "#1e6091"
BLUE_TEAL   = "#2a8a8a"
BLUE_LIGHT  = "#d0e8f2"
ARROW_BLUE  = "#4a90c2"
ACCENT_ORANGE = "#e6863e"
ACCENT_GREEN  = "#5fa66b"
TEXT_DARK   = "#1f2937"

fig, ax = plt.subplots(figsize=(14, 7.0))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# 제목
ax.text(0.4, 6.6, "GraphRAG", fontsize=22, fontweight="bold", color=TEXT_DARK)

# ───── ① Pairing 결과 입력 ─────
def draw_box(x, y, w, h, label, sublabel=None, fc=BLUE_TEAL, fontcolor="white", fontsize=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor="none")
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=fontcolor)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha="center", va="center",
                fontsize=8.5, color=fontcolor, style="italic")

def arrow(x1, y1, x2, y2, lw=3, mut=20):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=f"->,head_length={mut*0.55},head_width={mut*0.42}",
                        color=ARROW_BLUE, lw=lw)
    ax.add_patch(a)

# 입력: 변화 객체 표
ax.text(0.6, 5.7, "변화 탐지 결과", fontsize=10.5, color=BLUE_DARK,
        fontweight="bold")
table_x, table_y = 0.4, 4.0
ax.add_patch(patches.Rectangle((table_x, table_y), 2.7, 1.5,
              facecolor="white", edgecolor=BLUE_DARK, lw=1.5))
ax.text(table_x + 0.1, table_y + 1.25, "객체", fontsize=8.5, fontweight="bold", color=BLUE_DARK)
ax.text(table_x + 1.05, table_y + 1.25, "위치", fontsize=8.5, fontweight="bold", color=BLUE_DARK)
ax.text(table_x + 2.0, table_y + 1.25, "상태", fontsize=8.5, fontweight="bold", color=BLUE_DARK)
ax.plot([table_x + 0.05, table_x + 2.65],
        [table_y + 1.15, table_y + 1.15], color=BLUE_DARK, lw=0.5)
rows = [("tank", "37.50,127.00", "new"),
        ("APC",  "37.50,127.00", "new"),
        ("art.", "37.50,127.00", "new"),
        ("tank", "37.51,127.01", "moved")]
for i, (a, b, c) in enumerate(rows):
    y_r = table_y + 0.95 - i * 0.23
    ax.text(table_x + 0.1, y_r, a, fontsize=8, color=TEXT_DARK)
    ax.text(table_x + 1.05, y_r, b, fontsize=7.5, color=TEXT_DARK)
    ax.text(table_x + 2.0, y_r, c, fontsize=8, color=TEXT_DARK)

# ① → ② 화살표
arrow(3.2, 4.75, 4.0, 4.75)

# ───── ② 격자 양자화 ─────
draw_box(4.0, 4.25, 2.3, 1.05,
         "격자 양자화",
         "lat/lon → 1km 격자",
         fc=BLUE_TEAL)
ax.text(5.15, 4.05, "loc:37.50,127.00", fontsize=8, color=BLUE_DARK,
        ha="center", family="monospace")
ax.text(5.15, 3.85, "asset:tank:37.50,127.00", fontsize=7.5, color=BLUE_DARK,
        ha="center", family="monospace")

arrow(6.4, 4.75, 7.2, 4.75)

# ───── ③ 카운터 누적 (LLM 비호출 강조) ─────
draw_box(7.2, 4.25, 2.5, 1.05,
         "카운터 +1 누적",
         "AI 호출 ❌ · 결정론",
         fc=ACCENT_GREEN)
# small "no LLM" highlight
ax.text(8.45, 4.05, "new_count++", fontsize=8, color=ACCENT_GREEN,
        ha="center", family="monospace", fontweight="bold")
ax.text(8.45, 3.85, "matched_count++", fontsize=7.5, color=ACCENT_GREEN,
        ha="center", family="monospace")

arrow(9.8, 4.75, 10.6, 4.75)

# ───── ④ 공출현 엣지 + Louvain 군집 ─────
# 그래프 시각화
gx, gy = 10.85, 4.55
# 노드
for (nx, ny, label, color) in [
    (gx, gy + 0.6, "tank", ACCENT_ORANGE),
    (gx - 0.5, gy - 0.4, "APC", ACCENT_ORANGE),
    (gx + 0.5, gy - 0.4, "art.", ACCENT_ORANGE),
]:
    c = patches.Circle((nx, ny), 0.25, facecolor=color, edgecolor="white", lw=2, zorder=3)
    ax.add_patch(c)
    ax.text(nx, ny, label, ha="center", va="center", fontsize=8, color="white",
            fontweight="bold", zorder=4)
# 엣지 (굵기 = 공출현 빈도)
edges = [
    ((gx, gy + 0.6), (gx - 0.5, gy - 0.4), 3),
    ((gx, gy + 0.6), (gx + 0.5, gy - 0.4), 2.5),
    ((gx - 0.5, gy - 0.4), (gx + 0.5, gy - 0.4), 2),
]
for (x1, y1), (x2, y2), w in edges:
    ax.plot([x1, x2], [y1, y2], color=ACCENT_ORANGE, lw=w, alpha=0.5, zorder=2)

# 점선 박스로 군집 표시
ax.add_patch(patches.FancyBboxPatch((gx - 1.05, gy - 0.85), 2.1, 1.95,
              boxstyle="round,pad=0.05",
              facecolor="none", edgecolor=ACCENT_ORANGE, lw=2, linestyle="--"))
ax.text(gx, gy + 1.32, "Louvain 자동 군집화", fontsize=9, color=ACCENT_ORANGE,
        ha="center", fontweight="bold")
ax.text(gx, gy - 1.05, "= \"기갑 복합체\" doctrine", fontsize=8.5, color=ACCENT_ORANGE,
        ha="center", style="italic")

# ④ → ⑤ 화살표 (아래로)
arrow(11.0, 3.3, 11.0, 2.5)

# ───── ⑤ Local + Global Search ─────
draw_box(8.8, 1.4, 4.4, 1.05,
         "Local + Global Search",
         "지역 이력 + 군집 요약 → ~500 토큰",
         fc=BLUE_DARK)

arrow(8.8, 1.95, 7.6, 1.95)

# ───── ⑥ LLM 컨텍스트 주입 ─────
draw_box(3.0, 1.4, 4.5, 1.05,
         "LLM 프롬프트 prepend",
         "환각 차단 · 시계열 패턴 주입",
         fc=BLUE_TEAL)

arrow(5.25, 1.4, 5.25, 0.7)

# 결과
ax.text(5.25, 0.45, "IMINT 판독보고서 자율 생성", fontsize=11.5, color=BLUE_DARK,
        ha="center", fontweight="bold")

# ───── 좌하단 핵심 메시지 박스 ─────
ax.add_patch(patches.FancyBboxPatch((0.4, 0.05), 2.4, 1.05,
              boxstyle="round,pad=0.05",
              facecolor="#fef9c3", edgecolor="#f59e0b", lw=1.2))
ax.text(1.6, 0.92, "핵심 차별", ha="center", fontsize=10, fontweight="bold",
        color="#92400e")
ax.text(1.6, 0.62, "AI 호출 없이", ha="center", fontsize=8.5, color="#92400e")
ax.text(1.6, 0.40, "자동 누적·재현·감사", ha="center", fontsize=8.5, color="#92400e")
ax.text(1.6, 0.18, "→ 시계열 메모리", ha="center", fontsize=8.5, color="#92400e",
        fontweight="bold")

plt.tight_layout()
out_path = "/home/user/multi-source-intelligent-system/data/graphrag_architecture_slide.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
