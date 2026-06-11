"""Generate a visualization of the Pairing DB structure and example data."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib as mpl

# Korean font support
import matplotlib.font_manager as fm
for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        fm.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

STATUS_COLORS = {
    "new":          ("#16a34a", "신규 출현"),
    "matched":      ("#3b82f6", "정지"),
    "moved":        ("#f59e0b", "이동"),
    "disappeared":  ("#dc2626", "소실"),
    "past_not_included":    ("#a855f7", "과거 미포함\n(FOV 밖)"),
    "current_not_included": ("#fb923c", "현재 미포함\n(FOV 밖)"),
}

fig = plt.figure(figsize=(16, 22))
gs = fig.add_gridspec(4, 1, height_ratios=[1.0, 2.2, 1.4, 1.0], hspace=0.35)

# ===================================================================
# Panel 1 : Schema overview
# ===================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 4)
ax1.axis("off")
ax1.set_title("Pairing DB 구조 — object_pairings.db / pairing_records 테이블",
              fontsize=16, fontweight="bold", pad=14)

# Sensor DB box
sensor_box = FancyBboxPatch((0.2, 0.8), 2.8, 2.4,
                             boxstyle="round,pad=0.08",
                             facecolor="#e0f2fe", edgecolor="#0369a1", linewidth=2)
ax1.add_patch(sensor_box)
ax1.text(1.6, 2.9, "Sensor DB", ha="center", fontsize=12, fontweight="bold", color="#0369a1")
ax1.text(1.6, 2.45, "image_records", ha="center", fontsize=9)
ax1.text(1.6, 2.10, "detection_records", ha="center", fontsize=9)
ax1.text(1.6, 1.6, "(영상 + 객체 탐지)", ha="center", fontsize=8, color="#475569")

# Pairing DB box
pair_box = FancyBboxPatch((4.0, 0.4), 5.6, 3.1,
                          boxstyle="round,pad=0.08",
                          facecolor="#fef3c7", edgecolor="#b45309", linewidth=2)
ax1.add_patch(pair_box)
ax1.text(6.8, 3.2, "Pairing DB  (object_pairings.db)",
         ha="center", fontsize=12, fontweight="bold", color="#b45309")
ax1.text(6.8, 2.75, "pairing_records 테이블", ha="center", fontsize=10, fontweight="bold")
ax1.text(6.8, 2.35, "한 행 = '현재 객체 ↔ 과거 객체' 한 쌍의 매칭 결과",
         ha="center", fontsize=9, color="#475569")
ax1.text(6.8, 1.95, "│ pairing_time   lat_center / lon_center  status   session_id", ha="center",
         fontsize=8, family="monospace")
ax1.text(6.8, 1.65, "│ current_detection_id / current_class / current_lat / current_lon ...",
         ha="center", fontsize=8, family="monospace")
ax1.text(6.8, 1.35, "│ past_detection_id    / past_class    / past_lat    / past_lon    ...",
         ha="center", fontsize=8, family="monospace")
ax1.text(6.8, 1.0, "│ current_bbox (JSON)  / past_bbox (JSON)   / source_type",
         ha="center", fontsize=8, family="monospace")
ax1.text(6.8, 0.65, "(외래키 X — detection_id 문자열로만 Sensor DB 참조)",
         ha="center", fontsize=8, color="#94a3b8", style="italic")

# Arrow
arr = FancyArrowPatch((3.05, 2.0), (3.95, 2.0),
                       arrowstyle="->,head_length=8,head_width=6",
                       color="#475569", lw=2)
ax1.add_patch(arr)
ax1.text(3.5, 2.25, "detection_id\n참조", ha="center", fontsize=8, color="#475569")

# ===================================================================
# Panel 2 : Example rows for each status
# ===================================================================
ax2 = fig.add_subplot(gs[1])
ax2.axis("off")
ax2.set_title("상태별(status) 저장 패턴 — 6개 상태 예시 행",
              fontsize=15, fontweight="bold", pad=10)

# Headers
headers = ["status", "current_*\n(현재 프레임)", "past_*\n(과거 프레임)",
           "분석 의미", "보고서 포함?"]
col_x = [0.05, 0.20, 0.40, 0.60, 0.85]
col_w = [0.13, 0.18, 0.18, 0.21, 0.10]

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Header bar
ax2.add_patch(patches.Rectangle((0.02, 0.92), 0.96, 0.07,
                                 facecolor="#1e293b"))
for i, h in enumerate(headers):
    ax2.text(col_x[i] + col_w[i] / 2, 0.955, h,
             ha="center", va="center", fontsize=9.5,
             fontweight="bold", color="white")

# Rows
rows = [
    ("new",
     "tank, 0.87,\n(37.578, 126.971),\nbbox=[3120,2450,...]",
     "NULL\n(과거 없음)",
     "신규 출현 — 분석관 관심",
     "✅ 포함"),
    ("matched",
     "tank, 0.87,\n(37.578, 126.971)",
     "tank, 0.83,\n(37.578, 126.971)",
     "정지(geo_dist < 100m)",
     "❌ 제외"),
    ("moved",
     "tank, 0.87,\n(37.578, 126.971)",
     "tank, 0.83,\n(37.575, 126.969)",
     "이동(geo_dist ≥ 100m)",
     "△ 보조"),
    ("disappeared",
     "NULL\n(현재 없음)",
     "APC, 0.80,\n(37.571, 126.962)",
     "현재 영상에서 소실",
     "✅ 포함"),
    ("past_not_included",
     "tank, 0.87,\n(37.585, 126.978)",
     "NULL",
     "현재 객체가 \n과거 영상 FOV 밖\n→ 비교 불가",
     "❌ 제외"),
    ("current_not_included",
     "NULL",
     "APC, 0.80,\n(37.560, 126.955)",
     "과거 객체가 \n현재 영상 FOV 밖\n→ 비교 불가",
     "❌ 제외"),
]

row_top = 0.88
row_h = 0.135
for i, (status, cur, past, meaning, report) in enumerate(rows):
    y_top = row_top - row_h * i
    y_mid = y_top - row_h / 2
    # Row background
    color = STATUS_COLORS[status][0]
    ax2.add_patch(patches.Rectangle((0.02, y_top - row_h + 0.005), 0.96,
                                     row_h - 0.005,
                                     facecolor="#f8fafc",
                                     edgecolor=color, linewidth=1.5))
    # Status badge
    ax2.add_patch(patches.FancyBboxPatch(
        (col_x[0], y_mid - 0.04), col_w[0] - 0.01, 0.08,
        boxstyle="round,pad=0.005",
        facecolor=color, edgecolor="none"))
    ax2.text(col_x[0] + col_w[0] / 2 - 0.005, y_mid, status,
             ha="center", va="center", fontsize=9.5,
             fontweight="bold", color="white")
    # current_*
    null_color = "#94a3b8" if "NULL" in cur else "#1e293b"
    ax2.text(col_x[1] + col_w[1] / 2, y_mid, cur,
             ha="center", va="center", fontsize=8.5,
             family="monospace", color=null_color)
    # past_*
    null_color = "#94a3b8" if "NULL" in past else "#1e293b"
    ax2.text(col_x[2] + col_w[2] / 2, y_mid, past,
             ha="center", va="center", fontsize=8.5,
             family="monospace", color=null_color)
    # meaning
    ax2.text(col_x[3] + col_w[3] / 2, y_mid, meaning,
             ha="center", va="center", fontsize=9)
    # report
    ax2.text(col_x[4] + col_w[4] / 2, y_mid, report,
             ha="center", va="center", fontsize=10, fontweight="bold")

# ===================================================================
# Panel 3 : Geographic visualization (FOV diagram)
# ===================================================================
ax3 = fig.add_subplot(gs[2])
ax3.set_title("FOV 기반 상태 분류 시각화 — 같은 지역, 다른 촬영 범위",
              fontsize=14, fontweight="bold", pad=10)
ax3.set_xlim(126.94, 127.01)
ax3.set_ylim(37.55, 37.60)
ax3.set_xlabel("경도(lon)", fontsize=9)
ax3.set_ylabel("위도(lat)", fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_aspect(1.3)

# Past image FOV
past_fov = patches.Rectangle((126.952, 37.560), 0.028, 0.025,
                              facecolor="#dbeafe", edgecolor="#1d4ed8",
                              linewidth=2.5, alpha=0.5, label="과거 영상 FOV")
ax3.add_patch(past_fov)
ax3.text(126.966, 37.587, "과거 영상 FOV",
         color="#1d4ed8", fontsize=10, fontweight="bold")

# Current image FOV
cur_fov = patches.Rectangle((126.965, 37.568), 0.030, 0.028,
                             facecolor="#fef3c7", edgecolor="#b45309",
                             linewidth=2.5, alpha=0.5, label="현재 영상 FOV")
ax3.add_patch(cur_fov)
ax3.text(126.986, 37.598, "현재 영상 FOV",
         color="#b45309", fontsize=10, fontweight="bold")

# Objects with status
objects = [
    (126.978, 37.580, "matched",       "전차1\nmatched"),
    (126.970, 37.575, "moved",         "전차2\nmoved"),
    (126.985, 37.585, "new",           "전차3\nnew"),
    (126.960, 37.572, "disappeared",   "APC\ndisappeared"),
    (126.992, 37.575, "past_not_included",    "전차4\npast_not_\nincluded"),
    (126.957, 37.583, "current_not_included", "전차5\ncurrent_not_\nincluded"),
]
for lon, lat, st, label in objects:
    color = STATUS_COLORS[st][0]
    ax3.scatter(lon, lat, s=180, c=color, edgecolors="black", linewidths=1.5, zorder=5)
    ax3.annotate(label, (lon, lat), xytext=(8, 8), textcoords="offset points",
                  fontsize=8, fontweight="bold", color=color,
                  bbox=dict(boxstyle="round,pad=0.2",
                            facecolor="white",
                            edgecolor=color, alpha=0.95))

# Legend
legend_handles = [patches.Patch(facecolor=c, label=f"{k}: {lbl}")
                   for k, (c, lbl) in STATUS_COLORS.items()]
ax3.legend(handles=legend_handles, loc="lower right",
           fontsize=8, framealpha=0.95, ncol=2)

# ===================================================================
# Panel 4 : Status priority + storage flow
# ===================================================================
ax4 = fig.add_subplot(gs[3])
ax4.axis("off")
ax4.set_title("상태 결정 우선순위 + 저장 흐름",
              fontsize=14, fontweight="bold", pad=10)

# Priority pipeline
labels = ["matched", "moved", "disappeared",
          "past/current_\nnot_included", "new"]
xs = [0.07, 0.24, 0.42, 0.60, 0.83]
for i, lab in enumerate(labels):
    key = lab.split("\n")[0].strip() if "/" in lab else lab
    if key == "past/current_":
        color = STATUS_COLORS["past_not_included"][0]
    else:
        color = STATUS_COLORS.get(key, ("#64748b", ""))[0]
    bbox = FancyBboxPatch((xs[i] - 0.07, 0.55), 0.14, 0.20,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor="black", linewidth=1.5)
    ax4.add_patch(bbox)
    ax4.text(xs[i], 0.65, lab, ha="center", va="center",
             fontsize=9.5, fontweight="bold", color="white")
    if i < len(labels) - 1:
        arr = FancyArrowPatch((xs[i] + 0.07, 0.65), (xs[i + 1] - 0.07, 0.65),
                               arrowstyle="->,head_length=6,head_width=4",
                               color="#1e293b", lw=1.5)
        ax4.add_patch(arr)

ax4.text(0.5, 0.85, "우선순위 — 위 순서대로 적용 (먼저 결정되면 그 상태 확정)",
         ha="center", fontsize=10, style="italic", color="#475569")

# Storage flow
ax4.text(0.5, 0.35,
         "저장: insert_pairings_bulk()  →  pairing_records 테이블 INSERT  "
         "(src/database/pairing_db.py:39)",
         ha="center", fontsize=10, family="monospace")
ax4.text(0.5, 0.20,
         "조회: get_latest_pairings(session_id)  →  최신 pairing_time 배치 (보고서 생성용)",
         ha="center", fontsize=10, family="monospace")
ax4.text(0.5, 0.05,
         "모든 행: session_id 부여 → 단일 파이프라인 실행 단위로 그룹화",
         ha="center", fontsize=9.5, color="#475569", style="italic")

plt.suptitle("Pairing DB 시각화 — 영상 간 변화 페어링 결과의 저장 구조",
             fontsize=17, fontweight="bold", y=0.995)

out_path = "/home/user/multi-source-intelligent-system/data/pairing_db_visualization.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")
