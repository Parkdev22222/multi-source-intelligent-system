"""도 2 v3 통합본 — 객체 페어링 변화 탐지 상세 (고정형·이동형 이원 처리)

레이아웃:
  [1] 입력: 두 프레임 탐지 결과
  [2] 클래스 판정 (분기 노드)
    ├── 좌측 (고정형): 3단계 파이프라인 (노란 톤)
    └── 우측 (이동형): 5단계 파이프라인 (주황 톤)
  [3] 5상태 통합 분류
  [4] Pairing DB 저장
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Polygon, Circle
import matplotlib.font_manager as fm

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 색상 팔레트
FIX_MAIN   = "#f59e0b"    # 노란 (고정형 메인)
FIX_LIGHT  = "#fef3c7"    # 노란 배경
MOV_MAIN   = "#ea580c"    # 주황 (이동형 메인)
MOV_LIGHT  = "#fed7aa"    # 주황 배경
NEUTRAL    = "#334155"
STAGE_BG   = "#f8fafc"
STAGE_BD   = "#e2e8f0"
TEXT       = "#111827"
MUTED      = "#64748b"
DB         = "#475569"
COMMON     = "#3b82f6"    # 공통/통합

fig, ax = plt.subplots(figsize=(14, 17))
ax.set_xlim(0, 14); ax.set_ylim(0, 19)
ax.axis("off")

# ── 제목 ──
ax.text(7, 18.5, "도 2. 객체 페어링 변화 탐지 상세 (고정형·이동형 이원 처리)",
        ha="center", fontsize=15, fontweight="bold", color=TEXT)
ax.text(7, 18.1,
        "객체 클래스에 따라 두 파이프라인이 병렬 실행되어 5상태로 통합 분류",
        ha="center", fontsize=10, color=MUTED, style="italic")


# ══════════════════════════════════════════════════════════════
# 헬퍼 함수
# ══════════════════════════════════════════════════════════════
def box(x, y, w, h, label, sub=None, fc=MOV_MAIN, fs=11, ss=8.5, tc="white"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                       facecolor=fc, edgecolor="none")
    ax.add_patch(p)
    if sub:
        ax.text(x+w/2, y+h/2+0.18, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc)
        ax.text(x+w/2, y+h/2-0.24, sub, ha="center", va="center",
                fontsize=ss, color=tc)
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc)

def db_shape(x, y, w, h, label, sub=None, fc=DB):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="#1e293b", lw=1.2, zorder=1))
    ax.add_patch(Ellipse((x+w/2, y+h), w, h*0.28, facecolor=fc, edgecolor="#1e293b", lw=1.2, zorder=2))
    ax.add_patch(Ellipse((x+w/2, y), w, h*0.28, facecolor=fc, edgecolor="#1e293b", lw=1.2, zorder=3))
    if sub:
        ax.text(x+w/2, y+h/2+0.15, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
        ax.text(x+w/2, y+h/2-0.22, sub, ha="center", va="center", fontsize=8.5, color="white")
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")

def arrow(x1, y1, x2, y2, color=NEUTRAL, lw=2, mut=10, label=None, lx=None, ly=None,
          la=None):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=f"->,head_length={mut*0.55},head_width={mut*0.42}",
                        color=color, lw=lw)
    ax.add_patch(a)
    if label:
        ax.text(lx if lx is not None else (x1+x2)/2,
                ly if ly is not None else (y1+y2)/2,
                label, ha="center", fontsize=8.5, color=color, fontweight="bold")


# ══════════════════════════════════════════════════════════════
# [1단계] 입력
# ══════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((0.3, 16.5), 13.4, 1.1, facecolor=STAGE_BG, edgecolor=STAGE_BD, lw=1))
ax.text(0.55, 17.35, "[1] 입력 — Sensor DB로부터 두 시점 탐지 결과 조회",
        fontsize=10.5, color=MUTED, fontweight="bold")

box(1.8, 16.7, 4.0, 0.7, "현재 프레임 탐지 (N개)", fc="#93c5fd", tc=TEXT, fs=10.5)
box(8.2, 16.7, 4.0, 0.7, "과거 프레임 탐지 (M개)", fc="#93c5fd", tc=TEXT, fs=10.5)

# 세로 화살표 → 분기 판정으로
arrow(7, 16.5, 7, 15.7, lw=2.5, mut=12)


# ══════════════════════════════════════════════════════════════
# [2단계] 클래스 판정 (분기 노드)
# ══════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((0.3, 14.4), 13.4, 1.3, facecolor=STAGE_BG, edgecolor=STAGE_BD, lw=1))
ax.text(0.55, 15.5, "[2] 각 객체의 클래스 판정 — 사전 정의된 고정형 자산 리스트에 속하는가?",
        fontsize=10.5, color=MUTED, fontweight="bold")

# 다이아몬드 형태 분기 노드
diamond = Polygon([(7, 15.35), (8.7, 14.85), (7, 14.5), (5.3, 14.85)],
                   facecolor=COMMON, edgecolor="none")
ax.add_patch(diamond)
ax.text(7, 14.9, "class ∈\n_STATIC_CLASSES?",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="white")

# 분기 화살표
arrow(5.3, 14.75, 3.5, 14.0, color=FIX_MAIN, lw=2.5,
      label="예 (고정형)", lx=3.9, ly=14.5)
arrow(8.7, 14.75, 10.5, 14.0, color=MOV_MAIN, lw=2.5,
      label="아니오 (이동형)", lx=10.1, ly=14.5)

# 좌하단 참고 (STATIC_CLASSES 목록)
ax.add_patch(FancyBboxPatch((0.5, 14.55), 4.3, 0.5, boxstyle="round,pad=0.03",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=0.8))
ax.text(2.65, 14.8, "_STATIC_CLASSES = { building, command_post, fuel_storage,\n"
                    "                    supply_depot, military_building, radar }",
        ha="center", fontsize=7.5, color=TEXT, family="monospace")


# ══════════════════════════════════════════════════════════════
# [3-L] 좌측 파이프라인: 고정형 객체 알고리즘
# ══════════════════════════════════════════════════════════════
# 좌측 배경 (노란 톤)
ax.add_patch(Rectangle((0.3, 3.5), 6.5, 10.4, facecolor="#fffbeb", edgecolor=FIX_MAIN,
                       lw=1.5, linestyle="dashed"))
ax.text(0.55, 13.7, "[3-L] 좌측 파이프라인 — 고정형 객체 전용 (Step 0)",
        fontsize=10.5, color=FIX_MAIN, fontweight="bold")
ax.text(3.5, 13.3, "물리적으로 이동 불가 → 위경도 근접이 강력한 근거",
        ha="center", fontsize=9, color="#78350f", style="italic")

# 단계 1: 위경도 초근접 그리디 결합
box(0.8, 11.7, 5.5, 1.2,
    "① 위경도 초근접 그리디 결합",
    "동일 클래스 + 거리 ≤ 약 11m 인 쌍부터\n가까운 순서로 1:1 결합 (그리디)",
    fc=FIX_MAIN, fs=11, ss=8.5)
arrow(3.55, 11.7, 3.55, 10.5, color=FIX_MAIN, lw=2.5, mut=12)

# 단계 2: CLIP 외형 비교 → matched/changed
box(0.8, 9.1, 5.5, 1.4,
    "② CLIP 외형 비교 (변화 판정)",
    "결합된 쌍의 마스크 crop을 CLIP 인코더에 통과\n → 유사도 ≥ 0.5 → matched (변화 없음)\n → 유사도 < 0.5 → changed (구조 변화)",
    fc=FIX_MAIN, fs=11, ss=8)
arrow(3.55, 9.1, 3.55, 7.8, color=FIX_MAIN, lw=2.5, mut=12)

# 단계 3: 한쪽 유실 시 가상 탐지 합성
box(0.8, 6.0, 5.5, 1.8,
    "③ 한쪽 유실 시 가상 탐지 합성",
    "한쪽 프레임에만 탐지된 경우:\n· 상대 영상의 같은 위경도 영역 강제 crop\n· CLIP 유사도 재검증\n· 유사도 ≥ 0.5 → synthetic 레코드 주입\n· 유사도 < 0.5 → 진짜 disappeared/new",
    fc=FIX_MAIN, fs=11, ss=8)

# 특이 처리 안내 (changed 상태)
ax.add_patch(FancyBboxPatch((0.8, 4.5), 5.5, 1.1, boxstyle="round,pad=0.04",
                             facecolor=FIX_LIGHT, edgecolor=FIX_MAIN, lw=1))
ax.text(3.55, 5.3, "★ 고정형 전용 상태: changed",
        ha="center", fontsize=10, fontweight="bold", color="#78350f")
ax.text(3.55, 4.95, "위경도 동일하지만 외형 변화 감지 시 부여",
        ha="center", fontsize=8.5, color="#78350f", style="italic")
ax.text(3.55, 4.7, "(예: 건물 파손·증축)",
        ha="center", fontsize=8, color="#78350f", style="italic")
arrow(3.55, 6.0, 3.55, 5.6, color=FIX_MAIN, lw=2, mut=10)

# 좌측 → 통합 화살표
arrow(3.55, 4.5, 3.55, 3.35, color=FIX_MAIN, lw=2.5, mut=12)


# ══════════════════════════════════════════════════════════════
# [3-R] 우측 파이프라인: 이동형 객체 알고리즘
# ══════════════════════════════════════════════════════════════
# 우측 배경 (주황 톤)
ax.add_patch(Rectangle((7.2, 3.5), 6.5, 10.4, facecolor="#fff7ed", edgecolor=MOV_MAIN,
                       lw=1.5, linestyle="dashed"))
ax.text(7.4, 13.7, "[3-R] 우측 파이프라인 — 이동형 객체 전용 (Step 1~)",
        fontsize=10.5, color=MOV_MAIN, fontweight="bold")
ax.text(10.45, 13.3, "위치가 이동해도 외형이 유지 → CLIP 임베딩이 강력한 근거",
        ha="center", fontsize=9, color="#7c2d12", style="italic")

# 단계 1: 마스크 배경 제거
box(7.65, 12.15, 5.7, 0.9,
    "① 마스크 배경 제거",
    "SAM 마스크로 배경 픽셀 zeroing → bbox crop",
    fc=MOV_MAIN, fs=10.5, ss=8)
arrow(10.5, 12.15, 10.5, 11.4, color=MOV_MAIN, lw=2, mut=10)

# 단계 2: CLIP/ViT 임베딩
box(7.65, 10.55, 5.7, 0.9,
    "② CLIP/ViT 임베딩 산출",
    "L2 정규화 시각 특징 벡터 (N×D, M×D)",
    fc=MOV_MAIN, fs=10.5, ss=8)
arrow(10.5, 10.55, 10.5, 9.8, color=MOV_MAIN, lw=2, mut=10)

# 단계 3: N×M 유사도 행렬
box(7.65, 8.95, 5.7, 0.9,
    "③ N×M 코사인 유사도 행렬",
    "cur_embed × past_embedᵀ (단일 GEMM)",
    fc=MOV_MAIN, fs=10.5, ss=8)
arrow(10.5, 8.95, 10.5, 8.2, color=MOV_MAIN, lw=2, mut=10)

# 단계 4: 동일 클래스 필터 + 점수
box(7.65, 7.30, 5.7, 0.9,
    "④ 동일 클래스 필터 + 점수화",
    "score = 0.8·CLIP + 0.2·크기 (임계값 > 0.5)",
    fc=MOV_MAIN, fs=10.5, ss=8)
arrow(10.5, 7.30, 10.5, 6.5, color=MOV_MAIN, lw=2, mut=10)

# 단계 5: Gale-Shapley 안정 매칭
box(7.65, 5.55, 5.7, 1.0,
    "⑤ Gale-Shapley 안정 매칭",
    "후보 쌍 → 1:1 안정 매칭\n(그리디의 국소 최적 · 교차 페어링 회피)",
    fc=MOV_MAIN, fs=10.5, ss=8)

# 우측 결과 안내
ax.add_patch(FancyBboxPatch((7.65, 4.5), 5.7, 0.85, boxstyle="round,pad=0.04",
                             facecolor=MOV_LIGHT, edgecolor=MOV_MAIN, lw=1))
ax.text(10.5, 5.1, "★ 매칭 성공 시 → matched",
        ha="center", fontsize=10, fontweight="bold", color="#7c2d12")
ax.text(10.5, 4.75, "(위경도 거리 무관 — 이동해도 동일 객체 유지)",
        ha="center", fontsize=8.5, color="#7c2d12", style="italic")
arrow(10.5, 5.55, 10.5, 5.35, color=MOV_MAIN, lw=2, mut=10)

# 우측 → 통합 화살표
arrow(10.5, 4.5, 10.5, 3.35, color=MOV_MAIN, lw=2.5, mut=12)


# ══════════════════════════════════════════════════════════════
# [4] 5상태 통합 분류
# ══════════════════════════════════════════════════════════════
ax.add_patch(Rectangle((0.3, 1.9), 13.4, 1.4, facecolor=STAGE_BG, edgecolor=STAGE_BD, lw=1))
ax.text(0.55, 3.1, "[4] 5상태 통합 분류 (좌·우 파이프라인 결과 통합 + FOV 검증)",
        fontsize=10.5, color=MUTED, fontweight="bold")

# 5개 상태 박스
states = [
    ("matched", "동일 객체", MOV_MAIN),
    ("changed", "구조 변화", FIX_MAIN),
    ("new", "신규 출현", "#16a34a"),
    ("disappeared", "소실", "#dc2626"),
    ("FOV 공백", "촬영 범위 밖", "#94a3b8"),
]
xs = [0.7, 3.35, 6.0, 8.65, 11.3]
for (name, desc, c), xx in zip(states, xs):
    box(xx, 2.15, 2.15, 0.85, name, desc, fc=c, fs=10, ss=8)

# ══════════════════════════════════════════════════════════════
# [5] Pairing DB
# ══════════════════════════════════════════════════════════════
arrow(7, 1.9, 7, 1.55, color=NEUTRAL, lw=2.5, mut=12)
db_shape(4.5, 0.5, 5.0, 0.9, "Pairing DB", "insert_pairings_bulk()", fc=DB)

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/fig2_v3_unified.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
