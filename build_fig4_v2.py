"""도 4 v2 — 판독보고서 자율 생성 (이해하기 쉬운 시각화)

제목: '판독보고서 자율 생성 — 영문 초안 → 한국어 무오염 번역'

각 단계를 실제 그림으로 도식화:
1) 두 갈래 입력 (Graph DB 압축 컨텍스트 + Pairing DB 변화 객체)
2) 프롬프트 조립 (시스템 + 사용자, 실제 프롬프트 구조 카드로 표현)
3) 영문 정형 8섹션 판독보고서 생성
4) 동일 LLM 인스턴스로 한국어 번역, 좌표·수치·타임스탬프 보존
5) 최종 보고서 → Report DB
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Ellipse
import matplotlib.font_manager as fm

for fp in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try: fm.fontManager.addfont(fp)
    except Exception: pass
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 색상
GRAPH        = "#7c3aed"
GRAPH_LIGHT  = "#f3e8ff"
LLM          = "#dc2626"
LLM_LIGHT    = "#fee2e2"
PAIR         = "#ea580c"
PAIR_LIGHT   = "#fed7aa"
REPORT       = "#16a34a"
REPORT_LIGHT = "#dcfce7"
STAGE        = "#f8fafc"
STAGE_BD     = "#e2e8f0"
TEXT         = "#111827"
MUTED        = "#64748b"
DB           = "#475569"
BLUE         = "#3b82f6"
YELLOW       = "#f59e0b"
RED          = "#dc2626"

fig, ax = plt.subplots(figsize=(13, 19))
ax.set_xlim(0, 13); ax.set_ylim(0, 20)
ax.axis("off")

# 제목
ax.text(6.5, 19.45, "도 4. 판독보고서 자율 생성",
        ha="center", fontsize=15, fontweight="bold", color=TEXT)
ax.text(6.5, 19.05,
        "그래프 압축 컨텍스트 + 이번 회차 변화 객체를 LLM에 주입 → 영문 정형 보고서 → 무오염 한국어 번역",
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

def cylinder(cx, cy, w, h, color, label, sub=None):
    """DB 원기둥 아이콘."""
    ax.add_patch(Rectangle((cx, cy), w, h, facecolor=color,
                           edgecolor="#1e293b", lw=1))
    ax.add_patch(Ellipse((cx+w/2, cy+h), w, h*0.35,
                          facecolor=color, edgecolor="#1e293b", lw=1))
    ax.add_patch(Ellipse((cx+w/2, cy), w, h*0.35,
                          facecolor=color, edgecolor="#1e293b", lw=1))
    ax.text(cx+w/2, cy+h/2 + (0.08 if sub else 0), label,
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="white")
    if sub:
        ax.text(cx+w/2, cy+h/2 - 0.18, sub, ha="center", va="center",
                fontsize=7.8, color="white", style="italic")


# ═══════════════════════════════════════════════════════════════
# [1단계] 두 입력 소스 (병렬)
# ═══════════════════════════════════════════════════════════════
stage_bg(16.5, 2.4, "1단계",
         "두 갈래 입력: GraphRAG 압축 컨텍스트 + Pairing DB 변화 객체")

# 좌: Graph DB → 압축 컨텍스트
cylinder(1.5, 17.85, 3.2, 0.5, GRAPH, "Graph DB", "누적 지식 그래프")
arrow_down(3.1, 17.75, 17.35)
ax.text(3.35, 17.55, "Local + Global 검색", fontsize=7.5, color=MUTED,
        style="italic", va="center")

# 압축 컨텍스트 카드
ctx_x, ctx_y = 0.8, 16.6
ctx_w, ctx_h = 5.4, 0.75
ax.add_patch(FancyBboxPatch((ctx_x, ctx_y), ctx_w, ctx_h,
                             boxstyle="round,pad=0.04",
                             facecolor=GRAPH_LIGHT, edgecolor=GRAPH, lw=1.2))
ax.text(ctx_x + ctx_w/2, ctx_y + ctx_h - 0.14,
        "압축 컨텍스트 (~500 토큰)",
        ha="center", fontsize=9, fontweight="bold", color=GRAPH)
ax.text(ctx_x + 0.15, ctx_y + 0.35,
        "\"기갑 복합체 8회 반복 관측,",
        fontsize=7.5, color=TEXT, style="italic")
ax.text(ctx_x + 0.15, ctx_y + 0.20,
        " 지난 30일간 3회 재배치,",
        fontsize=7.5, color=TEXT, style="italic")
ax.text(ctx_x + 0.15, ctx_y + 0.05,
        " 반경 5km 내 지속 배치 자산 다수...\"",
        fontsize=7.5, color=TEXT, style="italic")

# 우: Pairing DB → 변화 객체
cylinder(8.3, 17.85, 3.2, 0.5, DB, "Pairing DB", "이번 회차 페어링 결과")
arrow_down(9.9, 17.75, 17.35)
ax.text(10.15, 17.55, "이번 회차 변화 목록 조회", fontsize=7.5, color=MUTED,
        style="italic", va="center")

# 변화 객체 카드
chg_x, chg_y = 6.8, 16.6
chg_w, chg_h = 5.4, 0.75
ax.add_patch(FancyBboxPatch((chg_x, chg_y), chg_w, chg_h,
                             boxstyle="round,pad=0.04",
                             facecolor=PAIR_LIGHT, edgecolor=PAIR, lw=1.2))
ax.text(chg_x + chg_w/2, chg_y + chg_h - 0.14,
        "변화 객체 (신뢰도 순 정렬)",
        ha="center", fontsize=9, fontweight="bold", color="#9a3412")

items = [
    ("new",         "tank @ (37.58, 126.97)   · 0.87", BLUE),
    ("disappeared", "bldg @ (37.57, 126.96)   · 0.85", RED),
    ("changed",     "facility @ (37.58, 126.97) · 0.79", YELLOW),
]
for i, (status, detail, color) in enumerate(items):
    yy = chg_y + 0.35 - i * 0.14
    ax.text(chg_x + 0.15, yy, status, fontsize=7, color=color,
            fontweight="bold")
    ax.text(chg_x + 1.6, yy, detail, fontsize=7, color=TEXT,
            family="monospace")

# 두 카드에서 아래로 통합 화살표 → 2단계
arrow_down(3.5, 16.5, 16.0, color=GRAPH)
arrow_down(9.5, 16.5, 16.0, color=PAIR)


# ═══════════════════════════════════════════════════════════════
# [2단계] 프롬프트 조립
# ═══════════════════════════════════════════════════════════════
stage_bg(12.5, 3.5, "2단계",
         "두 입력을 시스템 프롬프트 + 사용자 프롬프트로 조립하여 LLM에 주입할 최종 프롬프트 완성")

# 시스템 프롬프트 카드
sys_x, sys_y = 1.0, 14.05
sys_w, sys_h = 11.0, 1.35
ax.add_patch(FancyBboxPatch((sys_x, sys_y), sys_w, sys_h,
                             boxstyle="round,pad=0.06",
                             facecolor="#fef2f2", edgecolor=LLM, lw=1.4))
ax.text(sys_x + 0.25, sys_y + sys_h - 0.22, "시스템 프롬프트 (강제 규칙)",
        fontsize=10, fontweight="bold", color=LLM)
ax.text(sys_x + 0.4, sys_y + sys_h - 0.5,
        "· 역할: 위성영상 판독관",
        fontsize=8.2, color=TEXT)
ax.text(sys_x + 0.4, sys_y + sys_h - 0.75,
        "· \"DISAPPEARED ≠ destroyed\"  —  소실은 파괴 의미 아님 (도메인 가드레일)",
        fontsize=8.2, color=TEXT)
ax.text(sys_x + 0.4, sys_y + sys_h - 1.0,
        "· 정형 8개 섹션 구조 강제 · 좌표·수치·타임스탬프는 원본 그대로 유지",
        fontsize=8.2, color=TEXT)

# 사용자 프롬프트 카드
usr_x, usr_y = 1.0, 12.6
usr_w, usr_h = 11.0, 1.35
ax.add_patch(FancyBboxPatch((usr_x, usr_y), usr_w, usr_h,
                             boxstyle="round,pad=0.06",
                             facecolor="#fff7ed", edgecolor=PAIR, lw=1.4))
ax.text(usr_x + 0.25, usr_y + usr_h - 0.22, "사용자 프롬프트",
        fontsize=10, fontweight="bold", color="#9a3412")

# prepend 컨텍스트 표시 (보라 박스)
ax.add_patch(FancyBboxPatch((usr_x + 0.4, usr_y + 0.55), usr_w - 0.8, 0.42,
                             boxstyle="round,pad=0.03",
                             facecolor=GRAPH_LIGHT, edgecolor=GRAPH, lw=0.9))
ax.text(usr_x + 0.55, usr_y + 0.76,
        "① [prepend] 압축 컨텍스트 ~500 토큰   ← Graph DB 검색 결과",
        fontsize=8, color="#4c1d95", style="italic", fontweight="bold")

# 변화 목록 표시 (주황 박스)
ax.add_patch(FancyBboxPatch((usr_x + 0.4, usr_y + 0.08), usr_w - 0.8, 0.42,
                             boxstyle="round,pad=0.03",
                             facecolor=PAIR_LIGHT, edgecolor=PAIR, lw=0.9))
ax.text(usr_x + 0.55, usr_y + 0.29,
        "② [본문] 이번 회차 변화 객체 목록   ← Pairing DB (new / disappeared / changed 상세)",
        fontsize=8, color="#7c2d12", style="italic", fontweight="bold")

arrow_down(6.5, 12.5, 12.0)


# ═══════════════════════════════════════════════════════════════
# [3단계] 1차 LLM 호출 — 영문 보고서
# ═══════════════════════════════════════════════════════════════
stage_bg(9.0, 3.0, "3단계",
         "영문 정형 판독보고서 생성 (8개 섹션 구조 준수)")

# 좌: LLM 호출 아이콘 + 1차 배지
llm_x, llm_y = 0.9, 10.3
llm_w, llm_h = 3.2, 1.2
ax.add_patch(FancyBboxPatch((llm_x, llm_y), llm_w, llm_h,
                             boxstyle="round,pad=0.06",
                             facecolor=LLM, edgecolor="#7f1d1d", lw=1.5))
ax.text(llm_x + llm_w/2, llm_y + 0.8, "LLM (예: EXAONE)",
        ha="center", fontsize=11, fontweight="bold", color="white")
ax.text(llm_x + llm_w/2, llm_y + 0.4, "영문 판독보고서 생성",
        ha="center", fontsize=8.5, color="white", style="italic")

ax.add_patch(Circle((llm_x + llm_w - 0.15, llm_y + llm_h - 0.05), 0.28,
                    facecolor="white", edgecolor=LLM, lw=1.8))
ax.text(llm_x + llm_w - 0.15, llm_y + llm_h - 0.05, "1",
        ha="center", va="center", fontsize=12, fontweight="bold", color=LLM)

# 화살표 → 영문 보고서
arrow_right(llm_x + llm_w + 0.1, llm_y + llm_h/2, 5.6, color=LLM, lw=2)
ax.text((llm_x + llm_w + 5.6)/2, llm_y + llm_h/2 + 0.2, "영문 초안 생성",
        ha="center", fontsize=8, color=LLM, style="italic")

# 우: 영문 보고서 카드 (문서 모양)
rep_x, rep_y = 5.6, 9.15
rep_w, rep_h = 6.7, 2.55
# 문서 그림자
ax.add_patch(Rectangle((rep_x + 0.08, rep_y - 0.08), rep_w, rep_h,
                        facecolor="#e5e7eb", edgecolor="none"))
# 문서 본체
ax.add_patch(FancyBboxPatch((rep_x, rep_y), rep_w, rep_h,
                             boxstyle="round,pad=0.04",
                             facecolor="white", edgecolor=LLM, lw=1.3))
ax.text(rep_x + rep_w/2, rep_y + rep_h - 0.22,
        "English Interpretation Report (8 sections)",
        ha="center", fontsize=9.5, fontweight="bold", color=LLM)
# 얇은 구분선
ax.plot([rep_x + 0.2, rep_x + rep_w - 0.2], [rep_y + rep_h - 0.4, rep_y + rep_h - 0.4],
        color=LLM, lw=0.5, alpha=0.5)

# 8개 섹션 (2열 배치)
sections_en = [
    "1. Executive Summary",
    "2. Area & Timeframe",
    "3. Detected Assets",
    "4. Recent Changes",
    "5. Historical Pattern",
    "6. Co-occurrence Analysis",
    "7. Confidence & Limitations",
    "8. Recommendations",
]
for i, sec in enumerate(sections_en):
    col = i // 4
    row = i % 4
    xx = rep_x + 0.3 + col * (rep_w/2 - 0.15)
    yy = rep_y + rep_h - 0.7 - row * 0.32
    ax.text(xx, yy, sec, fontsize=8.2, color=TEXT)

# 하단 강조 문구
ax.text(rep_x + rep_w/2, rep_y + 0.18,
        "→ 좌표·수치·타임스탬프는 raw 값 그대로 삽입",
        ha="center", fontsize=8, color=MUTED, style="italic")

arrow_down(6.5, 9.0, 8.5)


# ═══════════════════════════════════════════════════════════════
# [4단계] 2차 LLM 호출 — 한국어 무오염 번역
# ═══════════════════════════════════════════════════════════════
stage_bg(5.0, 3.5, "4단계",
         "동일 LLM 인스턴스로 한국어 번역 — 좌표·수치·타임스탬프는 원본 그대로 유지")

# 좌: 같은 LLM 인스턴스 + 2차 배지
llm2_x, llm2_y = 0.9, 6.5
llm2_w, llm2_h = 3.2, 1.2
ax.add_patch(FancyBboxPatch((llm2_x, llm2_y), llm2_w, llm2_h,
                             boxstyle="round,pad=0.06",
                             facecolor=LLM, edgecolor="#7f1d1d", lw=1.5))
ax.text(llm2_x + llm2_w/2, llm2_y + 0.8, "LLM (동일 인스턴스)",
        ha="center", fontsize=10.5, fontweight="bold", color="white")
ax.text(llm2_x + llm2_w/2, llm2_y + 0.4, "한국어 번역",
        ha="center", fontsize=8.5, color="white", style="italic")

ax.add_patch(Circle((llm2_x + llm2_w - 0.15, llm2_y + llm2_h - 0.05), 0.28,
                    facecolor="white", edgecolor=LLM, lw=1.8))
ax.text(llm2_x + llm2_w - 0.15, llm2_y + llm2_h - 0.05, "2",
        ha="center", va="center", fontsize=12, fontweight="bold", color=LLM)

# 하단 콜아웃: 왜 영문→한국어 2단계로 하는지 요약
call_x, call_y = 0.5, 5.15
call_w, call_h = 4.0, 1.15
ax.add_patch(FancyBboxPatch((call_x, call_y), call_w, call_h,
                             boxstyle="round,pad=0.05",
                             facecolor="#fef9c3",
                             edgecolor="#ca8a04", lw=1.3))
ax.text(call_x + call_w/2, call_y + call_h - 0.2,
        "왜 영문 → 한국어 2단계?",
        ha="center", fontsize=9, fontweight="bold", color="#854d0e")
ax.text(call_x + 0.2, call_y + call_h - 0.5,
        "· 정형 출력·도메인 규칙 준수율이 영문에서 안정적",
        fontsize=7.5, color="#713f12")
ax.text(call_x + 0.2, call_y + call_h - 0.75,
        "· 번역만 좁게 잠가 좌표·수치·타임스탬프 raw 값 보존",
        fontsize=7.5, color="#713f12")
ax.text(call_x + 0.2, call_y + call_h - 1.0,
        "· 동일 인스턴스 재사용 → 모델·가중치 재로드 없음",
        fontsize=7.5, color="#713f12")

# 화살표 → 번역 결과
arrow_right(llm2_x + llm2_w + 0.1, llm2_y + llm2_h/2, 5.6, color=LLM, lw=2)

# 우: 영문 → 한국어 대조 카드
comp_x, comp_y = 5.6, 5.15
comp_w, comp_h = 6.7, 3.15
ax.add_patch(FancyBboxPatch((comp_x, comp_y), comp_w, comp_h,
                             boxstyle="round,pad=0.04",
                             facecolor="white", edgecolor=LLM, lw=1.3))

# 헤더
ax.text(comp_x + comp_w/4, comp_y + comp_h - 0.22, "Before (영문)",
        ha="center", fontsize=9, fontweight="bold", color=MUTED)
ax.text(comp_x + 3*comp_w/4, comp_y + comp_h - 0.22, "After (한국어)",
        ha="center", fontsize=9, fontweight="bold", color=LLM)
ax.plot([comp_x + comp_w/2, comp_x + comp_w/2],
        [comp_y + 0.9, comp_y + comp_h - 0.35],
        color=STAGE_BD, lw=0.8)

# 3개 예시 대조 라인
pairs = [
    ("1. Executive Summary",           "1. 요약"),
    ("4. Recent Changes",              "4. 최근 변화 사항"),
    ("\"3 new tanks @ 37.5765,126.9680\"",
     "\"신규 전차 3대 @ 37.5765,126.9680\""),
]
for i, (en, ko) in enumerate(pairs):
    yy = comp_y + comp_h - 0.55 - i * 0.4
    ax.text(comp_x + 0.2, yy, en, fontsize=7.8, color=TEXT)
    ax.text(comp_x + comp_w/2, yy, "→", ha="center", fontsize=11,
            color=LLM, fontweight="bold")
    ax.text(comp_x + comp_w/2 + 0.15, yy, ko, fontsize=7.8, color=TEXT)

# 보존 태그 (하단)
preserve_y = comp_y + 0.4
ax.text(comp_x + 0.2, preserve_y + 0.28,
        "원본 그대로 유지 (LLM이 손대지 않음):",
        fontsize=8, fontweight="bold", color="#166534")
tags = ["섹션 번호", "좌표", "수치", "타임스탬프"]
for i, tag in enumerate(tags):
    tx = comp_x + 0.3 + i * 1.55
    ax.add_patch(FancyBboxPatch((tx, preserve_y - 0.04), 1.4, 0.28,
                                 boxstyle="round,pad=0.02",
                                 facecolor=REPORT_LIGHT, edgecolor=REPORT, lw=0.8))
    ax.text(tx + 0.7, preserve_y + 0.1, tag, ha="center",
            fontsize=7.8, color="#166534", fontweight="bold")

arrow_down(6.5, 5.0, 4.5)


# ═══════════════════════════════════════════════════════════════
# [5단계] 최종 판독보고서 → Report DB
# ═══════════════════════════════════════════════════════════════
stage_bg(1.5, 3.0, "5단계",
         "최종 한국어 판독보고서 산출 → Report DB에 회차별 저장")

# 좌: 최종 보고서 문서 아이콘
doc_x, doc_y = 1.5, 1.9
doc_w, doc_h = 5.4, 2.0
# 문서 그림자
ax.add_patch(Rectangle((doc_x + 0.1, doc_y - 0.1), doc_w, doc_h,
                        facecolor="#d1d5db", edgecolor="none"))
# 문서 본체
ax.add_patch(Rectangle((doc_x, doc_y), doc_w, doc_h,
                        facecolor="white", edgecolor=REPORT, lw=1.8))
# 접힌 모서리 (문서 느낌)
corner = [[doc_x + doc_w - 0.4, doc_y + doc_h],
          [doc_x + doc_w, doc_y + doc_h - 0.4],
          [doc_x + doc_w, doc_y + doc_h]]
ax.add_patch(patches.Polygon(corner, facecolor=REPORT_LIGHT,
                              edgecolor=REPORT, lw=0.8))

# 문서 제목
ax.text(doc_x + doc_w/2 - 0.15, doc_y + doc_h - 0.28, "판독보고서 (한국어)",
        ha="center", fontsize=10.5, fontweight="bold", color=REPORT)
ax.plot([doc_x + 0.2, doc_x + doc_w - 0.2],
        [doc_y + doc_h - 0.48, doc_y + doc_h - 0.48],
        color=REPORT, lw=0.5, alpha=0.6)

# 8개 섹션 한글 목록
mini_secs = [
    "1. 요약",
    "2. 대상 지역 및 기간",
    "3. 탐지 자산 목록",
    "4. 최근 변화 사항",
    "5. 과거 이력 및 반복 패턴",
    "6. 자산 공출현 분석",
    "7. 신뢰도 및 한계",
    "8. 후속 조치 권장",
]
for i, s in enumerate(mini_secs):
    col = i // 4
    row = i % 4
    xx = doc_x + 0.3 + col * (doc_w/2 - 0.2)
    yy = doc_y + doc_h - 0.7 - row * 0.28
    ax.text(xx, yy, s, fontsize=7.8, color=TEXT)

# 화살표 → Report DB
arrow_right(doc_x + doc_w + 0.15, doc_y + doc_h/2, 8.7,
            color=REPORT, lw=2.2)
ax.text((doc_x + doc_w + 8.7)/2 + 0.15, doc_y + doc_h/2 + 0.22, "INSERT",
        ha="center", fontsize=9, color=REPORT, fontweight="bold")

# 우: Report DB 원기둥
cylinder(8.7, 2.1, 3.0, 0.4, DB, "Report DB", "회차별 보고서 저장")

# Report DB 아래 설명
ax.text(10.2, 1.75,
        "= 회차마다 보고서 1건씩 누적",
        ha="center", fontsize=8, color=MUTED, style="italic")

plt.tight_layout()
out = "/home/user/multi-source-intelligent-system/data/fig4_report_v2.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
