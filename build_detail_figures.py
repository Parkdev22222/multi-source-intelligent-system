"""특허 명세서 상세 도면 3종 — 각 핵심 모듈별.

도 2: Pairing Module 상세 (객체 페어링 변화 탐지)
도 3: GraphRAG 상세 (누적 인덱싱 + Local/Global 검색)
도 4: 판독보고서 생성 상세 (LLM 2회 호출)
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

IMG      = "#93c5fd"
SAM3     = "#0891b2"
PAIR     = "#ea580c"
GRAPH    = "#7c3aed"
LLM      = "#dc2626"
REPORT   = "#16a34a"
DB       = "#475569"
ARROW    = "#1e293b"
BG_STAGE = "#f8fafc"
TEXT     = "#111827"
SOFT     = "#e2e8f0"


def module(ax, x, y, w, h, label, sub=None, fc=SAM3, fs=12, ss=9, tc="white"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor="none")
    ax.add_patch(p)
    if sub:
        ax.text(x+w/2, y+h/2+0.20, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc)
        ax.text(x+w/2, y+h/2-0.28, sub, ha="center", va="center",
                fontsize=ss, color=tc)
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc)

def db(ax, x, y, w, h, label, sub=None, fc=DB):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#1e293b", lw=1.2, zorder=1))
    ax.add_patch(Ellipse((x+w/2, y+h), w, h*0.28, facecolor=fc, edgecolor="#1e293b", lw=1.2, zorder=2))
    ax.add_patch(Ellipse((x+w/2, y), w, h*0.28, facecolor=fc, edgecolor="#1e293b", lw=1.2, zorder=3))
    if sub:
        ax.text(x+w/2, y+h/2+0.15, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")
        ax.text(x+w/2, y+h/2-0.22, sub, ha="center", va="center", fontsize=8, color="white")
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")

def arrow_down(ax, x, y1, y2, color=ARROW, lw=2.5, label=None):
    a = FancyArrowPatch((x, y1), (x, y2),
                        arrowstyle="->,head_length=12,head_width=8",
                        color=color, lw=lw)
    ax.add_patch(a)
    if label:
        ax.text(x+0.25, (y1+y2)/2, label, fontsize=9, va="center",
                color=color, fontweight="bold")

def arrow_right(ax, x1, y, x2, color=ARROW, lw=2, label=None, above=True):
    a = FancyArrowPatch((x1, y), (x2, y),
                        arrowstyle="->,head_length=10,head_width=7",
                        color=color, lw=lw)
    ax.add_patch(a)
    if label:
        ax.text((x1+x2)/2, y + (0.25 if above else -0.35),
                label, ha="center", fontsize=9, color=color, fontweight="bold")

def stage(ax, y_bot, h, label, color=BG_STAGE):
    ax.add_patch(patches.Rectangle((0.3, y_bot), 11.4, h,
                                    facecolor=color, edgecolor="#e2e8f0", lw=1))
    ax.text(0.55, y_bot+h-0.28, label, fontsize=9,
            color="#64748b", fontweight="bold")


# ═══════════════════════════════════════════════════════════════
# 도 2: Pairing Module 상세
# ═══════════════════════════════════════════════════════════════
def build_fig2():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12); ax.set_ylim(0, 16.5); ax.axis("off")

    ax.text(6, 15.9, "도 2. Pairing Module 상세 (객체 페어링 변화 탐지)",
            ha="center", fontsize=15, fontweight="bold", color=TEXT)
    ax.text(6, 15.5,
            "두 시점 객체를 외형 임베딩과 안정 매칭으로 짝지어 5상태 정밀 분류",
            ha="center", fontsize=10, color="#475569", style="italic")

    # 입력
    stage(ax, 13.7, 1.3, "① 입력 (Sensor DB로부터 조회)")
    module(ax, 1.0, 13.9, 4.2, 1.0, "현재 프레임 탐지 (N개)",
           "detection_records", fc=IMG, tc=TEXT, fs=11, ss=8)
    module(ax, 6.8, 13.9, 4.2, 1.0, "과거 프레임 탐지 (M개)",
           "detection_records", fc=IMG, tc=TEXT, fs=11, ss=8)

    arrow_down(ax, 3.1, 13.7, 12.5)
    arrow_down(ax, 8.9, 13.7, 12.5)

    # ② 배경 제거 + 임베딩
    stage(ax, 10.9, 1.5, "② 마스크 기반 배경 제거 + 시각 임베딩")
    module(ax, 1.0, 11.3, 4.2, 1.1,
           "mask_rle → 배경 zeroing\n→ bbox crop → ViT/CLIP",
           "L2 정규화 임베딩 (N×D)", fc=PAIR, fs=10, ss=8)
    module(ax, 6.8, 11.3, 4.2, 1.1,
           "mask_rle → 배경 zeroing\n→ bbox crop → ViT/CLIP",
           "L2 정규화 임베딩 (M×D)", fc=PAIR, fs=10, ss=8)

    arrow_down(ax, 3.1, 10.9, 9.7)
    arrow_down(ax, 8.9, 10.9, 9.7)

    # ③ N×M 유사도 행렬
    stage(ax, 8.1, 1.5, "③ N×M 코사인 유사도 행렬 (단일 GEMM)")
    module(ax, 3.5, 8.5, 5.0, 1.1,
           "cur_embed × past_embedᵀ",
           "동일 클래스 필터 · score = 0.8·CLIP + 0.2·크기",
           fc=PAIR, fs=11, ss=8.5)

    arrow_down(ax, 6, 8.1, 6.9)

    # ④ Gale-Shapley
    stage(ax, 5.3, 1.5, "④ Gale-Shapley 지연 승인 안정 매칭")
    module(ax, 3.5, 5.7, 5.0, 1.1,
           "score > 0.5 후보 → 1:1 안정 매칭",
           "그리디 중복·교차 페어링 회피",
           fc=PAIR, fs=11, ss=8.5)

    arrow_down(ax, 6, 5.3, 4.4)

    # ⑤ 5상태 분류
    stage(ax, 2.5, 1.7, "⑤ 매칭 결과 + FOV 검증 → 5상태 분류")
    # 5개 상태 박스 (좌→우)
    states = [
        ("matched", "정지 (양쪽 존재)", PAIR),
        ("changed", "구조 변화 (고정형)", "#f59e0b"),
        ("new", "신규 (현재만·FOV내)", "#16a34a"),
        ("disappeared", "소실 (과거만·FOV내)", "#dc2626"),
        ("FOV 공백", "촬영 범위 밖", "#94a3b8"),
    ]
    xs = [0.6, 2.85, 5.1, 7.35, 9.6]
    for (name, desc, c), xx in zip(states, xs):
        module(ax, xx, 2.8, 2.05, 0.95, name, desc, fc=c, fs=10, ss=7.5)

    arrow_down(ax, 6, 2.5, 1.6)

    # ⑥ Pairing DB 저장
    db(ax, 4, 0.5, 4, 1.0, "Pairing DB", "insert_pairings_bulk()", fc=DB)

    plt.tight_layout()
    out = "/home/user/multi-source-intelligent-system/data/fig2_pairing_module.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════
# 도 3: GraphRAG 상세
# ═══════════════════════════════════════════════════════════════
def build_fig3():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12); ax.set_ylim(0, 16.5); ax.axis("off")

    ax.text(6, 15.9, "도 3. GraphRAG 상세 (누적 인덱싱 + Local·Global 검색)",
            ha="center", fontsize=15, fontweight="bold", color=TEXT)
    ax.text(6, 15.5,
            "LLM 호출 없이 결정론적으로 그래프에 누적하고, 시공간 컨텍스트를 압축 검색",
            ha="center", fontsize=10, color="#475569", style="italic")

    # ① 입력
    stage(ax, 13.7, 1.3, "① 입력")
    db(ax, 4.5, 13.85, 3.0, 1.05, "Pairing DB",
       "5상태 페어링 배치", fc=DB)

    arrow_down(ax, 6, 13.7, 12.9)

    # ② 격자 양자화 결정론 키 생성
    stage(ax, 11.2, 1.7, "② 격자 양자화 결정론 노드 키 생성 (LLM 호출 없음)")
    module(ax, 1.0, 11.7, 4.5, 1.0,
           "위경도 → 소수점 2자리 반올림",
           "loc:{lat:.2f},{lon:.2f}",
           fc=GRAPH, fs=10.5, ss=9)
    module(ax, 6.5, 11.7, 4.5, 1.0,
           "(class × loc) → asset 키",
           "asset:{class}:{loc}",
           fc=GRAPH, fs=10.5, ss=9)

    arrow_down(ax, 6, 11.2, 9.8)

    # ③ 노드·엣지 upsert
    stage(ax, 8.0, 3.0, "③ 노드/엣지 결정론 upsert (LLM 비호출)")
    module(ax, 0.8, 9.3, 3.4, 1.3,
           "Location Node",
           "loc:37.50,127.00", fc="#a78bfa", fs=10, ss=8)
    module(ax, 4.4, 9.3, 3.2, 1.3,
           "Asset Node",
           "new_count / matched_count\n/ changed_count / ...",
           fc="#a78bfa", fs=10, ss=7.5)
    module(ax, 7.8, 9.3, 3.4, 1.3,
           "Edge",
           "found_at (asset→loc)\nco_occurred_with (asset↔asset)",
           fc="#a78bfa", fs=10, ss=7.5)
    ax.text(6, 8.4,
            "동일 키 존재 → 카운터 +1 (upsert) · 신규 → INSERT",
            ha="center", fontsize=9, color=TEXT, style="italic")

    arrow_down(ax, 6, 8.0, 6.8)

    # ④ Louvain 군집 탐지
    stage(ax, 5.3, 1.5, "④ Louvain 군집화 (자산 공출현 패턴 자동 발견)")
    module(ax, 1.0, 5.7, 4.5, 1.1,
           "co_occurred_with 가중치\n→ Louvain 알고리즘",
           "군집 자동 발견",
           fc=GRAPH, fs=10, ss=8.5)
    module(ax, 6.5, 5.7, 4.5, 1.1,
           "member_summary 생성",
           "LLM 호출 없이 결정론 통계",
           fc=GRAPH, fs=10, ss=8.5)
    arrow_right(ax, 5.5, 6.25, 6.5, color=GRAPH, lw=1.8)

    arrow_down(ax, 6, 5.3, 4.5)

    # ⑤ Graph DB 저장
    stage(ax, 3.5, 1.0, "⑤ Graph DB에 저장")
    db(ax, 3.0, 3.6, 6.0, 0.9, "Graph DB",
       "graph_entities / graph_relations / graph_communities", fc=GRAPH)

    arrow_down(ax, 6, 3.5, 2.5)

    # ⑥ Local + Global 검색 (보고서 생성 시)
    stage(ax, 0.5, 2.0, "⑥ 보고서 생성 시 검색 (Local + Global)")
    module(ax, 0.6, 1.4, 3.4, 1.0,
           "Local Search",
           "반경 R 자산 이력",
           fc="#5b21b6", fs=10.5, ss=8.5)
    module(ax, 4.3, 1.4, 3.4, 1.0,
           "Global Search",
           "중첩 커뮤니티 요약",
           fc="#5b21b6", fs=10.5, ss=8.5)
    module(ax, 8.0, 1.4, 3.4, 1.0,
           "~500 토큰 컨텍스트",
           "LLM 프롬프트 prepend용",
           fc="#5b21b6", fs=10.5, ss=8.5)
    arrow_right(ax, 4.0, 1.9, 4.3, color="#5b21b6", lw=1.8)
    arrow_right(ax, 7.7, 1.9, 8.0, color="#5b21b6", lw=1.8)

    plt.tight_layout()
    out = "/home/user/multi-source-intelligent-system/data/fig3_graphrag.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════
# 도 4: 판독보고서 생성 상세 (LLM 2회 호출)
# ═══════════════════════════════════════════════════════════════
def build_fig4():
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.set_xlim(0, 12); ax.set_ylim(0, 15); ax.axis("off")

    ax.text(6, 14.4, "도 4. 판독보고서 자율 생성 상세 (LLM 2회 호출)",
            ha="center", fontsize=15, fontweight="bold", color=TEXT)
    ax.text(6, 14.0,
            "그래프 컨텍스트 + 변화 객체를 LLM에 주입, 영문→한국어 무오염 번역",
            ha="center", fontsize=10, color="#475569", style="italic")

    # ① 두 입력 소스 (병렬)
    stage(ax, 12.2, 1.3, "① 입력 소스 두 갈래")
    db(ax, 1.0, 12.4, 4.5, 1.0,
       "Graph DB (Retriever)",
       "Local + Global → 압축 컨텍스트", fc=GRAPH)
    db(ax, 6.5, 12.4, 4.5, 1.0,
       "Pairing DB",
       "변화 객체 (new + disappeared)", fc=DB)

    arrow_down(ax, 3.25, 12.2, 10.7)
    arrow_down(ax, 8.75, 12.2, 10.7)

    # ② 프롬프트 조립
    stage(ax, 8.7, 2.0, "② 프롬프트 조립 (시스템 + 사용자)")
    # 시스템 프롬프트
    module(ax, 0.8, 9.5, 5.0, 1.1,
           "시스템 프롬프트 (강제)",
           "· \"DISAPPEARED ≠ destroyed\"\n· 8개 서술 섹션 구조 강제",
           fc=LLM, fs=10.5, ss=8.5)
    # 사용자 프롬프트
    module(ax, 6.2, 9.5, 5.0, 1.1,
           "사용자 프롬프트",
           "① 컨텍스트 prepend (~500 토큰)\n② 변화 객체 목록 (신뢰도 순)",
           fc="#f97316", fs=10.5, ss=8.5)

    arrow_down(ax, 6, 8.7, 7.8)

    # ③ LLM 1차 호출 (영문)
    stage(ax, 6.1, 1.6, "③ [1차 호출] EXAONE (영문 IMINT 보고서)")
    module(ax, 2.5, 6.5, 7.0, 1.1,
           "EXAONE LLM (영문 생성)",
           "정형 8섹션 구조 준수 · IMINT 서술 스타일",
           fc=LLM, fs=11.5, ss=9)

    arrow_down(ax, 6, 6.1, 5.2)

    # ④ LLM 2차 호출 (한국어 번역)
    stage(ax, 3.5, 1.6, "④ [2차 호출] 동일 인스턴스로 한국어 무오염 번역")
    module(ax, 2.5, 3.9, 7.0, 1.1,
           "EXAONE LLM (한국어 번역)",
           "섹션 헤더·좌표·수치·타임스탬프 · 바이패스",
           fc="#b91c1c", fs=11.5, ss=8.5)

    arrow_down(ax, 6, 3.5, 2.6)

    # ⑤ 최종 판독보고서
    stage(ax, 1.5, 1.1, "⑤ 최종 판독보고서 산출")
    module(ax, 1.5, 1.75, 5.5, 0.85,
           "판독보고서 (정형 8섹션)",
           "지휘 결심에 즉시 활용", fc=REPORT, fs=11, ss=8.5)
    db(ax, 8.0, 1.65, 2.8, 1.0, "Report DB",
       "영속 저장", fc=DB)
    arrow_right(ax, 7.0, 2.15, 8.0, color=REPORT, lw=2, label="INSERT")

    plt.tight_layout()
    out = "/home/user/multi-source-intelligent-system/data/fig4_report_generation.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p2 = build_fig2()
    p3 = build_fig3()
    p4 = build_fig4()
    print(f"Saved:\n  {p2}\n  {p3}\n  {p4}")
