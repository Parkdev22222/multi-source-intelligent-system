# ICCE-Asia 2026 experiment harness

Everything in `icce/` exists to turn MSIS into a paper. It is additive: the
production pipeline (`pipeline.py`, `web_api.py`, `src/`) behaves exactly as
before unless you set `DOMAIN`.

---

## 1. What the paper claims

**Working title.** *Graph-Grounded Change Reporting for Consumer Satellite
Monitoring: Learned Instance Pairing and Factuality-Aware Evaluation*

**Setting.** A consumer-facing urban monitoring service: a subscriber pins a
neighbourhood, and after each satellite revisit the service returns a written
interpretation of what changed — new construction, demolition, redevelopment.
This is the civil re-framing of the existing military IMINT pipeline, and it is
what makes the work a fit for IEEE CTSoc / IEIE rather than a defence venue.

**Contributions.**

| | Claim | Evidence |
|---|---|---|
| C1 | End-to-end pipeline coupling open-vocabulary instance change detection with a persistent spatio-temporal knowledge graph for report generation | System description + E1–E6 |
| C2 | A 20k-parameter learned pairing head, supervised for free from CD masks, replaces the hand-tuned CLIP+geo heuristic | E1, E2, E5 |
| C3 | Graph grounding measurably reduces report hallucination, over and above flat RAG | E3, E4 |
| C4 | The added accuracy costs sub-millisecond latency in a pipeline dominated by segmentation and generation | E6 |

**What we do not claim.** We do not claim SOTA pixel-level change detection.
Supervised specialists trained on LEVIR-CD will beat an open-vocabulary
detector, and the paper says so. The claim is that an *open-vocabulary,
report-producing* system gets close enough to be useful, and that the report
it writes is more factual than the alternatives.

---

## 2. Experiments

| ID | Question | Dataset | Metrics | Table |
|----|----------|---------|---------|-------|
| E1 | How good is our change detection? | LEVIR-CD test (128) | P/R/F1/IoU pixel, P/R/F1 instance | `table_levir_cd_pixel.tex`, `..._instance.tex` |
| E2 | Does it transfer without retraining? | WHU-CD test | same | `table_whu_cd_pixel.tex` |
| E3 | Is the generated text competitive with change-captioning models? | LEVIR-CC test | BLEU-1/4, METEOR, ROUGE-L, CIDEr-D | `table_levir_cc_caption.tex` |
| E4 | Does graph grounding reduce hallucination? | LEVIR-CC test | CFS-P/R/F1, HalRate, ChgAcc, CountMAE | `table_factuality_caption.tex` |
| E5 | Does better pairing produce better reports? | LEVIR-CC test | CFS with pairing swapped, grounding fixed | `table_factuality_report.tex` |
| E6 | Can this be deployed? | LEVIR-CC test | ms/tile, peak GPU MB per stage | `table_efficiency.tex` |

### The grounding ladder (E4)

Each condition sees strictly more structure than the last, over an identical
evidence object, so a gain is attributable to grounding and nothing else:

```
template       no LLM, deterministic prose        -> zero hallucination by construction
llm_raw        + LLM, unaggregated detection dump
llm_struct     + aggregated change inventory      -> isolates aggregation
llm_flat_rag   + top-k retrieved past observations-> isolates "any history helps"
llm_graphrag   + entity history and community summaries (ours)
```

`llm_flat_rag` is the row that matters most to a reviewer: without it, a
GraphRAG gain could just be "retrieval helps".

### Change-Fact-Score

BLEU rewards phrasing overlap and is blind to a report confidently announcing
construction that never happened. CFS reduces a report to atomic claims
`(direction, object_class, count)` with a deterministic rule-based extractor,
then scores them against claims extracted the same way from the five human
references (majority vote, >= 2 annotators) and against the GT mask's instance
count. No LLM sits inside the metric, so the model being evaluated cannot game
it. See `icce/metrics/change_fact.py`.

---

## 3. Running it

### Datasets

```bash
export MSIS_DATA_ROOT=data/benchmarks
# LEVIR-CD  : https://chenhao.in/LEVIR/  -> $MSIS_DATA_ROOT/LEVIR-CD/<split>/{A,B,label}
# LEVIR-CC  : https://github.com/Chen-Yang-Liu/RSICC
#             -> $MSIS_DATA_ROOT/LEVIR-CC/{images/<split>/{A,B}, LevirCCcaptions.json}
# WHU-CD    : http://gpcv.whu.edu.cn/data/building_dataset.html
# S2Looking : https://github.com/S2Looking/Dataset   (optional, hardest split)
```

Loaders tolerate the common mirror layouts (`A`/`im1`/`before`, `label`/`OUT`/`gt`)
and raise with the exact expected tree when something is missing.

### Full run

```bash
DOMAIN=urban bash scripts/icce_runpod.sh
```

Stages are independently resumable:

```bash
STAGES="cache train" bash scripts/icce_runpod.sh
STAGES="caption"     bash scripts/icce_runpod.sh
```

### GPU-free sanity check

```bash
PYTHONPATH=. python tests/test_pairing_head.py
```

Builds synthetic scenes carrying the four real failure modes (registration
drift, look-alike terraces, detector false positives, missed detections),
trains a head and asserts it clears the production heuristic. Currently
0.851 vs 0.566 instance F1.

---

## 4. Baseline numbers

`icce/eval/baselines.json` holds the published comparisons with `verified:
false` and null values. **Fill each one by reading it off the cited paper**,
then flip `verified` to `true`. `icce/eval/tables.py` refuses to typeset an
unverified row and emits a `%% TODO` comment instead, so an unchecked number
cannot reach the submitted PDF.

---

## 5. Layout

```
icce/
  datasets/      LEVIR-CD / LEVIR-CC / WHU-CD / S2Looking loaders + registry
  convert/       georef (invertible px<->latlon), MSIS metadata, mask->instances
  metrics/       pixel CD, instance CD, caption (BLEU/ROUGE/CIDEr/METEOR), CFS
  pairing_head/  features, model, assignment, detection cache, train, infer
  report/        evidence, prompts, flat-RAG baseline, GraphRAG driver, LLM
  eval/          cache_detections, run_cd_eval, run_report_eval,
                 run_efficiency, tables, baselines.json
scripts/icce_runpod.sh
```

---

## 6. Domain switch

```bash
DOMAIN=urban     # LEVIR/WHU building vocabulary (paper default)
DOMAIN=disaster  # xBD-style damage vocabulary
DOMAIN=military  # the original IMINT vocabulary, unchanged
DOMAIN_CLASSES="building,road,vehicle"   # explicit override
```

`DOMAIN` also selects the wording of the GraphRAG context block; graph
contents and retrieval are identical across domains.
