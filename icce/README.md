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
| C2 | A 20k-parameter learned pairing head with cross-frame co-located evidence, supervised for free from CD masks, replaces the hand-tuned CLIP+geo heuristic | E1, E2, E5 |
| C3 | Graph grounding measurably reduces report hallucination, over and above flat RAG **and over a VLM shown the images directly** | E3, E4 |
| C4 | The added accuracy costs sub-millisecond latency in a pipeline dominated by segmentation and generation | E6 |

### Who we are actually competing with

This matters more than any single number. Methods carry a **tier**, and tables
group by tier rather than printing one undifferentiated block:

- **`supervised`** — trained on this dataset's own training split (BIT,
  ChangeFormer, RSICCformer, Chg2Cap). These are specialists with in-domain
  labels. They are printed as a **reference ceiling**, not as our opponent, and
  we expect to lose to them. Pretending otherwise in either direction is
  misleading.
- **`zero-shot`** — no training on this dataset. This is our league: AnyChange
  for detection, and a VLM shown both images for captioning.
- **`ours`** — this work, plus its ablations.

**What we do not claim.** We do not claim SOTA pixel-level change detection,
and we do not claim to beat models trained on LEVIR-CC captions at n-gram
overlap. The claims are: (i) among methods that have *not* been trained on the
benchmark, ours is stronger; (ii) the reports it writes state more correct
facts and fewer invented ones, which is what a monitoring service is actually
judged on; and (iii) it costs little to run.

The `vlm_direct` row carries most of the weight here. "Just show both pictures
to a capable VLM" is what a practitioner would try first, so the whole pipeline
has to justify itself against it. It is also where our design should genuinely
win: a VLM reading 256 px aerial tiles has no instance-level grounding, cannot
count roofs reliably, and produces confident well-formed prose anyway.
Change-Fact-Score is built to catch exactly that.

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
vlm_direct     EXTERNAL BASELINE: a VLM shown both images and none of our pipeline
llm_raw        + LLM, unaggregated detection dump
llm_struct     + aggregated change inventory      -> isolates aggregation
llm_flat_rag   + top-k retrieved past observations-> isolates "any history helps"
llm_graphrag   + entity history and community summaries (ours)
```

`llm_flat_rag` is the row that matters most to a reviewer: without it, a
GraphRAG gain could just be "retrieval helps". `vlm_direct` is the row that
matters most to a reader deciding whether to adopt any of this.

### Cross-frame co-located evidence

The dominant error of detect-then-pair: an object the detector missed in the
past frame becomes a false "new building" in the current one. Pairing cannot
fix it, because there is nothing to pair with — the evidence that the object
was already there lives in the past frame's *pixels*.

`icce/convert/cross_frame.py` crops each detection's footprint from both frames
and measures CLIP cosine, pixel difference, cross-correlation and signed
edge-density change, and hands all four to the learned verifier. On synthetic
scenes it is worth **+3.1 instance F1** on its own, without CLIP.

Ablating it means **retraining without it** (`--no-cross-frame`), not zeroing
it at inference: zero is not a neutral input after standardisation, so zeroing
measures a broken model rather than the feature's contribution.

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

### Start with the pilot

```bash
DOMAIN=urban STAGES=pilot bash scripts/icce_runpod.sh
```

Processes a few dozen tiles instead of a few hundred and produces the same
tables. Read `results/pilot_levir_cd/cd_results.json` and
`results/pilot_levir_cc/report_results_caption.json` **before** committing a
week of GPU time. If the pilot says the numbers are far off what the paper
needs, that is a decision to make on day 3, not day 9.

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

## 4. Evaluation integrity

`icce/eval/integrity.py` runs before every evaluation and **aborts** on:

- a pair id present in both the training and evaluation caches
- a parent scene straddling train and test (two 256 px crops of one 1024 px
  tile are near-duplicates)
- thresholds selected on the split being evaluated
- caption style exemplars drawn from anything but train

`--allow-leakage` exists for single-split debugging and stamps the result JSON
as unclean so a leaked run cannot be mistaken for a valid one. There is no way
to silently produce a leaked number.

The point is not bureaucratic. The only ways to guarantee a win are to rig the
evaluation or to actually be better; these checks close off the first so the
numbers that come out mean something.

## 5. Baseline numbers

`icce/eval/baselines.json` holds the published comparisons with `verified:
false` and null values. **Fill each one by reading it off the cited paper**,
then flip `verified` to `true`. `icce/eval/tables.py` refuses to typeset an
unverified row and emits a `%% TODO` comment instead, so an unchecked number
cannot reach the submitted PDF.

---

### External model outputs

Run a published model's released checkpoint on LEVIR-CC test, dump its
captions, then:

```bash
python -m icce.eval.score_external \
    --predictions external/chg2cap_levircc_test.json --name Chg2Cap \
    --cache data/cache/levir_cc_test --out results/external
```

This is the highest-value single experiment left. Change-Fact-Score is our own
metric, and a table where we are the only system on it is worth little — a
reviewer will say we invented a yardstick and reported that we are tall.
Scoring a specialist's own outputs with the same yardstick removes that
objection, and is the most likely route to the paper's central result: a model
that wins decisively on BLEU while stating fewer correct facts.

## 6. Layout

```
icce/
  datasets/      LEVIR-CD / LEVIR-CC / WHU-CD / S2Looking loaders + registry
  convert/       georef (invertible px<->latlon), MSIS metadata, mask->instances
  metrics/       pixel CD, instance CD, caption (BLEU/ROUGE/CIDEr/METEOR), CFS
  pairing_head/  features, model, assignment, detection cache, train, infer
  report/        evidence, prompts, flat-RAG baseline, GraphRAG driver, LLM
  report/        ... plus vlm.py (image-conditioned external baseline)
  eval/          cache_detections, run_cd_eval, run_report_eval,
                 run_efficiency, score_external, integrity, tables,
                 baselines.json
scripts/icce_runpod.sh
```

---

## 7. Domain switch

```bash
DOMAIN=urban     # LEVIR/WHU building vocabulary (paper default)
DOMAIN=disaster  # xBD-style damage vocabulary
DOMAIN=military  # the original IMINT vocabulary, unchanged
DOMAIN_CLASSES="building,road,vehicle"   # explicit override
```

`DOMAIN` also selects the wording of the GraphRAG context block; graph
contents and retrieval are identical across domains.
