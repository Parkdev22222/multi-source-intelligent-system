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
| C3 | Reports grounded in the pipeline's instance evidence are far more factual than a VLM shown the images directly; graph grounding adds to that only at neighbourhood level, where it halves the counting error of flat retrieval | E4 (VLM), E7 (graph, preliminary) |
| C4 | The added accuracy costs 1.5 ms/tile -- 0.01% of a pipeline dominated by segmentation | E6 |

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

**Where `ours` actually sits on LEVIR-CD, stated plainly.** The detector (SAM3)
has never seen LEVIR-CD, but the 20k-parameter pairing head is trained on
LEVIR-CD's own training split, from mask-derived labels. By the definition
above that is not `zero-shot`, and the paper must not print it inside that
group. It is a third position -- an open-vocabulary detector with a small head
supervised for free from the benchmark's masks -- and E2 (WHU-CD, head trained
on LEVIR-CD only) is the experiment that carries the genuinely zero-shot claim.

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
| E4 | Does graph grounding reduce hallucination? | LEVIR-CC test | CFS-P/R/F1, ChgAcc, CountMAE (HalRate = 1 - CFS-P, prose only) | `table_factuality_caption.tex` |
| E5 | Does better pairing produce better reports? | LEVIR-CC test | CFS with pairing swapped, grounding fixed | `table_factuality_report.tex` |
| E6 | Can this be deployed? | LEVIR-CC test | ms/tile, peak GPU MB per stage | `table_efficiency.tex` |
| E7 | Does graph grounding help where retrieval is lossy? | LEVIR-CC, whole neighbourhoods | CFS-P/R/F1, scene CountMAE | `scene_results.json` |

### Change detection (E1)

**Measured, full LEVIR-CD test split (128 tiles), head trained on all 445
training tiles. Integrity clean: pair ids and parent scenes disjoint from
train, thresholds selected on `val`.**

| method | P | R | F1 | IoU | inst-F1 |
|---|---|---|---|---|---|
| AnyChange, zero-shot | 13.70 | 83.00 | 23.40 | -- | -- |
| AnyChange-H, 3-point query *(human in the loop)* | 28.50 | 81.10 | 42.20 | -- | -- |
| geo-only | 14.34 | 64.73 | 23.47 | 13.30 | 35.62 |
| heuristic (production) | 16.08 | 62.67 | 25.59 | 14.67 | 35.72 |
| learned head, no verifier | 12.49 | 76.66 | 21.48 | 12.03 | 31.25 |
| learned head, no cross-frame | 47.26 | 51.52 | 49.30 | 32.71 | 50.56 |
| **learned head (ours)** | **56.22** | **51.97** | **54.01** | **36.99** | **54.28** |

The row worth pointing at is not ours, it is the cluster: geo-only 23.47,
heuristic 25.59 and AnyChange 23.40 all sit within two points of each other, at
precision 13-16 with high recall. Methods that do not learn to pair land in the
same place regardless of how good their segmentation is, and the 20k-parameter
head is what leaves that cluster -- 2.3x AnyChange's F1, and above even its
three-point human-assisted setting.

Both ablations are load-bearing. Without the verifier the system is worse than
geo-only (21.48): the head proposes matches and something has to reject them.
Cross-frame evidence, retrained without rather than zeroed, is worth +4.71
pixel F1.

We do not beat the supervised ceiling (86-91 F1) and do not claim to. See the
tier note above for where `ours` actually sits: the detector never saw
LEVIR-CD, the pairing head trained on its training split, and E2 is what
carries the zero-shot claim.

### The grounding ladder (E4)

Each condition sees strictly more structure than the last, over an identical
evidence object, so a gain is attributable to grounding and nothing else:

```
template       no LLM, deterministic prose        -> the detector's own error floor
vlm_direct     EXTERNAL BASELINE: a VLM shown both images and none of our pipeline
llm_raw        + LLM, unaggregated detection dump
llm_struct     + aggregated change inventory      -> isolates aggregation
llm_flat_rag   + top-k retrieved past observations-> isolates "any history helps"
llm_graphrag   + entity history and community summaries (ours)
```

`llm_flat_rag` is the row that matters most to a reviewer: without it, a
GraphRAG gain could just be "retrieval helps". `vlm_direct` is the row that
matters most to a reader deciding whether to adopt any of this.

`template` is not a zero-hallucination row, and the paper must not call it one.
It invents no *sentences*, but it states whatever the detector found, so every
detection error becomes an unsupported claim: measured HalRate is 30.9, not 0.
That is the point of the row. It fixes the error floor that our own detector
imposes, so the LLM conditions are read as *what generation adds on top of it*.

**Measured, 128 crops of 8 neighbourhoods (EXAONE-4.0-32B, Qwen2.5-VL-7B):**

| condition | CFS-P | CFS-R | CFS-F1 | ChgAcc | BLEU-4 |
|---|---|---|---|---|---|
| template | 69.12 | 40.52 | **51.09** | 85.16 | 25.85 |
| vlm_direct | 48.97 | 30.60 | 37.67 | **50.00** | 3.87 |
| llm_raw | 62.84 | 40.09 | 48.95 | 80.47 | 24.22 |
| llm_struct | 67.63 | 40.52 | 50.67 | 85.16 | 25.54 |
| llm_flat_rag | 67.63 | 40.52 | 50.67 | 85.16 | 26.56 |
| llm_graphrag | 67.63 | 40.52 | 50.67 | 85.16 | 27.66 |

Two things in this table have to be reported, not buried.

**`vlm_direct` decides 50.00 -- chance -- on whether anything changed at all**,
while writing fluent prose about it. That is the result the pipeline exists to
justify, and the one CFS was built to expose. BLEU sees part of it (3.87); only
ChgAcc shows how complete the failure is.

**The top three rows are identical to four decimals, and that is a property of
the design, not a bug.** All three receive the same change inventory for the
crop being described, and CFS scores only claims about that crop; retrieved
history concerns *other* crops, so it can move wording but never a claim. Of
128 generations, 77 differed between conditions and 0 differed in extracted
claims. A crop-level ladder therefore cannot measure C3's graph term, and
saying so is the honest reading -- the alternative is to present three copies
of one number as an ablation.

### Neighbourhood-level grounding (E7)

E7 (`icce.eval.run_scene_eval`) asks the same question where the answer can
differ: one report per neighbourhood, over all 16 crops of a LEVIR-CD tile. The
conditions stop being nested and become three *representations* of the same 16
observations -- full concatenation, top-k retrieval, graph aggregate. The
module docstring states this departure; a paper using the table must too.

**Measured, 8 neighbourhoods, mean 104.9 GT instances each:**

| condition | CFS-F1 | scene CountMAE |
|---|---|---|
| template | **41.18** | **20.25** |
| llm_struct | 35.14 | 81.50 |
| llm_flat_rag | 26.23 | 71.38 |
| llm_graphrag | 32.43 | 41.00 |

Graph aggregation halves flat retrieval's counting error (41.0 vs 71.4) and
beats raw concatenation by more (81.5), which is the predicted effect: top-k
truncation drops crops, and a count cannot be recovered from crops that were
never retrieved. But **no LLM condition beats the deterministic template on
either axis**, so this supports "graph structure preserves counts better than
retrieval", not "generation improves factuality". With n=8 and GT spanning
12-216 instances it is a preliminary result and must be labelled one.

### Cross-frame co-located evidence

The dominant error of detect-then-pair: an object the detector missed in the
past frame becomes a false "new building" in the current one. Pairing cannot
fix it, because there is nothing to pair with — the evidence that the object
was already there lives in the past frame's *pixels*.

`icce/convert/cross_frame.py` crops each detection's footprint from both frames
and measures CLIP cosine, pixel difference, cross-correlation and signed
edge-density change, and hands all four to the learned verifier.

Ablating it means **retraining without it** (`--no-cross-frame`), not zeroing
it at inference: zero is not a neutral input after standardisation, so zeroing
measures a broken model rather than the feature's contribution. Retrained on
the full LEVIR-CD training split it is worth **+4.71 pixel F1 and +3.72
instance F1** (see E1 below), which supersedes the +3.1 previously measured on
synthetic scenes.

### Change-Fact-Score

BLEU rewards phrasing overlap and is blind to a report confidently announcing
construction that never happened. CFS reduces a report to atomic claims
`(direction, object_class, count)` with a deterministic rule-based extractor,
then scores them against claims extracted the same way from the five human
references (majority vote, >= 2 annotators) and against the GT mask's instance
count. No LLM sits inside the metric, so the model being evaluated cannot game
it. See `icce/metrics/change_fact.py`.

**HalRate is not an independent number.** It is defined as `1 - CFS-P`
(`change_fact.py:383`), so printing both in one table gives a reviewer two
columns carrying one measurement and invites the charge that the metric set was
padded. Report CFS-P/R/F1 in the table and quote HalRate only in prose, where
"one claim in three is unsupported" is easier to read than a precision.

### Deployment cost (E6)

Measured on one A100 80GB, 64 pairs, EXAONE-4.0-32B served by vLLM:

| stage | ms/tile | peak GPU MB |
|---|---|---|
| SAM3 detection + CLIP (2 frames) | 13392.05 | -- |
| pairing: heuristic (production) | 4.63 | 0.08 |
| pairing: learned head (ours, 19,781 params) | 6.08 | 9.39 |
| knowledge graph: index + retrieve | 74.73 | 8.29 |
| report LLM (batched) | 453.01 | 8.29 |

Segmentation is 96% of the ~13.9 s/tile budget. The learned head costs **1.45
ms more than the heuristic it replaces** -- 0.01% of the pipeline -- which is
what makes C2's accuracy gain free in practice. Quote 1.45 ms, not
"sub-millisecond": the difference is checkable straight off this table.

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

STAGES="cache train cd" bash scripts/icce_runpod.sh   # E1 exactly as reported
```

Resumption keys on `cache_info.json`, which is written after the embeddings.
A split whose `samples.jsonl` exists without it is a killed run and is redone,
because `load_cache` accepts a cache with no `embeddings.npz` on a warning and
then hands zeroed CLIP features to everything downstream. A dataset that was
never downloaded (WHU-CD needs a manual fetch) is skipped with a message rather
than aborting the pipeline.

Caching the full LEVIR-CD split is the long pole: 445 + 64 + 128 tiles at
~33 s each is about 5.5 h on one A100, after which training both heads and
scoring E1 takes ten minutes. `CC_SCENES` (default 8) sets how many whole
LEVIR-CC neighbourhoods are cached; the split is selected by scene rather than
by `--limit` because a spread sample leaves roughly one crop per tile and
starves the per-scene graph that E4 and E7 measure.

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
