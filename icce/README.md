# ICCE-Asia 2026 experiment harness

Everything in `icce/` exists to turn MSIS into a paper. It is additive: the
production pipeline (`pipeline.py`, `web_api.py`, `src/`) behaves exactly as
before unless you set `DOMAIN`.

---

## 1. What the paper claims

**Working title.** *Grounding LLM Change Reports in Foundation-Model Instance
Evidence for Consumer Satellite Monitoring*

**Target track.** ICCE-Asia *Artificial Intelligence and Machine Learning for
CE Applications* (AIM). The title and abstract name the stack --- segmentation
foundation model, CLIP, knowledge graph, LLM --- because that is what the track
is for. They do **not** name GraphRAG. An earlier title led with
"Graph-Grounded"; it was dropped when the full-scale run came in with
`llm_flat_rag` at 48.51 CFS-F1 against `llm_graphrag` at 48.32. E7 later gave
the graph a real result at neighbourhood scale (C6), but it is a counting
result, not a factuality one, and it does not carry a title.

**Setting.** A consumer-facing urban monitoring service: a subscriber pins a
neighbourhood, and after each satellite revisit the service returns a written
interpretation of what changed — new construction, demolition, redevelopment.
This is the civil re-framing of the existing military IMINT pipeline, and it is
what makes the work a fit for IEEE CTSoc / IEIE rather than a defence venue.

**Contributions.**

Ordered by how well the evidence supports them, which is also the order the
paper argues them in. The knowledge graph is part of the system but is
deliberately **not** a headline claim: see C6 and C7 and the E4/E7 sections for why.

| | Claim | Evidence |
|---|---|---|
| C1 | An LLM change-reporting pipeline for a CE service, built end to end on a segmentation foundation model, CLIP, a persistent spatio-temporal graph and an LLM, evaluated on the factuality of its prose rather than on segmentation overlap alone | System description + E1–E7 |
| C2 | **Factuality is won upstream of the LLM.** Over 1929 crops, top-$k$ retrieval and graph aggregation over the same evidence move CFS-F1 by <0.2 points in either direction; aggregating detections into an inventory is worth +3.1; improving the pairing that produced the evidence is worth +12.97, with unsupported claims falling 65.99% → 38.28%, at 1.45 ms/tile. Spend the budget on the evidence, not the prompt | E4 (full scale) + E5 + E6 |
| C3 | Pairing, not segmentation quality, is what separates the methods we can compare. Every non-learned pairing strategy lands in the same 23--26 F1 band at precision 13--16; a 20k-parameter head, supervised for free from CD masks and fed cross-frame co-located evidence, leaves it | E1 + its two ablations |
| C4 | That head transfers with no retraining and no threshold tuning to a domain it has never seen, cutting instance count error from 14.8 to 1.3 | E2 |
| C5 | Change-Fact-Score, a claim-level factuality metric with no LLM inside it, exposes a failure BLEU registers only faintly: a capable VLM shown both images writes fluent prose while deciding at chance (45.78 ChgAcc, i.e. below it) whether anything changed | E4 (`vlm_direct`) |
| C6 | Representation decides whether a generated neighbourhood report can count. Over 218 scenes, concatenating every crop into the prompt gives a scene count error of 458.5 instances and top-$k$ retrieval 167.9; graph aggregation holds it to 18.5 -- better than the deterministic template (26.2) and the only generated condition that beats it on any axis. The claim is about counting, not factuality overall: graph aggregation still trails the template on CFS-F1 | E7 (n=218) |
| C7 | *Reported, not claimed.* A crop-level grounding ladder cannot measure a graph's contribution -- structural, not a bug: 68.1% of generations differ in text between conditions and 1.3% differ in extracted claims. And the temporal half of the graph is untested, because both benchmarks carry exactly two timepoints | E4 (null), and a gap in the benchmarks |

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
E2 is now measured: 57.92 pixel F1 on a domain the head never saw, against
28.86 for the heuristic it replaces. See "Transfer without retraining" below.

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
| E5 | Does better pairing produce better reports? | LEVIR-CC test | CFS with pairing swapped, grounding fixed | `table_factuality_caption.tex` in two run dirs |
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

### Transfer without retraining (E2)

**Measured, full WHU-CD test split (690 pairs), the LEVIR-CD head applied
unchanged.** No retraining, and the thresholds are the ones the checkpoint
carries from LEVIR-CD's `val` split -- nothing was tuned on WHU-CD. Integrity
clean.

| method | P | R | F1 | IoU | inst-F1 | CountMAE |
|---|---|---|---|---|---|---|
| geo-only | 23.35 | 42.57 | 30.16 | 17.76 | 8.73 | 15.4 |
| heuristic (production) | 21.73 | 42.98 | 28.86 | 16.87 | 9.06 | 14.8 |
| learned head, no verifier | 12.13 | 53.50 | 19.78 | 10.98 | 4.56 | 40.4 |
| **learned head (ours)** | **72.89** | **48.05** | **57.92** | **40.77** | **52.11** | **1.3** |

This is the experiment C2 rests on. The head is 20k parameters trained on one
dataset's masks; the question is whether it learned to pair or learned
LEVIR-CD. On a domain it has never seen it doubles the heuristic at pixel level
and clears it by 5.8x at instance level, and the verifier ablation collapses
below geo-only exactly as it does in E1.

The instance columns say it more plainly than F1 does. Against 987 ground-truth
instances the heuristic emits 11,204 predictions and ours emits 1,431:
**CountMAE 14.8 against 1.3**. For a monitoring service that reports how many
buildings appeared, that is the difference between a usable number and noise.

**Do not read 57.92 > 54.01 as "transfers better than in-domain".** WHU-CD test
averages 1.4 GT instances per tile against LEVIR-CD's 54.5 -- a 38x difference
in change density. The two numbers are not a difficulty comparison, and the
paper must print the densities beside them.

The WHU-CD tables still have no published rows: `baselines.json` has no
verified WHU-CD entries, so `latex_table` emits a TODO instead of inventing
them. Fill them before submission, or the table has our five rows and nothing
to compare against.

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
detection error becomes an unsupported claim: measured HalRate is 25.9, not 0.
That is the point of the row. It fixes the error floor that our own detector
imposes, so the LLM conditions are read as *what generation adds on top of it*.

**Measured, full LEVIR-CC test split -- 1929 crops (EXAONE-4.0-32B,
Qwen2.5-VL-7B), pairing head trained on all 445 LEVIR-CD training tiles** --
`results/levir_cc_caption/`, with `vlm_direct` merged in from
`results/levir_cc_caption_vlm/` by `icce.eval.merge_passes`:

| condition | CFS-P | CFS-R | CFS-F1 | Hal | ChgAcc | BLEU-4 |
|---|---|---|---|---|---|---|
| template | 61.55 | 39.60 | 48.20 | 38.45 | 76.46 | 26.03 |
| vlm_direct | 34.97 | 24.03 | 28.49 | 65.03 | **45.78** | 4.51 |
| llm_raw | 55.87 | 38.06 | 45.28 | 44.13 | 70.40 | 29.15 |
| llm_struct | 61.73 | 39.76 | 48.37 | 38.27 | 76.46 | 28.15 |
| llm_flat_rag | 61.96 | 39.85 | **48.51** | 38.04 | 76.57 | 29.19 |
| llm_graphrag | 61.72 | 39.70 | 48.32 | 38.28 | 76.52 | 29.56 |

Every row except `vlm_direct` consumes the pairing output. The JSON records
`checkpoint` per run and `_merged_from` per row; an earlier version of this
table was produced by the 60-tile pilot head and recorded that nowhere.

Three things here have to be reported, not buried.

**`vlm_direct` decides 45.78 -- below chance -- on whether anything changed at
all**, while writing fluent prose about it. That is the result the pipeline
exists to justify, and the one CFS was built to expose. BLEU sees part of it
(4.51); only ChgAcc shows how complete the failure is. The gap to the grounded
pipeline is 19.83 CFS-F1.

**Aggregation is worth +3.1; retrieval and graph aggregation on top of it are
worth nothing measurable.** `llm_raw` 45.28 → `llm_struct` 48.37 is a real
gain. `llm_flat_rag` 48.51 and `llm_graphrag` 48.32 are within 0.2 of it and of
each other, and the ordering does not favour the graph. Read the three as
indistinguishable.

The reason is structural. All three receive the same change inventory for the
crop being described, and CFS scores only claims about that crop; retrieved
history concerns *other* crops, so it moves wording far more often than it
moves a claim. Measured over the full split: **1313/1929 (68.1%) of generations
differ in text between the three conditions and 26/1929 (1.3%) differ in
extracted claims.** A crop-level ladder cannot decide the question it appears
to ask. E7 asks it where it can be decided.

**`template` and the LLM conditions are indistinguishable on factuality**
(48.20 against 48.37/48.51/48.32). The row fixes the detector's error floor so
the LLM conditions read as what generation adds; on factuality it adds nothing
measurable. What it adds is prose a person can read, and at neighbourhood level
a count they can act on -- see E7.

### Pairing, held against fixed grounding (E5)

E4 varies grounding and holds pairing fixed. E5 does the opposite: grounding is
pinned at `llm_graphrag` and only the pairing is swapped, so whatever moves is
attributable to pairing. Unlike the graph term above, the swapped variable
does reach the evidence -- every crop receives a different `CHANGE_COUNTS`
line.

**Measured, the same 1929 crops, same style examples, same LLM.** The learned
arm is the `llm_graphrag` row of `results/levir_cc_caption/`; the heuristic arm
is `results/levir_cc_caption_heuristic_pairing/` (no `--checkpoint`):

| pairing | CFS-P | CFS-R | CFS-F1 | Hal | ChgAcc |
|---|---|---|---|---|---|
| heuristic (production) | 34.01 | 36.79 | 35.35 | 65.99 | 65.01 |
| **learned head** | **61.72** | **39.70** | **48.32** | **38.28** | **76.52** |

**+12.97 CFS-F1**, and two of every three claims the heuristic-fed report makes
are unsupported (65.99%) against fewer than two in five (38.28%). This is the
experiment that connects the detection work to the reports: E1 and E2 show the
head detects better, and E5 shows the improvement survives all the way to what
the service actually shows a user. A paper that only had E1 would be claiming
that link rather than measuring it.

At the 128-crop pilot scale the gap read +21.02. It narrowed from both sides
when the full split was scored (heuristic 33.62 → 35.35, learned 54.64 →
48.32), because the pilot's eight scenes were the largest in the split and
therefore a favourable slice. The full-split number is the one to report.

Note what E5 does *not* show. It is not evidence for graph grounding -- the
grounding condition is held constant precisely so that it cannot be.

### Neighbourhood-level grounding (E7)

E7 (`icce.eval.run_scene_eval`) asks the same question where the answer can
differ: one report per neighbourhood, over every crop of a tile the split
provides. The conditions stop being nested and become three *representations*
of the same observations -- full concatenation, top-k retrieval, graph
aggregate. The module docstring states this departure; a paper using the table
must too.

**Measured, all 218 scenes of the LEVIR-CC test split, learned head** --
`results/levir_cc_scene/`. 1--16 crops per scene (mean 8.8; 129 scenes with
more than one crop), mean 37.7 GT instances per scene:

| condition | CFS-F1 | Hal | scene CountMAE |
|---|---|---|---|
| template | **40.99** | 56.46 | 26.15 |
| llm_struct (full concatenation) | 21.02 | 76.58 | 458.46 |
| llm_flat_rag (top-k) | 18.54 | 78.57 | 167.87 |
| **llm_graphrag (graph aggregate)** | **27.55** | **72.28** | **18.50** |

**This is where the graph earns a claim, and it is a counting claim.**
Concatenating every crop's inventory into one prompt gives a scene count error
of 458.46 instances: handed everything, the LLM does not aggregate, and the
error grows with the number of observations. Top-k retrieval does better only
because it truncates (167.87), and it cannot recover a count from crops it
never retrieved. Graph aggregation holds it to **18.50 -- better than the
deterministic template (26.15), the only generated condition that beats the
template on any axis.** On CFS-F1 the graph leads the other generated
conditions but still trails the template, so the claim is about counting, not
factuality overall.

**Scale is what made this measurable.** At n=8 with the pilot head, graph
aggregation sat *below* plain concatenation on CFS-F1 (32.43 against 35.14).
At n=218 the ordering reverses and the counting gap widens by more than an
order of magnitude. The two results are consistent: with few observations any
representation serves, and the cost of not aggregating only appears once there
are enough observations to lose track of.

**What E7 still does not test is the temporal half of the design.** The graph
is built to accumulate entity histories across many revisits, but LEVIR-CC and
WHU-CD both carry exactly two timepoints, so what is being measured here is
spatial aggregation over the crops of one tile. A dataset with a real revisit
history is what would settle the design's main hypothesis, and the paper should
say so rather than let "spatio-temporal" imply otherwise.

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

Measured on one A100 80GB, **512 pairs of the full LEVIR-CC cache** (the same
cache the report experiments use), EXAONE-4.0-32B served by vLLM --
`results/efficiency/`:

| stage | ms/tile | peak GPU MB |
|---|---|---|
| SAM3 detection + CLIP (2 frames) | 6099.45 | -- |
| pairing: heuristic (production) | 8.98 | 0.08 |
| pairing: learned head (ours, 19,781 params) | 9.69 | 9.39 |
| knowledge graph: index + retrieve | 141.56 | 8.29 |
| report LLM (batched) | 438.29 | 8.29 |

Segmentation is 91% of the ~6.7 s/tile budget. The learned head costs **0.70 ms
more than the heuristic it replaces** -- 0.01% of the pipeline -- which is what
makes the accuracy gain free in practice.

**This number is cache-dependent, and the paper has to say which cache.** The
same measurement over the 8-scene pilot cache gives 1.45 ms (heuristic 4.63,
head 6.08) because those crops carry a different number of detections and both
pairing strategies scale with that count. Two runs over the full cache agree to
0.08 ms (0.62 at n=64, 0.70 at n=512), so this is not run-to-run noise -- it is
a different tile population. Report the full-cache figure, and name the
population.

Do not characterise the delta ("sub-millisecond", "negligible") -- quote it.
`run_efficiency` now derives the caption's figure from the two rows above
rather than hard-coding a word, so the caption cannot drift from the table
again; an earlier version claimed "sub-millisecond" while the table showed
1.45 ms.

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

### Passes that cannot share a GPU

EXAONE-4.0-32B is 60 GB of weights and Qwen2.5-VL-7B another 16.6 GB; one 80 GB
card holds one of them. So the text conditions and `vlm_direct` are two
invocations, and they must write to **different** `--out` directories:
`run_report_eval` builds `results` from the modes of that invocation and
overwrites the JSON, so a second pass into the same directory replaces the
first pass's rows instead of extending them. `vlm_direct` also regenerates
rather than reusing cached generations, so running it first does not help.

Join them afterwards:

```bash
python -m icce.eval.merge_passes \
  --into results/levir_cc_caption \
  --from results/levir_cc_caption_vlm \
  --modes vlm_direct
```

It refuses to merge two runs that disagree on `dataset/split/style/n_pairs`,
and -- the check that matters -- compares the generated pair ids, because two
runs over 128 crops each are not necessarily the same 128 crops. Each row comes
out stamped with `_merged_from`: which run, which checkpoint, which model. The
merged JSON and the LaTeX are written by the same helpers `run_report_eval`
uses, so the table in the paper is one a reader can regenerate.

This replaces merging by hand. `results/pilot_cc_scenes8` was assembled that
way and could not say which pairing head produced its rows; it turned out to be
the 60-tile pilot head, and every conclusion drawn from those rows had to be
re-measured.

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
                 run_scene_eval, run_efficiency, score_external, integrity,
                 tables, merge_passes, baselines.json
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
