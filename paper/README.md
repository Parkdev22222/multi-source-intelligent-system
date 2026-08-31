# ICCE-Asia 2026 paper

Submission title:

> **Vision-Grounded LLM Reports for Satellite Change Monitoring: Learned
> Instance Pairing and Claim-Level Evaluation**

The paper studies a consumer-facing satellite monitoring pipeline that turns
two revisits into a written change report. Its contribution order is deliberate:

1. a 19,781-parameter learned instance-pairing head;
2. controlled evidence that pairing quality changes final report factuality;
3. a deterministic, domain-specific report-evaluation protocol;
4. a secondary analysis of retrieval and graph aggregation.

Change-Fact-Score (CFS) is an evaluation protocol, not the primary algorithmic
contribution. It matches claims by change direction and object class; numerical
count error is reported separately with CountMAE. The central result is the E5
controlled swap: with crops, prompt, LLM and GraphRAG fixed, learned pairing
improves CFS-F1 by 12.97 points and reduces unsupported claims from 65.99% to
38.28% at 0.70 ms additional latency per tile.

## Venue format

The source starts from the ICCE-Asia conference template and vendors the
provided `IEEEtran.cls`. ICCE-Asia 2026 requires regular papers to be 2--6
pages, A4, two-column, single-spaced, and at least 10pt. `main.tex` therefore
uses:

```tex
\documentclass[conference,a4paper]{IEEEtran}
```

The current layout is A4 and six pages in the checked build. The final upload
must be rebuilt after author information and the bibliography are present;
that final PDF, rather than an intermediate `pdflatex` pass, is the authority
for the page count.

The public submission instructions do not state an anonymous-review policy.
The supplied template contains an author block, but confirm blinding with the
secretariat if needed before uploading.

## Build

From the repository root:

```bash
python scripts/gen_paper_figure.py   # method figure, from the results
python paper/make_tables.py
cd paper
latexmk -pdf main.tex
```

The figure is generated, not drawn: its two annotations are read from
`results/*.json`, so it cannot drift from the tables. The old
`docs/architecture.png` is the repository's deployment diagram and stays in
the top-level README; the paper uses `docs/method_figure.pdf`.

Equivalent manual sequence:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**The Python environment must be complete before any re-run.** `decode_mask`
reaches the RLE decoder through `src.pairing`, whose import chain pulls in
`src.database` and therefore `sqlalchemy`. With that dependency missing the
import used to be swallowed silently, every detection fell back to
bounding-box rasterisation, and pixel metrics came out several points low
(WHU-CD by up to 9 F1) while the run still recorded
`pixel_scoring: sam3_masks`. `run_cd_eval` now refuses to start in that state;
`--bbox-pixels` remains the way to ask for the bounding-box measurement on
purpose. Install `requirements.txt` in full, not a subset.

Minimal TeX Live packages for a bare environment:

```bash
apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended \
    texlive-publishers texlive-fonts-recommended
```

`texlive-publishers` supplies `IEEEtran.bst`; the repository vendors the class
but not the bibliography style. A single `pdflatex` pass without the `.bbl`
produces unresolved citations and is **not submission-ready**. If source files
are ever requested, include the generated `main.bbl` as well.

## Reproducible numbers and tables

- Prose values come from macros in `numbers.tex`; all paper-facing macros are
  marked `FINAL`.
- Tables emitted by the evaluation harness are included directly from
  `results/`.
- `paper/make_tables.py` builds five cross-run, compacted or re-columned
  tables: `detection.tex`, `factuality.tex`, `e5_pairing.tex`, `e7_scene.tex`
  and `example.tex`. `detection.tex` and `factuality.tex` replace tables the
  harness also writes, because both needed a column the harness does not emit:
  a per-tile Count MAE on detection, and the crop-level CountMAE that
  `report_results_caption.json` has always carried but no table printed.
  `example.tex` is one worked crop, read from the generation dumps and the
  LEVIR-CC reference captions rather than transcribed.
- Generated table comments record the run directory, checkpoint and sample
  count. Missing inputs cause generation to stop rather than invent a row.

After rerunning an experiment: regenerate its result files, run
`paper/make_tables.py`, update the corresponding macros in `numbers.tex`, and
rebuild the PDF.

## Paper structure

| Section | Role |
|---|---|
| I. Introduction | consumer-facing failure modes and contribution order |
| II. Related Work | supervised ceiling, zero-shot context, captioning and grounding |
| III. Proposed Method | pipeline, learned pairing, cross-frame evidence, deterministic evaluation, grounding conditions |
| IV. Experiment | setup; detection; pairing-to-report result; cost; VLM/caption baselines; ablations |
| V. Discussion | where factuality is determined; narrower GraphRAG result; limitations and future work |

There is no separate Conclusion section; the concluding interpretation and
future work are folded into Discussion to stay within six pages.

Six tables remain. The detection table combines pixel- and instance-level
results on LEVIR-CD and WHU-CD. E5 is intentionally placed immediately after
the detection result because it connects upstream pairing to the final report.
The efficiency table is expressed in prose. The sixth is the worked example of
the invention failure; it is the only qualitative element in the paper and the
first thing to cut if the page count moves.

## Experiment status

| Experiment | Scale | Purpose |
|---|---:|---|
| E1: LEVIR-CD | 128 test tiles | in-domain pairing and change detection |
| E2: WHU-CD | 690 test pairs | transfer without retraining or threshold tuning, and the cross-frame ablation |
| E3/E4: LEVIR-CC | 1,929 crops | report and grounding conditions |
| E5: pairing swap | same 1,929 crops | main detection-to-report result |
| E6: deployment cost | 512 pairs | pairing latency within the full pipeline |
| E7: neighbourhood level | 218 scenes | count preservation across crop aggregation |

The Qwen VLM pass is merged from `results/levir_cc_caption_vlm/`; the heuristic
E5 arm is in `results/levir_cc_caption_heuristic_pairing/`; neighbourhood
results are in `results/levir_cc_scene/`. Pilot values remain only in the
`Pilot*` macro block and must not be typeset as final results.

## Submission checklist

### Required before initial submission

- [x] Use the ICCE-Asia template with A4, two columns, single spacing and 10pt.
- [x] Keep the regular paper within the 2--6 page limit in the current layout.
- [x] Include the 189-word abstract and IEEE keywords.
- [x] Use embedded fonts and a non-encrypted PDF in the checked build.
- [ ] Replace all five template author blocks in `main.tex` with the actual
      authors, affiliations, cities/countries, emails or ORCIDs; remove unused
      blocks and confirm author order.
- [ ] Use the first author's email for the ICCE-Asia submission account.
- [ ] Confirm whether the review is anonymous; if it is, anonymise the author
      block and any repository-identifying material.
- [ ] Run the complete BibTeX build and confirm that there are no `[?]`,
      undefined citations/references, or missing bibliography pages.
- [ ] Confirm that the **bibliography-inclusive, author-complete PDF** is still
      A4 and no more than six pages.
- [ ] Visually inspect the final PDF at 100%: title/author layout, architecture
      figure text, table width, page breaks, and the final reference list.
- [ ] Enter title, abstract, keywords, authors and author order in the portal
      exactly as they appear in the PDF.
- [ ] Select the closest AI/ML-for-consumer-electronics track; use the image/
      video or miscellaneous CE track only if the portal taxonomy differs.
- [ ] Upload the final PDF, complete submission, and retain the confirmation
      page and email.

### Bibliography and comparison checks

- [ ] Complete missing volume, issue and page metadata for older entries in
      `refs.bib` where publisher records are available.
- [ ] **`lewis2020retrieval`, `edge2024local` and `zheng2023judging` were
      written from knowledge of the literature and have not been checked
      against a publisher record.** Verify author lists, venues and pages
      before submitting; they are flagged in `refs.bib` as well.
- [ ] Read the rendered author lists for `sam3` and `qwen25vl`; they currently
      use `and others`, which IEEEtran renders as “et al.”
- [x] LEVIR-CC caption baselines are verified from Chg2Cap Table IX.
- [x] AnyChange is presented as zero-shot context, not as a
      supervision-matched baseline.
- [x] WHU-CD is reported as transfer without an unverified published baseline;
      differing WHU splits are not mixed into the table.

### Title and keywords

The title foregrounds the complete vision-to-language pipeline while retaining
the two contributions a reader should be able to cite: learned instance pairing
and claim-level evaluation. "Vision-grounded" means that SAM3, CLIP and the
pairing head establish explicit object evidence before the LLM generates a
report; it does not describe an end-to-end vision--language model. The
consumer-facing use case remains explicit in the abstract and Introduction.

The title deliberately does not assert the grounding null ("pairing, not
prompting"): the paper's own position is that the crop-level ladder *cannot*
decide that question and that graph aggregation earns a neighbourhood-level
counting claim. Such a title would overstate what the experiments support.

Rebuild the author-complete PDF and verify the rendered title and six-page limit
before submission.

### Known measurement caveats

- **Hungarian assignment does not reproduce bit-for-bit.** Re-measuring E2 on
  2026-08-31 reproduced `geo-only`, the heuristic and the greedy arm exactly,
  while the two Hungarian arms moved: ours 57.92 -> 57.97 pixel F1, the
  no-verifier ablation 19.78 -> 20.84. Ties in the assignment are broken
  non-deterministically. The E2 block is quoted from that single later run so
  the table has one provenance; the LEVIR-CD block is unchanged and no claim
  in the paper turns on the difference.

### Deliberately deferred, not submission blockers

- Independent manual validation of the rule-based claim extractor.
- Rescoring released outputs from an external specialist captioning model.
- Paired confidence intervals or repeated stochastic generation passes.
- Evaluation on true multi-revisit histories and change types beyond buildings.

These are stated as limitations or future work and should not be presented as
completed experiments.
