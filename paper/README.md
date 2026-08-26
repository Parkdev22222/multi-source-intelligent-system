# Paper source

ICCE-Asia 2026 submission. Written against a **six-page, non-blind** assumption
-- neither has been checked against the CFP. Confirm both before going far:
the page limit decides how much of Related Work and how many tables survive,
and blinding decides how the repository and the author block are handled.

## Build

```bash
python paper/make_tables.py     # builds the two tables the harness does not emit
cd paper && latexmk -pdf main.tex
```

There is no LaTeX toolchain on this pod, so **nothing here has been compiled**.
Expect to fix the first round of errors by hand. `IEEEtran.cls` is not vendored
either; get it from the CFP's template package rather than from CTAN, so the
class version matches what the venue expects.

## How numbers get into the paper

Nothing is typed by hand, in either direction.

- **Tables** are `\input` straight from the directories the harness writes
  (`../results/...`). Re-run an experiment and the table changes with it.
- **Numbers in prose** come from macros in `numbers.tex`, one per measurement,
  each tagged with the run it was read from and marked `FINAL` or
  `PROVISIONAL`. Prose never contains a literal number that also appears in a
  table.
- `make_tables.py` builds the two tables that span runs: E5 compares two run
  directories, and E7's scene results ship as JSON with no LaTeX writer. It
  refuses to invent a table whose inputs are missing, and it prints the
  provenance of every row it does write into the `.tex` as a comment.

If you re-run an experiment, the sequence is: re-run, then `make_tables.py`,
then update the affected macros in `numbers.tex`. The macro block names the
source run for exactly this reason.

## What is provisional right now

| Section | Status |
|---|---|
| E1 (LEVIR-CD) | final -- 128 test tiles |
| E2 (WHU-CD) | final -- 690 test pairs |
| E3/E4 (grounding ladder) | **provisional** -- 128 crops of 8 neighbourhoods |
| E5 (pairing swap) | **provisional** -- same 128 crops |
| E6 (deployment cost) | final |
| E7 (neighbourhood level) | **provisional twice over** -- $n=8$, and produced by the 60-tile pilot head |

The full-scale LEVIR-CC runs replace the E3/E4/E5/E7 rows. `experiments.tex`
already points at `../results/levir_cc_caption/`, which the full-scale run
creates; until it lands those `\input` lines have no file and the build will
fail on them. Either run the experiments first or comment those two lines while
drafting.

## Before submission

- [ ] Confirm page limit and blinding against the CFP
- [ ] Fill the author block in `main.tex`
- [ ] `baselines.json`: WHU-CD has **no verified rows**, so `latex_table`
      emits a TODO comment instead of a comparison block. LEVIR-CC has four
      unverified supervised rows. Fill and verify both, or those tables ship
      with our rows and nothing to compare against.
- [ ] `refs.bib`: every entry needs checking; volume/issue/pages were omitted
      rather than guessed, and entries marked `TODO-CITE` need the authors'
      preferred citation form.
- [ ] Re-read `numbers.tex` and confirm nothing is still marked `PROVISIONAL`
- [ ] Figure: `docs/architecture.png` is referenced as
      `fig:architecture` in `method.tex` but not yet included -- add the
      `figure` environment or drop the reference
