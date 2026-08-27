# Paper source

ICCE-Asia 2026 submission.

**This is not built on the venue's template package.** No `.cls` is vendored
here; `main.tex` calls plain `IEEEtran` and the build uses TeX Live's copy of
it. Download the template the conference publishes and build against that
before submitting -- a class version mismatch is the kind of thing that is
caught at upload, not before.

The conference's submission page states regular papers are **2--6 pages,
two-column, at least 10pt, on A4**, which is why `main.tex` passes `a4paper`
(IEEEtran defaults to US Letter). That was read off a search summary of
`icce-asia2026.org`, not off the page itself -- confirm it. **Blinding is still
unknown**, and it decides how the author block and the repository link are
handled.

## Build

```bash
python paper/make_tables.py     # builds the two tables the harness does not emit
cd paper && latexmk -pdf main.tex
```

On a bare pod that needs a TeX installation first:

```bash
apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended \
    texlive-publishers texlive-fonts-recommended   # texlive-publishers has IEEEtran
```

The document **compiles clean** with that toolchain: 8 pages, no errors, no
undefined references, no overfull boxes. `latexmk` runs BibTeX itself; by hand
it is pdflatex, bibtex, pdflatex, pdflatex.

`IEEEtran.cls` is still not vendored. The build above uses TeX Live's copy,
which is fine for reading a draft; for the actual submission get the class from
the CFP's template package, so the version matches what the venue expects.

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

## Experiment status

Every experiment now runs at the scale the paper reports, and `numbers.tex`
carries no `PROVISIONAL` macro.

| Section | Scale |
|---|---|
| E1 (LEVIR-CD) | 128 test tiles |
| E2 (WHU-CD) | 690 test pairs |
| E3/E4 (grounding ladder) | 1929 crops, `results/levir_cc_caption/` (+ the Qwen pass merged in from `results/levir_cc_caption_vlm/`) |
| E5 (pairing swap) | same 1929 crops, heuristic arm in `results/levir_cc_caption_heuristic_pairing/` |
| E6 (deployment cost) | 512 pairs of the full LEVIR-CC cache |
| E7 (neighbourhood level) | 218 scenes, `results/levir_cc_scene/` |

The pilot-scale values (128 crops of the 8 largest scenes) survive only in the
`Pilot*` macro block at the bottom of `numbers.tex`, and only to support the
sentence that compares the two scales. Do not typeset them as results.

Every `\input` in `experiments.tex` resolves against a file that exists in
`results/`, so the build no longer depends on an experiment landing first.

## Before submission

- [ ] **The draft is 8 pages against a stated 6-page maximum.** Two pages have
      to go: Related Work compresses well, and E3/E4 ship two tables
      (factuality and caption metrics) where the argument needs one
- [ ] Build against the venue's own template package, not TeX Live's IEEEtran
- [ ] Confirm the page limit and blinding on the submission page itself
- [ ] Fill the author block in `main.tex`
- [ ] `baselines.json`: WHU-CD has **no verified rows**, so `latex_table`
      emits a TODO comment instead of a comparison block. LEVIR-CC has four
      unverified supervised rows. Fill and verify both, or those tables ship
      with our rows and nothing to compare against.
- [ ] `refs.bib`: the model and recent-paper entries have been checked against
      a primary source (each names its source in a comment above it), but the
      older entries still lack volume, issue and page numbers -- these were
      omitted rather than guessed. Add them from the publisher's record.
- [ ] Compile once and read the reference list: `sam3` and `qwen25vl` are
      deliberately abbreviated to `and others`, which IEEEtran renders as
      "et al."; confirm the venue accepts that for a 38-author paper rather
      than requiring the full list.

Done since the first draft, kept here so it is not re-checked:

- `numbers.tex` is fully `FINAL`
- `docs/architecture.png` is included by the `figure` environment at the top of
  `method.tex`, so `fig:architecture` resolves
- every entry in `refs.bib` is cited, and every `\cite` key resolves to an entry
