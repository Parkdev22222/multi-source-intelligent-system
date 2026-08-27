# Paper source

ICCE-Asia 2026 submission.

This is built on the venue's own template package
(`Conference-LaTeX-template_10-17-19`, the IEEE conference template ICCE-Asia
distributes). `IEEEtran.cls` in this directory is that package's copy,
byte-for-byte, so the build does not depend on whichever IEEEtran a local TeX
installation carries -- and `main.tex` starts from the template's preamble,
with its two departures marked in the file.

Two things about the template are worth knowing before submitting:

- **Paper size is unresolved.** The template is `[conference]`, which is US
  Letter, and that is what this builds as. The conference's submission page
  says A4. That was read off a search summary of `icce-asia2026.org`, not off
  the page itself, so confirm it: if A4 is right, `[conference,a4paper]` is the
  whole change (tried; still 8 pages, still no overfull boxes).
- **The template's bibliography is a hand-written `thebibliography`; this paper
  keeps BibTeX.** Every entry in `refs.bib` records the source it was verified
  against, and that trail is worth more than matching the template's example.
  `IEEEtran.bst` is not in the template package -- it ships with TeX Live. If
  the venue wants LaTeX sources rather than a PDF, send the generated
  `main.bbl` with them.

**Blinding is still unknown**, and it decides how the author block and the
repository link are handled. The author block is currently the template's, with
one author and its placeholder fields intact.

## Build

```bash
python paper/make_tables.py     # builds the two tables the harness does not emit
cd paper && latexmk -pdf main.tex
```

On a bare pod that needs a TeX installation first:

```bash
apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended \
    texlive-publishers texlive-fonts-recommended
```

`texlive-publishers` is still needed, not for the class -- that one is vendored
here -- but for `IEEEtran.bst`.

The document **compiles clean** against the vendored class: 8 pages, no errors,
no undefined references, no overfull boxes, no font substitutions. `latexmk`
runs BibTeX itself; by hand it is pdflatex, bibtex, pdflatex, pdflatex.

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
- [ ] Confirm the page limit, the paper size and blinding on the submission
      page itself -- all three are currently second-hand
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
