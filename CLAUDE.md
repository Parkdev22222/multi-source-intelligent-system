# Working in this repository

## The paper has no slack — check the page count after every edit

`paper/` is an ICCE-Asia submission with a hard 6-page limit and a layout
that sits exactly at it. A single added sentence has cost a seventh page
repeatedly, and a 7-page build has been pushed more than once because the
count was not checked after an edit.

**After changing anything under `paper/` — including `numbers.tex`, a table,
or the figure — run:**

```bash
scripts/check_paper.sh          # pages / overfull / undefined; non-zero on failure
```

Do not commit a failing build. When an edit costs a page, pay for it by
cutting prose that restates something the paper already says. Never pay for
it with the figure, a table row, a result, or a caveat.

Also re-run it **after every rebase**: another session's edit and yours can
each fit alone and not together.

## The figure and tables are generated, not drawn

The figure is a chain, and the `.pptx` is its source:

```bash
node scripts/gen_figure_pptx.js       # docs/method_figure.pptx  <- edit here
python scripts/figure_from_pptx.py    # docs/method_figure.{pdf,png} <- the paper
python paper/make_tables.py           # paper/tables/*.tex from results/*.json
```

`scripts/gen_paper_figure.py` is the layout of record that the `.pptx`
transcribes, and the reasoning behind each edge lives in its comments. It
writes `docs/method_figure_layout.png` under its own name and no longer
touches what the paper includes.

Edit the generator, never the output. Numbers in the prose come from macros
in `paper/numbers.tex`, which trace to files under `results/`; do not type a
figure into the text.

## Results are evidence, not decoration

Every number in the paper traces to a file in `results/`. If a re-run
changes a value the paper quotes, that is a finding to diagnose — something
believed deterministic is not — and never something to absorb by updating
the paper to match. Report it.

`docs/verifier_ablation.md` is the worked example of how an experiment gets
run, read against rules fixed beforehand, and written up whichever way it
comes out.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```
