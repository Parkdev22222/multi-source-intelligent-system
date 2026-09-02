# verifier_ablation

**One question:** how much of 25.59 → 54.01 on LEVIR-CD is *learning to match*,
and how much is *having a verifier at all*?

Table I compares the production heuristic against the learned head, but four
things differ between those rows: the match rule, the state rule, greedy versus
Hungarian assignment, and whether anything suppresses spurious detections. The
heuristic has no verifier — `HeuristicHead.change_probability` returns ones, so
every unmatched detection is reported, which is why it emits 11,204 predictions
against 987 ground-truth instances. The headline gain therefore confounds two
things, and a reviewer will ask which one carries it.

This experiment adds one row that answers it. `HybridHead`
(`icce/pairing_head/model.py`) pairs with the hand-tuned rule and takes state
and verify from the trained model, so **only the match branch differs from the
learned head**.

The substitution is clean rather than off-distribution: `verify_labels`
(`icce/pairing_head/cache.py`) come from each detection's coverage of the
ground-truth change mask and never refer to the matcher, so the verifier does
not assume its leftovers were produced by the learned one.

---

## 1. Run it

Nothing is re-segmented. This reads the cached detections and runs the MLP plus
assignment, so it is minutes, not hours. Write to a scratch directory first —
step 2 explains why.

```bash
git pull origin icce-asia

python -m icce.eval.run_cd_eval \
    --cache data/cache/levir_cd_test \
    --checkpoint data/checkpoints/pairing_head.pt \
    --checkpoint-no-xf data/checkpoints/pairing_head_no_xf.pt \
    --dataset levir_cd --split test \
    --out results/_verifier_ablation/levir_cd \
    --device cuda --verifier-ablation

python -m icce.eval.run_cd_eval \
    --cache data/cache/whu_cd_test \
    --checkpoint data/checkpoints/pairing_head.pt \
    --checkpoint-no-xf data/checkpoints/pairing_head_no_xf.pt \
    --dataset whu_cd --split test \
    --out results/_verifier_ablation/whu_cd \
    --device cuda --verifier-ablation
```

Adjust `--cache` to wherever the cached detections actually live. Keep
`--checkpoint-no-xf`: without it the *no cross-frame* row disappears from the
new result file and would drop out of the table in step 3.

The new row prints as `heuristic matching, learned verifier`.

## 2. Check the old rows reproduce before adopting anything

The run recomputes every existing row. It should be deterministic — fixed
checkpoint, cached detections, deterministic assignment — so the rows the paper
already quotes must come back **identical**.

```bash
python - <<'PY'
import json, pathlib
for name, old, new in [
    ("levir_cd", "results/levir_cd_test/cd_results.json",
     "results/_verifier_ablation/levir_cd/cd_results.json"),
    ("whu_cd", "results/whu_cd_test/cd_results.json",
     "results/_verifier_ablation/whu_cd/cd_results.json"),
]:
    o = {r["name"]: r for r in json.loads(pathlib.Path(old).read_text())["results"]}
    n = {r["name"]: r for r in json.loads(pathlib.Path(new).read_text())["results"]}
    print(f"\n== {name} ==")
    for k in o:
        if k not in n:
            print(f"  MISSING  {k}"); continue
        drift = [f for f in ("f1", "iou", "inst_f1")
                 if abs(o[k][f] - n[k][f]) > 1e-9]
        print(f"  {'DRIFT ' + ','.join(drift) if drift else 'same  '}  {k}")
    for k in n.keys() - o.keys():
        print(f"  NEW      {k}: f1={n[k]['f1']:.4f} inst_f1={n[k]['inst_f1']:.4f} "
              f"count_mae={n[k]['instance']['count_mae']:.1f}")
PY
```

**If anything drifts, stop and report it.** Do not update the paper's numbers to
match a re-run — every quoted figure in `paper/numbers.tex` traces to the
existing result files, and a silent shift means something is nondeterministic
that we believed was not. That is a finding in its own right and needs
diagnosing, not absorbing.

If the old rows are identical, adopt the new files:

```bash
cp results/_verifier_ablation/levir_cd/cd_results.json results/levir_cd_test/
cp results/_verifier_ablation/whu_cd/cd_results.json results/whu_cd_test/
```

## 3. Put it in the paper

`paper/make_tables.py` already knows the row (`DET_ORDER`), so it appears
automatically once the result files carry it:

```bash
python paper/make_tables.py
```

Add the LEVIR-CD number to `paper/numbers.tex` next to the other E1 macros:

```tex
\newcommand{\EOneHybridFOne}{<pixel F1>}       % heuristic matching, learned verifier
\newcommand{\EOneHybridInstFOne}{<instance F1>}
```

Then write **one** sentence in `paper/sections/experiments.tex`, in the
**Ablation Study** subsection beside the existing verifier sentence. Which
sentence depends on the number — see step 4.

## 4. Decide what it means, by a rule set before seeing it

Let **H** = heuristic (25.59), **X** = the new row, **L** = learned head (54.01),
all LEVIR-CD pixel F1.

- **L − X ≥ 15.** Learning to match carries the gain. Say so: the verifier is
  necessary but not sufficient, and the match branch is worth L − X on its own.
- **5 ≤ L − X < 15.** Both matter. Report the split honestly — suppression is
  worth X − H, learned matching a further L − X — and leave the contribution
  wording as it is.
- **L − X < 5.** Suppression carries the gain, and the paper currently
  overstates the match branch. This is the case that needs real edits:
  1. Contribution 1 in `paper/sections/introduction.tex` and the abstract sentence
     must say *pairing and verification*, not pairing alone.
  2. The Ablation Study must state the decomposition outright with both numbers.
  3. Do **not** delete the row, soften it, or move it to a footnote.

Whatever the number, it goes in the table and is quoted in the text. An
experiment run to answer a reviewer's question and then omitted because the
answer was inconvenient is worse than never running it. **E5 is unaffected in
every case** — that swap replaces the whole head, so the report-factuality
result (12.97 CFS-F1) does not depend on this decomposition.

## 5. Finish

The paper is exactly six pages with no slack, so every edit needs the page count
re-checked. `paper/README.md` has the full build; the short loop is:

```bash
python scripts/gen_paper_figure.py
python paper/make_tables.py
cd paper && latexmk -pdf main.tex
```

Then confirm, from `paper/main.log`: **6 pages**, zero `Overfull`, zero
undefined references. If the new row costs a seventh page, cut prose elsewhere —
never the row, and never the figure.

Run the tests (`PYTHONPATH=. python -m pytest tests/ -q`; `tests/`
`test_hybrid_ablation.py` pins that the row is genuinely a hybrid and has not
collapsed onto either parent), then commit and push to `icce-asia`.
