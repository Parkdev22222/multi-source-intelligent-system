"""
Score a published model's released outputs with our metrics.

    python -m icce.eval.score_external \
        --predictions external/chg2cap_levircc_test.json \
        --name Chg2Cap --cache data/cache/levir_cc_test \
        --out results/external_chg2cap

Why this exists: Change-Fact-Score is our own metric, and a table where we are
the only system measured on it is worth very little. A reviewer will say we
invented a yardstick and then reported that we are tall. Running the same
yardstick over a published model's own released captions removes that
objection, and it is the experiment most likely to produce the paper's central
result -- a specialist that wins decisively on BLEU while stating fewer correct
facts, because n-gram overlap does not penalise a fluent wrong answer.

The predictions file is whatever the upstream repo emits, in any of:
    {"pair_id": "caption", ...}
    [{"pair_id": ..., "caption": ...}, ...]
    JSONL of {"key": ..., "text": ...}
Ground truth (references, change flags, GT instance counts) comes from our
detection cache, so the external model is scored on exactly the samples and
exactly the references our own rows are scored on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

_ID_KEYS = ("pair_id", "key", "id", "image_id", "name", "filename")
_TEXT_KEYS = ("caption", "text", "pred", "prediction", "hypothesis",
              "sentence", "sentences", "captions", "raw")


def load_predictions(path: Path) -> Dict[str, str]:
    """Accept the several shapes upstream repos emit, without editing them.

    Detection is by structure, not by sniffing the text: a whole-file JSON
    parse decides between JSON and JSONL, and a record is told apart from a
    flat id->caption mapping by whether it carries both an id key and a text
    key. Guessing from punctuation silently mis-parses a one-line JSONL file.
    """
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"{path} is empty")

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        return _normalise_keys(_from_jsonl(raw, path))

    if isinstance(doc, dict):
        if "images" in doc and isinstance(doc["images"], list):
            doc = doc["images"]                        # coco-style
        elif _is_record(doc):
            return _normalise_keys({_pick(doc, _ID_KEYS): _pick(doc, _TEXT_KEYS)})
        elif all(isinstance(v, str) for v in doc.values()):
            return _normalise_keys(doc)                # flat id -> caption
        else:
            raise ValueError(
                f"{path}: dict is neither a record nor an id->caption mapping "
                f"(keys: {sorted(doc)[:8]})"
            )

    if isinstance(doc, list):
        return _normalise_keys(
            {_pick(r, _ID_KEYS): _pick(r, _TEXT_KEYS) for r in doc}
        )
    raise ValueError(f"unrecognised prediction format in {path}")


def _has(rec: Dict, keys: Sequence[str]) -> bool:
    return any(k in rec and rec[k] is not None for k in keys)


def _is_record(d: Dict) -> bool:
    """A single prediction carries both an identifier and a caption."""
    return _has(d, _ID_KEYS) and _has(d, _TEXT_KEYS)


def _from_jsonl(raw: str, path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} is neither JSON nor JSONL: {exc}") from exc
        out[_pick(rec, _ID_KEYS)] = _pick(rec, _TEXT_KEYS)
    if not out:
        raise ValueError(f"{path}: no records found")
    return out


def _pick(rec: Dict, keys: Sequence[str]) -> str:
    for k in keys:
        if k in rec and rec[k] is not None:
            v = rec[k]
            if isinstance(v, list):
                v = v[0] if v else ""
            if isinstance(v, dict):
                # coco-style: {"sentences": [{"raw": "..."}]}
                v = _pick(v, _TEXT_KEYS)
            return str(v).strip()
    raise KeyError(f"no key from {keys} in record with keys {sorted(rec)}")


def _normalise_keys(d: Dict[str, str]) -> Dict[str, str]:
    """Strip directory and extension so `test_000042_0_1.png` matches our ids."""
    return {Path(str(k)).stem: v for k, v in d.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score external model outputs")
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--name", required=True, help="method name for the table")
    ap.add_argument("--cache", required=True, type=Path,
                    help="detection cache supplying references and GT")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tier", default="supervised",
                    choices=("supervised", "zero-shot", "ours"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from icce.metrics.caption_metrics import score_corpus
    from icce.metrics.change_fact import ChangeFactEvaluator, gt_claims_from_captions
    from icce.pairing_head.cache import load_cache

    preds = load_predictions(args.predictions)
    samples, _ = load_cache(args.cache)
    logger.info("%d predictions, %d cached samples", len(preds), len(samples))

    hyps, refs = [], []
    fact = ChangeFactEvaluator()
    missing = 0
    for s in samples:
        if not s.captions:
            continue
        text = preds.get(s.pair_id)
        if text is None:
            missing += 1
            continue
        hyps.append(text)
        refs.append(s.captions)
        fact.update(text, gt_claims_from_captions(s.captions),
                    s.gt_change_present, len(s.gt_instances))

    if missing:
        logger.warning("%d cached samples had no prediction and were skipped; "
                       "caption metrics are computed on the %d that overlap",
                       missing, len(hyps))
    if not hyps:
        logger.error("no pair ids matched between predictions and cache. "
                     "Sample prediction ids: %s | sample cache ids: %s",
                     list(preds)[:3], [s.pair_id for s in samples[:3]])
        return 1

    row: Dict = {"name": args.name, "tier": args.tier,
                 "n_scored": len(hyps), "n_missing": missing}
    row.update(score_corpus(hyps, refs))
    row.update(fact.compute().as_dict())

    from icce.eval.tables import print_console_table, save_json
    save_json({"predictions": str(args.predictions), **row},
              Path(args.out) / f"external_{args.name.lower()}.json")
    print_console_table(f"{args.name} (external, {args.tier})",
                        ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-L", "CIDEr-D"], [row])
    print_console_table(f"{args.name} factuality",
                        ["cfs_precision", "cfs_recall", "cfs_f1",
                         "hallucination_rate", "change_accuracy"], [row])
    return 0


if __name__ == "__main__":
    sys.exit(main())
