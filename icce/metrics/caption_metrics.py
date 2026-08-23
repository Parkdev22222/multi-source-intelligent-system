"""
Natural-language change-description metrics: BLEU-n, ROUGE-L, CIDEr-D, METEOR.

Why re-implement instead of importing `pycocoevalcap`?
  * the canonical package shells out to a Java jar (PTBTokenizer, METEOR 1.5),
    which is a nuisance inside a slim RunPod container;
  * we need to score thousands of *long* generated reports, not just captions.

The implementations below follow the coco-caption reference code exactly
(BLEU with 'closest' effective reference length, ROUGE-L with beta = 1.2,
CIDEr-D with n = 4 and sigma = 6.0) so the numbers stay comparable with the
published RSICC leaderboard. When `pycocoevalcap` *is* installed,
`score_corpus(..., prefer_pycoco=True)` will use it instead and the returned
dict records which backend produced the numbers -- the paper reports that.
"""

from __future__ import annotations

import logging
import math
import re
import string
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PUNCT = re.compile(r"[%s]" % re.escape(string.punctuation))
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------
# tokenisation
# --------------------------------------------------------------------------
def simple_tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, collapse whitespace (PTB-tokenizer proxy)."""
    t = text.lower().replace("\n", " ")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip().split()


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _all_ngrams(tokens: Sequence[str], n_max: int) -> Counter:
    c: Counter = Counter()
    for n in range(1, n_max + 1):
        c.update(_ngrams(tokens, n))
    return c


# --------------------------------------------------------------------------
# BLEU
# --------------------------------------------------------------------------
def bleu(
    hyps: Sequence[Sequence[str]],
    refs: Sequence[Sequence[Sequence[str]]],
    n_max: int = 4,
) -> List[float]:
    """Corpus BLEU-1..n_max, coco-caption convention ('closest' ref length)."""
    matches = [0] * n_max
    totals = [0] * n_max
    hyp_len = 0
    ref_len = 0

    for hyp, ref_list in zip(hyps, refs):
        hyp_len += len(hyp)
        # 'closest' effective reference length; ties break to the shorter ref
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in ref_list)[1]

        for n in range(1, n_max + 1):
            h_ng = _ngrams(hyp, n)
            max_ref: Counter = Counter()
            for r in ref_list:
                r_ng = _ngrams(r, n)
                for g, c in r_ng.items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            matches[n - 1] += sum(min(c, max_ref[g]) for g, c in h_ng.items())
            totals[n - 1] += max(0, len(hyp) - n + 1)

    # brevity penalty
    if hyp_len == 0:
        return [0.0] * n_max
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)

    scores = []
    log_sum = 0.0
    for n in range(n_max):
        p = (matches[n] / totals[n]) if totals[n] else 0.0
        # coco-caption guards zeros with a tiny epsilon rather than smoothing
        log_sum += math.log(p) if p > 0 else math.log(1e-15)
        scores.append(bp * math.exp(log_sum / (n + 1)))
    return scores


# --------------------------------------------------------------------------
# ROUGE-L
# --------------------------------------------------------------------------
def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(
    hyps: Sequence[Sequence[str]],
    refs: Sequence[Sequence[Sequence[str]]],
    beta: float = 1.2,
) -> float:
    """Mean over samples of max-over-references F_lcs (coco-caption Rouge)."""
    scores = []
    for hyp, ref_list in zip(hyps, refs):
        best = 0.0
        for ref in ref_list:
            l = _lcs(hyp, ref)
            if l == 0:
                continue
            p = l / len(hyp) if hyp else 0.0
            r = l / len(ref) if ref else 0.0
            if p > 0 and r > 0:
                best = max(best, ((1 + beta ** 2) * r * p) / (r + (beta ** 2) * p))
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------
# CIDEr-D
# --------------------------------------------------------------------------
def cider_d(
    hyps: Sequence[Sequence[str]],
    refs: Sequence[Sequence[Sequence[str]]],
    n_max: int = 4,
    sigma: float = 6.0,
) -> float:
    """CIDEr-D with document frequencies estimated on the reference corpus."""
    if not hyps:
        return 0.0

    df: Dict[Tuple[str, ...], float] = defaultdict(float)
    for ref_list in refs:
        seen = set()
        for r in ref_list:
            seen.update(_all_ngrams(r, n_max).keys())
        for g in seen:
            df[g] += 1.0
    log_ref_len = math.log(float(len(refs)))

    def counts2vec(cnts: Counter):
        vec = [defaultdict(float) for _ in range(n_max)]
        norm = [0.0] * n_max
        length = 0
        for gram, tf in cnts.items():
            idf = log_ref_len - math.log(max(1.0, df[gram]))
            k = len(gram) - 1
            vec[k][gram] = float(tf) * idf
            norm[k] += vec[k][gram] ** 2
            # coco-caption measures sentence length in bigrams (k == 1); kept
            # verbatim so our CIDEr-D matches published values.
            if k == 1:
                length += tf
        return vec, [math.sqrt(x) for x in norm], length

    scores = []
    for hyp, ref_list in zip(hyps, refs):
        vec_h, norm_h, len_h = counts2vec(_all_ngrams(hyp, n_max))
        val = np.zeros(n_max)
        for ref in ref_list:
            vec_r, norm_r, len_r = counts2vec(_all_ngrams(ref, n_max))
            delta = float(len_h - len_r)
            for k in range(n_max):
                acc = 0.0
                for gram, v in vec_h[k].items():
                    acc += min(v, vec_r[k][gram]) * vec_r[k][gram]
                if norm_h[k] > 0 and norm_r[k] > 0:
                    acc /= norm_h[k] * norm_r[k]
                val[k] += acc * math.exp(-(delta ** 2) / (2 * sigma ** 2))
        s = float(np.mean(val)) / max(1, len(ref_list)) * 10.0
        scores.append(s)
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# METEOR
# --------------------------------------------------------------------------
def _get_stemmer():
    try:                                    # pragma: no cover - optional dep
        from nltk.stem.porter import PorterStemmer
        return PorterStemmer().stem
    except Exception:
        _SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness",
                     "ing", "edly", "ed", "es", "s", "ly", "ment", "ness")

        def _stem(w: str) -> str:
            for suf in _SUFFIXES:
                if len(w) - len(suf) >= 3 and w.endswith(suf):
                    return w[: -len(suf)]
            return w
        return _stem


_STEM = _get_stemmer()


def _align(hyp: Sequence[str], ref: Sequence[str]) -> List[Tuple[int, int]]:
    """Greedy 3-stage alignment: exact -> stem -> (optional) synonym."""
    used_h, used_r = set(), set()
    alignment: List[Tuple[int, int]] = []

    for key in (lambda w: w, _STEM):
        h_keys = [key(w) for w in hyp]
        r_keys = [key(w) for w in ref]
        for i, hk in enumerate(h_keys):
            if i in used_h:
                continue
            for j, rk in enumerate(r_keys):
                if j in used_r or hk != rk:
                    continue
                used_h.add(i)
                used_r.add(j)
                alignment.append((i, j))
                break
    return sorted(alignment)


def meteor(
    hyps: Sequence[Sequence[str]],
    refs: Sequence[Sequence[Sequence[str]]],
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    """METEOR 1.0 (exact + stem modules). Reported as 'METEOR*' in the paper."""
    scores = []
    for hyp, ref_list in zip(hyps, refs):
        best = 0.0
        for ref in ref_list:
            al = _align(hyp, ref)
            m = len(al)
            if m == 0:
                continue
            p = m / len(hyp) if hyp else 0.0
            r = m / len(ref) if ref else 0.0
            if p == 0 or r == 0:
                continue
            fmean = (p * r) / (alpha * p + (1 - alpha) * r)

            chunks = 1
            for k in range(1, m):
                if not (al[k][0] == al[k - 1][0] + 1 and al[k][1] == al[k - 1][1] + 1):
                    chunks += 1
            penalty = gamma * (chunks / m) ** beta
            best = max(best, fmean * (1 - penalty))
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def score_corpus(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    prefer_pycoco: bool = False,
) -> Dict[str, float]:
    """Score generated change descriptions against multi-reference ground truth.

    Returns BLEU-1..4, METEOR, ROUGE-L, CIDEr-D plus a `_backend` marker.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses vs {len(references)} reference sets")
    if not hypotheses:
        return {"_backend": "none", "n_samples": 0}

    if prefer_pycoco:
        got = _try_pycoco(hypotheses, references)
        if got is not None:
            return got

    hyp_tok = [simple_tokenize(h) for h in hypotheses]
    ref_tok = [[simple_tokenize(r) for r in rs] for rs in references]

    b = bleu(hyp_tok, ref_tok, n_max=4)
    out = {
        "BLEU-1": b[0], "BLEU-2": b[1], "BLEU-3": b[2], "BLEU-4": b[3],
        "METEOR": meteor(hyp_tok, ref_tok),
        "ROUGE-L": rouge_l(hyp_tok, ref_tok),
        "CIDEr-D": cider_d(hyp_tok, ref_tok),
        "n_samples": len(hypotheses),
        "_backend": "msis-native",
    }
    return out


def _try_pycoco(hypotheses, references) -> Optional[Dict[str, float]]:  # pragma: no cover
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except Exception as exc:
        logger.info("pycocoevalcap unavailable (%s); using native metrics", exc)
        return None

    gts = {i: list(rs) for i, rs in enumerate(references)}
    res = {i: [h] for i, h in enumerate(hypotheses)}
    out: Dict[str, float] = {"n_samples": len(hypotheses), "_backend": "pycocoevalcap"}
    try:
        b, _ = Bleu(4).compute_score(gts, res)
        for i, v in enumerate(b, 1):
            out[f"BLEU-{i}"] = float(v)
        out["METEOR"] = float(Meteor().compute_score(gts, res)[0])
        out["ROUGE-L"] = float(Rouge().compute_score(gts, res)[0])
        out["CIDEr-D"] = float(Cider().compute_score(gts, res)[0])
    except Exception as exc:
        logger.warning("pycocoevalcap failed mid-way (%s); falling back", exc)
        return None
    return out
