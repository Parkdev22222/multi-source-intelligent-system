"""
One-to-one assignment between past and current detections.

The production pipeline assigns greedily in descending score, which cannot undo
an early mistake: a high-scoring but wrong pair steals a detection from its true
partner and cascades. We solve the global optimum with the Hungarian algorithm
(SciPy) and keep the greedy path both as a fallback and as an explicit ablation
row -- "Hungarian vs greedy" is one of the numbers the paper reports.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Cost used for (past, cur) combinations that were never generated as candidates.
_FORBIDDEN = 1e6


def _score_matrix(
    pairs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    n_past: int,
    n_cur: int,
) -> np.ndarray:
    mat = np.zeros((n_past, n_cur), dtype=np.float64)
    for (i, j), s in zip(pairs, scores):
        mat[i, j] = max(mat[i, j], float(s))
    return mat


def greedy_assign(
    pairs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Descending-score greedy matching (the production behaviour)."""
    order = np.argsort(-np.asarray(scores))
    used_p, used_c = set(), set()
    out: List[Tuple[int, int, float]] = []
    for k in order:
        i, j = pairs[k]
        s = float(scores[k])
        if s < threshold or i in used_p or j in used_c:
            continue
        used_p.add(i)
        used_c.add(j)
        out.append((i, j, s))
    return out


def hungarian_assign(
    pairs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    n_past: int,
    n_cur: int,
    threshold: float,
) -> List[Tuple[int, int, float]]:
    """Globally optimal one-to-one matching; falls back to greedy without SciPy."""
    if n_past == 0 or n_cur == 0 or len(pairs) == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return greedy_assign(pairs, scores, threshold)

    mat = _score_matrix(pairs, scores, n_past, n_cur)
    cost = np.where(mat > 0, -mat, _FORBIDDEN)
    rows, cols = linear_sum_assignment(cost)

    out: List[Tuple[int, int, float]] = []
    for i, j in zip(rows, cols):
        s = float(mat[i, j])
        if s >= threshold and cost[i, j] < _FORBIDDEN:
            out.append((int(i), int(j), s))
    out.sort(key=lambda t: -t[2])
    return out


def assign(
    pairs: Sequence[Tuple[int, int]],
    scores: np.ndarray,
    n_past: int,
    n_cur: int,
    threshold: float = 0.5,
    method: str = "hungarian",
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Returns (matches, unmatched_past_idx, unmatched_cur_idx)."""
    if method == "greedy":
        matches = greedy_assign(pairs, scores, threshold)
    else:
        matches = hungarian_assign(pairs, scores, n_past, n_cur, threshold)

    used_p = {i for i, _, _ in matches}
    used_c = {j for _, j, _ in matches}
    return (
        matches,
        [i for i in range(n_past) if i not in used_p],
        [j for j in range(n_cur) if j not in used_c],
    )
