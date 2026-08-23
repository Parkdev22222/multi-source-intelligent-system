"""
Inference wrapper: detections in, change statuses out.

`LearnedPairer` is the drop-in replacement for the greedy CLIP+geo matcher in
`src/pairing/temporal_pairing.py`. It emits the same status vocabulary the rest
of MSIS already speaks (new / matched / moved / changed / disappeared), so the
knowledge-graph indexer and the report builder need no changes.

Set `head=None` to run the production heuristic through the identical code
path -- that is the ablation baseline in the paper, and sharing the path means
the comparison isolates the head rather than incidental plumbing differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from icce.pairing_head.assign import assign
from icce.pairing_head.features import (
    Det,
    candidate_pairs,
    pair_features,
    unary_features,
)

STATUS_MATCHED = "matched"
STATUS_MOVED = "moved"
STATUS_CHANGED = "changed"
STATUS_NEW = "new"
STATUS_DISAPPEARED = "disappeared"


@dataclass
class PairingOutcome:
    """One row of the pairing table, mirroring `PairingRecord`."""

    status: str
    past_idx: Optional[int]
    current_idx: Optional[int]
    score: float
    lat: float
    lon: float
    object_class: str
    confidence: float
    bbox_px: Tuple[float, float, float, float]
    state_probs: Dict[str, float] = field(default_factory=dict)


@dataclass
class PairingResult:
    outcomes: List[PairingOutcome]
    n_candidates: int
    n_suppressed: int = 0        # detections the verifier rejected as spurious

    def by_status(self, status: str) -> List[PairingOutcome]:
        return [o for o in self.outcomes if o.status == status]

    def change_instances(self, geo_to_px=None):
        """Change instances (new + disappeared + changed) for instance metrics."""
        from icce.metrics.instance_metrics import ChangeInstance

        out = []
        for o in self.outcomes:
            if o.status in (STATUS_MATCHED, STATUS_MOVED):
                continue
            ctype = {
                STATUS_NEW: "appeared",
                STATUS_DISAPPEARED: "disappeared",
                STATUS_CHANGED: "modified",
            }[o.status]
            out.append(ChangeInstance(
                bbox=o.bbox_px, change_type=ctype,
                score=o.score, object_class=o.object_class,
            ))
        return out


class LearnedPairer:
    def __init__(
        self,
        head=None,
        match_radius_deg: float = 0.001,
        match_threshold: float = 0.5,
        verify_threshold: float = 0.5,
        move_threshold_deg: float = 0.0002,
        assignment: str = "hungarian",
        max_candidates: int = 8,
        device: str = "cpu",
    ) -> None:
        self.head = head
        self.match_radius_deg = match_radius_deg
        self.match_threshold = match_threshold
        self.verify_threshold = verify_threshold
        self.move_threshold_deg = move_threshold_deg
        self.assignment = assignment
        self.max_candidates = max_candidates
        self.device = device

    # -- factory -----------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: Optional[Path], device: str = "cpu", **kw) -> "LearnedPairer":
        """`path=None` builds the heuristic baseline (no learned weights)."""
        if path is None:
            from icce.pairing_head.model import HeuristicHead
            return cls(head=HeuristicHead(), device="cpu", **kw)

        from icce.pairing_head.model import PairingHead
        model, extra = PairingHead.load(Path(path), map_location=device)
        model.to(device)
        kw.setdefault("match_threshold", extra.get("match_threshold", 0.5))
        kw.setdefault("verify_threshold", extra.get("verify_threshold", 0.5))
        return cls(head=model, device=device, **kw)

    # -- scoring -----------------------------------------------------------
    def _scores(
        self,
        pair_x: np.ndarray,
        unary_past: np.ndarray,
        unary_cur: np.ndarray,
        cls_past: np.ndarray,
        cls_cur: np.ndarray,
        cls_past_all: np.ndarray,
        cls_cur_all: np.ndarray,
    ):
        from icce.pairing_head.model import HeuristicHead

        if self.head is None or isinstance(self.head, HeuristicHead):
            h = self.head or HeuristicHead()
            return (
                h.match_probability(pair_x),
                None,
                h.change_probability(unary_past),
                h.change_probability(unary_cur),
            )

        import torch

        def t(a, dtype=torch.float32):
            return torch.as_tensor(a, dtype=dtype, device=self.device)

        with torch.no_grad():
            if pair_x.shape[0]:
                logit, state = self.head.forward_pair(
                    t(pair_x), t(cls_past, torch.long), t(cls_cur, torch.long)
                )
                match_p = torch.sigmoid(logit).cpu().numpy()
                state_p = torch.softmax(state, dim=-1).cpu().numpy()
            else:
                match_p = np.zeros((0,), np.float32)
                state_p = np.zeros((0, 3), np.float32)

            def verify(x, cls):
                if x.shape[0] == 0:
                    return np.zeros((0,), np.float32)
                return torch.sigmoid(
                    self.head.forward_unary(t(x), t(cls, torch.long))
                ).cpu().numpy()

            return match_p, state_p, verify(unary_past, cls_past_all), verify(unary_cur, cls_cur_all)

    # -- main --------------------------------------------------------------
    def pair(
        self,
        past: Sequence[Det],
        current: Sequence[Det],
        image_size: Tuple[int, int] = (256, 256),
    ) -> PairingResult:
        from icce.pairing_head.model import STATE_LABELS

        cands = candidate_pairs(past, current, self.match_radius_deg, self.max_candidates)
        pair_x = pair_features(past, current, cands, self.match_radius_deg)
        unary_past = unary_features(past, current, image_size)
        unary_cur = unary_features(current, past, image_size)

        cls_p = np.array([past[i].class_id for i, _ in cands], dtype=np.int64)
        cls_c = np.array([current[j].class_id for _, j in cands], dtype=np.int64)
        cls_p_all = np.array([d.class_id for d in past], dtype=np.int64)
        cls_c_all = np.array([d.class_id for d in current], dtype=np.int64)

        match_p, state_p, verify_past, verify_cur = self._scores(
            pair_x, unary_past, unary_cur, cls_p, cls_c, cls_p_all, cls_c_all
        )

        matches, un_past, un_cur = assign(
            cands, match_p, len(past), len(current),
            threshold=self.match_threshold, method=self.assignment,
        )
        pair_index = {(i, j): k for k, (i, j) in enumerate(cands)}

        outcomes: List[PairingOutcome] = []
        for i, j, s in matches:
            p, c = past[i], current[j]
            probs: Dict[str, float] = {}
            if state_p is not None:
                k = pair_index[(i, j)]
                probs = {lbl: float(state_p[k][m]) for m, lbl in enumerate(STATE_LABELS)}
                status = {
                    "stationary": STATUS_MATCHED,
                    "moved": STATUS_MOVED,
                    "modified": STATUS_CHANGED,
                }[max(probs, key=probs.get)]
            else:
                # heuristic baseline: distance threshold decides matched vs moved
                dist = float(np.hypot(p.lat - c.lat, p.lon - c.lon))
                status = STATUS_MOVED if dist > self.move_threshold_deg else STATUS_MATCHED

            outcomes.append(PairingOutcome(
                status=status, past_idx=i, current_idx=j, score=float(s),
                lat=c.lat, lon=c.lon, object_class=c.object_class,
                confidence=c.confidence, bbox_px=c.bbox_px, state_probs=probs,
            ))

        suppressed = 0
        for j in un_cur:
            if verify_cur[j] < self.verify_threshold:
                suppressed += 1
                continue
            c = current[j]
            outcomes.append(PairingOutcome(
                status=STATUS_NEW, past_idx=None, current_idx=j,
                score=float(verify_cur[j]), lat=c.lat, lon=c.lon,
                object_class=c.object_class, confidence=c.confidence, bbox_px=c.bbox_px,
            ))

        for i in un_past:
            if verify_past[i] < self.verify_threshold:
                suppressed += 1
                continue
            p = past[i]
            outcomes.append(PairingOutcome(
                status=STATUS_DISAPPEARED, past_idx=i, current_idx=None,
                score=float(verify_past[i]), lat=p.lat, lon=p.lon,
                object_class=p.object_class, confidence=p.confidence, bbox_px=p.bbox_px,
            ))

        return PairingResult(outcomes=outcomes, n_candidates=len(cands), n_suppressed=suppressed)
