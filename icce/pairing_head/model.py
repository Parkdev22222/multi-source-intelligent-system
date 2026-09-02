"""
The learned pairing head: three small MLPs over the features in `features.py`.

  match   : P(past_i and cur_j are the same physical object)
  state   : given a match, {stationary, moved, modified}
  verify  : P(an unmatched detection is a genuine change instance)
            -- this is what suppresses SAM3's open-vocabulary false positives,
               which dominate the pixel-F1 loss of the zero-shot baseline.

Total parameter count is 19,781, i.e. the head trains in minutes on cached
features and adds negligible latency next to SAM3 and the report LLM. That
matters for the deployment argument in the paper: the accuracy gain does not
cost the consumer service anything meaningful.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from icce.pairing_head.features import N_PAIR_FEATURES, N_UNARY_FEATURES

STATE_LABELS = ("stationary", "moved", "modified")


@dataclass
class HeadConfig:
    n_pair: int = N_PAIR_FEATURES
    n_unary: int = N_UNARY_FEATURES
    n_classes: int = 64
    class_dim: int = 8
    hidden: int = 64
    dropout: float = 0.1


def _mlp(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


class PairingHead(nn.Module):
    def __init__(self, cfg: Optional[HeadConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or HeadConfig()
        c = self.cfg

        self.class_emb = nn.Embedding(c.n_classes, c.class_dim)
        pair_in = c.n_pair + 2 * c.class_dim
        self.match = _mlp(pair_in, c.hidden, 1, c.dropout)
        self.state = _mlp(pair_in, c.hidden, len(STATE_LABELS), c.dropout)
        self.verify = _mlp(c.n_unary + c.class_dim, c.hidden, 1, c.dropout)

        # standardisation statistics, fitted on the training split
        self.register_buffer("pair_mean", torch.zeros(c.n_pair))
        self.register_buffer("pair_std", torch.ones(c.n_pair))
        self.register_buffer("unary_mean", torch.zeros(c.n_unary))
        self.register_buffer("unary_std", torch.ones(c.n_unary))

    # -- normalisation ------------------------------------------------------
    def fit_normalisation(self, pair_x: np.ndarray, unary_x: np.ndarray) -> None:
        def stats(a: np.ndarray, n: int):
            if a.size == 0:
                return torch.zeros(n), torch.ones(n)
            m = torch.tensor(a.mean(0), dtype=torch.float32)
            s = torch.tensor(a.std(0), dtype=torch.float32).clamp_min(1e-3)
            return m, s

        pm, ps = stats(pair_x, self.cfg.n_pair)
        um, us = stats(unary_x, self.cfg.n_unary)
        self.pair_mean.copy_(pm)
        self.pair_std.copy_(ps)
        self.unary_mean.copy_(um)
        self.unary_std.copy_(us)

    def _pair_input(self, x: torch.Tensor, cls_a: torch.Tensor, cls_b: torch.Tensor) -> torch.Tensor:
        x = (x - self.pair_mean) / self.pair_std
        n = self.cfg.n_classes
        return torch.cat([x, self.class_emb(cls_a % n), self.class_emb(cls_b % n)], dim=-1)

    # -- forward ------------------------------------------------------------
    def forward_pair(
        self, x: torch.Tensor, cls_past: torch.Tensor, cls_cur: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (match_logit [N], state_logits [N, 3])."""
        h = self._pair_input(x, cls_past, cls_cur)
        return self.match(h).squeeze(-1), self.state(h)

    def forward_unary(self, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """Returns change-instance logit [N]."""
        x = (x - self.unary_mean) / self.unary_std
        h = torch.cat([x, self.class_emb(cls % self.cfg.n_classes)], dim=-1)
        return self.verify(h).squeeze(-1)

    # -- persistence --------------------------------------------------------
    def save(self, path: Path, extra: Optional[Dict] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": asdict(self.cfg), "extra": extra or {}},
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> Tuple["PairingHead", Dict]:
        blob = torch.load(Path(path), map_location=map_location, weights_only=False)
        model = cls(HeadConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model, blob.get("extra", {})

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HybridHead:
    """Heuristic matching with the learned state and verify branches.

    The paper reports the production heuristic at one number and the learned
    head at another, but four things differ between those rows: the match
    rule, the state rule, the presence of a verifier at all, and greedy versus
    Hungarian assignment. The heuristic has no verifier -- its
    `change_probability` returns ones, so every unmatched detection is
    reported -- which leaves open how much of the gain is learning to match
    and how much is having any suppression step. This head isolates the match
    branch: everything downstream comes from the trained model, and only the
    pair score is the hand-tuned rule.

    The substitution is legitimate rather than a distribution mismatch. The
    verifier's labels come from each detection's coverage of the ground-truth
    change mask (`verify_labels`), which never refers to the matcher, so its
    decision does not assume the leftovers were produced by the learned
    matcher. It is scored for every detection in any case, and only applied to
    what the assignment leaves over.

    `_scores` dispatches on `isinstance(head, HeuristicHead)`, so this class
    deliberately is not one: it takes the learned branch and has to present
    the same `forward_pair` / `forward_unary` surface, returning logits that
    the caller squashes.
    """

    # Probabilities of exactly 0 or 1 -- the heuristic emits both, the first
    # from its hard radius gate -- have infinite logits, and sigmoid(inf) is
    # only finite by luck of the float. Clamp before inverting.
    _EPS = 1e-6

    def __init__(self, model: "PairingHead", heuristic: Optional["HeuristicHead"] = None) -> None:
        self.model = model
        self.heuristic = heuristic or HeuristicHead()
        self.cfg = model.cfg

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def forward_pair(self, x: torch.Tensor, cls_past: torch.Tensor,
                     cls_cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The heuristic reads raw feature columns; x is unstandardised here,
        # standardisation happening inside the model's own _pair_input.
        p = self.heuristic.match_probability(x.detach().cpu().numpy())
        p = np.clip(p, self._EPS, 1.0 - self._EPS)
        logit = torch.as_tensor(np.log(p / (1.0 - p)), dtype=x.dtype, device=x.device)
        _, state = self.model.forward_pair(x, cls_past, cls_cur)
        return logit, state

    def forward_unary(self, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        return self.model.forward_unary(x, cls)


class HeuristicHead:
    """The production rule as a drop-in baseline, so ablation rows share code.

    Reproduces `score = w_clip*clip + w_size*size + (1-w_clip-w_size)*geo`
    with the hard geodesic gate, using the same feature matrix as the MLP.
    """

    def __init__(self, w_clip: float = 0.7, w_size: float = 0.2,
                 match_threshold: float = 0.5) -> None:
        self.w_clip = w_clip
        self.w_size = w_size
        self.w_geo = max(0.0, 1.0 - w_clip - w_size)
        self.match_threshold = match_threshold

    def match_probability(self, pair_x: np.ndarray) -> np.ndarray:
        if pair_x.size == 0:
            return np.zeros((0,), dtype=np.float32)
        clip = pair_x[:, 0]
        geo_norm = pair_x[:, 1]
        log_area_ratio = pair_x[:, 3]

        size_sim = np.exp(-np.abs(log_area_ratio))
        geo_score = np.clip(1.0 - geo_norm, 0.0, 1.0)
        score = self.w_clip * clip + self.w_size * size_sim + self.w_geo * geo_score
        score = np.where(geo_norm > 1.0, 0.0, score)      # hard gate, as in production
        return score.astype(np.float32)

    def change_probability(self, unary_x: np.ndarray) -> np.ndarray:
        """No verifier in the production path: every unmatched detection counts."""
        return np.ones((unary_x.shape[0],), dtype=np.float32)
