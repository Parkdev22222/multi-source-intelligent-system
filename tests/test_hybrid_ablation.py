"""The hybrid row must actually be a hybrid.

`run_cd_eval --hybrid-ablation` adds a row that pairs with the hand-tuned rule
and verifies with the trained model, to separate learning to match from having
any verifier at all -- the production heuristic has none, so the gap between it
and the learned head confounds the two.

A row that silently collapsed onto one of its parents would answer that
question wrongly and look like a real measurement while doing it. These tests
pin the three properties that make it a hybrid:

  * its match scores are the heuristic's, not the model's;
  * its verify scores are the model's, not the heuristic's constant one;
  * it therefore suppresses detections, which the heuristic never does.
"""

from __future__ import annotations

import numpy as np
import torch

from icce.pairing_head.features import N_PAIR_FEATURES, N_UNARY_FEATURES
from icce.pairing_head.model import HeuristicHead, HybridHead, PairingHead

RNG = np.random.default_rng(11)


def _inputs(n=64):
    pair_x = RNG.random((n, N_PAIR_FEATURES)).astype(np.float32)
    pair_x[:, 1] *= 0.9                      # keep most inside the radius gate
    unary_x = RNG.random((n, N_UNARY_FEATURES)).astype(np.float32)
    cls = np.zeros(n, dtype=np.int64)
    return pair_x, unary_x, cls


def _model():
    torch.manual_seed(0)
    m = PairingHead()
    m.eval()
    return m


def test_match_comes_from_the_heuristic():
    pair_x, _, cls = _inputs()
    model, heur = _model(), HeuristicHead()
    hybrid = HybridHead(model)

    with torch.no_grad():
        logit, _ = hybrid.forward_pair(torch.from_numpy(pair_x),
                                       torch.from_numpy(cls), torch.from_numpy(cls))
    got = torch.sigmoid(logit).numpy()
    want = heur.match_probability(pair_x)

    # Round-tripping through a logit costs a little precision at the clamp, so
    # compare at the tolerance the clamp allows rather than exactly.
    assert np.allclose(got, want, atol=1e-4), "match score is not the heuristic's"


def test_verify_comes_from_the_model():
    _, unary_x, cls = _inputs()
    model = _model()
    hybrid = HybridHead(model)

    with torch.no_grad():
        got = hybrid.forward_unary(torch.from_numpy(unary_x), torch.from_numpy(cls))
        want = model.forward_unary(torch.from_numpy(unary_x), torch.from_numpy(cls))
    assert torch.allclose(got, want), "verify score is not the model's"

    # The heuristic's own verifier is the constant 1, which is the behaviour
    # this row exists to remove.
    assert not np.allclose(torch.sigmoid(got).numpy(),
                           HeuristicHead().change_probability(unary_x)), \
        "hybrid kept the heuristic's pass-everything verifier"


def test_hard_gate_survives_the_logit_round_trip():
    pair_x, _, cls = _inputs(16)
    pair_x[:, 1] = 2.0                        # every pair outside the radius
    hybrid = HybridHead(_model())

    with torch.no_grad():
        logit, _ = hybrid.forward_pair(torch.from_numpy(pair_x),
                                       torch.from_numpy(cls), torch.from_numpy(cls))
    p = torch.sigmoid(logit).numpy()
    assert np.all(np.isfinite(logit.numpy())), "gate produced a non-finite logit"
    assert np.all(p < 1e-5), "hard radius gate did not survive as a near-zero score"


if __name__ == "__main__":
    test_match_comes_from_the_heuristic()
    test_verify_comes_from_the_model()
    test_hard_gate_survives_the_logit_round_trip()
    print("OK")
