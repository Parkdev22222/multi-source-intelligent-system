"""End-to-end smoke test for the learned pairing head on synthetic scenes.

The synthetic generator deliberately reproduces the four failure modes that
cost the production heuristic accuracy on real LEVIR-CD tiles:

  1. **Registration drift** -- a per-scene global pixel shift between the two
     acquisitions, which pushes true matches past the hard geodesic gate.
  2. **Look-alike rows** -- terraces of identical townhouses whose CLIP
     embeddings are nearly the same, so appearance alone cannot disambiguate
     them and greedy assignment steals partners.
  3. **Detector false positives** -- low-confidence, poorly-segmented SAM3
     detections present in one frame only. The heuristic has no verifier, so
     every one of them is reported as a new building.
  4. **Missed detections** -- a stable building occasionally absent from one
     frame, which must not be reported as a change.

Ground truth is known by construction, so this test tells us whether the head
architecture can express the decision at all before we spend GPU hours on it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from icce.metrics.instance_metrics import ChangeInstance, InstanceEvaluator
from icce.pairing_head.cache import CacheWriter, CachedDet, CachedSample, load_cache, sample_dets
from icce.pairing_head.infer import LearnedPairer
from icce.pairing_head.train import build_tensors, select_thresholds, train

RNG = np.random.default_rng(7)
IMG = 256
GSD_DEG = 0.5 / 111_320.0     # 0.5 m/px expressed in degrees


def _det(idx, x, y, w, h, emb, coverage, cls_id=0, conf=None, mask_frac=0.8):
    lat = 36.0 - (y + h / 2) * GSD_DEG
    lon = 127.0 + (x + w / 2) * GSD_DEG
    return CachedDet(
        det_id=f"d{idx}", object_class="building", class_id=cls_id,
        confidence=float(conf if conf is not None
                         else np.clip(RNG.normal(0.8, 0.08), 0.3, 0.99)),
        bbox_px=[float(x), float(y), float(x + w), float(y + h)],
        lat=lat, lon=lon,
        geo_bbox=[lat - h / 2 * GSD_DEG, lon - w / 2 * GSD_DEG,
                  lat + h / 2 * GSD_DEG, lon + w / 2 * GSD_DEG],
        mask_area=int(w * h * mask_frac), coverage=coverage,
    )


def _bare_ground(h=IMG, w=IMG):
    """Smooth ground with mild sensor grain, as bare land looks at 0.5 m GSD."""
    base = np.linspace(100, 140, w, dtype=np.float32)[None, :].repeat(h, 0)
    base = base + RNG.normal(0, 5, (h, w))
    return base.clip(0, 255).astype(np.uint8)[..., None].repeat(3, -1)


def _draw_building(img, x, y, w, h, tone=None):
    """A bright roof with an outline and a ridge -- enough real edge structure
    for the cross-frame extractor to have something to measure."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    x2, y2 = min(IMG, x + w), min(IMG, y + h)
    if x2 <= x or y2 <= y:
        return
    tone = tone if tone is not None else RNG.integers(180, 225)
    img[y:y2, x:x2] = tone
    img[y:y2, x:min(IMG, x + 1)] = tone - 45          # outline
    img[y:min(IMG, y + 1), x:x2] = tone - 45
    mid = (y + y2) // 2
    img[mid:min(IMG, mid + 1), x:x2] = tone - 30      # ridge


def make_sample(pid: str, split: str, n_stable=8, n_changed=3,
                n_spurious=3, drift_px=6.0, n_lookalike=4):
    """One synthetic scene, rendered as real imagery.

    Both frames are drawn as actual pixel arrays and the cross-frame features
    are then computed by the production extractor rather than being written by
    hand. A hand-authored feature vector would let the verifier learn a signal
    that does not exist in real data; rendering forces the test to measure what
    the extractor can actually recover.
    """
    from icce.convert import cross_frame

    past_img = _bare_ground()
    cur_img = past_img.copy()

    past, cur, past_emb, cur_emb, gt = [], [], [], [], []
    dx, dy = RNG.normal(scale=drift_px), RNG.normal(scale=drift_px)

    def add(lst, embs, x, y, w, h, e, coverage, **kw):
        lst.append(_det(len(lst), x, y, w, h, e, coverage, **kw))
        embs.append(e)

    # (2) a terrace of look-alike townhouses, present in both frames
    proto = RNG.normal(size=16); proto /= np.linalg.norm(proto)
    row_x, row_y = RNG.integers(10, 100), RNG.integers(10, 180)
    for k in range(n_lookalike):
        x, y, w, h = row_x + k * 26, row_y, 22, 22
        tone = RNG.integers(190, 215)
        _draw_building(past_img, x, y, w, h, tone)
        _draw_building(cur_img, x + dx, y + dy, w, h, tone)
        e = proto + RNG.normal(scale=0.03, size=16); e /= np.linalg.norm(e)
        add(past, past_emb, x, y, w, h, e, 0.0)
        e2 = proto + RNG.normal(scale=0.03, size=16); e2 /= np.linalg.norm(e2)
        add(cur, cur_emb, x + dx, y + dy, w, h, e2, 0.0)

    # ordinary stable buildings, sometimes missed by the detector in one frame
    for k in range(n_stable):
        x, y = RNG.integers(0, 200), RNG.integers(0, 200)
        w, h = RNG.integers(16, 40), RNG.integers(16, 40)
        tone = RNG.integers(180, 225)
        _draw_building(past_img, x, y, w, h, tone)
        _draw_building(cur_img, x + dx, y + dy, w, h, tone)

        e = RNG.normal(size=16); e /= np.linalg.norm(e)
        miss_past = RNG.random() < 0.08
        if not miss_past:
            add(past, past_emb, x, y, w, h, e, 0.0)
        e2 = e + RNG.normal(scale=0.10, size=16); e2 /= np.linalg.norm(e2)
        if miss_past or RNG.random() >= 0.08:
            add(cur, cur_emb, x + dx + RNG.normal(scale=1.5),
                y + dy + RNG.normal(scale=1.5), w, h, e2, 0.0)

    # genuine new construction -- drawn only in the current frame
    for k in range(n_changed):
        x, y = RNG.integers(0, 200), RNG.integers(0, 200)
        w, h = RNG.integers(16, 40), RNG.integers(16, 40)
        _draw_building(cur_img, x, y, w, h)
        e = RNG.normal(size=16); e /= np.linalg.norm(e)
        add(cur, cur_emb, x, y, w, h, e, 0.95)
        gt.append([float(x), float(y), float(x + w), float(y + h)])

    # (3) detector false positives: boxes over ground that did not change
    for k in range(n_spurious):
        x, y = RNG.integers(0, 220), RNG.integers(0, 220)
        w, h = RNG.integers(8, 18), RNG.integers(8, 18)
        e = RNG.normal(size=16); e /= np.linalg.norm(e)
        target, embs = (cur, cur_emb) if RNG.random() < 0.6 else (past, past_emb)
        add(target, embs, x, y, w, h, e, 0.0,
            conf=float(np.clip(RNG.normal(0.38, 0.06), 0.05, 0.55)),
            mask_frac=0.35)

    s = CachedSample(
        pair_id=pid, dataset="SYNTH", split=split, image_size=(IMG, IMG),
        past=past, current=cur, gt_instances=gt, gt_change_present=bool(gt),
        parent_scene=pid.rsplit("_", 2)[0],
    )

    # cross-frame evidence, computed by the production extractor on the
    # rendered pixels (no CLIP available offline, so the pixel statistics
    # carry the signal alone -- an under-estimate of the deployed system)
    for rows, own, other in ((s.past, past_img, cur_img), (s.current, cur_img, past_img)):
        if not rows:
            continue
        xf = cross_frame.compute(own, other, [d.bbox_px for d in rows])
        for d, v in zip(rows, xf):
            d.xf = [float(x) for x in v]

    return s, np.array(past_emb, np.float32), np.array(cur_emb, np.float32)


def build_cache(root: Path, split: str, n: int) -> Path:
    d = root / split
    w = CacheWriter(d)
    for i in range(n):
        s, pe, ce = make_sample(f"{split}_{i//4}_{i%4}_0", split)
        w.add(s, pe, ce)
    w.close()
    return d


def instance_f1(pairer, samples, emb) -> float:
    ev = InstanceEvaluator(iou_thr=0.5, score_types=False)
    for s in samples:
        past, cur = sample_dets(s, emb)
        res = pairer.pair(past, cur, image_size=s.image_size)
        ev.update(res.change_instances(), [ChangeInstance(bbox=tuple(b)) for b in s.gt_instances])
    return ev.compute().f1


def test_head_beats_heuristic():
    root = Path(tempfile.mkdtemp())
    tr_dir = build_cache(root, "train", 60)
    va_dir = build_cache(root, "val", 20)
    te_dir = build_cache(root, "test", 20)

    tr_s, tr_e = load_cache(tr_dir)
    va_s, va_e = load_cache(va_dir)
    te_s, te_e = load_cache(te_dir)

    radius = 0.001
    train_t = build_tensors(tr_s, tr_e, radius)
    val_t = build_tensors(va_s, va_e, radius)
    print("train:", train_t.summary())

    model, info = train(train_t, val_t, epochs=40, lr=3e-3, device="cpu", seed=0)
    thr = select_thresholds(model, va_s, va_e, radius, "cpu")

    learned = LearnedPairer(head=model, match_radius_deg=radius,
                            match_threshold=thr["match_threshold"],
                            verify_threshold=thr["verify_threshold"])
    heuristic = LearnedPairer.from_checkpoint(None, match_radius_deg=radius)

    f1_learned = instance_f1(learned, te_s, te_e)
    f1_heur = instance_f1(heuristic, te_s, te_e)
    print(f"test instance F1 -- heuristic {f1_heur:.4f} | learned {f1_learned:.4f}")

    # The synthetic task is built so the heuristic *cannot* solve it: without a
    # verifier every spurious detection is reported as new construction.
    assert f1_learned > 0.6, f"head failed to learn the synthetic task (F1={f1_learned:.3f})"
    assert f1_learned > f1_heur + 0.05, (
        f"head must clear the heuristic by a real margin "
        f"(learned {f1_learned:.3f} vs heuristic {f1_heur:.3f})"
    )


if __name__ == "__main__":
    test_head_beats_heuristic()
    print("OK")
