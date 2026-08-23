"""
Binary change mask -> discrete change instances.

Public CD benchmarks ship *pixel* labels, but a report speaks in objects
("two new buildings"). To score instance-level change detection and the count
claims inside reports we convert GT masks to connected components with a small
amount of morphological cleanup.

`connected_components` uses `scipy.ndimage.label` when SciPy is available and
otherwise falls back to a pure-NumPy two-pass union-find, so the harness runs
in a minimal container.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


def _label_scipy(binm: np.ndarray) -> Optional[Tuple[np.ndarray, int]]:
    try:                                       # pragma: no cover - optional dep
        from scipy import ndimage
    except Exception:
        return None
    structure = np.ones((3, 3), dtype=bool)    # 8-connectivity
    lab, n = ndimage.label(binm, structure=structure)
    return lab, int(n)


def _label_numpy(binm: np.ndarray) -> Tuple[np.ndarray, int]:
    """Two-pass connected-component labelling with union-find (8-connectivity)."""
    h, w = binm.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: List[int] = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(h):
        row = binm[y]
        for x in range(w):
            if not row[x]:
                continue
            neigh = []
            if y > 0:
                for dx in (-1, 0, 1):
                    xx = x + dx
                    if 0 <= xx < w and labels[y - 1, xx]:
                        neigh.append(labels[y - 1, xx])
            if x > 0 and labels[y, x - 1]:
                neigh.append(labels[y, x - 1])

            if not neigh:
                parent.append(nxt)
                labels[y, x] = nxt
                nxt += 1
            else:
                m = min(neigh)
                labels[y, x] = m
                for nb in neigh:
                    union(m, nb)

    # second pass: flatten
    remap: Dict[int, int] = {}
    out = np.zeros_like(labels)
    count = 0
    nz = np.nonzero(labels)
    for y, x in zip(*nz):
        root = find(int(labels[y, x]))
        if root not in remap:
            count += 1
            remap[root] = count
        out[y, x] = remap[root]
    return out, count


def label_mask(binm: np.ndarray) -> Tuple[np.ndarray, int]:
    got = _label_scipy(binm)
    return got if got is not None else _label_numpy(binm)


def _binarize(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if m.dtype == bool:
        return m
    return m > 127 if m.max() > 1 else m.astype(bool)


def connected_components(
    mask: np.ndarray,
    min_area: int = 32,
) -> List[Dict]:
    """Return [{bbox, area, mask, centroid}] for every change blob >= min_area."""
    binm = _binarize(mask)
    if not binm.any():
        return []

    labels, n = label_mask(binm)
    out: List[Dict] = []
    for lid in range(1, n + 1):
        comp = labels == lid
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(comp)
        out.append({
            "bbox": (float(xs.min()), float(ys.min()),
                     float(xs.max() + 1), float(ys.max() + 1)),
            "area": area,
            "centroid": (float(xs.mean()), float(ys.mean())),
            "mask": comp,
        })
    out.sort(key=lambda d: -d["area"])
    return out


def instances_from_mask(
    mask: np.ndarray,
    min_area: int = 32,
    change_type: str = "changed",
    object_class: str = "building",
    keep_masks: bool = False,
):
    """GT `ChangeInstance` list for `icce.metrics.instance_metrics`.

    `change_type` defaults to the direction-agnostic "changed": LEVIR-CD and
    WHU-CD annotate *that* a building changed, not whether it appeared or was
    demolished. Direction-aware evaluation is only run on the LEVIR-CC subset,
    where the human captions state the direction explicitly.
    """
    from icce.metrics.instance_metrics import ChangeInstance

    comps = connected_components(mask, min_area=min_area)
    return [
        ChangeInstance(
            bbox=c["bbox"],
            change_type=change_type,
            score=1.0,
            object_class=object_class,
            mask=c["mask"] if keep_masks else None,
        )
        for c in comps
    ]


def instances_to_mask(
    instances: Sequence,
    shape: Tuple[int, int],
) -> np.ndarray:
    """Rasterise predicted instances back to a binary mask for pixel metrics.

    Uses each instance's mask when present, otherwise fills its bounding box.
    """
    out = np.zeros(shape, dtype=bool)
    for inst in instances:
        m = getattr(inst, "mask", None)
        if m is not None and m.shape == shape:
            out |= m.astype(bool)
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in inst.bbox]
        x1 = max(0, min(shape[1], x1))
        x2 = max(0, min(shape[1], x2))
        y1 = max(0, min(shape[0], y1))
        y2 = max(0, min(shape[0], y2))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = True
    return out
