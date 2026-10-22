"""VOI report as voi.py:67-159 computes it (26-connected components of the two binary
masks, union-foreground voxels only, log base 2, score 1/(1+alpha*VI)). SPEC section 8."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class VOIReport:
    voi_total: float
    voi_split: float   # first skimage output, H(PR|GT)
    voi_merge: float   # second skimage output, H(GT|PR)
    voi_score: float
    n_foreground: int
    connectivity: int


_STRUCT = {6: ndimage.generate_binary_structure(3, 1),
           18: ndimage.generate_binary_structure(3, 2),
           26: ndimage.generate_binary_structure(3, 3)}


def _cond_entropies(a: np.ndarray, b: np.ndarray):
    """(H(b|a), H(a|b)) in bits, the order skimage.metrics.variation_of_information(a, b) returns."""
    n = a.size
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ai = ai.astype(np.int64).ravel()
    bi = bi.astype(np.int64).ravel()
    nb = int(bi.max()) + 1
    key, c = np.unique(ai * nb + bi, return_counts=True)
    ka, kb = key // nb, key % nb
    pxy = c / n
    px = np.bincount(ai).astype(np.float64) / n
    py = np.bincount(bi).astype(np.float64) / n
    h_b_given_a = float(-(pxy * np.log2(pxy / px[ka])).sum())
    h_a_given_b = float(-(pxy * np.log2(pxy / py[kb])).sum())
    return h_b_given_a, h_a_given_b


def voi_report(pred, gt, *, connectivity=26, alpha=0.3, transform="one_over_one_plus") -> VOIReport:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    if connectivity not in _STRUCT:
        raise ValueError("connectivity must be 6, 18 or 26")
    if transform not in ("one_over_one_plus", "exp"):
        raise ValueError("bad transform")
    union = gt | pred
    if not union.any():
        return VOIReport(0.0, 0.0, 0.0, 1.0, 0, connectivity)
    idx = [np.flatnonzero(union.any(axis=tuple(i for i in range(3) if i != ax))) for ax in range(3)]
    sl = tuple(slice(int(i[0]), int(i[-1]) + 1) for i in idx)
    g = gt[sl]
    p = pred[sl]
    u = union[sl]
    gl, _ = ndimage.label(g, structure=_STRUCT[connectivity])
    pl, _ = ndimage.label(p, structure=_STRUCT[connectivity])
    a = gl[u]
    b = pl[u]
    del gl, pl
    # official call: variation_of_information(a=GT labels, b=PR labels)
    split, merge = _cond_entropies(a, b)
    total = float(split + merge)
    score = float(np.exp(-alpha * total)) if transform == "exp" else float(1.0 / (1.0 + alpha * total))
    return VOIReport(total, float(split), float(merge), score, int(u.sum()), connectivity)
