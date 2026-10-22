"""Surface Dice at a tolerance, as leaderboard.py:194-200 computes it with the
surface_distance library's algorithm (SPEC section 7). Own implementation, numpy + scipy."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ._kernels import jit
from ._surface_table import SURFACE_AREA

_AREA = np.array(SURFACE_AREA, dtype=np.float64)


def _bbox(mask):
    lo, hi = [], []
    for ax in range(3):
        idx = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != ax)))
        if idx.size == 0:
            return None
        lo.append(int(idx[0]))
        hi.append(int(idx[-1]) + 1)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def neighbour_codes(m: np.ndarray) -> np.ndarray:
    """uint8 code per position p: bit weight 128>>(4a+2b+c) for m[p+(a-1,b-1,c-1)]
    (out of range = 0). Equals ndimage.correlate(m, [[[128,64],[32,16]],[[8,4],[2,1]]],
    mode='constant', cval=0). surface_dice zero-pads the crop by one voxel on the high side
    first (metrics.py _crop_to_bounding_box), so every window touching a voxel exists."""
    P = np.pad(m.astype(np.uint8), ((1, 0), (1, 0), (1, 0)))
    n0, n1, n2 = m.shape
    code = np.zeros(m.shape, np.uint8)
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = 128 >> (4 * a + 2 * b + c)
                code += P[a:a + n0, b:b + n1, c:c + n2] * np.uint8(w)
    return code


def _ball_offsets(tolerance):
    """All integer offsets whose Euclidean length is <= tolerance (same sqrt/compare as the
    EDT route: scipy returns sqrt of an integer sum of squares)."""
    r = int(np.floor(tolerance))
    g = np.arange(-r, r + 1)
    dz, dy, dx = np.meshgrid(g, g, g, indexing="ij")
    ss = (dz * dz + dy * dy + dx * dx)
    keep = np.sqrt(ss.astype(np.float64)) <= tolerance
    return np.stack([dz[keep], dy[keep], dx[keep]], axis=1).astype(np.int64)


@jit
def _mark_within(border, offs, out):
    n0, n1, n2 = border.shape
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                if border[i, j, k]:
                    for q in range(offs.shape[0]):
                        a = i + offs[q, 0]
                        b = j + offs[q, 1]
                        c = k + offs[q, 2]
                        if a >= 0 and a < n0 and b >= 0 and b < n1 and c >= 0 and c < n2:
                            out[a, b, c] = True


def within_distance(border, tolerance):
    """Boolean array: voxel is within Euclidean distance `tolerance` of some True voxel of
    `border`. Exactly distance_transform_edt(~border) <= tolerance, without computing the
    distance map (the Euclidean ball is a fixed set of integer offsets)."""
    out = np.zeros(border.shape, np.bool_)
    if border.any():
        _mark_within(np.ascontiguousarray(border), _ball_offsets(float(tolerance)), out)
    return out


def surface_dice(pred, gt, tolerance=2.0, spacing=(1.0, 1.0, 1.0)) -> float:
    """Surface Dice (pred, gt boolean 3D). Both empty -> 1.0; exactly one empty -> 0.0
    (leaderboard.py:194-198)."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    a_gt, a_pr = gt.any(), pred.any()
    if not a_gt and not a_pr:
        return 1.0
    if a_gt != a_pr:
        return 0.0
    if tuple(float(s) for s in spacing) != (1.0, 1.0, 1.0):
        raise NotImplementedError("only unit spacing is implemented (leaderboard default)")
    sl = _bbox(gt | pred)
    sides = []
    for m in (gt[sl], pred[sl]):
        m = np.pad(m, ((0, 1), (0, 1), (0, 1)))
        code = neighbour_codes(m)
        border = (code != 0) & (code != 255)
        sides.append((code, border))
    (cg, bg), (cp, bp) = sides
    tot_gt = _AREA[cg[bg]].sum()
    tot_pr = _AREA[cp[bp]].sum()
    # distance of each GT surface element to the pred surface, and vice versa
    near_pr = within_distance(bp, tolerance)
    ov_gt = _AREA[cg[bg & near_pr]].sum()
    del near_pr
    near_gt = within_distance(bg, tolerance)
    ov_pr = _AREA[cp[bp & near_gt]].sum()
    return float((ov_gt + ov_pr) / (tot_gt + tot_pr))
