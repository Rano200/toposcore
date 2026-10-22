"""Drop-in for the `betti_matching` pybind module's `compute_matching`
(Betti-Matching-3D, _BettiMatching.cpp:426-540), 3D inputs only.

    compute_matching(a, b, include_input1_unmatched_pairs=True,
                     include_input2_unmatched_pairs=True) -> BettiMatchingResult
    compute_matching([a1, a2], [b1, b2], ...)            -> [BettiMatchingResult, ...]

Inputs are any real-valued 3D numpy arrays (cast to C-contiguous float64, as the
binding does). Results carry the binding's attribute names, dtypes (int64) and
shapes ((n, 3) per homology dimension 0, 1, 2), with coordinates given as the
"parent voxel" of each birth/death cell (data_structures.cpp:172-261) and the
pairs listed in the order the reference appends them. Unmatched lists are None
when not requested, like the binding's std::optional fields.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import _kernels as K
from . import _pairs as P
from .metric import default_workers
from .betti import _cell_values, _ranks, _sorted_valid

NDIM = 3


@dataclass
class BettiMatchingResult:
    input1_matched_birth_coordinates: List[np.ndarray]
    input1_matched_death_coordinates: List[np.ndarray]
    input2_matched_birth_coordinates: List[np.ndarray]
    input2_matched_death_coordinates: List[np.ndarray]
    num_matched: np.ndarray
    input1_unmatched_birth_coordinates: Optional[List[np.ndarray]] = None
    input1_unmatched_death_coordinates: Optional[List[np.ndarray]] = None
    num_unmatched_input1: Optional[np.ndarray] = None
    input2_unmatched_birth_coordinates: Optional[List[np.ndarray]] = None
    input2_unmatched_death_coordinates: Optional[List[np.ndarray]] = None
    num_unmatched_input2: Optional[np.ndarray] = None


# --------------------------------------------------------------------------
# parent voxels (CubicalGridComplex::getParentVoxel)
# --------------------------------------------------------------------------
def _parent_voxels(ids, dim, cell_vals, vol, shape):
    ids = np.asarray(ids, np.int64)
    X, Y, Z = shape
    if ids.size == 0:
        return np.zeros((0, NDIM), np.int64)
    if dim == 3:
        mY, mZ = Y - 1, Z - 1
        x = ids // (mY * mZ)
        r = ids - x * mY * mZ
        y = r // mZ
        z = r - y * mZ
        t = np.zeros_like(ids)
    else:
        vid = ids // 3 if dim in (1, 2) else ids
        t = ids - vid * 3 if dim in (1, 2) else np.zeros_like(ids)
        x = vid // (Y * Z)
        r = vid - x * Y * Z
        y = r // Z
        z = r - y * Z
    birth = cell_vals[ids]
    base = np.stack([x, y, z], axis=1)
    if dim == 0:
        return base

    def v(off):
        c = base + np.asarray(off, np.int64)
        return vol[c[:, 0], c[:, 1], c[:, 2]], c

    out = base.copy()
    if dim == 1:
        for typ, off in ((0, (1, 0, 0)), (1, (0, 1, 0)), (2, (0, 0, 1))):
            sel = t == typ
            if sel.any():
                c = base[sel] + np.asarray(off, np.int64)
                hit = vol[c[:, 0], c[:, 1], c[:, 2]] == birth[sel]
                o = out[sel]
                o[hit] = c[hit]
                out[sel] = o
        return out
    if dim == 2:
        cand = {0: [(0, 1, 1), (0, 1, 0), (0, 0, 1)],
                1: [(1, 0, 1), (1, 0, 0), (0, 0, 1)],
                2: [(1, 1, 0), (1, 0, 0), (0, 1, 0)]}
        for typ, offs in cand.items():
            sel = np.nonzero(t == typ)[0]
            if sel.size == 0:
                continue
            done = np.zeros(sel.size, bool)
            for off in offs:
                c = base[sel] + np.asarray(off, np.int64)
                hit = (~done) & (vol[c[:, 0], c[:, 1], c[:, 2]] == birth[sel])
                out[sel[hit]] = c[hit]
                done |= hit
        return out
    # dim 3: (x,y,z),(x,y,z+1),(x,y+1,z),(x,y+1,z+1),(x+1,y,z),... else (x+1,y+1,z+1)
    done = np.zeros(ids.size, bool)
    for off in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0)]:
        c = base + np.asarray(off, np.int64)
        hit = (~done) & (vol[c[:, 0], c[:, 1], c[:, 2]] == birth)
        out[hit] = c[hit]
        done |= hit
    out[~done] = base[~done] + 1
    return out


def _empty():
    return np.zeros((0, NDIM), np.int64)


# --------------------------------------------------------------------------
def _match_one(a, b):
    """Returns per-dimension lists: pairs (birth_ids, death_ids) for inputs 0/1
    and match index arrays (idx0, idx1)."""
    X, Y, Z = a.shape
    c = np.minimum(a, b)
    vols = (a, b, c)
    E, S, C = zip(*(_cell_values(v) for v in vols))
    vb = [v.reshape(-1) for v in vols]
    ne = E[0].shape[0]
    levels = [np.unique(v) for v in vols]
    levels = [lv if lv.shape[0] <= 16 else None for lv in levels]
    eorder = [_sorted_valid(e, lv) for e, lv in zip(E, levels)]
    sorder = [_sorted_valid(s, lv) for s, lv in zip(S, levels)]
    mX, mY, mZ = X - 1, Y - 1, Z - 1
    pairs = [[None] * 3, [None] * 3]   # pairs[k][d] = (birth_ids, death_ids)
    matches = [None] * 3               # matches[d] = (idx0, idx1)
    ii = lambda n: np.empty(max(n, 1), np.int64)

    # dimension 2
    removed = [np.zeros(ne, np.bool_) for _ in range(3)]
    if X > 1 and Y > 1 and Z > 1:
        nc = C[2].shape[0]
        keys = []
        for k in range(2):
            key = np.full(nc, -1, np.int64)
            bo, do = ii(nc), ii(nc)
            n = P.dim2_input_pairs(sorder[k], S[k], C[k], C[2], Y, Z, mX, mY, mZ, key,
                                   removed[k], bo, do)
            pairs[k][2] = (bo[:n].copy(), do[:n].copy())
            keys.append(key)
        m0, m1 = ii(nc), ii(nc)
        n = P.dim2_comp_pairs(sorder[2], S[2], C[2], Y, Z, mX, mY, mZ, keys[0], keys[1],
                              removed[2], m0, m1)
        matches[2] = (m0[:n].copy(), m1[:n].copy())
    else:
        for k in range(2):
            pairs[k][2] = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        matches[2] = (np.zeros(0, np.int64), np.zeros(0, np.int64))

    # dimension 1
    cols = [o[~r[o]] for o, r in zip(sorder, removed)]
    rank = [_ranks(o, ne) for o in eorder]
    piv = []
    for k in range(3):
        pr = np.empty(cols[k].shape[0], np.int64)
        K.reduce_dim1(cols[k], rank[k], eorder[k].shape[0], Y, Z, pr)
        piv.append(pr)
    keyidx = []
    for k in range(2):
        pr = piv[k]
        ok = pr >= 0
        be = eorder[k][pr[ok]]
        de = cols[k][ok]
        pers = E[k][be] != S[k][de]
        be, de = be[pers], de[pers]
        pairs[k][1] = (be, de)
        ki = np.full(ne, -1, np.int64)
        ki[be] = np.arange(be.shape[0])
        keyidx.append(ki)
    prc = piv[2]
    okc = prc >= 0
    cb = eorder[2][prc[okc]]
    cpos = np.nonzero(okc)[0]
    pers = E[2][cb] != S[2][cols[2][cpos]]
    cpos = cpos[pers]  # comparison pairs, in pairsComp order
    sel = np.ones(cpos.shape[0], bool)
    idx = []
    for k in range(2):
        pr = np.empty(cols[2].shape[0], np.int64)
        K.reduce_dim1(cols[2], rank[k], eorder[k].shape[0], Y, Z, pr)
        r = pr[cpos]
        hit = r >= 0
        e = eorder[k][np.where(hit, r, 0)]
        j = np.where(hit, keyidx[k][e], -1)
        sel &= j >= 0
        idx.append(j)
    matches[1] = (idx[0][sel], idx[1][sel])

    # dimension 0
    n_v = vb[0].shape[0]
    keys = []
    for k in range(2):
        key = np.full(n_v, -1, np.int64)
        bo, do = ii(n_v), ii(n_v)
        n = P.dim0_input_pairs(eorder[k], E[k], vb[k], Y, Z, key, bo, do)
        pairs[k][0] = (bo[:n].copy(), do[:n].copy())
        keys.append(key)
    m0, m1 = ii(n_v), ii(n_v)
    n = P.dim0_comp_pairs(eorder[2], E[2], vb[2], vb[0], vb[1], Y, Z, keys[0], keys[1], m0, m1)
    matches[0] = (m0[:n].copy(), m1[:n].copy())
    return pairs, matches, (E, S, C)


def _result(a, b, inc1, inc2):
    shape = a.shape
    pairs, matches, (E, S, C) = _match_one(a, b)
    vols = (a, b)
    cellvals = {0: lambda k: vols[k].reshape(-1), 1: lambda k: E[k], 2: lambda k: S[k], 3: lambda k: C[k]}
    pv = lambda ids, dim, k: _parent_voxels(ids, dim, cellvals[dim](k), vols[k], shape)
    mb = [[None] * 3, [None] * 3]
    md = [[None] * 3, [None] * 3]
    ub = [[None] * 3, [None] * 3]
    ud = [[None] * 3, [None] * 3]
    for d in range(3):
        for k in range(2):
            bi, di = pairs[k][d]
            mi = matches[d][k]
            mb[k][d] = pv(bi[mi], d, k)
            md[k][d] = pv(di[mi], d + 1, k)
            un = np.ones(bi.shape[0], bool)
            un[mi] = False
            ub[k][d] = pv(bi[un], d, k)
            ud[k][d] = pv(di[un], d + 1, k)
    res = BettiMatchingResult(
        mb[0], md[0], mb[1], md[1],
        np.array([matches[d][0].shape[0] for d in range(3)], np.int64))
    if inc1:
        res.input1_unmatched_birth_coordinates = ub[0]
        res.input1_unmatched_death_coordinates = ud[0]
        res.num_unmatched_input1 = np.array([x.shape[0] for x in ub[0]], np.int64)
    if inc2:
        res.input2_unmatched_birth_coordinates = ub[1]
        res.input2_unmatched_death_coordinates = ud[1]
        res.num_unmatched_input2 = np.array([x.shape[0] for x in ub[1]], np.int64)
    return res


def _as_volume(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("inputs must be numpy arrays (the binding uses noconvert)")
    v = np.ascontiguousarray(x, dtype=np.float64)
    if v.ndim != 3:
        raise NotImplementedError(f"only 3D volumes are implemented, got ndim={v.ndim}")
    if np.isnan(v).any():
        raise ValueError("NaN values are not supported")
    return v


def compute_matching(input1, input2, include_input1_unmatched_pairs=True,
                     include_input2_unmatched_pairs=True, workers=None):
    """workers: threads over batch items (None -> min(8, cpu_count); 1 -> serial)."""
    single = isinstance(input1, np.ndarray) and isinstance(input2, np.ndarray)
    in1 = [input1] if single else list(input1)
    in2 = [input2] if single else list(input2)
    if len(in1) != len(in2):
        raise ValueError(f"Different numbers of inputs where provided: {len(in1)} (inputs1) "
                         f"and {len(in2)} (inputs2)")
    pairs = []
    for a, b in zip(in1, in2):
        a, b = _as_volume(a), _as_volume(b)
        if a.shape != b.shape:
            raise ValueError(f"The shapes of the input volumes must agree. Got {a.shape} and {b.shape}")
        pairs.append((a, b))
    big = max((a.size for a, _ in pairs), default=0)
    nw = min(default_workers(big) if workers is None else int(workers), max(1, len(pairs)))
    if nw < 1:
        raise ValueError("workers must be >= 1")
    f = lambda ab: _result(ab[0], ab[1], include_input1_unmatched_pairs, include_input2_unmatched_pairs)
    if nw == 1:
        out = [f(ab) for ab in pairs]
    else:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            out = list(ex.map(f, pairs))
    return out[0] if single else out
