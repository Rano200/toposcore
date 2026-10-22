"""Betti matching of two 3D volumes (sublevel filtration, V-construction), a
re-implementation of the counts that Betti-Matching-3D's dim3::BettiMatching
returns. See SPEC.md for the reference file:line behind every rule.

match_counts(a, b) -> int64 array (3, 3): rows = homology dimension 0..2,
columns = (matched, input-a pairs, input-b pairs), counting only pairs with
birth != death, exactly like the reference's matched / unmatched lists.
"""
from __future__ import annotations

import numpy as np

from . import _kernels as K

INF = np.inf


def _padded(v):
    p = np.full((v.shape[0] + 1, v.shape[1] + 1, v.shape[2] + 1), INF, dtype=v.dtype)
    p[:-1, :-1, :-1] = v
    return p


def _cell_values(v, edges=True):
    """Filtration values for every cell id, INF for ids that are not cells.
    Returns (edge_vals[3N], square_vals[3N], cube_vals[mX*mY*mZ + 1])."""
    X, Y, Z = v.shape
    p = _padded(v)
    c = p[:-1, :-1, :-1]
    ex = np.maximum(c, p[1:, :-1, :-1])
    ey = np.maximum(c, p[:-1, 1:, :-1])
    ez = np.maximum(c, p[:-1, :-1, 1:])
    if edges:
        evals = np.empty(ex.shape + (3,), v.dtype)
        evals[..., 0], evals[..., 1], evals[..., 2] = ex, ey, ez
        evals = evals.reshape(-1)
    else:
        evals = None  # caller recomputes them later to keep peak memory down
    s0 = np.maximum(ey, p[:-1, :-1, 1:]).copy()  # spans y,z
    s0 = np.maximum(s0, p[:-1, 1:, 1:])
    s1 = np.maximum(ex, p[:-1, :-1, 1:])          # spans x,z
    s1 = np.maximum(s1, p[1:, :-1, 1:])
    s2 = np.maximum(ex, p[:-1, 1:, :-1])          # spans x,y
    s2 = np.maximum(s2, p[1:, 1:, :-1])
    svals = np.empty(s0.shape + (3,), v.dtype)
    svals[..., 0], svals[..., 1], svals[..., 2] = s0, s1, s2
    svals = svals.reshape(-1)
    # cubes (x<mX, y<mY, z<mZ) + outside node
    if X > 1 and Y > 1 and Z > 1:
        cub = np.maximum(s2[: X - 1, : Y - 1, : Z - 1], s2[: X - 1, : Y - 1, 1:Z]).reshape(-1)
    else:
        cub = np.empty(0, dtype=v.dtype)
    cvals = np.concatenate([cub, np.array([INF], dtype=v.dtype)])
    return evals, svals, cvals


_I32_MAX = 2**31 - 1


def _idt(n):
    """Smallest of int32/int64 that holds ids and ranks below n (halves memory; values unchanged)."""
    return np.int32 if n < _I32_MAX else np.int64


def _sorted_valid(vals, levels=None):
    """Valid ids (finite value) sorted by (value, id), i.e. a stable sort by
    value. With few distinct levels (binary masks) a bucket pass is used, which
    yields the identical order."""
    if levels is not None and len(levels) <= 16:
        dt = _idt(vals.shape[0])
        return np.concatenate([np.flatnonzero(vals == u).astype(dt) for u in levels])
    order = np.argsort(vals, kind="stable")
    nvalid = int(np.count_nonzero(np.isfinite(vals)))
    return order[:nvalid].astype(_idt(vals.shape[0]))


def _ranks(order, n):
    dt = _idt(n)
    r = np.full(n, -1, dt)
    r[order] = np.arange(order.shape[0], dtype=dt)
    return r


def match_counts(a, b) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError(f"need two 3D arrays of equal shape, got {a.shape} {b.shape}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("NaN values are not supported (the leaderboard maps NaN to 0 first)")
    X, Y, Z = a.shape
    # Filtration values only ever get compared (max/min/==), so float32 gives identical
    # results whenever every input value is exactly representable (always true for masks).
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    if np.array_equal(a32, a) and np.array_equal(b32, b):
        a, b = a32, b32
    del a32, b32
    c = np.minimum(a, b)
    out = np.zeros((3, 3), np.int64)
    vols = (a, b, c)
    _, S, C = zip(*(_cell_values(v, edges=False) for v in vols))
    vb = [v.reshape(-1) for v in vols]
    ne = 3 * vb[0].shape[0]
    levels = [np.unique(v) for v in vols]  # cell values are a subset of these
    levels = [lv if lv.shape[0] <= 16 else None for lv in levels]
    sorder = [_sorted_valid(s, lv) for s, lv in zip(S, levels)]
    mX, mY, mZ = X - 1, Y - 1, Z - 1

    # ---- dimension 2 ------------------------------------------------------
    removed = [np.zeros(ne, np.bool_) for _ in range(3)]
    if X > 1 and Y > 1 and Z > 1:
        key0 = np.zeros(C[2].shape[0], np.bool_)
        key1 = np.zeros(C[2].shape[0], np.bool_)
        p0 = K.dim2_input(sorder[0], S[0], C[0], C[2], Y, Z, mX, mY, mZ, key0, removed[0])
        p1 = K.dim2_input(sorder[1], S[1], C[1], C[2], Y, Z, mX, mY, mZ, key1, removed[1])
        _, m2 = K.dim2_comp_match(sorder[2], S[2], C[2], Y, Z, mX, mY, mZ, key0, key1, removed[2])
        out[2] = (m2, p0, p1)

    del C
    S_in = list(S)
    del S
    # ---- dimension 1 ------------------------------------------------------
    cols = [o[~r[o]] for o, r in zip(sorder, removed)]
    del sorder, removed
    # Everything below is staged so that only one input's order/rank/pivot tables are alive at a
    # time (peak memory per tile); the arithmetic is the same as computing all three up front.
    # comparison complex first: its pivots define which pairs the image reductions must match
    E2 = _cell_values(vols[2])[0]
    eo2 = _sorted_valid(E2, levels[2])
    rank = _ranks(eo2, ne)
    prc = np.empty(cols[2].shape[0], rank.dtype)
    K.reduce_dim1(cols[2], rank, eo2.shape[0], Y, Z, prc)
    okc = prc >= 0
    comp_birth = eo2[prc[okc]]
    comp_pers = E2[comp_birth] != S_in[2][cols[2][okc]]
    comp_cols_idx = np.nonzero(okc)[0][comp_pers]  # positions in cols[2]
    del rank, prc, okc, comp_birth, comp_pers
    S_in[2] = None
    img_ok = np.ones(comp_cols_idx.shape[0], np.bool_)
    k_vert = []
    for k in range(2):  # input 0, input 1
        Ek = _cell_values(vols[k])[0]
        eo = _sorted_valid(Ek, levels[k])
        rank = _ranks(eo, ne)
        pr = np.empty(cols[k].shape[0], rank.dtype)
        K.reduce_dim1(cols[k], rank, eo.shape[0], Y, Z, pr)
        ok = pr >= 0
        birth_e = eo[pr[ok]]
        persistent = Ek[birth_e] != S_in[k][cols[k][ok]]
        key = np.zeros(ne, np.bool_)
        key[birth_e[persistent]] = True
        out[1, 1 + k] = int(persistent.sum())
        del pr, ok, birth_e, persistent
        S_in[k] = None
        # image reduction: comparison columns, rows in input-k order
        pr = np.empty(cols[2].shape[0], rank.dtype)
        K.reduce_dim1(cols[2], rank, eo.shape[0], Y, Z, pr)
        r = pr[comp_cols_idx]
        hit = r >= 0
        e = np.where(hit, eo[np.where(hit, r, 0)], 0)
        img_ok &= hit & key[e]
        del pr, r, hit, e, key, rank
        # dimension 0 input barcode (needs the same edge order)
        kv = np.zeros(vb[0].shape[0], np.bool_)
        q = K.dim0_input(eo, Ek, vb[k], Y, Z, kv)
        k_vert.append(kv)
        out[0, 1 + k] = q
        del eo, Ek
    out[1, 0] = int(img_ok.sum())

    # ---- dimension 0 comparison ------------------------------------------
    _, m0 = K.dim0_comp_match(eo2, E2, vb[2], vb[0], vb[1], Y, Z, k_vert[0], k_vert[1])
    out[0, 0] = m0
    return out
