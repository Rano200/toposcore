"""Independent brute-force reference for small volumes (tests only).

Builds the V-construction cubical complex explicitly (cells as tuples), orders
each dimension by (value, packed index), and runs textbook Z/2 column
reductions with Python sets. Shares no code with toposcore.

* barcode(v, d): persistence pairs of dimension d with birth != death.
* image_matched(a, b, d): Betti matching count for d in {0, 1} via image
  persistence (comparison columns, input-k rows), i.e. the matrix form of the
  rules in SPEC.md.
"""
from __future__ import annotations

import itertools

import numpy as np

INF = float("inf")


def _cells(shape):
    X, Y, Z = shape
    cells = {0: [], 1: [], 2: [], 3: []}
    for x, y, z in itertools.product(range(X), range(Y), range(Z)):
        cells[0].append((x, y, z, 0))
        for t, (dx, dy, dz) in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
            if x + dx < X and y + dy < Y and z + dz < Z:
                cells[1].append((x, y, z, t))
        # squares: type 0 spans y,z ; 1 spans x,z ; 2 spans x,y
        for t, (dx, dy, dz) in enumerate([(0, 1, 1), (1, 0, 1), (1, 1, 0)]):
            if x + dx < X and y + dy < Y and z + dz < Z:
                cells[2].append((x, y, z, t))
        if x + 1 < X and y + 1 < Y and z + 1 < Z:
            cells[3].append((x, y, z, 0))
    return cells


def _verts(c, d):
    x, y, z, t = c
    if d == 0:
        return [(x, y, z)]
    if d == 1:
        dx, dy, dz = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][t]
        return [(x, y, z), (x + dx, y + dy, z + dz)]
    if d == 2:
        span = [(0, 1, 1), (1, 0, 1), (1, 1, 0)][t]
        return [(x + i * span[0], y + j * span[1], z + k * span[2])
                for i in range(2) for j in range(2) for k in range(2)
                if (span[0] or i == 0) and (span[1] or j == 0) and (span[2] or k == 0)]
    return [(x + i, y + j, z + k) for i in range(2) for j in range(2) for k in range(2)]


def _faces(c, d):
    x, y, z, t = c
    if d == 1:
        dx, dy, dz = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][t]
        return [(x, y, z, 0), (x + dx, y + dy, z + dz, 0)]
    if d == 2:
        if t == 0:
            return [(x, y, z, 1), (x, y, z, 2), (x, y, z + 1, 1), (x, y + 1, z, 2)]
        if t == 1:
            return [(x, y, z, 0), (x, y, z, 2), (x, y, z + 1, 0), (x + 1, y, z, 2)]
        return [(x, y, z, 0), (x, y, z, 1), (x, y + 1, z, 0), (x + 1, y, z, 1)]
    # cube faces: x-normal (type 0) at x and x+1, y-normal (1), z-normal (2)
    return [(x, y, z, 0), (x + 1, y, z, 0), (x, y, z, 1), (x, y + 1, z, 1),
            (x, y, z, 2), (x, y, z + 1, 2)]


def _val(v, c, d):
    return max(float(v[p]) for p in _verts(c, d))


def _order(v, cells, d):
    return sorted(cells[d], key=lambda c: (_val(v, c, d), c))


def _reduce(cols, rowkey, faces_of):
    """Textbook reduction. cols: column cells in order. Returns {col: pivot}."""
    pivot_owner = {}
    reduced = {}
    out = {}
    for c in cols:
        col = set(faces_of(c))
        while col:
            low = max(col, key=rowkey)
            if low in pivot_owner:
                col ^= reduced[pivot_owner[low]]
            else:
                pivot_owner[low] = c
                reduced[c] = col
                out[c] = low
                break
    return out


def barcode(v, d):
    """Pairs (birth_cell, death_cell) in dimension d with birth != death."""
    cells = _cells(v.shape)
    rows = _order(v, cells, d)
    rk = {c: i for i, c in enumerate(rows)}
    cols = _order(v, cells, d + 1)
    piv = _reduce(cols, rk.__getitem__, lambda c: _faces(c, d + 1))
    return [(b, c) for c, b in piv.items() if _val(v, b, d) != _val(v, c, d + 1)]


def image_matched(a, b, d):
    """Matched count for dimension d (0 or 1)."""
    c_ = np.minimum(a, b)
    cells = _cells(a.shape)
    births = []
    for v in (a, b):
        births.append({bc for bc, _ in barcode(v, d)})
    comp_pairs = barcode(c_, d)
    comp_cols = _order(c_, cells, d + 1)
    imgs = []
    for v in (a, b):
        rows = _order(v, cells, d)
        rk = {c: i for i, c in enumerate(rows)}
        imgs.append(_reduce(comp_cols, rk.__getitem__, lambda c: _faces(c, d + 1)))
    m = 0
    for _, death in comp_pairs:
        e0 = imgs[0].get(death)
        e1 = imgs[1].get(death)
        if e0 in births[0] and e1 in births[1]:
            m += 1
    return m


def betti_fg(fg):
    """Betti numbers of the V-construction complex on FG voxels, from cell counts
    (Euler characteristic) plus component / cavity counts via BFS."""
    from scipy import ndimage

    X, Y, Z = fg.shape
    b0 = ndimage.label(fg)[1]
    bg = ~fg
    lab, n = ndimage.label(bg, np.ones((3, 3, 3), bool))
    border = set(np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                           lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])))
    b2 = len([i for i in range(1, n + 1) if i not in border])
    cells = _cells(fg.shape)
    chi = 0
    for d in range(4):
        cnt = sum(1 for c in cells[d] if all(fg[p] for p in _verts(c, d)))
        chi += (-1) ** d * cnt
    b1 = b0 + b2 - chi
    return b0, b1, b2


def dim2_matched(a, b):
    """Literal re-statement of the reference's dimension-2 rule (dual
    union-find, squares in decreasing (value, index) order, input pairs keyed
    by the comparison-image younger root), written with tuples and dicts."""
    c_ = np.minimum(a, b)
    X, Y, Z = a.shape
    if min(X, Y, Z) < 2:
        return 0
    cells = _cells(a.shape)
    OUT = ("out",)

    def cube_val(v, q):
        return INF if q == OUT else _val(v, q, 3)

    def cofaces(s):
        x, y, z, t = s
        d = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][t]
        lo = (x - d[0], y - d[1], z - d[2], 0)
        hi = (x, y, z, 0)
        ok = lambda q: 0 <= q[0] < X - 1 and 0 <= q[1] < Y - 1 and 0 <= q[2] < Z - 1
        return (lo if ok(lo) else OUT), (hi if ok(hi) else OUT)

    def key(v, q):  # dual order: larger (value, index) is older; OUT is oldest
        return (cube_val(v, q), (1,) if q == OUT else (0,) + q)

    def run(v, vc=None):
        par, parc = {}, {}

        def find(p, q):
            while p.get(q, q) != q:
                q = p[q]
            return q
        keys = set()
        for s in reversed(_order(v, cells, 2)):
            u, w = cofaces(s)
            ru, rw = find(par, u), find(par, w)
            if ru == rw:
                continue
            young, old = (ru, rw) if key(v, ru) < key(v, rw) else (rw, ru)
            par[young] = old
            if vc is not None:
                cu, cw = find(parc, u), find(parc, w)
                cyoung, cold = (cu, cw) if key(vc, cu) < key(vc, cw) else (cw, cu)
                parc[cyoung] = cold
                if _val(v, s, 2) != cube_val(v, young):
                    keys.add(cyoung)
            else:
                if _val(v, s, 2) != cube_val(v, young):
                    keys.add(young)
        return keys
    k0 = run(a, c_)
    k1 = run(b, c_)
    kc = run(c_)
    return len(kc & k0 & k1)


def parent_voxel(v, cell, d):
    """CubicalGridComplex::getParentVoxel restated on tuples."""
    x, y, z, t = cell
    b = _val(v, cell, d)
    if d == 0:
        return (x, y, z)
    if d == 1:
        o = [(x + 1, y, z), (x, y + 1, z), (x, y, z + 1)][t]
        return o if v[o] == b else (x, y, z)
    if d == 2:
        cands = {0: [(x, y + 1, z + 1), (x, y + 1, z), (x, y, z + 1)],
                 1: [(x + 1, y, z + 1), (x + 1, y, z), (x, y, z + 1)],
                 2: [(x + 1, y + 1, z), (x + 1, y, z), (x, y + 1, z)]}[t]
    else:
        cands = [(x, y, z), (x, y, z + 1), (x, y + 1, z), (x, y + 1, z + 1),
                 (x + 1, y, z), (x + 1, y, z + 1), (x + 1, y + 1, z)]
    for c in cands:
        if v[c] == b:
            return c
    return (x, y, z) if d == 2 else (x + 1, y + 1, z + 1)


def matched_pairs(a, b, d):
    """Set of ((birth0, death0), (birth1, death1)) matched cell pairs for d in {0,1}."""
    c_ = np.minimum(a, b)
    cells = _cells(a.shape)
    bars = [{bc: dc for bc, dc in barcode(v, d)} for v in (a, b)]
    comp_cols = _order(c_, cells, d + 1)
    imgs = []
    for v in (a, b):
        rk = {c: i for i, c in enumerate(_order(v, cells, d))}
        imgs.append(_reduce(comp_cols, rk.__getitem__, lambda c: _faces(c, d + 1)))
    out = set()
    for _, death in barcode(c_, d):
        e0, e1 = imgs[0].get(death), imgs[1].get(death)
        if e0 in bars[0] and e1 in bars[1]:
            out.add(((e0, bars[0][e0]), (e1, bars[1][e1])))
    return out
