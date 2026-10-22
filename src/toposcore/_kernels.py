"""Sequential kernels (union-find passes and the dimension-1 column reduction).

Every function here is plain Python restricted to constructs numba can compile.
If numba is importable they are JIT-compiled; otherwise (or with
NUMBA_DISABLE_JIT=1) they run as ordinary Python, which is exact but only
practical for small volumes.

Index conventions (shape X, Y, Z; C order, x slowest), mirroring
Betti-Matching-3D src_3D (see SPEC.md):
  vertex  vid = (x*Y + y)*Z + z
  edge    eid = vid*3 + type    type 0/1/2 = +x/+y/+z direction
  square  sid = vid*3 + type    type 0 spans (y,z), 1 spans (x,z), 2 spans (x,y)
  cube    cid = (x*mY + y)*mZ + z, with m = shape - 1; the outside node is mX*mY*mZ
Ordering by (value, linear id) equals the reference's (birth, packed index)
order because the packed index is lexicographic in (x, y, z, type).
"""
from __future__ import annotations

import heapq

import numpy as np

try:  # numba is optional; it only changes speed, not results
    from numba import njit as _njit

    def jit(f):
        return _njit(cache=True, nogil=True)(f)

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    def jit(f):
        return f

    HAVE_NUMBA = False


# --------------------------------------------------------------------------
# Union-find, primal (vertices). Root = minimum (birth, index). link() returns
# the younger root (the one that dies), exactly like dim3::UnionFind::link.
# --------------------------------------------------------------------------
@jit
def _uf_find(parent, x):
    y = x
    z = parent[y]
    while z != y:
        y = z
        z = parent[y]
    y = parent[x]
    while z != y:
        parent[x] = z
        x = y
        y = parent[x]
    return z


@jit
def _uf_link(parent, birth, x, y):
    if birth[x] > birth[y]:
        parent[x] = y
        return x
    elif birth[x] < birth[y]:
        parent[y] = x
        return y
    else:
        if x > y:
            parent[x] = y
            return x
        else:
            parent[y] = x
            return y


# Dual union-find (cubes + outside node). Root = maximum (birth, index).
@jit
def _ufd_link(parent, birth, x, y):
    if birth[x] < birth[y]:
        parent[x] = y
        return x
    elif birth[x] > birth[y]:
        parent[y] = x
        return y
    else:
        if x < y:
            parent[x] = y
            return x
        else:
            parent[y] = x
            return y


@jit
def _edge_ends(eid, Y, Z):
    vid = eid // 3
    t = eid - vid * 3
    if t == 0:
        return vid, vid + Y * Z
    elif t == 1:
        return vid, vid + Z
    return vid, vid + 1


# --------------------------------------------------------------------------
# Dimension 0
# --------------------------------------------------------------------------
@jit
def dim0_input(order, evals, vbirth, Y, Z, is_key):
    """Union-find barcode of one input. order: valid edge ids sorted by
    (value, eid). Marks is_key[birth vertex] for every pair with
    birth != death. Returns the number of such pairs."""
    n = vbirth.shape[0]
    parent = np.arange(n)
    npairs = 0
    for k in range(order.shape[0]):
        e = order[k]
        a, b = _edge_ends(e, Y, Z)
        ra = _uf_find(parent, a)
        rb = _uf_find(parent, b)
        if ra != rb:
            bi = _uf_link(parent, vbirth, ra, rb)
            if vbirth[bi] != evals[e]:
                is_key[bi] = True
                npairs += 1
    return npairs


@jit
def dim0_comp_match(order, evals_c, vb_c, vb_0, vb_1, Y, Z, key0, key1):
    """Comparison barcode with simultaneous image union-finds
    (Dimension0::computeImagePairsAndMatch). Returns (n_comp_pairs, n_matched)."""
    n = vb_c.shape[0]
    pc = np.arange(n)
    p0 = np.arange(n)
    p1 = np.arange(n)
    ncomp = 0
    nmatch = 0
    for k in range(order.shape[0]):
        e = order[k]
        a, b = _edge_ends(e, Y, Z)
        ra = _uf_find(pc, a)
        rb = _uf_find(pc, b)
        if ra != rb:
            bic = _uf_link(pc, vb_c, ra, rb)
            r0a = _uf_find(p0, a)
            r0b = _uf_find(p0, b)
            bi0 = _uf_link(p0, vb_0, r0a, r0b)
            r1a = _uf_find(p1, a)
            r1b = _uf_find(p1, b)
            bi1 = _uf_link(p1, vb_1, r1a, r1b)
            if vb_c[bic] != evals_c[e]:
                ncomp += 1
                if key0[bi0] and key1[bi1]:
                    nmatch += 1
    return ncomp, nmatch


# --------------------------------------------------------------------------
# Dimension 2 (dual union-find over cubes, squares processed in reverse order)
# --------------------------------------------------------------------------
@jit
def _square_cubes(sid, Y, Z, mX, mY, mZ):
    vid = sid // 3
    t = sid - vid * 3
    x = vid // (Y * Z)
    r = vid - x * Y * Z
    y = r // Z
    z = r - y * Z
    outside = mX * mY * mZ
    if t == 0:
        a = outside if x == 0 else ((x - 1) * mY + y) * mZ + z
        b = outside if x == mX else (x * mY + y) * mZ + z
    elif t == 1:
        a = outside if y == 0 else (x * mY + (y - 1)) * mZ + z
        b = outside if y == mY else (x * mY + y) * mZ + z
    else:
        a = outside if z == 0 else (x * mY + y) * mZ + (z - 1)
        b = outside if z == mZ else (x * mY + y) * mZ + z
    return a, b


@jit
def dim2_input(order, svals, cb_k, cb_c, Y, Z, mX, mY, mZ, key, removed):
    """Dimension2::computeInputAndImagePairs for one input. order: valid square
    ids sorted by (value_k, sid); processed in reverse. key[comp cube] is set
    for pairs with birth != death (keyed by the comparison-image younger root).
    removed[sid] marks squares consumed (cleared for dimension 1)."""
    n = cb_k.shape[0]
    pk = np.arange(n)
    pc = np.arange(n)
    npairs = 0
    for k in range(order.shape[0] - 1, -1, -1):
        s = order[k]
        a, b = _square_cubes(s, Y, Z, mX, mY, mZ)
        ra = _uf_find(pk, a)
        rb = _uf_find(pk, b)
        if ra != rb:
            bi = _ufd_link(pk, cb_k, ra, rb)
            rca = _uf_find(pc, a)
            rcb = _uf_find(pc, b)
            bic = _ufd_link(pc, cb_c, rca, rcb)
            if svals[s] != cb_k[bi]:
                key[bic] = True
                npairs += 1
            removed[s] = True
    return npairs


@jit
def dim2_comp_match(order, svals, cb_c, Y, Z, mX, mY, mZ, key0, key1, removed):
    n = cb_c.shape[0]
    pc = np.arange(n)
    ncomp = 0
    nmatch = 0
    for k in range(order.shape[0] - 1, -1, -1):
        s = order[k]
        a, b = _square_cubes(s, Y, Z, mX, mY, mZ)
        ra = _uf_find(pc, a)
        rb = _uf_find(pc, b)
        if ra != rb:
            bi = _ufd_link(pc, cb_c, ra, rb)
            if svals[s] != cb_c[bi]:
                ncomp += 1
                if key0[bi] and key1[bi]:
                    nmatch += 1
            removed[s] = True
    return ncomp, nmatch


# --------------------------------------------------------------------------
# Dimension 1: left-to-right column reduction of the square->edge boundary
# matrix. Columns in the given order; rows ranked by `rank` (rank of an edge in
# the row filtration order). Pivot = largest rank (CubeComparator max).
# By the pairing-uniqueness lemma the pivots equal those of the reference
# (emergent pairs + cached reduced columns are shortcuts of the same
# left-to-right reduction).
# --------------------------------------------------------------------------
@jit
def _square_faces(sid, Y, Z, out):
    vid = sid // 3
    t = sid - vid * 3
    if t == 0:
        out[0] = vid * 3 + 1
        out[1] = vid * 3 + 2
        out[2] = (vid + 1) * 3 + 1
        out[3] = (vid + Z) * 3 + 2
    elif t == 1:
        out[0] = vid * 3 + 0
        out[1] = vid * 3 + 2
        out[2] = (vid + 1) * 3 + 0
        out[3] = (vid + Y * Z) * 3 + 2
    else:
        out[0] = vid * 3 + 0
        out[1] = vid * 3 + 1
        out[2] = (vid + Z) * 3 + 0
        out[3] = (vid + Y * Z) * 3 + 1


@jit
def _pop_pivot(heap):
    # heap holds negated ranks (min-heap => max rank first); equal entries cancel
    if len(heap) == 0:
        return -1
    p = -heapq.heappop(heap)
    while len(heap) > 0 and -heap[0] == p:
        heapq.heappop(heap)
        if len(heap) == 0:
            return -1
        p = -heapq.heappop(heap)
    return p


@jit
def reduce_dim1(cols, rank, n_rows, Y, Z, col_pivot_rank):
    """Reduce columns `cols` (square ids, in column order). Writes, for column i,
    col_pivot_rank[i] = rank of its pivot row, or -1 if it reduces to zero."""
    ncol = cols.shape[0]
    idt = col_pivot_rank.dtype  # int32 for any volume below 2^31 cells: halves these tables
    pivot_col = np.full(n_rows, -1, idt)  # by row rank
    # cache of reduced columns (flat pool)
    c_start = np.full(ncol, -1, idt)
    c_len = np.zeros(ncol, idt)
    pool = np.empty(1 << 16, np.int64)
    pool_n = 0
    faces = np.empty(4, np.int64)
    for i in range(ncol):
        _square_faces(cols[i], Y, Z, faces)
        # emergent pair: the largest face of the unreduced column
        mx = -1
        for f in range(4):
            r = rank[faces[f]]
            if r > mx:
                mx = r
        if pivot_col[mx] == -1:
            pivot_col[mx] = i
            col_pivot_rank[i] = mx
            continue
        heap = [np.int64(0)]
        heap.pop()
        for f in range(4):
            heapq.heappush(heap, -rank[faces[f]])
        j = pivot_col[mx]
        while True:
            # add (reduced) column j
            if c_start[j] >= 0:
                st = c_start[j]
                for q in range(c_len[j]):
                    heapq.heappush(heap, -pool[st + q])
            else:
                _square_faces(cols[j], Y, Z, faces)
                for f in range(4):
                    heapq.heappush(heap, -rank[faces[f]])
            p = _pop_pivot(heap)
            if p == -1:
                col_pivot_rank[i] = -1
                break
            heapq.heappush(heap, -p)
            if pivot_col[p] == -1:
                pivot_col[p] = i
                col_pivot_rank[i] = p
                # cache the cleaned reduced column
                need = pool_n + len(heap)
                if need > pool.shape[0]:
                    newsize = pool.shape[0] * 2
                    while newsize < need:
                        newsize *= 2
                    np2 = np.empty(newsize, np.int64)
                    np2[:pool_n] = pool[:pool_n]
                    pool = np2
                c_start[i] = pool_n
                cnt = 0
                while len(heap) > 0:
                    c = -heapq.heappop(heap)
                    if len(heap) > 0 and -heap[0] == c:
                        heapq.heappop(heap)
                    else:
                        pool[pool_n] = c
                        pool_n += 1
                        cnt += 1
                c_len[i] = cnt
                break
            j = pivot_col[p]
    return 0
