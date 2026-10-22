"""Pair-recording variants of the dimension-0 and dimension-2 union-find
passes (used by binding.compute_matching). Same traversal and tie rules as
_kernels.py; additionally record each persistence pair and each match, in the
order the reference appends them to its vectors."""
from __future__ import annotations

import numpy as np

from ._kernels import _edge_ends, _square_cubes, _uf_find, _uf_link, _ufd_link, jit


@jit
def dim0_input_pairs(order, evals, vbirth, Y, Z, key_idx, birth_out, death_out):
    """Dimension0::computePairs. key_idx[birth vertex] = pair number.
    birth_out/death_out receive (birth vertex id, death edge id) per pair."""
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
                key_idx[bi] = npairs
                birth_out[npairs] = bi
                death_out[npairs] = e
                npairs += 1
    return npairs


@jit
def dim0_comp_pairs(order, evals_c, vb_c, vb_0, vb_1, Y, Z, key0, key1, m0_out, m1_out):
    """Dimension0::computeImagePairsAndMatch; m0_out/m1_out receive the input-0
    and input-1 pair numbers of each match, in match order."""
    n = vb_c.shape[0]
    pc = np.arange(n)
    p0 = np.arange(n)
    p1 = np.arange(n)
    nmatch = 0
    for k in range(order.shape[0]):
        e = order[k]
        a, b = _edge_ends(e, Y, Z)
        ra = _uf_find(pc, a)
        rb = _uf_find(pc, b)
        if ra != rb:
            bic = _uf_link(pc, vb_c, ra, rb)
            bi0 = _uf_link(p0, vb_0, _uf_find(p0, a), _uf_find(p0, b))
            bi1 = _uf_link(p1, vb_1, _uf_find(p1, a), _uf_find(p1, b))
            if vb_c[bic] != evals_c[e]:
                if key0[bi0] >= 0 and key1[bi1] >= 0:
                    m0_out[nmatch] = key0[bi0]
                    m1_out[nmatch] = key1[bi1]
                    nmatch += 1
    return nmatch


@jit
def dim2_input_pairs(order, svals, cb_k, cb_c, Y, Z, mX, mY, mZ, key_idx, removed,
                     birth_out, death_out):
    """Dimension2::computeInputAndImagePairs. Pairs are (square, cube);
    key_idx[comparison younger root] = pair number."""
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
            bic = _ufd_link(pc, cb_c, _uf_find(pc, a), _uf_find(pc, b))
            if svals[s] != cb_k[bi]:
                key_idx[bic] = npairs
                birth_out[npairs] = s
                death_out[npairs] = bi
                npairs += 1
            removed[s] = True
    return npairs


@jit
def dim2_comp_pairs(order, svals, cb_c, Y, Z, mX, mY, mZ, key0, key1, removed, m0_out, m1_out):
    n = cb_c.shape[0]
    pc = np.arange(n)
    nmatch = 0
    for k in range(order.shape[0] - 1, -1, -1):
        s = order[k]
        a, b = _square_cubes(s, Y, Z, mX, mY, mZ)
        ra = _uf_find(pc, a)
        rb = _uf_find(pc, b)
        if ra != rb:
            bi = _ufd_link(pc, cb_c, ra, rb)
            if svals[s] != cb_c[bi]:
                if key0[bi] >= 0 and key1[bi] >= 0:
                    m0_out[nmatch] = key0[bi]
                    m1_out[nmatch] = key1[bi]
                    nmatch += 1
            removed[s] = True
    return nmatch
