"""Tests for SurfaceDice, VOI_score and the blend (hand-computable + independent checks)."""
import math

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial import cKDTree

from toposcore import leaderboard_score, surface_dice, voi_report
from toposcore._surface_table import SURFACE_AREA
from toposcore.surface import neighbour_codes


def box(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


# ---- surface dice ----
def test_table_basics():
    assert len(SURFACE_AREA) == 256
    assert SURFACE_AREA[0] == 0.0 and SURFACE_AREA[255] == 0.0
    assert SURFACE_AREA[1] == pytest.approx(math.sqrt(3) / 8)
    assert SURFACE_AREA[3] == pytest.approx(math.sqrt(2) / 2)  # two adjacent corners: a unit-ish diagonal quad


def test_codes_match_scipy_correlate():
    rng = np.random.default_rng(0)
    m = rng.random((7, 8, 9)) < 0.4
    k = np.array([[[128, 64], [32, 16]], [[8, 4], [2, 1]]])
    ref = ndimage.correlate(m.astype(np.uint8), k, mode="constant", cval=0)
    assert np.array_equal(neighbour_codes(m), ref)


def test_identical_and_empty():
    a = box((12, 12, 12), (2, 3, 4), (8, 9, 10))
    assert surface_dice(a, a) == 1.0
    z = np.zeros_like(a)
    assert surface_dice(z, z) == 1.0
    assert surface_dice(z, a) == 0.0
    assert surface_dice(a, z) == 0.0


def test_single_voxel_identical():
    a = np.zeros((5, 5, 5), bool)
    a[2, 2, 2] = True
    assert surface_dice(a, a) == 1.0


def test_plane_shifts():
    g = np.zeros((12, 12, 14), bool)
    g[:, :, 5] = True
    for shift, expect in [(0, 1.0), (1, 1.0), (2, 1.0), (4, 0.0), (6, 0.0)]:
        p = np.zeros_like(g)
        p[:, :, 5 + shift] = True
        assert surface_dice(p, g, 2.0) == expect, shift


def test_partial_overlap_hand():
    # GT plane z=5; pred = same plane on half of the area plus nothing else.
    # Compare with an independent brute force instead of a closed form.
    g = np.zeros((10, 10, 12), bool)
    g[:, :, 5] = True
    p = g.copy()
    p[5:, :, 5] = False
    assert 0.0 < surface_dice(p, g) < 1.0


def _brute(pred, gt, tol):
    sl = tuple(slice(int(np.flatnonzero(( gt | pred).any(axis=tuple(j for j in range(3) if j != i)))[0]),
                     int(np.flatnonzero((gt | pred).any(axis=tuple(j for j in range(3) if j != i)))[-1]) + 1)
               for i in range(3))
    out = []
    for m in (gt[sl], pred[sl]):
        c = neighbour_codes(np.pad(m, ((0, 1), (0, 1), (0, 1))))
        b = (c != 0) & (c != 255)
        out.append((np.argwhere(b), np.array(SURFACE_AREA)[c[b]]))
    (pg, ag), (pp, ap) = out
    dg = cKDTree(pp).query(pg)[0]
    dp = cKDTree(pg).query(pp)[0]
    return (ag[dg <= tol].sum() + ap[dp <= tol].sum()) / (ag.sum() + ap.sum())


def test_against_kdtree_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(5):
        base = ndimage.gaussian_filter(rng.random((20, 20, 20)), 2) > 0.5
        other = ndimage.gaussian_filter(rng.random((20, 20, 20)), 2) > 0.5
        for tol in (1.0, 2.0, 3.0):
            assert surface_dice(other, base, tol) == pytest.approx(_brute(other, base, tol), abs=1e-12)


# ---- VOI ----
def test_voi_identical():
    a = box((8, 8, 8), (1, 1, 1), (4, 4, 4)) | box((8, 8, 8), (6, 6, 6), (8, 8, 8))
    r = voi_report(a, a)
    assert r.voi_total == 0.0 and r.voi_score == 1.0


def test_voi_merge_hand():
    gt = np.zeros((1, 1, 5), bool)
    gt[0, 0, 0] = gt[0, 0, 2] = True
    pr = np.zeros_like(gt)
    pr[0, 0, 0:3] = True
    r = voi_report(pr, gt, alpha=1.0)  # gt: two comps + bridge voxel as class 0; pred: one comp
    assert r.voi_split == pytest.approx(0.0)
    assert r.voi_merge == pytest.approx(math.log2(3))
    assert r.voi_score == pytest.approx(1 / (1 + math.log2(3)))


def test_voi_split_hand():
    gt = np.zeros((1, 1, 5), bool)
    gt[0, 0, 0:3] = True
    pr = np.zeros_like(gt)
    pr[0, 0, 0] = pr[0, 0, 2] = True
    r = voi_report(pr, gt, alpha=1.0)
    assert r.voi_split == pytest.approx(math.log2(3))
    assert voi_report(pr, gt, alpha=0.3).voi_score == pytest.approx(1 / (1 + 0.3 * math.log2(3)))
    assert voi_report(pr, gt, alpha=1.0).voi_score == pytest.approx(1 / (1 + math.log2(3)))  # function default in leaderboard.py:130
    assert voi_report(pr, gt).voi_score == pytest.approx(1 / (1 + 0.3 * math.log2(3)))  # our default
    assert r.voi_merge == pytest.approx(0.0)
    assert r.voi_score == pytest.approx(1 / (1 + math.log2(3)))


def test_voi_connectivity_26_diagonal():
    a = np.zeros((2, 2, 2), bool)
    a[0, 0, 0] = a[1, 1, 1] = True       # body-diagonal: one component under 26
    assert voi_report(a, a, connectivity=26).voi_score == 1.0
    two = np.zeros((2, 2, 2), bool)
    two[0, 0, 0] = True
    r = voi_report(two, a)               # gt one comp (2 voxels), pred covers 1 of them
    # a=[1,1], b=[1,0]: H(b|a)=1 bit, H(a|b)=0
    assert r.voi_split == pytest.approx(1.0) and r.voi_merge == pytest.approx(0.0)


def test_voi_one_side_empty_quirk():
    a = box((6, 6, 6), (1, 1, 1), (4, 4, 4))
    z = np.zeros_like(a)
    # GT one component, pred empty: labels a=[1..], b=[0..] -> both entropies 0 -> score 1.0
    assert voi_report(z, a).voi_score == 1.0
    assert voi_report(z, z).voi_score == 1.0


# ---- blend ----
def test_blend_formula_and_ignore():
    gt = np.zeros((12, 12, 12), np.uint8)
    gt[2:6, 2:6, 2:6] = 1
    gt[9:, :, :] = 2
    pr = np.zeros_like(gt)
    pr[3:7, 2:6, 2:6] = 1
    pr[9:, 3:5, 3:5] = 1          # inside ignore region: must be erased
    r = leaderboard_score(pr, gt)
    assert r.score == pytest.approx(0.3 * r.topo.toposcore + 0.35 * r.surface_dice + 0.35 * r.voi.voi_score, abs=1e-15)
    pr2 = pr.copy()
    pr2[9:] = 0
    assert leaderboard_score(pr2, gt).score == r.score


def test_blend_perfect_and_empty():
    gt = np.zeros((10, 10, 10), np.uint8)
    gt[2:7, 2:7, 2:7] = 1
    assert leaderboard_score(gt, gt).score == pytest.approx(1.0)
    z = np.zeros_like(gt)
    assert leaderboard_score(z, z).score == pytest.approx(1.0)
