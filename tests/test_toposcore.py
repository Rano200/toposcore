import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import brute  # noqa: E402
from toposcore import from_counts, match_counts, octant_slices, toposcore  # noqa: E402


def inv(fg):
    return (~np.asarray(fg, bool)).astype(np.uint8)


def betti_pairs(fg):
    c = match_counts(inv(fg), inv(fg))
    return tuple(int(c[d, 1]) for d in range(3))


# ---- hand-countable shapes: finite pairs = (b0 - 1, b1, b2) of the FG ----
def test_empty_and_full():
    z = np.zeros((4, 5, 6), bool)
    assert betti_pairs(z) == (0, 0, 0)
    assert betti_pairs(~z) == (0, 0, 0)


def test_two_blobs():
    fg = np.zeros((8, 8, 8), bool)
    fg[1:3, 1:3, 1:3] = True
    fg[5:7, 5:7, 5:7] = True
    assert betti_pairs(fg) == (1, 0, 0)


def test_diagonal_is_not_connected():
    fg = np.zeros((4, 4, 4), bool)
    fg[1, 1, 1] = fg[2, 2, 2] = True  # 26- but not 6-adjacent
    assert betti_pairs(fg) == (1, 0, 0)


def test_ring_has_one_tunnel():
    fg = np.zeros((7, 7, 3), bool)
    fg[1:6, 1:6, 1] = True
    fg[2:5, 2:5, 1] = False
    assert betti_pairs(fg) == (0, 1, 0)


def test_hollow_cube_has_one_cavity():
    fg = np.zeros((9, 9, 9), bool)
    fg[1:8, 1:8, 1:8] = True
    fg[3:6, 3:6, 3:6] = False
    assert betti_pairs(fg) == (0, 0, 1)


def test_cavity_must_not_touch_border():
    fg = np.ones((5, 5, 5), bool)
    fg[0, 2, 2] = False  # border hole: connected to the outside, no cavity
    assert betti_pairs(fg) == (0, 0, 0)


def test_two_rings_and_cavity_with_blobs():
    fg = np.zeros((12, 12, 12), bool)
    fg[1:6, 1:6, 1] = True; fg[2:5, 2:5, 1] = False
    fg[1:6, 1:6, 4] = True; fg[2:5, 2:5, 4] = False
    fg[6:11, 6:11, 6:11] = True; fg[8, 8, 8] = False
    fg[1, 10, 10] = True
    assert betti_pairs(fg) == (3, 2, 1)


def test_identical_inputs_match_everything():
    rng = np.random.default_rng(1)
    for _ in range(5):
        fg = rng.random((9, 8, 7)) < 0.55
        c = match_counts(inv(fg), inv(fg))
        assert (c[:, 0] == c[:, 1]).all() and (c[:, 1] == c[:, 2]).all()


@pytest.mark.parametrize("seed", range(12))
def test_betti_numbers_vs_euler(seed):
    rng = np.random.default_rng(seed)
    shape = tuple(rng.integers(3, 8, size=3))
    fg = rng.random(shape) < rng.uniform(0.3, 0.8)
    b0, b1, b2 = brute.betti_fg(fg)
    assert betti_pairs(fg) == ((b0 - 1) if b0 else 0, b1, b2)


@pytest.mark.parametrize("seed", range(30))
def test_matching_vs_bruteforce(seed):
    rng = np.random.default_rng(100 + seed)
    shape = tuple(rng.integers(2, 7, size=3))
    if seed % 3 == 2:  # non-binary values with ties
        a = rng.integers(0, 4, size=shape).astype(float)
        b = rng.integers(0, 4, size=shape).astype(float)
    else:
        a = (rng.random(shape) < rng.uniform(0.2, 0.8)).astype(float)
        b = (rng.random(shape) < rng.uniform(0.2, 0.8)).astype(float)
    c = match_counts(a, b)
    for d in range(3):
        assert c[d, 1] == len(brute.barcode(a, d))
        assert c[d, 2] == len(brute.barcode(b, d))
    assert c[0, 0] == brute.image_matched(a, b, 0)
    assert c[1, 0] == brute.image_matched(a, b, 1)
    assert c[2, 0] == brute.dim2_matched(a, b)


def test_degenerate_axis():
    a = np.zeros((1, 6, 6)); a[0, 1:5, 1:5] = 1; a[0, 2:4, 2:4] = 0
    b = np.zeros((1, 6, 6))
    c = match_counts(a, b)
    assert c[2].tolist() == [0, 0, 0]


def test_from_counts_formula():
    r = from_counts({0: (2, 4, 3), 1: (0, 2, 0), 2: (0, 0, 0)})
    w = np.array([0.34, 0.33]); w = w / w.sum()
    assert r.toposcore == float((w * np.array([4 / 7, 0.5 / 2.5])).sum())
    assert np.isnan(r.topoF1_by_dim[2])
    assert from_counts({0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0)}).toposcore == 1.0


def test_octants_and_ignore():
    assert len(octant_slices((5, 4, 3))) == 8
    assert octant_slices((3, 3, 3))[0] == (slice(0, 2), slice(0, 2), slice(0, 2))
    gt = np.zeros((6, 6, 6), np.uint8); gt[1:5, 1:5, 2] = 1; gt[0] = 2
    pr = gt.copy(); pr[pr == 2] = 1  # predictions inside ignore are erased
    assert toposcore(pr, gt).toposcore == 1.0
    pr2 = pr.copy(); pr2[3, 3, 2] = 0  # punch a hole -> one unmatched tunnel
    rep = toposcore(pr2, gt, splits=(1, 1, 1))
    assert rep.counts_by_dim[1] == (0, 1, 0)


def _perturbed_pair(seed):
    """Porous base volume with many tunnels/cavities, and a copy with a few
    voxels flipped, so dimensions 1 and 2 have partial (non-trivial) matchings."""
    rng = np.random.default_rng(1000 + seed)
    shape = tuple(rng.integers(6, 9, size=3))
    fg = np.ones(shape, bool)
    holes = rng.random(shape) < rng.uniform(0.15, 0.35)
    fg &= ~holes
    fg[0] = fg[-1] = False if seed % 2 else fg[0]
    other = fg.copy()
    flips = rng.random(shape) < 0.04
    other ^= flips
    return inv(fg).astype(float), inv(other).astype(float)


@pytest.mark.parametrize("seed", range(20))
def test_partial_matching_dims_1_2(seed):
    a, b = _perturbed_pair(seed)
    c = match_counts(a, b)
    assert c[1, 0] == brute.image_matched(a, b, 1)
    assert c[2, 0] == brute.dim2_matched(a, b)
    assert c[0, 0] == brute.image_matched(a, b, 0)
    for d in range(3):
        assert c[d, 1] == len(brute.barcode(a, d))
        assert c[d, 2] == len(brute.barcode(b, d))
