"""workers=1 (the serial path) and workers>1 must give identical results; the memory-lean
float32/int32 paths must equal the float64/int64 paths."""
import numpy as np

from toposcore import leaderboard_score, match_counts, toposcore
from toposcore.binding import compute_matching
from toposcore.metric import default_workers


def _case(seed=0, n=24):
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    gt = ndimage.gaussian_filter(rng.random((n, n, n)), 1.5) > 0.5
    pr = ndimage.gaussian_filter(rng.random((n, n, n)), 1.5) > 0.5
    return pr.astype(np.uint8), gt.astype(np.uint8)


def test_toposcore_workers_identical():
    pr, gt = _case()
    r1, t1 = toposcore(pr, gt, workers=1, return_tiles=True)
    r4, t4 = toposcore(pr, gt, workers=4, return_tiles=True)
    r8, t8 = toposcore(pr, gt, workers=8, return_tiles=True)
    assert r1.toposcore == r4.toposcore == r8.toposcore
    assert r1.counts_by_dim == r4.counts_by_dim == r8.counts_by_dim
    assert t1 == t4 == t8


def test_default_workers_and_memory_cap():
    assert default_workers() >= 1
    assert default_workers(10**12) == 1
    pr, gt = _case(1)
    assert toposcore(pr, gt).toposcore == toposcore(pr, gt, workers=1).toposcore


def test_leaderboard_workers_identical():
    pr, gt = _case(2)
    a = leaderboard_score(pr, gt, workers=1)
    b = leaderboard_score(pr, gt, workers=4)
    assert a.score == b.score and a.topo.counts_by_dim == b.topo.counts_by_dim


def test_compute_matching_batch_workers_identical():
    xs = [_case(s, 12)[0].astype(float) for s in range(3)]
    ys = [_case(s + 10, 12)[1].astype(float) for s in range(3)]
    r1 = compute_matching(xs, ys, workers=1)
    r4 = compute_matching(xs, ys, workers=4)
    for p, q in zip(r1, r4):
        assert (p.num_matched == q.num_matched).all()
        for d in range(3):
            assert (p.input1_matched_birth_coordinates[d] == q.input1_matched_birth_coordinates[d]).all()
            assert (p.input2_unmatched_death_coordinates[d] == q.input2_unmatched_death_coordinates[d]).all()


def test_float32_path_equals_float64_path():
    # values exact in float32 take the lean path; adding a non-representable epsilon forces float64
    rng = np.random.default_rng(5)
    a = rng.integers(0, 4, (10, 10, 10)).astype(float) / 4.0
    b = rng.integers(0, 4, (10, 10, 10)).astype(float) / 4.0
    exact = match_counts(a, b)
    eps = 2.0 ** -40  # a power of two but 1 + 2^-40 is not a float32; order is unchanged
    wide = match_counts(a * (1 + eps), b * (1 + eps))
    assert (exact == wide).all()


def test_within_distance_equals_edt():
    from scipy import ndimage
    from toposcore.surface import within_distance
    rng = np.random.default_rng(9)
    for shape, dens in (((9, 11, 13), 0.02), ((16, 15, 14), 0.1), ((7, 7, 7), 0.5), ((6, 5, 4), 0.0)):
        b = rng.random(shape) < dens
        for tol in (0.0, 1.0, 2.0, 2.5, 3.7):
            ref = ndimage.distance_transform_edt(~b) <= tol if b.any() else np.zeros(shape, bool)
            assert (within_distance(b, tol) == ref).all(), (shape, dens, tol)


def test_surface_dice_unchanged_vs_edt_formulation():
    from scipy import ndimage
    from toposcore import surface_dice
    from toposcore import surface as S
    pr, gt = _case(3, 20)
    pr, gt = pr.astype(bool), gt.astype(bool)

    def old(pred, gt, tol=2.0):
        sl = S._bbox(gt | pred)
        sides = []
        for m in (gt[sl], pred[sl]):
            m = np.pad(m, ((0, 1), (0, 1), (0, 1)))
            code = S.neighbour_codes(m)
            sides.append((code, (code != 0) & (code != 255)))
        (cg, bg), (cp, bp) = sides
        A = S._AREA
        d_pr = ndimage.distance_transform_edt(~bp)
        d_gt = ndimage.distance_transform_edt(~bg)
        return float((A[cg[bg & (d_pr <= tol)]].sum() + A[cp[bp & (d_gt <= tol)]].sum())
                     / (A[cg[bg]].sum() + A[cp[bp]].sum()))
    for tol in (1.0, 2.0, 3.0):
        assert surface_dice(pr, gt, tol) == old(pr, gt, tol)
