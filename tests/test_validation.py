"""Input validation: every rejected input has a clear message; every accepted odd input has a
defined, documented value (SPEC sections 2, 3, 5, 7, 8)."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toposcore import leaderboard_score, toposcore  # noqa: E402


def _blob(n=8):
    a = np.zeros((n, n, n), np.uint8)
    a[2:6, 2:6, 2:6] = 1
    return a


def test_shape_mismatch():
    with pytest.raises(ValueError, match=r"same shape.*\(8, 8, 8\).*\(8, 8, 9\)"):
        leaderboard_score(_blob(), np.zeros((8, 8, 9), np.uint8))


@pytest.mark.parametrize("shape", [(8, 8), (2, 8, 8, 8), (8,)])
def test_non_3d(shape):
    bad = np.zeros(shape, np.uint8)
    with pytest.raises(ValueError, match="3D array"):
        toposcore(bad, bad)
    with pytest.raises(ValueError, match="3D array"):
        leaderboard_score(_blob(), bad)


def test_zero_size():
    with pytest.raises(ValueError, match="zero size"):
        toposcore(np.zeros((0, 4, 4)), np.zeros((0, 4, 4)))


def test_unsupported_dtype():
    with pytest.raises(ValueError, match="dtype"):
        toposcore(np.zeros((4, 4, 4), complex), np.zeros((4, 4, 4)))


def test_empty_prediction_and_empty_gt_follow_official_rules():
    # SPEC s3 (no essential classes -> no pairs), s5 (no active dim -> 1.0), s7 (exactly one empty
    # -> SurfaceDice 0.0), s8 (union mask holds one class -> VOI 0 -> score 1.0)
    z, o = np.zeros((8, 8, 8), np.uint8), _blob()
    for pred, gt in ((z, o), (o, z)):
        r = leaderboard_score(pred, gt)
        assert r.topo.toposcore == 1.0
        assert r.surface_dice == 0.0
        assert r.voi.voi_score == 1.0
        assert r.score == pytest.approx(0.3 + 0.35)
    both = leaderboard_score(z, z)  # both empty: SPEC s7 -> 1.0, s8 -> 1.0
    assert both.surface_dice == 1.0 and both.voi.voi_score == 1.0 and both.topo.toposcore == 1.0
    assert both.score == pytest.approx(1.0)


def test_all_ignore_volume_warns_and_scores_as_both_empty():
    gt = np.full((8, 8, 8), 2, np.uint8)
    with pytest.warns(UserWarning, match="every voxel is ignored"):
        r = leaderboard_score(_blob(), gt)
    assert r.score == pytest.approx(1.0)
    # with ignore disabled the same GT is a plain all-foreground mask: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        leaderboard_score(_blob(), gt, ignore_label=None)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_by_default(bad):
    p = _blob().astype(np.float32)
    p[3, 3, 3] = bad
    with pytest.raises(ValueError, match="NaN.*inf"):
        leaderboard_score(p, _blob())
    with pytest.raises(ValueError, match="NaN.*inf"):
        toposcore(_blob(), p)


def test_nonfinite_official_behaviour_opt_in():
    p = _blob().astype(np.float32)
    p[0, 0, 0] = np.nan  # official: NaN -> 0
    a = toposcore(p, _blob(), check_finite=False)
    assert a.counts_by_dim == toposcore(_blob(), _blob()).counts_by_dim
    q = _blob().astype(np.float32)
    q[0, 0, 0] = np.inf  # official: inf != 0 -> foreground
    b = toposcore(q, _blob(), check_finite=False)
    ref = _blob()
    ref[0, 0, 0] = 1
    assert b.counts_by_dim == toposcore(ref, _blob()).counts_by_dim


def test_dtypes_binarize_consistently():
    rng = np.random.default_rng(1)
    pb = rng.random((10, 10, 10)) > 0.5
    gb = rng.random((10, 10, 10)) > 0.5
    ref = leaderboard_score(pb, gb)
    for dt in (np.uint8, np.int16, np.int64, np.float32, np.float64):
        r = leaderboard_score(pb.astype(dt), gb.astype(dt))
        assert r.score == ref.score and r.topo.counts_by_dim == ref.topo.counts_by_dim
    # the nonzero scale does not matter (x != 0): 0.25 and 7 are foreground like 1
    r = leaderboard_score(pb.astype(np.float32) * 0.25, gb.astype(np.int32) * 7)
    assert r.score == ref.score


def test_fg_threshold_changes_probabilities_only_when_given():
    p = np.full((8, 8, 8), 0.2, np.float32)  # all nonzero
    g = _blob()
    assert leaderboard_score(p, g).surface_dice != leaderboard_score(p, g, fg_threshold=0.5).surface_dice


def test_memory_cap_message(monkeypatch):
    monkeypatch.setenv("TOPOSCORE_MEM_MB", "0.001")
    with pytest.raises(MemoryError, match="TOPOSCORE_MEM_MB.*workers"):
        toposcore(_blob(), _blob())
    # an explicit workers= skips the estimate
    assert toposcore(_blob(), _blob(), workers=1).toposcore == 1.0
    monkeypatch.setenv("TOPOSCORE_MEM_MB", "1500")
    assert toposcore(_blob(), _blob()).toposcore == 1.0
