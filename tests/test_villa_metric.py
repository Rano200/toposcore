"""villa-compatible metric class, numpy inputs only (torch is never imported)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from toposcore import leaderboard_score  # noqa: E402
from toposcore.integrations.villa_metric import ToposcoreMetric  # noqa: E402


def _vols(n=12, seed=0):
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    f = lambda: ndimage.gaussian_filter(rng.random((n, n, n)), 1.5)
    return f(), (f() > 0.5).astype(np.uint8)


def test_interface_and_value():
    p, g = _vols()
    out = ToposcoreMetric().compute(p[None, None], g[None, None])  # (B,1,Z,Y,X) float: > 0.5
    assert all(isinstance(v, float) for v in out.values())
    ref = leaderboard_score((p > 0.5).astype(np.uint8), g)
    assert out["score"] == pytest.approx(ref.score)
    assert out["toposcore"] == pytest.approx(ref.topo.toposcore)
    assert out["voi_score"] == pytest.approx(ref.voi.voi_score)


def test_update_aggregate_reset_and_batch_mean():
    p1, g1 = _vols(seed=1)
    p2, g2 = _vols(seed=2)
    m = ToposcoreMetric()
    r1 = m.update(p1[None, None], g1[None, None])
    r2 = m.update(p2[None, None], g2[None, None])
    agg = m.aggregate()
    assert agg["score"] == pytest.approx((r1["score"] + r2["score"]) / 2)
    batch = ToposcoreMetric().compute(np.stack([p1, p2])[:, None], np.stack([g1, g2])[:, None])
    assert batch["score"] == pytest.approx(agg["score"])
    m.reset()
    assert m.aggregate() == {}


def test_multichannel_argmax_and_mask():
    p, g = _vols()
    logits = np.stack([1 - p, p])[None]  # (1,2,Z,Y,X): class 1 where p > 0.5
    out = ToposcoreMetric().compute(logits, g)
    ref = leaderboard_score((p > 0.5).astype(np.uint8), g)
    assert out["score"] == pytest.approx(ref.score)
    mask = np.ones_like(g, bool)
    mask[:3] = False
    out2 = ToposcoreMetric().compute(logits, g, mask=mask)
    ref2 = leaderboard_score((p > 0.5).astype(np.uint8), g, ignore_mask=~mask)
    assert out2["score"] == pytest.approx(ref2.score)
