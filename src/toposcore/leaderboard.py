"""Full Vesuvius surface-detection leaderboard score:
0.3 TopoScore + 0.35 SurfaceDice@2 + 0.35 VOI_score (leaderboard.py:115-238)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .metric import TopoReport, prepare, toposcore_from_binary
from .surface import surface_dice
from .voi import VOIReport, voi_report


@dataclass
class LeaderboardReport:
    score: float
    topo: TopoReport
    surface_dice: float
    voi: VOIReport
    tiles: Optional[list] = None  # per-octant counts, only when return_tiles=True


def _norm_weights(w):
    w = [0.0 if (not np.isfinite(x) or x < 0) else float(x) for x in w]
    s = sum(w)
    if s <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (w[0] / s, w[1] / s, w[2] / s)


def leaderboard_score(pred, gt, *, ignore_label=2, ignore_mask=None, fg_threshold=None, check_finite=True,
                      surface_tolerance=2.0, voi_connectivity=26, voi_transform="one_over_one_plus",
                      voi_alpha=0.3, combine_weights=(0.3, 0.35, 0.35), topo_weights=None, workers=None, return_tiles=False) -> LeaderboardReport:
    """pred, gt: 3D arrays of identical shape (GT label 2 = ignore by default).
    voi_alpha defaults to 0.3: PLatS (kaggle_surface.py:30) and ciscoriordan/mednext-vs-umamba-scroll
    (README ~l.25) use 0.3 for the Kaggle metric; the official function default
    (leaderboard.py:130) and a speakerdeck recap say 1.0. Pass voi_alpha=1.0 for that. See SPEC section 8.
    """
    pr_bin, gt_bin = prepare(pred, gt, ignore_label=ignore_label, ignore_mask=ignore_mask,
                             fg_threshold=fg_threshold, check_finite=check_finite)
    res = toposcore_from_binary(pr_bin, gt_bin, topo_weights=topo_weights, workers=workers,
                                return_tiles=return_tiles)
    topo, tiles = res if return_tiles else (res, None)
    sd = surface_dice(pr_bin, gt_bin, surface_tolerance)
    voi = voi_report(pr_bin, gt_bin, connectivity=voi_connectivity, alpha=voi_alpha,
                     transform=voi_transform)
    wt, ws, wv = _norm_weights(combine_weights)
    return LeaderboardReport(wt * float(topo.toposcore) + ws * sd + wv * float(voi.voi_score), topo, sd, voi, tiles)
