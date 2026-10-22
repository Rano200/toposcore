"""Vesuvius leaderboard score as a metric object for villa's evaluation interface
(``vesuvius.models.evaluation.base_metric.BaseMetric``: ``compute(pred, gt, **kwargs) -> Dict[str, float]``,
plus ``update`` / ``aggregate`` / ``reset``).

If the ``vesuvius`` package is installed, ``BaseMetric`` is imported from it; otherwise a minimal
class with the same methods is used, so this module works without villa and without torch.
Torch tensors (any dtype, including bfloat16, any device) and numpy arrays are accepted;
torch is only imported if a tensor is passed.

Shapes follow villa's other metrics (e.g. ``VOIMetric``):
  pred  (Z,Y,X), (B,Z,Y,X), or (B,C,Z,Y,X). C > 1 is argmax-ed over channels and foreground is
        class != 0; C == 1 is squeezed and, for float input, foreground is ``> threshold`` (default 0.5);
        integer or bool input is foreground where != 0.
  gt    (Z,Y,X), (B,Z,Y,X) or (B,1,Z,Y,X); label ``ignore_label`` (default 2) is erased from both.
  mask  optional kwarg, True = valid voxel (villa's convention); invalid voxels are ignored.
Each batch item is scored separately and the metrics are averaged.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..leaderboard import leaderboard_score

try:  # real interface when villa's package is importable
    from vesuvius.models.evaluation.base_metric import BaseMetric as _Base
except Exception:  # ImportError, or a vesuvius install whose own imports (torch...) fail
    class _Base:  # minimal duck-typed stand-in with the same methods
        def __init__(self, name: str):
            self.name = name
            self.results = []

        def compute(self, pred, gt, **kwargs) -> Dict[str, float]:
            raise NotImplementedError

        def update(self, pred, gt, **kwargs):
            result = self.compute(pred, gt, **kwargs)
            self.results.append(result)
            return result

        def aggregate(self) -> Dict[str, float]:
            if not self.results:
                return {}
            keys = set()
            for r in self.results:
                keys.update(r.keys())
            return {k: float(np.mean([r[k] for r in self.results if k in r])) for k in keys}

        def reset(self):
            self.results = []


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch tensor, duck-typed so torch is never imported here
        x = x.detach()
        if str(x.dtype) == "torch.bfloat16":
            x = x.float()
        return x.cpu().numpy()
    return np.asarray(x)


class ToposcoreMetric(_Base):
    """0.3 TopoScore + 0.35 SurfaceDice@2 + 0.35 VOI_score (see the toposcore README)."""

    def __init__(self, ignore_label=2, voi_alpha: float = 0.3, threshold: float = 0.5,
                 workers=None):
        super().__init__("toposcore")
        self.ignore_label = ignore_label
        self.voi_alpha = voi_alpha
        self.threshold = threshold
        self.workers = workers

    def _binarize_pred(self, p):
        if p.ndim == 5:
            if p.shape[1] > 1:
                return (np.argmax(p, axis=1) != 0).astype(np.uint8)
            p = p[:, 0]
        if p.dtype.kind == "f":
            return (p > self.threshold).astype(np.uint8)
        return (p != 0).astype(np.uint8)

    def compute(self, pred, gt, **kwargs) -> Dict[str, float]:
        p = _to_numpy(pred)
        g = _to_numpy(gt)
        mask = kwargs.get("mask")
        m = None if mask is None else _to_numpy(mask).astype(bool)
        if g.ndim == 5:
            if g.shape[1] != 1:
                raise ValueError(f"gt with 5 dims must have a single channel, got shape {g.shape}")
            g = g[:, 0]
            if m is not None and m.ndim == 5:
                m = m[:, 0]
        if g.ndim == 3:
            g = g[None]
            if m is not None and m.ndim == 3:
                m = m[None]
        if p.ndim == 3:
            p = p[None]
        pb = self._binarize_pred(p)
        if pb.shape != g.shape:
            raise ValueError(f"after channel handling pred {pb.shape} and gt {g.shape} must match")
        rows: List[Dict[str, float]] = []
        for i in range(g.shape[0]):
            rep = leaderboard_score(
                pb[i], g[i], ignore_label=self.ignore_label,
                ignore_mask=None if m is None else ~m[i],
                voi_alpha=self.voi_alpha, workers=self.workers)
            row = dict(score=rep.score, toposcore=rep.topo.toposcore,
                       surface_dice=rep.surface_dice, voi_score=rep.voi.voi_score,
                       voi_total=rep.voi.voi_total)
            for k, v in rep.topo.topoF1_by_dim.items():
                row[f"topo_f1_dim{k}"] = float(v)  # NaN when the dimension is inactive
            rows.append(row)
        out = {}
        for k in rows[0]:
            vals = np.array([r[k] for r in rows], float)
            out[k] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        return out
