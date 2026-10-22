"""TopoScore exactly as topometrics computes it (toposcore.py + leaderboard.py),
on top of our own Betti matching counts (betti.match_counts)."""
from __future__ import annotations

from dataclasses import dataclass
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .betti import match_counts

_DEFAULT_WEIGHTS = (0.34, 0.33, 0.33)


@dataclass
class TopoReport:
    toposcore: float
    topoF1_by_dim: Dict[int, float]
    counts_by_dim: Dict[int, Tuple[int, int, int]]  # (m_k, p_k, g_k)
    dims: List[int]


def _weights_for_dims(dims_list, weights_in=None):
    default_map = {0: _DEFAULT_WEIGHTS[0], 1: _DEFAULT_WEIGHTS[1], 2: _DEFAULT_WEIGHTS[2]}
    if weights_in is None:
        return {k: default_map.get(k, 0.0) for k in dims_list}
    if isinstance(weights_in, dict):
        out = {k: default_map.get(k, 0.0) for k in dims_list}
        out.update({int(k): float(v) for k, v in weights_in.items()})
        return out
    seq = list(weights_in)
    if len(seq) >= (max(dims_list) + 1 if dims_list else 0):
        return {k: float(seq[k]) for k in dims_list}
    return {k: (float(seq[i]) if i < len(seq) else 0.0) for i, k in enumerate(dims_list)}


def from_counts(counts_by_dim, dims=None, weights=None) -> TopoReport:
    """Mirror of TopoScore.from_counts (toposcore.py:205-245), same float ops."""
    dims_list = sorted(counts_by_dim.keys()) if dims is None else list(dims)
    f1: Dict[int, float] = {}
    active: List[int] = []
    for k in dims_list:
        m_k, p_k, g_k = counts_by_dim.get(k, (0, 0, 0))
        denom = p_k + g_k
        if denom > 0:
            if g_k != 0:
                v = (2.0 * m_k) / float(denom)
            else:
                v = 0.5 / (float(denom) + 0.5)
            f1[k] = float(v)
            active.append(k)
        else:
            f1[k] = float("nan")
    if not active:
        score = 1.0
    else:
        w = _weights_for_dims(dims_list, weights)
        w_active = np.array([w.get(k, 0.0) for k in active], dtype=float)
        if not np.isfinite(w_active).all() or w_active.sum() <= 0:
            w_active = np.ones(len(active), dtype=float)
        w_active = w_active / w_active.sum()
        vals = np.array([f1[k] for k in active], dtype=float)
        score = float((w_active * vals).sum())
    return TopoReport(score, f1, {k: tuple(counts_by_dim.get(k, (0, 0, 0))) for k in dims_list}, dims_list)


def tile_counts(pred_topo, gt_topo) -> Dict[int, Tuple[int, int, int]]:
    """(m_k, p_k, g_k) for one tile, as TopoScore._counts_for_dim derives them:
    p_k = matched + unmatched(pred), g_k = matched + unmatched(gt)."""
    c = match_counts(pred_topo, gt_topo)
    return {k: (int(c[k, 0]), int(c[k, 1]), int(c[k, 2])) for k in range(3)}


def _axis_cuts(n, parts):
    parts = max(1, min(parts, n if n > 0 else 1))
    base, extra = n // parts, n % parts
    cuts, start = [], 0
    for i in range(parts):
        size = base + (1 if i < extra else 0)
        cuts.append((start, start + size))
        start += size
    return cuts


def octant_slices(shape, splits=(2, 2, 2)):
    out = []
    for z0, z1 in _axis_cuts(shape[0], splits[0]):
        for y0, y1 in _axis_cuts(shape[1], splits[1]):
            for x0, x1 in _axis_cuts(shape[2], splits[2]):
                if z1 > z0 and y1 > y0 and x1 > x0:
                    out.append((slice(z0, z1), slice(y0, y1), slice(x0, x1)))
    return out


def _check_volume(name, a):
    if a.ndim != 3:
        raise ValueError(f"{name} must be a 3D array (z, y, x), got {a.ndim}D with shape {a.shape}. "
                         "For a batch, score each volume separately.")
    if a.size == 0:
        raise ValueError(f"{name} has zero size (shape {a.shape})")
    if a.dtype.kind not in "biuf":
        raise ValueError(f"{name} has unsupported dtype {a.dtype}; use bool, integer or float")


def prepare(predictions, labels, *, ignore_label=2, ignore_mask=None, fg_threshold=None,
            check_finite=True):
    """Validation, ignore erasure and binarization as compute_leaderboard_score does
    (leaderboard.py:153-169). Returns (pr_bin, gt_bin).

    Binarization is the same for bool, integer and float input (SPEC section 2): foreground is
    ``x != 0`` after the ignore voxels are set to 0, or ``x > fg_threshold`` when given. A float
    probability map is therefore NOT thresholded at 0.5 unless you pass fg_threshold=0.5.

    check_finite (default True): NaN or +-inf anywhere in pred or gt raises ValueError. The official
    scorer instead maps NaN to 0 and counts inf as foreground (SPEC section 2, leaderboard.py:23-33);
    pass check_finite=False for that behaviour."""
    pr = np.asarray(predictions)
    gt = np.asarray(labels)
    _check_volume("prediction", pr)
    _check_volume("ground truth", gt)
    if pr.shape != gt.shape:
        raise ValueError(f"prediction and ground truth must have the same shape, got {pr.shape} vs {gt.shape}")
    if check_finite:
        for name, a in (("prediction", pr), ("ground truth", gt)):
            if a.dtype.kind == "f" and not np.isfinite(a).all():
                n_nan = int(np.isnan(a).sum())
                n_inf = int(np.isinf(a).sum())
                raise ValueError(f"{name} contains {n_nan} NaN and {n_inf} inf values. The official scorer "
                                 "silently treats NaN as 0 and inf as foreground; fix the input or pass "
                                 "check_finite=False (CLI: --allow-nonfinite) to reproduce that.")
    if ignore_mask is not None:
        ign = np.asarray(ignore_mask).astype(bool)
        if ign.shape != gt.shape:
            raise ValueError(f"ignore_mask shape {ign.shape} != volume shape {gt.shape}")
    elif ignore_label is None:
        ign = np.zeros(gt.shape, bool)
    else:
        ign = gt == ignore_label
    if ign.all():
        warnings.warn("every voxel is ignored (GT == ignore_label everywhere): both masks are empty "
                      "and the score is 1.0 by the official rules (SPEC sections 3, 7, 8)", UserWarning,
                      stacklevel=3)
    pr = np.nan_to_num(np.where(ign, 0, pr), nan=0.0)
    gt = np.nan_to_num(np.where(ign, 0, gt), nan=0.0)
    if fg_threshold is None:
        return pr != 0, gt != 0
    return pr > float(fg_threshold), gt > float(fg_threshold)


BYTES_PER_VOXEL = 160  # measured peak of match_counts per tile voxel (631 MB for a 160^3 tile)


def _check_memory_cap(tile_voxels, shape):
    """Refuse (with advice) when ONE octant tile alone is estimated above the memory cap."""
    budget_mb = float(os.environ.get("TOPOSCORE_MEM_MB", "1500"))
    need_mb = BYTES_PER_VOXEL * tile_voxels / 2**20
    if need_mb > budget_mb:
        raise MemoryError(
            f"a {tuple(shape)} volume needs an estimated {need_mb:.0f} MB for a single octant tile, above the "
            f"memory cap of {budget_mb:.0f} MB. Raise the cap with the environment variable TOPOSCORE_MEM_MB "
            f"(e.g. TOPOSCORE_MEM_MB={int(need_mb * 1.2)}), or pass workers=N explicitly "
            "(CLI: --workers N) to skip this check, or score a downsampled/cropped volume. "
            "Parallel workers multiply the footprint; --workers 1 is the smallest.")


def default_workers(tile_voxels: int = 0) -> int:
    """Default thread count: min(8, os.cpu_count()), capped so that the estimated sum of the
    concurrent tiles' working sets stays under TOPOSCORE_MEM_MB (default 1500 MB). One tile of
    160^3 voxels peaks at about 630 MB, so a 320^3 case defaults to 2 threads and an 80^3
    case to 8. Pass workers= explicitly to override the memory cap."""
    n = max(1, min(8, os.cpu_count() or 1))
    if tile_voxels > 0:
        budget = float(os.environ.get("TOPOSCORE_MEM_MB", "1500")) * 2**20
        n = min(n, max(1, int(budget // (BYTES_PER_VOXEL * tile_voxels))))
    return n


def toposcore(predictions, labels, *, ignore_label=2, ignore_mask=None, fg_threshold=None, check_finite=True,
              dims=(0, 1, 2), topo_weights=None, splits=(2, 2, 2), return_tiles=False, workers=None):
    """TopoScore component of the Vesuvius surface-detection leaderboard.
    Topology runs on the inverted masks (FG=0, BG=1), tiled into octants, with
    counts summed over tiles and weights applied once (leaderboard.py:171-189)."""
    pr_bin, gt_bin = prepare(predictions, labels, ignore_label=ignore_label,
                             ignore_mask=ignore_mask, fg_threshold=fg_threshold, check_finite=check_finite)
    return toposcore_from_binary(pr_bin, gt_bin, dims=dims, topo_weights=topo_weights,
                                 splits=splits, return_tiles=return_tiles, workers=workers)


def toposcore_from_binary(pr_bin, gt_bin, *, dims=(0, 1, 2), topo_weights=None, splits=(2, 2, 2),
                          return_tiles=False, workers=None):
    """TopoScore from already-prepared boolean foreground masks.
    workers: threads over octant tiles (None -> default_workers(tile size), memory-capped; 1 -> the serial path).
    The numba kernels release the GIL, tiles share the input arrays, and per-tile integer
    counts are summed in tile order either way, so the result is identical for any workers."""
    topo_pr = (~pr_bin).astype(np.uint8)
    topo_gt = (~gt_bin).astype(np.uint8)
    slices = octant_slices(topo_pr.shape, splits)
    biggest = max((int(np.prod([(sl_.stop - sl_.start) for sl_ in sl])) for sl in slices), default=0)
    if workers is None:
        _check_memory_cap(biggest, topo_pr.shape)
    nw = default_workers(biggest) if workers is None else int(workers)
    if nw < 1:
        raise ValueError("workers must be >= 1")
    nw = min(nw, max(1, len(slices)))
    work = lambda sl: tile_counts(topo_pr[sl], topo_gt[sl])
    try:
        if nw == 1:
            results = [work(sl) for sl in slices]
        else:
            with ThreadPoolExecutor(max_workers=nw) as ex:
                results = list(ex.map(work, slices))  # map preserves tile order
    except MemoryError as e:
        raise MemoryError(f"out of memory scoring a {topo_pr.shape} volume with {nw} worker(s) "
                          f"(about {BYTES_PER_VOXEL * biggest / 2**20:.0f} MB per octant tile). "
                          "Pass workers=1 (CLI: --workers 1) or free memory.") from e
    tiles = []
    combined: Dict[int, Tuple[int, int, int]] = {}
    for tc in results:
        tc = {k: tc[k] for k in dims}
        tiles.append(tc)
        for k, (m, p, g) in tc.items():
            M, P, G = combined.get(k, (0, 0, 0))
            combined[k] = (M + m, P + p, G + g)
    rep = from_counts(combined, dims=dims, weights=topo_weights)
    return (rep, tiles) if return_tiles else rep
