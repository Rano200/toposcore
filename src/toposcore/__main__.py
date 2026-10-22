"""Command line: toposcore PRED GT [--ignore-label 2] [--voi-alpha 0.3] [--workers N] [--json]

PRED and GT are 3D volumes of identical shape and axis order: .npy, .tif/.tiff (needs the
`tifffile` extra), or .nii/.nii.gz (needs `nibabel`). Prints the leaderboard score
0.3 TopoScore + 0.35 SurfaceDice@2 + 0.35 VOI_score and its three parts."""
import argparse
import json
import sys

import numpy as np

from .leaderboard import leaderboard_score


def load_volume(path):
    low = str(path).lower()
    if low.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if low.endswith((".tif", ".tiff")):
        try:
            import tifffile
        except ImportError:
            raise ImportError("reading TIFF needs the optional 'tifffile' package: "
                              "pip install tifffile imagecodecs  (or pip install 'toposcore[tiff]')") from None
        return tifffile.imread(path)
    if low.endswith((".nii", ".nii.gz")):
        try:
            import nibabel
        except ImportError:
            raise ImportError("reading NIfTI (.nii/.nii.gz) needs the optional 'nibabel' package: "
                              "pip install nibabel  (or pip install 'toposcore[nifti]')") from None
        return np.asanyarray(nibabel.load(path).dataobj)
    raise ValueError(f"unsupported file type: {path} (use .npy, .tif/.tiff, .nii or .nii.gz)")


def build_report(rep):
    topo = rep.topo
    return dict(
        score=rep.score,
        toposcore=topo.toposcore,
        surface_dice=rep.surface_dice,
        voi_score=rep.voi.voi_score,
        voi_total=rep.voi.voi_total,
        voi_split=rep.voi.voi_split,
        voi_merge=rep.voi.voi_merge,
        topoF1_by_dim={str(k): v for k, v in topo.topoF1_by_dim.items()},
        # counts per dimension: [matched, prediction pairs, GT pairs]
        counts_by_dim={str(k): list(v) for k, v in topo.counts_by_dim.items()},
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="toposcore", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred")
    ap.add_argument("gt")
    ap.add_argument("--ignore-label", type=float, default=2, help="GT value to erase from both volumes (default 2)")
    ap.add_argument("--no-ignore", action="store_true", help="do not erase any label")
    ap.add_argument("--voi-alpha", type=float, default=0.3,
                    help="VOI_score = 1/(1+alpha*VOI); 0.3 reproduces the published values, "
                         "1.0 is the official function default (see SPEC section 8)")
    ap.add_argument("--workers", type=int, default=None,
                    help="threads over octant tiles (default: memory-capped, see TOPOSCORE_MEM_MB)")
    ap.add_argument("--allow-nonfinite", action="store_true",
                    help="NaN -> 0 and inf -> foreground like the official scorer instead of an error")
    ap.add_argument("--json", action="store_true", help="print one JSON object, including per-dimension counts")
    a = ap.parse_args(argv)
    try:
        rep = leaderboard_score(load_volume(a.pred), load_volume(a.gt),
                                ignore_label=None if a.no_ignore else a.ignore_label,
                                voi_alpha=a.voi_alpha, workers=a.workers,
                                check_finite=not a.allow_nonfinite)
    except (ValueError, ImportError, MemoryError, OSError) as e:
        print(f"toposcore: error: {e}", file=sys.stderr)
        return 2
    out = build_report(rep)
    if a.json:
        print(json.dumps(out))
    else:
        print(f"score        {out['score']!r}\n"
              f"  TopoScore    {out['toposcore']!r}\n"
              f"  SurfaceDice  {out['surface_dice']!r}\n"
              f"  VOI_score    {out['voi_score']!r}  (alpha={a.voi_alpha}, VOI={out['voi_total']!r})")
        for k in rep.topo.dims:
            print(f"  dim {k}: F1={rep.topo.topoF1_by_dim[k]!r} (matched, pred, gt)={rep.topo.counts_by_dim[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
