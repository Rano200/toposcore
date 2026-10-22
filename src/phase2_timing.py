"""Phase-2 runtime: compute_matching on float volumes of 64^3 and 128^3.
Two kinds: smooth (Gaussian-filtered noise, sigmoid -> probability-like, the
typical loss input) and white noise (worst case). Writes results/phase2_timing.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toposcore.binding import compute_matching  # noqa: E402


def smooth(rng, n):
    f = ndimage.gaussian_filter(rng.standard_normal((n, n, n)), 2.0)
    return 1.0 / (1.0 + np.exp(-f / f.std() * 2))


def main():
    rng = np.random.default_rng(0)
    compute_matching(rng.random((8, 8, 8)), rng.random((8, 8, 8)))  # JIT warm-up
    rows = []
    for n in (64, 128):
        for kind in ("smooth", "noise"):
            if kind == "smooth":
                base = smooth(rng, n)
                a = base
                b = np.clip(base + 0.05 * rng.standard_normal(base.shape), 0, 1)
            else:
                a, b = rng.random((n, n, n)), rng.random((n, n, n))
            t = time.time()
            r = compute_matching(1.0 - a, 1.0 - b)  # superlevel, as the loss's default
            dt = time.time() - t
            row = dict(n=n, kind=kind, seconds=round(dt, 2), num_matched=r.num_matched.tolist(),
                       num_unmatched_input1=r.num_unmatched_input1.tolist(),
                       num_unmatched_input2=r.num_unmatched_input2.tolist())
            print(row, flush=True)
            rows.append(row)
    (HERE.parent / "results" / "phase2_timing.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
