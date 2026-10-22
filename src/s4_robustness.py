"""Robustness pass on the 36 Scroll-4 GT patches of Jinhojeong's public eval
pool (release patches-v2, labels.tar; data only). The official C++ matcher is
reported to segfault on one s4 patch; the crashing prediction is not public, so
we score every s4 GT against itself shifted by one voxel along z (a cheap
synthetic 'prediction') and record that a finite value is returned.
Writes results/s4_robustness.jsonl."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toposcore import toposcore  # noqa: E402

D = HERE.parent / "refdata" / "jinho_patches" / "labelsTr"
OUT = HERE.parent / "results" / "s4_robustness.jsonl"


def main():
    import tifffile

    done = set()
    if OUT.exists():
        done = {json.loads(l)["patch"] for l in OUT.read_text().splitlines() if l.strip()}
    for f in sorted(D.glob("s4_*.tif")):
        if f.stem in done:
            continue
        t = time.time()
        gt = tifffile.imread(str(f))
        vals = np.unique(gt).tolist()
        pred = np.roll(gt != 0, 1, axis=0).astype(np.uint8)
        rep = toposcore(pred, gt)
        row = dict(patch=f.stem, shape=list(gt.shape), label_values=vals,
                   toposcore=rep.toposcore, finite=bool(np.isfinite(rep.toposcore)),
                   counts=[list(rep.counts_by_dim[k]) for k in (0, 1, 2)],
                   seconds=round(time.time() - t, 1))
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
