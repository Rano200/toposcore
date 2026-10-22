"""Parity run: our TopoScore vs PLatS published per-case toposcore
(0058 automatic, binary_union arm). Appends one JSON line per case to
results/parity_0058.jsonl. Usage: python -I src/parity.py [case ...]"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from refio import LABELS, REFDATA, load_label_xyz, load_metrics, load_pred_xyz  # noqa: E402
from toposcore import toposcore  # noqa: E402

ARM = "0058"
if len(sys.argv) > 2 and sys.argv[1] == "--arm":
    ARM = sys.argv[2]
    del sys.argv[1:3]
ROOT = REFDATA / "plats_release" / f"PLatS_hidden106_{ARM}_automatic" / "cases"
OUT = HERE.parent / "results" / f"parity_{ARM}.jsonl"


def done_cases():
    if not OUT.exists():
        return set()
    return {json.loads(l)["case_id"] for l in OUT.read_text().splitlines() if l.strip()}


def main(cases):
    OUT.parent.mkdir(exist_ok=True)
    skip = done_cases()
    for cid in (c.strip() for c in cases):
        if not cid or cid in skip or not (LABELS / f"{cid}.tif").exists():
            continue
        t = time.time()
        lab = load_label_xyz(cid)
        pred = load_pred_xyz(cid, ROOT)
        ref = load_metrics(cid, ROOT)["binary_union"]["toposcore"]
        rep, tiles = toposcore(pred, lab, return_tiles=True)
        row = dict(case_id=cid, arm=ARM, shape=list(lab.shape), ours=rep.toposcore, ref=ref,
                   absdiff=abs(rep.toposcore - ref),
                   topoF1=[rep.topoF1_by_dim[k] for k in (0, 1, 2)],
                   counts=[list(rep.counts_by_dim[k]) for k in (0, 1, 2)],
                   tiles=[[list(t_[k]) for k in (0, 1, 2)] for t_ in tiles],
                   seconds=round(time.time() - t, 1))
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(cid, rep.toposcore, ref, row["absdiff"], row["counts"], row["seconds"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
