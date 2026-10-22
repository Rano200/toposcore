"""Parity run: our SurfaceDice@2 / VOI_score / blend vs PLatS published binary_union fields
(0058 automatic). One process, sequential, resumable -> results/leaderboard_0058.jsonl.

  python -I src/leaderboard_parity.py [--arm 0058] [--full-topo] [case ...]

TopoScore is taken from results/parity_<arm>.jsonl (already bit-identical to the published
value) unless --full-topo, which recomputes it end to end (about 48 s per case).
Holds results/.leaderboard.lock (mkdir); a second instance exits immediately."""
import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from refio import LABELS, REFDATA, load_label_xyz, load_metrics, load_pred_xyz  # noqa: E402
from toposcore import leaderboard_score, voi_report, surface_dice  # noqa: E402
from toposcore.leaderboard import _norm_weights  # noqa: E402
from toposcore.metric import prepare  # noqa: E402

args = sys.argv[1:]
ARM = "0058"
FULL = False
if "--arm" in args:
    i = args.index("--arm")
    ARM = args[i + 1]
    del args[i:i + 2]
if "--full-topo" in args:
    FULL = True
    args.remove("--full-topo")
ROOT = REFDATA / "plats_release" / f"PLatS_hidden106_{ARM}_automatic" / "cases"
RES = HERE.parent / "results"
OUT = RES / f"leaderboard_{ARM}.jsonl"
LOCK = RES / ".leaderboard.lock"
MIN_FREE_MB = 1500
VOI_ALPHA = 0.3  # PLatS configuration (kaggle_surface.py:30); also our default (the official function default is 1.0)


def available_mb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("tot", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong),
                    ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong),
                    ("ext", ctypes.c_ulonglong)]
    s = MS()
    s.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.avail / 2**20


def wait_memory():
    while available_mb() < MIN_FREE_MB:
        print("low memory, waiting 60 s", flush=True)
        time.sleep(60)


def done_cases():
    if not OUT.exists():
        return set()
    return {json.loads(l)["case_id"] for l in OUT.read_text().splitlines() if l.strip()}


def cached_topo():
    f = RES / f"parity_{ARM}.jsonl"
    if not f.exists():
        return {}
    return {r["case_id"]: r["ours"] for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip())}


def run_case(cid, topo_cache):
    ref = load_metrics(cid, ROOT)["binary_union"]
    t = time.time()
    lab = load_label_xyz(cid)
    pred = load_pred_xyz(cid, ROOT)
    if FULL or cid not in topo_cache:
        rep = leaderboard_score(pred, lab, voi_alpha=VOI_ALPHA)
        topo, sdv, v, score = rep.topo.toposcore, rep.surface_dice, rep.voi, rep.score
        topo_src = "recomputed"
    else:
        pr_bin, gt_bin = prepare(pred, lab)
        del pred, lab
        gc.collect()
        sdv = surface_dice(pr_bin, gt_bin, 2.0)
        v = voi_report(pr_bin, gt_bin, alpha=VOI_ALPHA)
        topo = topo_cache[cid]
        wt, ws, wv = _norm_weights((0.3, 0.35, 0.35))
        score = wt * float(topo) + ws * sdv + wv * float(v.voi_score)
        topo_src = "parity_0058.jsonl"
    row = dict(case_id=cid, arm=ARM, topo_source=topo_src, toposcore=topo,
               surface_dice=sdv, voi_score=v.voi_score, voi_split=v.voi_split, voi_merge=v.voi_merge,
               score=score,
               ref_surface_dice=ref["surface_dice_tau2"], ref_voi_score=ref["voi_score"],
               ref_voi_split=ref["voi_split"], ref_voi_merge=ref["voi_merge"],
               ref_score=ref["leaderboard_formula_score"], ref_toposcore=ref["toposcore"],
               d_surface_dice=abs(sdv - ref["surface_dice_tau2"]),
               d_voi_score=abs(v.voi_score - ref["voi_score"]),
               d_score=abs(score - ref["leaderboard_formula_score"]),
               d_voi_split=abs(v.voi_split - ref["voi_split"]),
               d_voi_merge=abs(v.voi_merge - ref["voi_merge"]),
               seconds=round(time.time() - t, 1))
    return row


def main(cases):
    RES.mkdir(exist_ok=True)
    try:
        os.mkdir(LOCK)
    except FileExistsError:
        print("lock held, exiting", LOCK)
        return
    try:
        topo_cache = cached_topo()
        skip = done_cases()
        for cid in (c.strip() for c in cases):
            if not cid or cid in skip or not (LABELS / f"{cid}.tif").exists():
                continue
            wait_memory()
            row = run_case(cid, topo_cache)
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(cid, row["surface_dice"], row["ref_surface_dice"], row["d_surface_dice"],
                  row["d_voi_score"], row["d_score"], row["seconds"], flush=True)
            gc.collect()
    finally:
        os.rmdir(LOCK)


if __name__ == "__main__":
    main(args)
