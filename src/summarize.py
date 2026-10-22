"""Summarize parity results (results/parity_*.jsonl) and cross-check that the
archive metrics.json references equal PLatS evidence/04 rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
EV = ROOT / "refdata" / "plats_evidence" / "04_paper_automatic_hidden106_gpu0_gpu2.json"
ARMS = {"0058": "0058_kaggle", "0076": "original_0076", "0076_plus_100k": "0076_plus_100k"}

ev = json.loads(EV.read_text())
ev_ref = {(r["arm"], r["case_id"]): r["binary_union"]["toposcore"] for r in ev["rows"]}
total = 0
worst = 0.0
for arm, evarm in ARMS.items():
    f = RES / f"parity_{arm}.jsonl"
    if not f.exists():
        print(arm, "no results")
        continue
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    # a case can be scored twice if two runs overlapped; identical rows, keep one per case
    rows = list({r["case_id"]: r for r in rows}.values())
    n = len(rows)
    exact = sum(r["absdiff"] == 0.0 for r in rows)
    within = sum(r["absdiff"] <= 1e-9 for r in rows)
    mx = max(r["absdiff"] for r in rows)
    ev_agree = sum(ev_ref.get((evarm, r["case_id"])) == r["ref"] for r in rows)
    dims_active = [sum(1 for r in rows if (r["counts"][d][1] + r["counts"][d][2]) > 0) for d in range(3)]
    dims_matched = [sum(1 for r in rows if r["counts"][d][0] > 0) for d in range(3)]
    secs = sum(r["seconds"] for r in rows) / n
    print(f"{arm}: cases={n} bit-identical={exact} within1e-9={within} max|diff|={mx:.3e} "
          f"ref==evidence04={ev_agree}/{n} active-dims(0,1,2)={dims_active} m_k>0 cases={dims_matched} "
          f"mean_s={secs:.0f}")
    for r in rows:
        if r["absdiff"] > 1e-9:
            print("   MISMATCH", r["case_id"], r["ours"], r["ref"], r["counts"])
    total += n
    worst = max(worst, mx)
print("TOTAL", total, "worst", worst)

# ---- full leaderboard parity (results/leaderboard_<arm>.jsonl) ----
for arm in ARMS:
    f = RES / f"leaderboard_{arm}.jsonl"
    if not f.exists():
        continue
    rows = list({r["case_id"]: r for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip())}.values())
    n = len(rows)
    print(f"LEADERBOARD {arm}: cases={n}")
    for name, key in [("SurfaceDice@2", "d_surface_dice"), ("VOI_score", "d_voi_score"),
                      ("blended score", "d_score"), ("voi_split (info)", "d_voi_split"),
                      ("voi_merge (info)", "d_voi_merge")]:
        d = [r[key] for r in rows]
        print(f"  {name:18s} within1e-9={sum(x <= 1e-9 for x in d)}/{n} exact={sum(x == 0.0 for x in d)} max|diff|={max(d):.3e}")
    for r in rows:
        if max(r["d_surface_dice"], r["d_voi_score"], r["d_score"]) > 1e-9:
            print("   MISMATCH", r["case_id"], r["d_surface_dice"], r["d_voi_score"], r["d_score"])
