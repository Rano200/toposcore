"""Download GT label TIFFs (plain data) for the PLatS case list from the public
HF bucket into refdata/labels. Retries; skips files already present.
Usage: python -I src/get_labels.py"""
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "refdata" / "labels"
IDS = (ROOT / "refdata" / "plats_evidence" / "case_ids.txt").read_text().split()
URL = "https://huggingface.co/buckets/scrollprize/datasets/resolve/surfaces/kaggle/labels/{}.tif"

D.mkdir(parents=True, exist_ok=True)
for cid in IDS:
    f = D / f"{cid}.tif"
    if f.exists() and f.stat().st_size > 0:
        continue
    for attempt in range(6):
        try:
            with urllib.request.urlopen(URL.format(cid), timeout=60) as r:
                data = r.read()
            if data[:2] in (b"II", b"MM"):
                f.write_bytes(data)
                print(cid, len(data), flush=True)
                break
            print(cid, "not a tiff", data[:40], flush=True)
        except Exception as e:  # noqa: BLE001
            print(cid, "retry", attempt, repr(e)[:120], flush=True)
            time.sleep(3 * (attempt + 1))
print("have", len(list(D.glob("*.tif"))), "of", len(IDS))
