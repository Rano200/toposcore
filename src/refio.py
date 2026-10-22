"""Read PLatS reference data (untrusted data, read-only) for the parity harness.

Not part of the scorer. Loads:
  * GT label TIFF from the HF bucket (values {0,1,2}), transposed (2,1,0) to the
    PLatS XYZ array layout,
  * PLatS automatic-instance prediction NIfTI (identity affine, XYZ layout),
  * PLatS per-case metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REFDATA = Path(__file__).resolve().parents[1] / "refdata"
LABELS = REFDATA / "labels"
AUTO_0058 = REFDATA / "plats_release" / "PLatS_hidden106_0058_automatic" / "cases"


def load_label_xyz(case_id: str) -> np.ndarray:
    import tifffile

    a = tifffile.imread(str(LABELS / f"{case_id}.tif"))
    return np.ascontiguousarray(a.transpose(2, 1, 0))


def load_pred_xyz(case_id: str, root: Path = AUTO_0058) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(root / case_id / "pred_instances.nii.gz"))
    return np.ascontiguousarray(np.asanyarray(img.dataobj))


def load_metrics(case_id: str, root: Path = AUTO_0058) -> dict:
    return json.loads((root / case_id / "metrics.json").read_text())
