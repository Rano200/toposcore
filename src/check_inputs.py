"""Gate 0: reconstruct PLatS scored masks and check the integer fields
(pred/gt voxels scored, component counts) plus Dice before any topology."""
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else __file__.rsplit("/", 1)[0])
from refio import load_label_xyz, load_pred_xyz, load_metrics  # noqa: E402


def main(cases):
    s26 = np.ones((3, 3, 3), bool)
    for cid in cases:
        lab = load_label_xyz(cid)
        pred = load_pred_xyz(cid)
        m = load_metrics(cid)["binary_union"]
        ign = lab == 2
        pb = (pred != 0) & ~ign
        gb = (lab != 0) & ~ign
        inter = int((pb & gb).sum())
        dice = 2 * inter / (pb.sum() + gb.sum())
        c_raw6 = ndimage.label(pred != 0)[1]
        c_raw26 = ndimage.label(pred != 0, s26)[1]
        c_p26 = ndimage.label(pb, s26)[1]
        c_p6 = ndimage.label(pb)[1]
        c_g26 = ndimage.label(gb, s26)[1]
        c_g6 = ndimage.label(gb)[1]
        print(cid, lab.shape, pred.shape, pred.dtype, np.unique(lab))
        print("  pred_vox", int(pb.sum()), m["pred_voxels_scored"], " gt_vox", int(gb.sum()), m["gt_voxels_scored"])
        print("  dice", dice, m["dice"])
        print("  comps raw6/26", c_raw6, c_raw26, "ref raw", m["pred_components_raw"],
              "| scored p6/26", c_p6, c_p26, "ref", m["pred_components_scored"],
              "| gt6/26", c_g6, c_g26, "ref", m["gt_components_scored"])


if __name__ == "__main__":
    main(sys.argv[1:])
