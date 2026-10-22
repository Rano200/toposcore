"""CLI (python -m toposcore / toposcore entry point) on small .npy files."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toposcore import leaderboard_score  # noqa: E402
from toposcore.__main__ import load_volume, main  # noqa: E402


def _pair(tmp_path, n=16, seed=3):
    rng = np.random.default_rng(seed)
    from scipy import ndimage
    gt = (ndimage.gaussian_filter(rng.random((n, n, n)), 1.5) > 0.5).astype(np.uint8)
    pr = (ndimage.gaussian_filter(rng.random((n, n, n)), 1.5) > 0.5).astype(np.uint8)
    gt[0, 0, :4] = 2  # some ignore voxels
    np.save(tmp_path / "pred.npy", pr)
    np.save(tmp_path / "gt.npy", gt)
    return pr, gt, str(tmp_path / "pred.npy"), str(tmp_path / "gt.npy")


def test_cli_json_matches_api(tmp_path, capsys):
    pr, gt, p, g = _pair(tmp_path)
    assert main([p, g, "--json", "--workers", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    ref = leaderboard_score(pr, gt)
    assert out["score"] == ref.score
    assert out["toposcore"] == ref.topo.toposcore
    assert out["surface_dice"] == ref.surface_dice
    assert out["voi_score"] == ref.voi.voi_score
    assert set(out["counts_by_dim"]) == {"0", "1", "2"}
    assert all(len(v) == 3 for v in out["counts_by_dim"].values())


def test_cli_voi_alpha_and_ignore(tmp_path, capsys):
    pr, gt, p, g = _pair(tmp_path)
    assert main([p, g, "--json", "--voi-alpha", "1.0", "--no-ignore"]) == 0
    out = json.loads(capsys.readouterr().out)
    ref = leaderboard_score(pr, gt, voi_alpha=1.0, ignore_label=None)
    assert out["score"] == ref.score and out["voi_score"] == ref.voi.voi_score


def test_cli_text_output(tmp_path, capsys):
    _, _, p, g = _pair(tmp_path)
    assert main([p, g]) == 0
    txt = capsys.readouterr().out
    for word in ("score", "TopoScore", "SurfaceDice", "VOI_score", "dim 0", "dim 2"):
        assert word in txt


def test_cli_error_exit_code(tmp_path, capsys):
    _, _, p, g = _pair(tmp_path)
    np.save(tmp_path / "bad.npy", np.zeros((4, 4), np.uint8))
    assert main([str(tmp_path / "bad.npy"), g]) == 2
    assert "3D" in capsys.readouterr().err
    assert main([p, str(tmp_path / "x.txt")]) == 2
    assert "unsupported file type" in capsys.readouterr().err


def test_cli_subprocess_entry(tmp_path):
    _, _, p, g = _pair(tmp_path, n=10)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-m", "toposcore", p, g, "--json"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert 0.0 <= json.loads(r.stdout)["score"] <= 1.0


def test_tiff_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    a = (np.arange(4 * 5 * 6).reshape(4, 5, 6) % 3).astype(np.uint8)
    tifffile.imwrite(tmp_path / "a.tif", a, photometric="minisblack")
    assert np.array_equal(load_volume(str(tmp_path / "a.tif")), a)


def test_nifti_roundtrip_or_clear_message(tmp_path):
    try:
        import nibabel
    except ImportError:
        with pytest.raises(ImportError, match="nibabel"):
            load_volume("x.nii.gz")
        return
    a = (np.arange(4 * 5 * 6).reshape(4, 5, 6) % 3).astype(np.uint8)
    for name in ("a.nii", "a.nii.gz"):
        nibabel.save(nibabel.Nifti1Image(a, np.eye(4)), str(tmp_path / name))
        assert np.array_equal(load_volume(str(tmp_path / name)), a)
