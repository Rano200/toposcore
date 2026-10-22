import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import brute  # noqa: E402
from toposcore import match_counts  # noqa: E402
from toposcore.binding import compute_matching  # noqa: E402


def _rows(b, d):
    return np.asarray(b, np.int64).reshape(-1, 3)


def _vol(seed, kind):
    rng = np.random.default_rng(seed)
    shape = tuple(rng.integers(2, 7, size=3))
    if kind == "binary":
        return [(rng.random(shape) < rng.uniform(0.2, 0.8)).astype(float) for _ in range(2)]
    if kind == "ties":
        return [rng.integers(0, 4, size=shape).astype(float) for _ in range(2)]
    if kind == "float":
        return [rng.random(shape) for _ in range(2)]
    # smooth-ish float pair sharing structure
    base = rng.random(shape)
    return [base, np.clip(base + 0.15 * rng.standard_normal(shape), 0, 1)]


KINDS = ["binary", "ties", "float", "related"]


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("kind", KINDS)
def test_binding_vs_bruteforce(seed, kind):
    a, b = _vol(seed, kind)
    res = compute_matching(a, b)
    # counts agree with the phase-1 counter
    c = match_counts(a, b)
    for d in range(3):
        assert res.num_matched[d] == c[d, 0]
        assert res.num_matched[d] + res.num_unmatched_input1[d] == c[d, 1]
        assert res.num_matched[d] + res.num_unmatched_input2[d] == c[d, 2]
        for arr in (res.input1_matched_birth_coordinates[d], res.input2_unmatched_death_coordinates[d]):
            assert arr.dtype == np.int64 and arr.ndim == 2 and arr.shape[1] == 3
    # full barcodes (matched + unmatched), as parent-voxel coordinate pairs
    for k, (v, mb, md, ub, ud) in enumerate([
            (a, res.input1_matched_birth_coordinates, res.input1_matched_death_coordinates,
             res.input1_unmatched_birth_coordinates, res.input1_unmatched_death_coordinates),
            (b, res.input2_matched_birth_coordinates, res.input2_matched_death_coordinates,
             res.input2_unmatched_birth_coordinates, res.input2_unmatched_death_coordinates)]):
        for d in range(3):
            ours = sorted(map(tuple, np.concatenate([np.hstack([_rows(mb[d], d), _rows(md[d], d)]),
                                                     np.hstack([_rows(ub[d], d), _rows(ud[d], d)])]).tolist()))
            ref = sorted(brute.parent_voxel(v, bc, d) + brute.parent_voxel(v, dc, d + 1)
                         for bc, dc in brute.barcode(v, d))
            assert ours == ref, (k, d)
    # matched pairs for dims 0 and 1
    for d in (0, 1):
        ours = sorted(tuple(r) for r in np.hstack([
            res.input1_matched_birth_coordinates[d], res.input1_matched_death_coordinates[d],
            res.input2_matched_birth_coordinates[d], res.input2_matched_death_coordinates[d]]).tolist())
        ref = sorted(brute.parent_voxel(a, p0[0], d) + brute.parent_voxel(a, p0[1], d + 1)
                     + brute.parent_voxel(b, p1[0], d) + brute.parent_voxel(b, p1[1], d + 1)
                     for p0, p1 in brute.matched_pairs(a, b, d))
        assert ours == ref, d
    assert res.num_matched[2] == brute.dim2_matched(a, b)


def test_batch_and_optional_fields():
    rng = np.random.default_rng(0)
    a = [rng.random((5, 6, 7)) for _ in range(3)]
    b = [rng.random((5, 6, 7)) for _ in range(3)]
    out = compute_matching(a, b, include_input1_unmatched_pairs=True, include_input2_unmatched_pairs=False)
    assert isinstance(out, list) and len(out) == 3
    assert out[0].input2_unmatched_birth_coordinates is None
    assert out[0].num_unmatched_input2 is None
    single = compute_matching(a[1], b[1])
    for d in range(3):
        assert np.array_equal(single.input1_matched_birth_coordinates[d], out[1].input1_matched_birth_coordinates[d])
    with pytest.raises(ValueError):
        compute_matching(a[0], b[0][:4])


def test_binary_counts_unchanged_from_phase1():
    rng = np.random.default_rng(7)
    for _ in range(5):
        fg0 = rng.random((12, 11, 10)) < 0.5
        fg1 = rng.random((12, 11, 10)) < 0.5
        a, b = (~fg0).astype(np.uint8), (~fg1).astype(np.uint8)
        c = match_counts(a, b)
        r = compute_matching(a, b)
        assert r.num_matched.tolist() == c[:, 0].tolist()
        assert (r.num_matched + r.num_unmatched_input1).tolist() == c[:, 1].tolist()
