# RESULTS: TopoScore parity

**What this is.** Our TopoScore against published per-case values on all 106 released Kaggle test cubes, plus self-tests, a robustness pass on 36 Scroll 4 patches, and timing for the Betti-matching drop-in.

**What it is not.** It is not a comparison with the official compiled `betti_matching` binary. We never built or ran third-party code, and no public case was scored by both the official binary and an independent tool.

## Pre-registered bar

Every TopoScore value must be within 1e-9 of the published value, on every available case.

## Reference

PLatS v0.1.0-progress-prize publishes, per case, an exact TopoScore for its automatic instance predictions on the 106 released Kaggle test cubes. The checkpoint 0058 arm is `binary_union` in each case's `metrics.json`, and it is identical to `evidence/04_paper_automatic_hidden106_gpu0_gpu2.json`.

- **Inputs:**
  - Predictions: `pred_instances.nii.gz` from the release archive (sha256 checked against the release SHA256SUMS).
  - GT: `surfaces/kaggle/labels/<case>.tif` from the scrollprize HF bucket, transposed (2,1,0) to PLatS's XYZ layout. Label 2 means ignore.
- **Gate 0 passed before any topology was run.** The reconstructed masks reproduce PLatS's `pred_voxels_scored`, `gt_voxels_scored` and Dice exactly, and the 26-connected component counts, on sample_00808 and sample_00809.
- **Which backend PLatS used.** PLatS computed its values with its `compact_exact` module. That is the official Betti-Matching-3D src_3D code with mechanical storage patches (see SPEC.md section 1). It is not the reference pybind binding. Our code was written from the official binding's source.

## Result (checkpoint 0058, binary union, source-ignore erasure)

| | value |
|---|---|
| cases scored | **106 / 106** |
| bit-identical (\|ours - ref\| = 0.0) | **106 / 106** |
| within 1e-9 | 106 / 106 |
| max abs diff | 0.0 |
| published value equals evidence/04 row | 106 / 106 |
| cases with an active dimension (p_k + g_k > 0), dims 0 / 1 / 2 | 106 / 94 / 14 |
| cases with a non-zero matched count m_k, dims 0 / 1 / 2 | 106 / 3 / 0 |
| mean runtime | 48 s per 320^3 case (8 octants, one core, numba) |

Per-case rows, with per-octant (m, p, g) counts, are in `results/parity_0058.jsonl`. Two overlapping runs scored a few cases twice; the duplicate rows are identical, and `src/summarize.py` keeps one per case.

**Verdict: PASS.** All 106 cases are bit-identical to the published values.

### What the real data exercises, and what it does not

- Real GT has almost no tunnels or cavities, so the real-data parity mainly tests:
  - dimension-0 matching, which is non-trivial and order-dependent, and is matched in all 106 cases;
  - dim-1/dim-2 pair counts, which feed the g = 0 pseudocount branch.
- Dimension-1 matching (m_1 > 0) is exercised in 3 cases only. Dimension-2 matching is never exercised by real data here.
- Those paths are covered by the self-tests instead: brute-force image-persistence matching in dims 0 and 1, and a literal dual-union-find rule in dim 2. That includes 20 partial-matching cases built to have many tunnels.

## Self-tests (161 pass, `python -m pytest tests`)

The suite passes both with numba and in plain Python (`NUMBA_DISABLE_JIT=1`).

- **Hand-countable shapes:** blobs, a diagonal (non-6-adjacent) pair, a ring (b1 = 1), a hollow cube (b2 = 1), a border notch (no cavity), and a mixed volume with (b0 - 1, b1, b2) = (3, 2, 1).
- **Betti numbers** on random volumes, checked against Euler characteristic plus scipy component and cavity counts.
- **Matching counts** in dims 0, 1 and 2 against an independent set-based brute force: binary, tied integer levels, and perturbed porous volumes.
- **TopoScore formula,** octant cutting, and ignore erasure.
- **CLI, input validation, villa-style metric class:** `tests/test_cli.py`, `tests/test_validation.py`, `tests/test_villa_metric.py`.
- **Phase-2 binding** (see PHASE2.md): barcodes and matched pairs as parent-voxel coordinates, compared with brute force on float volumes.

## Scroll 4 robustness

Jinhojeong's diagnostic README (line 156) says the official C++ Betti matching segfaults on one s4 evaluation patch. No reproducer or prediction is public, and no file names the evaluated subset, so **the crashing patch is unidentified**.

Proxy pass (`src/s4_robustness.py`): every one of the 36 public s4 GT patches (236^3) scored against a copy shifted by one voxel in z. All 36 return a finite score, at most 22 s per patch. 18 of them exercise dimension-1 matching (m_1 > 0), which the Kaggle cubes barely do. This shows the tool runs on s4 data; it does not show it handles the patch that crashed the official one.

## Drop-in timing (`src/phase2_timing.py`, real-valued volumes, one core)

| volume | smooth field | white noise |
|---|---|---|
| 64^3 | 1.6 s | 4.6 s |
| 128^3 | 16.9 s | 47.4 s |

White noise is the worst case: it has hundreds of thousands of pairs per dimension at 128^3.

## Not run

Arms 0076 and 0076_plus_100k (another 212 published cases) are downloaded and hash-verified but were not scored: the laptop ran short of memory. Run with `python -I src/parity.py --arm 0076 $(cat refdata/plats_evidence/case_ids.txt)`, and the same with `--arm 0076_plus_100k`.

---

# Full leaderboard score parity (SurfaceDice@2, VOI_score, blend)

## Pre-registered bar (written before any SurfaceDice/VOI value was computed on the reference data)

Arm: checkpoint 0058, `PLatS_hidden106_0058_automatic/cases/<case>/metrics.json`, object `binary_union`, all 106 cases. For every case:

- `|surface_dice - binary_union.surface_dice_tau2| <= 1e-9`
- `|voi_score - binary_union.voi_score| <= 1e-9`
- `|score - binary_union.leaderboard_formula_score| <= 1e-9`, where score = 0.3 toposcore + 0.35 surface_dice + 0.35 voi_score recomputed by us (our toposcore is already bit-identical to `binary_union.toposcore`)

Also reported, not part of the bar: `voi_split`, `voi_merge`. Gate 0: sample_00808 and sample_00809 must pass first. Any miss is reported by case and field. No tolerance change after seeing results.

## Result (checkpoint 0058, 106 cases, `binary_union` fields)

Run: `src/leaderboard_parity.py`, one process, sequential, memory-gated, with `voi_alpha=0.3` (PLatS's configuration). TopoScore for the 104 non-gate cases is the bit-identical value from `results/parity_0058.jsonl`; on the two Gate 0 cases (sample_00808, 00809) the full `leaderboard_score` was recomputed end to end and agrees (SurfaceDice diff 2e-16, VOI_score diff 3e-13, score diff 1e-13). Per-case rows: `results/leaderboard_0058.jsonl`.

| metric (published field) | cases within 1e-9 | exactly equal | max abs diff |
|---|---|---|---|
| SurfaceDice@2 (`surface_dice_tau2`) | 106 / 106 | 63 | 3.3e-16 |
| VOI_score (`voi_score`) | 106 / 106 | 0 | 1.0e-12 |
| blended score (`leaderboard_formula_score`) | 106 / 106 | 0 | 3.5e-13 |
| voi_split / voi_merge (info) | 106 / 106 | 0 | 5.7e-12 / 9.1e-12 |

**Verdict: PASS on the pre-registered bar** (the bar was met; the margin is about 1e-12, float summation order).

### Deviations found at Gate 0 (disclosed)

Gate 0 attempt 1 failed (kept as `results/leaderboard_0058_gate0_attempt1_FAILED_alpha1_nopad.jsonl`). Two causes, both fixed from source before the full run; the bar itself was not changed:

1. SurfaceDice off by 0.003: `surface_distance` zero-pads the bounding-box crop by one voxel on the high side (`metrics.py:97-119`). I had read only the correlate step.
2. VOI_score off by 0.28: the published values use `voi_alpha=0.3`. voi_split and voi_merge matched to 2e-12 from the first attempt, so the labelling and entropy code was right.

### The voi_alpha disagreement

Sources disagree: official `compute_leaderboard_score` default 1.0 (`leaderboard.py:130`) and a recap (speakerdeck sugupoko/vesuvius2-combined slide 4) say 1.0; PLatS (`kaggle_surface.py:30`) and ciscoriordan/mednext-vs-umamba-scroll README (~line 25) say 0.3. The Kaggle evaluation page is JS-only and was not read. **This parity is against PLatS's configuration. It is not proof of the Kaggle leaderboard's alpha.** `voi_alpha` is a parameter; our default is 0.3.

### Caveats

- The 256-entry surface-area table was extracted from the text of surface_distance's lookup table (data, not code). Parity over 106 cases agrees to 3e-16, which is what a wrong table entry could not do.
- Only unit spacing is implemented.
- The `scipy.ndimage.label` 26-connectivity replaces `cc3d`; VOI is invariant to label values.
- Tests: 161 pass, with and without `NUMBA_DISABLE_JIT=1`.

---

# Parallel / memory-lean run: pre-registered bar (written 2026-10-22, before any parallel parity case was run)

Change under test: octant tiles run on a thread pool (`workers=`), and `match_counts` was made memory-lean (float32 cell values when every input value is exactly representable in float32, int32 ids and ranks below 2^31 cells, staged per-input tables). Neither change is meant to alter any number.

Arm 0058, all 106 cases, new default `workers`, one foreground sequential process, memory gate (wait while < 1500 MB available), lock `results/.parallel.lock`. Outputs: `results/parity_0058_parallel.jsonl` (TopoScore) and `results/leaderboard_0058_parallel.jsonl` (full leaderboard score, TopoScore recomputed end to end).

PASS requires, on every case:
- TopoScore: `|new - serial| = 0.0` against `results/parity_0058.jsonl`, and also `|new - PLatS published| = 0.0`; per-dimension counts and per-octant counts identical to the serial rows.
- Leaderboard: `|surface_dice, voi_score, score - our serial values in results/leaderboard_0058.jsonl| <= 1e-12`, and `<= 1e-9` against the PLatS `binary_union` fields.
Any miss is reported by case and field; the bar is not changed after seeing results.

## Result of the parallel / memory-lean run (106 / 106 cases, default workers)

Run: `src/parallel_parity.py`, one foreground sequential process, memory-gated, `results/.parallel.lock`. Default workers for these cases: 2 threads (320^3 volumes, 160^3 tiles). Mean 17.8 s per case for the full leaderboard score including data loading (the serial run was ~48 s for TopoScore alone).

| check | result |
|---|---|
| TopoScore vs our serial rows, \|diff\| | 0.0 on 106 / 106 |
| TopoScore vs PLatS published, \|diff\| | 0.0 on 106 / 106 |
| (m, p, g) counts and all per-octant counts equal serial | 106 / 106 |
| SurfaceDice, VOI_score, score vs our serial values | max diff 0.0 (bit-identical; bar was 1e-12) |
| SurfaceDice / VOI_score / score vs PLatS | 3.3e-16 / 1.0e-12 / 3.5e-13 (bar 1e-9; same as the serial run) |

**Verdict: PASS on the pre-registered bar, with margin: every number is bit-identical to the serial results.** Rows: `results/parity_0058_parallel.jsonl`, `results/leaderboard_0058_parallel.jsonl`.

Note: SurfaceDice now uses an offset-ball marking instead of two Euclidean distance maps (same mask, tested equal to `distance_transform_edt <= tol`), so the leaderboard parity above also covers that change.

## Speed and memory

Timing table, per-config peak working set and caveats are in README.md ("Speed"); raw rows in `results/timing_parallel.jsonl` (`src/bench_speed.py`; peak = Win32 `PeakWorkingSetSize` through ctypes, psutil not installed; it includes the loaded case, so small-volume rows show the load peak). Per-tile peak of `match_counts` on a 160^3 tile fell from 2.4 GB to 0.63 GB (`src/bench_tile.py`). With the old code 8 threads at 320^3 peaked at 10.4 GB and gave no speedup (swapping), so the default is capped by estimated memory (about 2 threads at 320^3). The compiled C++ was not run; no comparison is made.
