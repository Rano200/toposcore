# toposcore

This is an independent implementation of the **TopoScore** term of the Vesuvius Challenge surface-detection metric. TopoScore is the 0.3-weighted topology part of `0.3 TopoScore + 0.35 SurfaceDice + 0.35 VOI`, computed by Betti matching. It needs no C++ build: it is numpy plus numba.

**What it is**
- A re-implementation, from source, of the counts that the official `topometrics` package gets from its compiled `betti_matching` module (Betti-Matching-3D). It covers the ignore erasure, binarization and inversion, the hard-coded 2x2x2 octant tiling, and the weighted Topological-F1. Every rule is cited to file:line in [SPEC.md](SPEC.md).
- It is checked per case against published exact TopoScore values. See [RESULTS.md](RESULTS.md).

**What it is not**
- It is not pure Python. numba (a PyPI wheel, no compiler) JIT-compiles the union-find and column-reduction loops. With `NUMBA_DISABLE_JIT=1` the same code runs as plain Python. The full test suite passes in both modes, but plain Python is only practical for small volumes.
- TopoScore is the core; SurfaceDice and VOI are in the full leaderboard score below.
- It has not been run side by side with the official compiled binary. Doing so would mean building third-party C++, which this project does not do. Parity is against PLatS's published outputs; see RESULTS.md for exactly what that does and does not establish.

## Install and use

```bash
uv venv --python 3.12 && uv pip install .               # extras: ".[tiff]" (tifffile), ".[nifti]" (nibabel)
toposcore pred.tif gt.tif                                 # same as: python -m toposcore pred.tif gt.tif
```

### Command line

```bash
toposcore pred.npy gt.npy [--ignore-label 2] [--voi-alpha 0.3] [--workers N] [--json]
```

Prints the full leaderboard score and its three parts (TopoScore, SurfaceDice@2, VOI_score), then the per-dimension TopoScore F1 and counts `(matched, pred pairs, gt pairs)`. `--json` prints one JSON object with the same fields (`score`, `toposcore`, `surface_dice`, `voi_score`, `voi_total`, `voi_split`, `voi_merge`, `topoF1_by_dim`, `counts_by_dim`). Other flags: `--no-ignore`, `--allow-nonfinite` (see Input rules). Input files: `.npy`, `.tif/.tiff` (needs `tifffile`), `.nii/.nii.gz` (needs `nibabel`; a clear message is printed if it is missing). Errors print one line to stderr and exit with code 2.

```python
from toposcore import toposcore, match_counts
rep = toposcore(pred, gt)            # same defaults as compute_leaderboard_score
rep.toposcore, rep.topoF1_by_dim, rep.counts_by_dim   # counts = (matched, pred pairs, gt pairs)
match_counts(a, b)                   # raw Betti matching counts of two float volumes, shape (3, 3)
```

### Input rules

`pred` and `gt` are checked before scoring (SPEC section 10). All errors are `ValueError` unless noted.

- **Shape / dimensions:** both must be 3D and have the same shape (zero-size arrays are rejected). For a batch, score each volume separately.
- **dtype:** bool, integer and float all work the same way: foreground is `x != 0` after the ignore voxels are set to 0 (official rule, SPEC section 2). A float probability map is **not** thresholded at 0.5 unless you pass `fg_threshold=0.5`.
- **NaN / inf:** rejected by default. The official scorer instead maps NaN to 0 and counts inf as foreground; pass `check_finite=False` (CLI `--allow-nonfinite`) to get that.
- **Empty prediction or empty GT:** not an error; scored exactly as the official code does. TopoScore is 1.0 (no pairs in any dimension), SurfaceDice is 0.0 (exactly one mask empty), VOI_score is 1.0 (one class only), so the blend is 0.65. Both empty gives 1.0. A model that predicts nothing therefore scores 0.65 on its own, which is a property of the official metric, not a bug here.
- **All-ignore volume** (GT equals the ignore label everywhere): both masks become empty, the score is 1.0, and a `UserWarning` is issued.
- **Memory:** if one octant tile alone is estimated above `TOPOSCORE_MEM_MB` (default 1500) a `MemoryError` explains how to raise the cap or pass `workers=N` (CLI `--workers N`), which skips the estimate.

As in the official scorer, classes that never die are not counted: one solid component gives zero dimension-0 pairs (components minus one per octant tile), and a volume whose tiles have no features in any dimension scores 1.0.

`pred` and `gt` must be 3D arrays of the same shape in the same axis order. Changing the axis order changes the octants and the tie-break order, and with them the score. The official scorer has the same property.

## Speed

Real case from the 0058 arm (sample_00808), one laptop (22 logical cores, 3 GB free RAM when measured), numba kernels cached, wall seconds, `src/bench_speed.py` (raw rows in `results/timing_parallel.jsonl`). 80^3 and 160^3 are strided subsamples of the 320^3 case, i.e. the 4x and 2x downsampling people used.

| volume | TopoScore, workers=1 | TopoScore, default | full leaderboard score, workers=1 | full score, default |
|---|---|---|---|---|
| 80^3  (4x down) | 0.40 s | 0.11 s (8 threads) | 0.44 s | 0.15 s |
| 160^3 (2x down) | 6.5 s  | 1.6 s (8 threads)  | 6.8 s  | 1.9 s |
| 320^3 (real)    | 27.2 s | 14.7 s (2 threads) | 28.5 s | 16.3 s |

Peak memory at 320^3 (Win32 peak working set, whole process, includes the loaded case): 0.97 GB at workers=1, 1.54 GB (TopoScore) / 1.54 GB (full score) at the default 2 threads. The old code peaked at 2.8 GB serial and about 10 GB with 8 threads.

- **Parallelism:** the 8 octant tiles are independent, so `toposcore(..., workers=N)`, `leaderboard_score(..., workers=N)` and `compute_matching(..., workers=N)` (over batch items) use a thread pool; the numba kernels release the GIL and the tiles share the input arrays. `workers=1` is the serial path. Results are identical for every `workers` (integer counts summed in tile order; checked in tests and on all 106 cases, RESULTS.md).
- **Default is memory-capped:** `min(8, cpu_count)`, lowered so that the estimated working sets of the concurrent tiles stay under 1.5 GB (`TOPOSCORE_MEM_MB` to change). One 160^3 octant tile peaks at about 630 MB, so a 320^3 case defaults to 2 threads and an 80^3 case to 8. On a machine with RAM to spare, `workers=8` at 320^3 should take roughly 5 GB and is not measured here.
- **Single-thread wins:** the per-tile memory is 3.9x smaller than before (float32 filtration values when every input value is exact in float32, int32 ids, staged tables); SurfaceDice no longer builds two Euclidean distance maps (an offset-ball marking that is exactly `edt <= tolerance`), 8.0 s to 1.0 s at 320^3. No number changed.
- **No comparison with the compiled C++:** we have not built or run the official `betti_matching` binary, so we cannot say how this compares in speed. These are our own timings only.

The first call in a fresh install adds about 20 s of JIT compilation, which is then cached.

**Pitfall: do not downsample to make the topology term fast.** mgoldfield/vesuvius-kaggle-competition (README ~line 40) reports that people scored at 4x downsampling because the topology part was slow, and that this inflated their scores by +0.16; at full resolution this implementation scores a 320^3 case in about 15 s on two threads, so there is no reason to.

### Full leaderboard score (SurfaceDice@2, VOI_score, blend)

```python
from toposcore import leaderboard_score
rep = leaderboard_score(pred, gt)   # 3D arrays; GT label 2 = ignore
rep.score, rep.topo.toposcore, rep.surface_dice, rep.voi.voi_score
```

`score = 0.3 TopoScore + 0.35 SurfaceDice@2 + 0.35 VOI_score`; needs numpy, numba, scipy. Parity against PLatS's published per-case values: 106/106 cases within 1e-9 for all three (see RESULTS.md).

**voi_alpha:** `VOI_score = 1/(1 + alpha * VOI)`. Sources disagree on alpha. The official `compute_leaderboard_score` defaults to 1.0 (`leaderboard.py:130`), as does a competition recap slide; PLatS and ciscoriordan/mednext-vs-umamba-scroll use 0.3 for the Kaggle metric. We default to 0.3 and keep it a parameter (`voi_alpha=1.0` for the function default). The parity is against PLatS's configuration, not a proof of the Kaggle leaderboard's alpha.

### Drop-in for Betti-matching losses

`toposcore.binding.compute_matching(input1, input2, include_input1_unmatched_pairs=..., include_input2_unmatched_pairs=...)` takes real-valued 3D arrays (or lists of them) and returns objects with the attribute names of the compiled `betti_matching` binding (`input1_matched_birth_coordinates[d]`, `input2_unmatched_death_coordinates[d]`, ...; int64 `(n, 3)` coordinates, `None` for unrequested unmatched lists). That is what villa's `BettiMatchingLoss` (`vesuvius/models/training/loss/betti_losses.py`) reads, so the loss can run without `build_betti.py`. Speed is the limit: see the verdict in [PHASE2.md](PHASE2.md) (fine for validation, too slow to run every training step on CPU). PHASE2.md also has the checks: barcodes and matchings equal a brute-force reference on binary, tied and real-valued volumes.

### Use as a villa metric (optional)

`toposcore.integrations.villa_metric.ToposcoreMetric` has the interface of villa's `vesuvius.models.evaluation.base_metric.BaseMetric` (`compute(pred, gt, **kwargs) -> Dict[str, float]`, `update`, `aggregate`, `reset`). It subclasses villa's `BaseMetric` when `vesuvius` is installed and a same-shaped stand-in otherwise. Torch tensors (any device, bfloat16 too) or numpy arrays work; torch is never imported by this package.

```python
from toposcore.integrations.villa_metric import ToposcoreMetric
metric = ToposcoreMetric()                      # ignore_label=2, voi_alpha=0.3, threshold=0.5
out = metric.compute(pred, gt)                  # pred (B,C,Z,Y,X) or (B,Z,Y,X) or (Z,Y,X); gt (B,1,Z,Y,X) or (Z,Y,X)
print(out["score"], out["toposcore"], out["surface_dice"], out["voi_score"])
metric.update(pred, gt); print(metric.aggregate())   # running mean over calls
```

Shape conventions follow villa's `VOIMetric`: C > 1 is argmax-ed (foreground is class != 0), C == 1 float input is thresholded at `threshold`, an optional `mask=` kwarg marks valid voxels, and batch items are scored one by one and averaged. Tested with numpy only; no villa file is modified.

## Layout

- `src/toposcore/`: the scorer.
  - `betti.py`: cell values, orders and the per-dimension driver.
  - `_kernels.py`: the union-find and reduction kernels.
  - `metric.py`: TopoScore exactly as in `toposcore.py` and `leaderboard.py`, with the thread pool over octants.
- `tests/`: hand-countable shapes, Euler-characteristic checks, and an independent brute-force (set-based Z/2 reduction) for the matching in dimensions 0, 1 and 2. Run `python -m pytest tests` (161 tests; also with `NUMBA_DISABLE_JIT=1`).
  - `__main__.py`: the command line. `integrations/villa_metric.py`: the villa-style metric class.
- `src/parity.py`, `src/check_inputs.py`, `src/refio.py`, `src/get_labels.py`: the parity harness. It reads downloaded data only.
- `src/bench_speed.py`, `src/parallel_parity.py`: timing / peak-memory runs and the parallel parity run.
- `src/s4_robustness.py`: robustness pass on public Scroll-4 eval patches.
- `SPEC.md`, `RESULTS.md`, `LOG.md`.

## Provenance and safety

No third-party code was cloned, built, installed, imported or executed. Official and PLatS sources were read as text. Reference data (PLatS release archive, HF GT labels, Jinhojeong's patch labels) was downloaded into `refdata/` and treated as untrusted data. Licences: topometrics and Betti-Matching-3D are MIT; this re-implementation is MIT.

Written with AI assistance (Claude). I reviewed the code, the tests and the parity runs.
