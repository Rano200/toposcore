# Phase 2: a no-C++-build `compute_matching` for villa's Betti matching loss

**What this is.** `toposcore.binding.compute_matching` is a drop-in replacement for the `betti_matching` pybind module's `compute_matching`, as consumed by villa `vesuvius/src/vesuvius/models/training/loss/betti_losses.py` (origin/main). It needs numpy plus numba; there is nothing to compile.

**What it is not.**
- It only handles 3D volumes. 1D, 2D and nD raise `NotImplementedError`. The reference has separate 1D/2D/nD code paths, and their tie rules were not ported.
- It is not a villa change. No villa file was edited.
- It is not verified against the compiled binary, which this project never builds or runs. It is verified against an independent brute force and against phase-1 parity.

**Possible use in villa:** `_load_betti_module()` could fall back to this implementation when the compiled extension is missing. No such PR has been opened.

## What the loss reads, and what we return

These are the uses in `betti_losses.py`:

| Use (line in origin/main file) | Provided |
|---|---|
| `bm.compute_matching(list_pred, list_tgt, include_input1_unmatched_pairs=True, include_input2_unmatched_pairs=flag)` returns a list, one result per pair of float64 C-contiguous volumes (519-524, 626-636, 686-691) | yes. A list in gives a list out; two arrays in give a single result (pybind overloads, `_BettiMatching.cpp:426-540`) |
| `res.input{1,2}_matched_{birth,death}_coordinates[d]`, d = 0..ndim-1 (142-145, 377-380) | list of 3 `int64` arrays of shape `(n_d, 3)` |
| `res.input{1,2}_unmatched_{birth,death}_coordinates` checked against `None`, then `[d]` (173-191, 382-385) | the list when the flag is set, `None` otherwise (std::optional semantics, `_BettiMatching.cpp:342-353`) |
| `num_matched`, `num_unmatched_input1/2` (binding fields, `:1111-1115`) | `int64` arrays of shape `(3,)`, or `None` when that side's unmatched list was not requested |
| Inputs are real-valued: sigmoid/softmax probabilities, with `1 - p` for superlevel | any finite float; NaN raises |

Two further details follow the reference:
- **Coordinates** are the parent voxel of each birth and death cell, using the reference's tie order (`data_structures.cpp:172-261`).
- **Pair order** within each list follows the order in which the reference appends pairs:
  - matched pairs in match order: comparison edge order (dim 0), comparison column order (dim 1), reverse comparison order (dim 2);
  - unmatched pairs in barcode order.
  - Order only changes the float summation order of the loss.

## Verification

All tests are in `tests/test_binding.py`; 42 pass.

- **Volumes tested.** 40 random volumes of shapes 2..6 per axis, of four kinds:
  - binary;
  - integer levels 0..3 (heavy ties);
  - uniform floats;
  - a float volume paired with a noisy copy of itself.
- **Checks on each volume:**
  - Each input's full barcode (matched plus unmatched), as parent-voxel coordinate pairs, equals a set-based Z/2 brute-force reduction with an independently restated `getParentVoxel` (`tests/brute.py`).
  - Matched pairs in dims 0 and 1, with both inputs' birth and death coordinates, equal brute-force image-persistence matching.
  - The dim-2 matched count equals a literal restatement of the reference's dual union-find rule.
  - Counts equal phase-1 `match_counts`, and dtypes and shapes are checked.
- **Binary inputs:** the counts equal phase 1, and phase 1 reproduced PLatS's published TopoScore bit-for-bit (RESULTS.md). The binding and phase 1 share the same kernels; the binding only adds pair recording.
- **Batch API:** list in gives list out, `include_input2_unmatched_pairs=False` gives `None` fields, and mismatched shapes raise ValueError like the binding.

## Runtime (one laptop core, numba JIT warm)

The inputs are `1 - p` (superlevel), as in the loss default:
- **smooth:** sigmoid of Gaussian-filtered noise, paired with a noisy copy;
- **noise:** independent uniform noise, the worst case.

**64^3 and 128^3** (`python -I src/phase2_timing.py`, `results/phase2_timing.json`): smooth 1.6 s and 16.9 s; white noise 4.6 s and 47.4 s.

Small sizes, uniform-noise float pairs, warm JIT:

| size | seconds | matched (d0, d1, d2) |
|---|---|---|
| 24^3 | 0.17 | 971, 808, 83 |
| 32^3 | 0.47 | 2245, 1875, 237 |

Two reference points for scaling:
- Binary 160^3 tiles take about 8 s each (phase 1).
- Noise is the worst case because its pair counts grow with volume.

Peak memory is roughly 1 GB per 160^3 tile. The free RAM on this laptop (about 1-3 GB) is the binding constraint.

The first call in a fresh environment JIT-compiles the kernels (about 20-30 s) and caches the result on disk (`cache=True`).

## Verdict: training-loss backend on CPU?

Measured with `python -I src/phase2_batch_timing.py` (`results/phase2_batch_timing.json`): `compute_matching` on float32 smooth fields (sigmoid of Gaussian-filtered noise vs a noisy copy, `1 - p`), numba warm, one laptop with 22 logical cores. Wall seconds per batch; `workers=1` is serial, the default uses threads over batch items (min(8, batch size) here).

| patch | batch | workers=1 | default workers |
|---|---|---|---|
| 64^3 | 2 | 1.8 s | 1.2 s |
| 64^3 | 4 | 3.5 s | 1.4 s |
| 96^3 | 2 | 7.4 s | 5.2 s |
| 96^3 | 4 | 15.2 s | 5.8 s |

Honest reading:
- This is the matching only (the forward part of the loss). It excludes the gradient scatter, the model and any GPU transfer. Smooth fields are the favourable case; white noise was about 3x slower per item at 64^3 (4.6 s vs 1.6 s, above), and an untrained network's output looks more like noise than like these fields.
- Parallelism helps only across batch items, so a batch of 1 gets no speedup, and a 96^3 item costs roughly 4 to 7 s however many cores there are.
- A GPU training step on such patches usually takes well under a second (not measured here). Calling this every step would make the loss the whole cost, a slowdown of one to two orders of magnitude. **It is not practical as a per-step training-loss backend on CPU at 96^3.** At 64^3 with a batch of 2 to 4 it costs 1 to 1.5 s per batch, usable only if the loss is applied every N steps, in a fine-tuning run where throughput does not matter, or on a subsample of the batch.
- It is practical for **testing and validation**: running villa's `BettiMatchingLoss` code path without building `betti_matching`, checking a build against it, and scoring held-out patches.
- We have not run the compiled C++ and so cannot say how its speed compares.

## Not done or open

- 1D/2D/nD inputs.
- Representative cycles (`BettiMatching` class API).
- `compute_barcode`.
- Thread-parallel batches: the reference uses `std::async` per batch item; we run items sequentially.
