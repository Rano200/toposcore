# SPEC: the TopoScore component of the Vesuvius surface-detection metric

This is the specification our implementation follows. Every rule cites the official source it comes from.

**Source.** `topometrics` and its vendored Betti-Matching-3D, as copied into PLatS:
`https://github.com/huanghoujing/PLatS/blob/main/third_party/topometrics/` (abbreviated `TM/` below), with the C++ under `TM/external/Betti-Matching-3D/src/` (abbreviated `BM/`).

**Hash check.** These files hash-match the Kaggle dataset `sohier/vesuvius-metric-resources`: `TM/src/topometrics/{__init__,_bm_loader,leaderboard,toposcore,voi}.py`, `BM/_BettiMatching.cpp` and `BM/../CMakeLists.txt`. The reference hashes are the ones recorded in `aviad12g/tifxyz-doctor` `experiments/fusion-aware-surface/official_metric.py`. The `BM/src_3D/*.cpp` files are PLatS's vendored copy of upstream Betti-Matching-3D. No published Kaggle hash exists for them.

## 1. Which matcher the official metric calls

- `toposcore.py:8`: `bm = load_betti_matching()`. In `_bm_loader.py:13` that is `importlib.import_module("betti_matching")`, the full pybind binding built from `BM/`. It is **not** a count-only variant.
- `toposcore.py:52`: `bm.compute_matching(predictions, labels)`. This uses the defaults `include_input1_unmatched_pairs=True` and `include_input2_unmatched_pairs=True` (`BM/_BettiMatching.cpp:426-438`).
- `BM/_BettiMatching.cpp:25,154-155`: inputs are cast to C-contiguous `float64` (`array_t<double, c_style>`), so non-contiguous octant views are copied.
- `BM/config.h` active switches:
  - `COMPUTE_COMPARISON`
  - `USE_EMERGENT_PAIRS`
  - `USE_CACHE`
  - `USE_CLEARING_DIM0`
  - `USE_STABLE_SORT_OR_STABLE_PARTITION`
  - `PARALLELIZE_INDEPENDENT_BARCODES_DIM1`
  - `value_t = double`; threshold = +inf

**PLatS's numbers.** PLatS scored its published results with its own `compact_exact` module, not this binding. That module is a chain of patches: an upstream `betti_matching_count_only` base, then `provenance/betti_matching_binary_exact.patch`, then `provenance/betti_matching_compact_exact.patch`. In the src_3D hunks of those patches, the pre-image lines equal the official source. The changes are mechanical storage changes (`birth`->`filtration()`, `index`->`identity()`, 2-bit births, 32-bit identities) plus a counts-only output. PLatS publishes no case scored by both modules.

## 2. Inputs to the matcher (leaderboard.py)

- Ignore handling: GT voxels with value `ignore_label=2`, or an explicit mask, are set to 0 in BOTH the prediction and the GT (`leaderboard.py:159-165`).
- Binarize with `x != 0`, NaN counts as 0. With `fg_threshold` set it is `x > t` instead (`:23-33, 168-169`).
- **Invert**: `topo = (~bin).astype(uint8)`, so foreground = 0 and background = 1 (`:173-174`). Homology therefore describes the FOREGROUND as a sublevel set at t = 0.
- **Tiling, hard-coded**: the volume is split 2x2x2 into octants with `_axis_cuts`; sizes differ by at most 1 and the first chunks are the larger ones (`:84-111, 176-179`).
- Each octant is matched independently: `TopoScore.compute(pred_tile, gt_tile)`, with input1 = prediction and input2 = GT (`:181-188`).
- Counts are summed over the tiles and the weights are applied once (`:189`, `toposcore.py:252-264` -> `from_counts` `:205-245`).

## 3. The cubical complex (V-construction)

- Voxels are vertices. The grid is padded with +inf (`BM/src_3D/data_structures.cpp:318-333`).
- A cell's value is the max over its vertices: edges, squares and cubes (`:129-168`).
- Cells with value +inf, i.e. those leaving the grid, are dropped by the `birth < threshold` tests.
- Foreground connectivity is therefore 6-connectivity, and complement components (cavities) use 26-connectivity.
- Cell identity is the packed index `x<<44 | y<<24 | z<<4 | type` (`:12-16`).
  - Edge type 0/1/2 = +x/+y/+z.
  - Square type 0 spans (y,z), type 1 spans (x,z), type 2 spans (x,y).
- **Total order within a dimension** = (value, packed index):
  - `CubeComparator` (`:42-48`).
  - `stable_partition`/`stable_sort` by value over the x,y,z,type enumeration (`dimension_0.cpp:186-217`, `dimension_1.cpp:239-282`, `dimension_2.cpp:124-172`).
  - These two orders agree because enumeration order equals packed-index order.
  - Ties at equal value fall to the lower dimension first; each dimension is reduced separately.
- Only pairs with **birth != death** are emitted:
  - dim 0: `dimension_0.cpp:93, 141`
  - dim 1: `dimension_1.cpp:742`
  - dim 2: `dimension_2.cpp:265, 318`
- Essential classes are never emitted. For a binary tile, the dim-0 pairs number (#6-components of FG) - 1 when FG is non-empty.

## 4. Per-dimension algorithms

The comparison image is `min(input1, input2)`, the union of the foregrounds (`BM/BettiMatching.cpp:26-29`). The dimensions run in the order 2, 1, 0 (`src_3D/BettiMatching.cpp:50-110`).

**Dimension 2: dual union-find** (`dimension_2.cpp`)

- Nodes are the cubes plus one outside node with value +inf (`data_structures.cpp:423-438`).
- Squares are processed in **decreasing** (value, index) order (`dimension_2.cpp:254, 310`).
- When two roots differ, the younger dies. The root is the max of (value, index) (`data_structures.cpp:455-471`). Squares on the box faces connect to the outside node (`:481-530`).
- For input k (`:234-287`):
  - A comparison union-find receives the same unions.
  - A pair with birth != death is keyed by the **comparison-image younger root** (`matchMap.emplace(birthIdxComp)`, `:271`).
- Comparison pass (`:289-350`):
  - A pair is matched iff its younger root is a key in both input maps (`:326-329`).
- Every merging square is removed from that complex's dimension-1 column list (clearing, `:273, 338`).
- Dimension 2 is skipped entirely if any tile axis has length 1 (`:39-46`).

**Dimension 1: column reduction** (`dimension_1.cpp`)

- Columns are the squares that survive dimension-2 clearing, in (value, index) order. Rows are edges.
- The pivot is the maximum (value, index) face. A working column is a max-heap with Z/2 cancellation (`popPivot`, `:310-335`).
- Emergent pairs (`:409-475`): the boundary enumerator yields faces in decreasing packed index (`enumerators.cpp:18-102`). The first face whose value equals the column's value is therefore the column's largest face. If no column owns it yet, it becomes the pivot. Otherwise reduction continues with the owner's reduced column, which is cached (`:370-404, 626-794`). This is ordinary left-to-right reduction, so by pairing uniqueness its pivots equal those of any left-to-right reduction with the same row and column orders.
- Input-k and comparison pairs require pivot value != column value. An input pair is keyed by its birth edge (`:740-746`).
- Image-k reduction (`:88-118, 754-760`):
  - The columns are the comparison's columns (`ctrImage = ctrComp`, `:82`).
  - Rows and values come from input k's complex.
  - For every column with a pivot, `matchMapIm_k[column] = pivot`. There is no persistence filter here.
- Matching (`:191-216`): a comparison pair (e, s) is matched iff `matchMapIm_0[s]` and `matchMapIm_1[s]` exist and each is the birth edge of a persistent pair of input 0 and input 1 respectively.

**Dimension 0: union-find** (`dimension_0.cpp`)

- Edges are processed in (value, index) order. Under `USE_CLEARING_DIM0`, dimension-1 pivot edges are skipped (`dimension_1.cpp:59,70,84`). Those edges close cycles, so skipping them is a no-op for union-find.
- The root is the min of (value, index); the younger root dies (`data_structures.cpp:365-381`).
- Input pass: a pair with birth != death is keyed by its birth vertex (`:68-109`).
- Comparison pass (`:111-163`):
  - On each comparison merge, the input-0 and input-1 union-finds link the same two endpoints' roots. Each yields its own younger root.
  - The comparison pair (birth != death) is matched iff both of those roots are keys.

**Order dependence.** For binary inputs, the matched counts depend on these tie-break orders. Example: the sequence in which components merge at t = 1 decides which bars are matched. Reproducing the exact orders is therefore required, not optional.

## 5. From pairs to the score (toposcore.py)

- Matched pairs m_k = the number of matches in dimension k.
- Prediction pairs p_k = m_k + unmatched(pred); GT pairs g_k = m_k + unmatched(gt) (`toposcore.py:189-194`). Unmatched = the emitted pairs not marked matched (`src_3D/BettiMatching.cpp:112-140`), so p_k and g_k are simply the numbers of persistent pairs.
- Per-dimension F1 (`:73-80`, `:222-229`):
  - If p_k + g_k > 0 and g_k != 0: F1_k = 2 m_k / (p_k + g_k).
  - If g_k = 0: F1_k = 0.5 / (p_k + 0.5), a pseudocount.
  - Otherwise F1_k = NaN, and dimension k is inactive.
- TopoScore (`:84-93`, `:233-242`):
  - The weights (0.34, 0.33, 0.33) are renormalized over the active dimensions: `w / w.sum()` then `(w * F1).sum()` in float64.
  - If no dimension is active, TopoScore = 1.0.
- Leaderboard blend (`leaderboard.py:133, 216-218`): 0.3 TopoScore + 0.35 SurfaceDice@2 + 0.35 VOI_score. The blend is specified in sections 7-9 below.

## 7. SurfaceDice@2 (leaderboard.py:191-200 and the surface_distance algorithm)

Source: `TM/src/topometrics/leaderboard.py` (PLatS `third_party/topometrics`); the library it imports (`import surface_distance as sd`, `leaderboard.py:6`) is google-deepmind/surface-distance `surface_distance/metrics.py` and `lookup_tables.py`, read as text only. Our code: `src/toposcore/surface.py`.

- Inputs are the already-ignored, binarized masks `pr_bin`, `gt_bin` (`leaderboard.py:164-169`; section 2 here). Spacing (1,1,1) (`:125, 192`), tolerance 2.0 (`:124`), "voxels".
- Edge cases (`:194-198`): both empty -> 1.0; exactly one empty -> 0.0. Otherwise `sd.compute_surface_distances(gt_bin, pr_bin, sp)` then `compute_surface_dice_at_tolerance(., 2.0)` (`:199-200`).
- Crop to the bounding box of `gt | pred` (`metrics.py:213-224`), then zero-pad ONE voxel on the high side of every axis (`_crop_to_bounding_box`, google-deepmind/surface-distance `surface_distance/metrics.py:97-119`, the library `leaderboard.py:6,199` imports, not a PLatS wrapper; its first 2018 commit e1c972f0 pads the same way, comment "zeropad ... right and the back side"), so the 2x2x2 windows on the far side exist. (Verified against the published values: without this pad SurfaceDice is off by about 0.003.)
- Neighbour code: `ndimage.correlate(mask.uint8, kernel, mode="constant", cval=0)` with `kernel = [[[128,64],[32,16]],[[8,4],[2,1]]]` (`lookup_tables.py:23-24`, `metrics.py:231-234`). With a size-2 kernel the window of output p covers inputs p-1..p; thanks to the high-side pad the output has one more position per axis than the crop. We reproduce this exactly (`neighbour_codes` on the padded crop).
- Surface elements = positions with code not 0 and not 255 (`metrics.py:237-240`).
- Per-code area = sum of the lengths of the marching-cubes triangle normals (`lookup_tables.py:36-292`, `296-327`, unit spacing). Our table `_surface_table.py` holds those 256 sums (values extracted by script from the text; the file was not imported). Single corner = sqrt(3)/8.
- Distance of each element to the other mask's element set: `distance_transform_edt(~borders)` (`metrics.py:244-258`), compared with `<= tolerance` (`:421-422`).
- Dice = (area of GT elements within 2 of pred + area of pred elements within 2 of GT) / (total GT area + total pred area) (`:423-424`). Sorting of the elements in the library is irrelevant to the sums.

## 8. VOI_score (voi.py:67-159, called at leaderboard.py:204-214)

Our code: `src/toposcore/voi.py`.

- Called on the binarized, ignore-erased masks with `connectivity=26`, `use_union_mask=True`, `score_transform="one_over_one_plus"`, ignore off (`leaderboard.py:204-214`, `voi.py:67-78`). **alpha: the three sources disagree.** (a) The official `compute_leaderboard_score` defaults `voi_alpha=1.0` (`leaderboard.py:130`, passed at `:204-210`). (b) A public competition recap (speakerdeck.com/sugupoko/vesuvius2-combined, slide 4) says "VOI Score (0.35) ... 26-connectivity, alpha=1.0". (c) Only PLatS `src/vesuvius_p2sd/eval/kaggle_surface.py:30, 165-168` says 0.3, claiming it is the setting of the public Kaggle metric notebook and rejecting anything else. The Kaggle evaluation page is JS-only and could not be read, so no primary evidence for 0.3 was found. PLatS's published `voi_score` values are reproduced exactly only with alpha = 0.3 (0.3 x 2.849 = 0.855 -> 0.5391; alpha 1.0 gives 0.2598). (d) A third source, github.com/ciscoriordan/mednext-vs-umamba-scroll README ~line 25, runs the official metric from sohier/vesuvius-metric-resources and states VOI_score = 1/(1 + 0.3*VOI) over 26-connected components. Tally: 0.3 = PLatS + ciscoriordan; 1.0 = function default + speakerdeck. Most likely the Kaggle notebook passes `voi_alpha=0.3` to a function whose default is 1.0, but no primary text (the notebook itself) was read. **Our `leaderboard_score` / `voi_report` keep `voi_alpha` as a parameter and default to 0.3; pass 1.0 for the function default. Parity runs use 0.3, PLatS's configuration, which agrees with the published numbers; that is not primary proof of the Kaggle leaderboard's alpha.**
- Both masks empty -> voi_total 0, voi_score 1.0 (`voi.py:115-118`). Note: this case is reached in the leaderboard only if both masks are empty, where SurfaceDice is already 1.0.
- Crop to the union bounding box (`voi.py:121-128`; exact, labelling is translation invariant, so we always crop).
- 26-connected component labelling of each mask separately, `cc3d` (`:131-132`). Label values are arbitrary, VI is invariant to relabelling, so scipy `ndimage.label` with a full 3x3x3 structure is equivalent.
- Union mask: only voxels with GT label > 0 or PR label > 0 are used; a label of 0 in one image is a real class paired against the other's component (`:134-138`).
- `variation_of_information(a=GT labels, b=PR labels)` (`:144`) of scikit-image: entropies in bits (log2, `skimage/metrics/_variation_of_information.py`), `ignore_labels=()`, no normalization. It returns [H(b|a), H(a|b)]. Published `voi_split` = first, `voi_merge` = second (the source comments say the opposite; the code is what runs).
- `voi_total = split + merge`; `voi_score = 1/(1 + alpha * voi_total)`, with alpha a parameter (our default 0.3; function default 1.0; see above) (`:145-150`).

## 9. The blend (leaderboard.py:216-218)

Weights `(0.3, 0.35, 0.35)` are clamped (non-finite or negative -> 0) and renormalized by their sum (`:56-61`). `score = w_topo*TopoScore + w_surf*SurfaceDice + w_voi*VOI_score`. Published field names (metrics.json `binary_union`): `surface_dice_tau2`, `voi_score`, `voi_split`, `voi_merge`, `toposcore`, `leaderboard_formula_score`.

## 10. Input validation of this implementation (not part of the official scorer)

- 3D only, identical shapes, no zero-size arrays, dtype bool/int/float. The official code has no such checks; wrong shapes would fail or mis-tile inside it.
- Binarization is `x != 0` for every dtype, after the ignore voxels are zeroed (section 2, `leaderboard.py:23-33, 159-169`).
- NaN and inf raise by default. Official behaviour (NaN -> 0, inf -> foreground, section 2) is available with `check_finite=False`.
- Empty prediction / empty GT / both empty are not errors. The values follow from the official rules: no essential classes are emitted, so TopoScore has no active dimension and is 1.0 (section 3 last bullet, section 5 last bullet); SurfaceDice is 0.0 when exactly one mask is empty and 1.0 when both are (section 7, `leaderboard.py:194-198`); VOI sees one class only, VOI 0, score 1.0 (section 8; the both-empty case is `voi.py:115-118`). One empty mask therefore scores 0.3 + 0.35 = 0.65.
- All-ignore GT: both masks are empty after erasure, score 1.0 (the same rules), with a `UserWarning` from this implementation.
- Memory: the working set is estimated at 160 bytes per voxel of one octant tile; see README.
