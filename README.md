# PAM 2D mosaic

This project reads the Col2a1-tdTomato mouse 58 `.dat` sequence, recovers two
bidirectional 500×500 MAP frames per file, registers acquisition positions in
2D, and fuses connected fields of view into floating-point mosaics.

SIFT point matches estimate an SE(2) pose and vessel-edge distance refines it.
The successful frame-to-frame constraints are jointly optimized with LM. No
scale or perspective change is permitted. File numbering is acquisition order,
not a known spatial position. The four scan restarts do not imply spatial
continuity.

## Run

```bash
python3 -m pip install -e .
pam-mosaic --data PAM_Skull_FYH_20260929 --output outputs/lm_full
pam-stitch-tiles --components outputs/lm_full --output outputs/lm_stitched --min-frames 50
```

For a quick trial:

```bash
PYTHONPATH=src python3 -m pam_mosaic.cli --start 1 --stop 100 --output outputs/pilot
```

`--even-direction auto` picks one fixed odd/even line orientation from the
first five selected files. `--forward-only` skips backward-frame averaging.
`--temporal-window` and `--appearance-neighbors` control overlap candidates.
Use `--logs` when log files are in one directory, or repeat `--log PATH` for
logs stored separately. Timestamps are included in the component report.
The pipeline uses LM for pairwise pose refinement and sparse damped LM for the
pose graph, with Huber robust residuals. It does not include a local-map backend.

## Outputs

Each connected component has `mosaic_float64.npy` (scientific values),
`weight_map.npy`, `mosaic_preview.png` (0.02–0.25 display range), and
`poses_xytheta.npy` (pixel translation and rotation in radians). `report.json`
lists source file indices and registration quality. Distinct components are
deliberately not placed together without a measured overlap.
The second command registers and fuses larger connected components into
`outputs/lm_stitched/mosaic_*/` using the same floating-point data and weights.
Its `report.json` lists which component tiles were joined.

## Data interpretation and limits

The reader skips the first eight bytes, reads 500,000 little-endian doubles,
and reshapes them as MATLAB's `[NumAline, NumBscan] = [500, 1000]`. Each half
is transposed so rows are Y/B-scans and columns are X samples. X samples are
resampled from the cosine scan trajectory. The backward frame is reversed in Y
and aligned with a rigid transform calibrated from up to five files.

The provided MATLAB program additionally estimates a fixed, nonuniform
odd/even scan trajectory error. This Python version currently corrects the
sinusoidal sampling and chooses the odd/even direction, but does not reproduce
that full nonuniform trajectory calibration. Inspect a small output against
MATLAB before treating its pixel geometry as quantitatively equivalent.
Log PRR and scan voltage do not determine physical pixel size, so mosaic poses
remain in pixels. Pixel size and physical orientation need acquisition
calibration before labeling axes in millimeters.
