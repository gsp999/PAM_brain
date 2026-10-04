# PAM 2D mosaic

This project reconstructs and stitches the 1301 Col2a1-tdTomato mouse 58
photoacoustic acquisitions in `PAM_Skull_FYH_20260929`.

Each `.dat` file contains a 500×1000 MATLAB matrix: two 500×500 MAP scans of
the same acquisition position, one forward and one backward. The current
pipeline therefore assigns **one global pose to each `.dat` file**. The two
directions are handled inside that file; they are never inserted into the
global mosaic as two independent positions.

## Current pipeline

For every acquisition, the pipeline:

1. reads the floating-point values after the eight-byte header and reconstructs
   the two scan directions using the calibrated raster trajectory;
2. mildly removes compact positive outliers from both directions;
3. registers backward to forward and fuses the original values only when the
   residual alignment is well supported and close to identity;
4. otherwise keeps the direction with the stronger usable vessel structure;
5. uses only the resulting filtered image for all inter-file registration—there
   is no fallback to an unfiltered image;
6. registers neighboring acquisitions with SIFT/RANSAC and rigid SE(2)
   refinement, conservatively retries missed adjacent pairs, then jointly
   optimizes each connected pose graph with robust damped LM;
7. fuses the original floating-point MAP values and writes a separate mosaic
   for every reliably connected component.

The extra experimental backend in `refine_backend.py` is not part of the
current result. The final connection of large components still uses the
ordinary tile matcher and LM pose graph; rotation-constrained tile stitching is
the next planned improvement.

## Install and run

```bash
python3 -m pip install -e .
pam-mosaic
pam-stitch-tiles --min-frames 50
```

Those defaults reproduce the current configuration and write:

- frontend components to `outputs/moderate_denoise_full`;
- the connected large mosaic to `outputs/moderate_denoise_stitched`.

Equivalent explicit commands are:

```bash
pam-mosaic --data PAM_Skull_FYH_20260929 \
  --validated-roundtrip --frontend-recovery --suppress-speckles \
  --output outputs/moderate_denoise_full

pam-stitch-tiles \
  --components outputs/moderate_denoise_full \
  --output outputs/moderate_denoise_stitched \
  --min-frames 50
```

For a short trial, use for example:

```bash
pam-mosaic --start 1 --stop 100 --output outputs/pilot
```

`--no-frontend-recovery` and `--no-suppress-speckles` disable the corresponding
current-stage options. `--temporal-window` and `--appearance-neighbors` control
inter-file candidate generation. Use `--logs DIR`, or repeat `--log PATH`, to
attach acquisition timestamps from scan logs.

## Baseline comparison

The preserved baseline directly averages the two scan directions after one
fixed correction. It is available only when explicitly requested:

```bash
pam-mosaic --baseline --output outputs/lm_full
pam-stitch-tiles --components outputs/lm_full \
  --output outputs/lm_stitched --min-frames 50
```

Baseline mode automatically disables frontend recovery and bright-point
suppression. `--forward-only` and manual `--even-direction` choices apply to
this mode and are intentionally incompatible with validated round-trip mode.

## Outputs

Each component directory contains:

- `mosaic_float64.npy`: scientific floating-point mosaic values;
- `weight_map.npy`: accumulated valid-pixel weights;
- `mosaic_preview.png`: display image using the fixed 0.02–0.25 range;
- `poses_xytheta.npy`: pixel translations and rotation in radians.

The frontend `report.json` records source files, timestamps, registration
quality, forward/backward decisions, filtering changes, recovered adjacent
links, and connected components. The tile-stitch report records selected large
components, block matches, and final groups.

Generated outputs and raw acquisitions are excluded by `.gitignore`.

## Interpretation and limits

The reader skips the first eight bytes, reads 500,000 little-endian doubles,
and reshapes them as MATLAB `[NumAline, NumBscan] = [500, 1000]`. Each half is
transposed so rows are Y/B-scans and columns are X samples. X positions are
resampled from the calibrated cosine-like scan trajectory, and the backward
scan is reversed in Y before within-file validation.

OpenCV registration is not numerically identical to MATLAB `imregcorr`.
Acquisition logs provide PRR, scan voltage and timestamps but not physical
pixel size or stage coordinates, so poses remain in pixels. Physical axes and
millimetre scale require a separate acquisition calibration. A successful
image match also does not by itself prove anatomical correctness; block seams
and newly recovered adjacent links must still be checked for duplicated or
broken vessels.
