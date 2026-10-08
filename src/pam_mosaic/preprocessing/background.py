"""Shared scalar acquisition background estimation without clipping."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def correct_background(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Subtract the smoothed histogram mode of the lower intensity distribution.

    Estimate a per-frame offset from the 0.5th to 70th percentiles. This is an
    estimated baseline, not a physical calibration; preserve negative residuals
    and vessel contrast rather than truncating them to zero.
    """
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError('Expected a finite 2D image for background correction')
    lo, hi = np.percentile(source, [.5, 70])
    if hi <= lo:
        baseline = float(lo)
    else:
        counts, bins = np.histogram(source, bins=256, range=(float(lo), float(hi)))
        peak = int(np.argmax(gaussian_filter1d(counts.astype(float), 2)))
        baseline = float((bins[peak] + bins[peak + 1]) / 2)
    return (source - baseline).astype(np.float32), baseline
