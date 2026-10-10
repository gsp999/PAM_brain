"""Optional per-frame histogram-mode subtraction without clipping."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def correct_background(image: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 2 or not source.size or not np.isfinite(source).all():
        raise ValueError('Expected a finite nonempty 2D image')
    lo, hi = np.percentile(source, [.5, 70])
    if hi <= lo:
        baseline = float(lo)
    else:
        counts, bins = np.histogram(source, bins=256, range=(float(lo), float(hi)))
        peak = int(np.argmax(gaussian_filter1d(counts.astype(float), 2)))
        baseline = float((bins[peak] + bins[peak+1])/2)
    return (source-baseline).astype(np.float32), baseline
