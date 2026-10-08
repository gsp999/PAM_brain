"""Intensity views for registration; source/fused float arrays are untouched."""
import numpy as np


def registration_view(image: np.ndarray, lo: float = .02, hi: float = .25) -> np.ndarray:
    clipped = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return np.uint8(np.rint(255 * clipped))


def matching_image(image: np.ndarray, intensity_range: tuple[float, float]) -> np.ndarray:
    """Map the acquisition window to the existing matcher window only."""
    if tuple(intensity_range) == (.02, .25):
        return image
    lo, hi = intensity_range
    if not hi > lo:
        raise ValueError('Intensity upper bound must exceed lower bound')
    return (.02 + (image.astype(np.float32) - lo) * (.23 / (hi - lo))).astype(np.float32)
