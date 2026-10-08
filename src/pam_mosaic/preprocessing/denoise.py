"""Shared conservative removal of isolated impulse artifacts in MAP frames."""
from __future__ import annotations

import cv2
import numpy as np


def bright_point_mask(image: np.ndarray, *, excess: float = .012,
                      max_area: int = 28, max_extent: int = 9,
                      median_size: int = 5) -> np.ndarray:
    """Locate compact positive outliers while excluding elongated vessels."""
    source = np.ascontiguousarray(image, dtype=np.float32)
    median = cv2.medianBlur(source, median_size)
    candidates = ((source - median) > excess).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    selected = np.zeros(count, dtype=bool)
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        selected[label] = area <= max_area and width <= max_extent and height <= max_extent
    return selected[labels]


def remove_isolated_bright_points(image: np.ndarray, *, excess: float = .012,
                                  max_area: int = 28, max_extent: int = 9,
                                  median_size: int = 5,
                                  preserve: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Replace only small bright outlier islands with the 3x3 local median.

    Continuous thin structures are retained when their high-pass response
    forms a component larger than ``max_area``. This is not a general image
    smoothing or a claim that every removed pixel is non-biological.
    """
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite 2D MAP frame")
    source = np.ascontiguousarray(image, dtype=np.float32)
    median = cv2.medianBlur(source, median_size)
    replace = bright_point_mask(source, excess=excess, max_area=max_area,
                                max_extent=max_extent, median_size=median_size)
    if preserve is not None:
        if preserve.shape != image.shape:
            raise ValueError("Preservation mask shape differs from image")
        replace &= ~preserve
    filtered = source.copy()
    filtered[replace] = median[replace]
    return filtered, int(replace.sum())
