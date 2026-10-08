"""Turn one forward/backward acquisition into one validated MAP image."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .reconstruction import (Calibration, calibrate_even_trajectory,
                             choose_even_direction, reconstruct)
from ..frontend.registration import register, registration_view
from .denoise import bright_point_mask, remove_isolated_bright_points


@dataclass(frozen=True)
class PairDecision:
    accepted: bool
    inliers: int
    median_error_px: float | None
    residual_shift_px: float | None
    residual_angle_deg: float | None
    selected: str = "forward"
    changed_pixels: int = 0


def calibrate(paths: list[Path]) -> Calibration:
    """Estimate scan correction once; never use the result as per-file evidence."""
    sample = paths[:5]
    reverse = choose_even_direction(sample)
    shift = calibrate_even_trajectory(sample, reverse)
    basic = Calibration(reverse_even_x=reverse, even_time_shift_samples=shift)
    candidates = []
    for path in sample:
        forward, backward = reconstruct(path, basic)
        match = register(forward, backward)
        if match.valid:
            angle = np.degrees(np.arctan2(match.matrix[1, 0], match.matrix[0, 0]))
            if np.linalg.norm(match.matrix[:, 2]) <= 20 and abs(angle) <= 5:
                candidates.append(match.matrix)
    if not candidates:
        raise ValueError("Cannot calibrate backward scan from the first five files")
    fixed = np.median(np.stack(candidates), axis=0)
    angle = np.arctan2(fixed[1, 0], fixed[0, 0])
    fixed[:, :2] = [[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]]
    return Calibration(reverse_even_x=reverse, backward_matrix=fixed,
                       even_time_shift_samples=shift)


def _quality_score(image: np.ndarray) -> float:
    """Rank two already despiked views by supported vessel structure."""
    view = registration_view(image)
    valid = image > 0
    if valid.sum() < 100:
        return 0.0
    sift = cv2.SIFT_create(nfeatures=600)
    keypoints = len(sift.detect(view, None))
    gx = cv2.Sobel(view, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(view, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    contrast = float(np.percentile(image[valid], 99) - np.median(image[valid]))
    structure = float(np.percentile(gradient[valid], 90))
    return contrast * structure * np.log1p(keypoints)


def fuse_if_aligned(forward: np.ndarray, backward: np.ndarray, *,
                    suppress_bright_points: bool = False) -> tuple[np.ndarray, PairDecision]:
    """Use backward only if its residual rigid alignment is well supported."""
    changed = 0
    if suppress_bright_points:
        # Mild pre-filtering prevents isolated impulses becoming keypoints,
        # while retaining nearly all vessel structure for round-trip alignment.
        mild = dict(excess=.02, max_area=4, max_extent=4, median_size=3)
        registration_forward, count_f = remove_isolated_bright_points(forward, **mild)
        registration_backward, count_b = remove_isolated_bright_points(backward, **mild)
        changed += count_f + count_b
    else:
        registration_forward, registration_backward = forward, backward
    match = register(registration_forward, registration_backward)
    if match.valid:
        shift = float(np.linalg.norm(match.matrix[:, 2]))
        angle = float(np.degrees(np.arctan2(match.matrix[1, 0], match.matrix[0, 0])))
        accepted = (match.inliers >= 20 and match.median_error <= 2 and
                    shift <= 5 and abs(angle) <= 1 and match.overlap >= .9)
    else:
        shift = angle = None
        accepted = False
    if not accepted:
        if suppress_bright_points and _quality_score(registration_backward) > _quality_score(registration_forward):
            selected, image = "backward", registration_backward
        else:
            selected, image = "forward", registration_forward
        if suppress_bright_points:
            image, count_final = remove_isolated_bright_points(image)
            changed += count_final
        decision = PairDecision(False, match.inliers,
                                float(match.median_error) if np.isfinite(match.median_error) else None,
                                shift, angle, selected, changed)
        return image.astype(np.float32), decision
    h, w = forward.shape
    matrix = match.matrix.astype(np.float32)
    aligned = cv2.warpAffine(backward, matrix, (w, h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine((backward > 0).astype(np.uint8), matrix, (w, h),
                           flags=cv2.INTER_NEAREST) > 0
    if suppress_bright_points:
        forward_spots = bright_point_mask(forward, **mild)
        backward_spots = cv2.warpAffine(
            bright_point_mask(backward, **mild).astype(np.uint8), matrix, (w, h),
            flags=cv2.INTER_NEAREST) > 0
        neighborhood = np.ones((3, 3), np.uint8)
        persistent = forward_spots & (cv2.dilate(backward_spots.astype(np.uint8), neighborhood) > 0)
        forward_clean, count_fused_f = remove_isolated_bright_points(
            forward, preserve=persistent, **mild)
        aligned_clean, count_fused_b = remove_isolated_bright_points(
            aligned, preserve=cv2.dilate(persistent.astype(np.uint8), neighborhood) > 0,
            **mild)
        changed += count_fused_f + count_fused_b
    else:
        forward_clean, aligned_clean = forward, aligned
        persistent = np.zeros_like(forward, dtype=bool)
    fused = forward_clean.copy()
    fused[valid] = (forward_clean[valid] + aligned_clean[valid]) / 2
    if suppress_bright_points:
        fused, count_final = remove_isolated_bright_points(fused, preserve=persistent)
        changed += count_final
    decision = PairDecision(True, match.inliers,
                            float(match.median_error) if np.isfinite(match.median_error) else None,
                            shift, angle, "fused", changed)
    return fused.astype(np.float32), decision
