"""SIFT point-to-point initialization followed by explicit vessel line alignment."""
from dataclasses import dataclass
import cv2
import numpy as np
from .legacy import (Match, registration_view, sift_inliers, _solve,
                            _matrix_from_params, _params_from_matrix, _overlap)
from .geometry import Lines, line_elements, refine_lines


@dataclass
class Features:
    sift: tuple
    lines: Lines
    map_points: np.ndarray


def prepare(image):
    view = registration_view(image)
    y, x = np.nonzero(cv2.Canny(view, 35, 90))
    dense = np.column_stack((x, y)).astype(float)
    return Features(cv2.SIFT_create(nfeatures=1200).detectAndCompute(view, None),
                    line_elements(dense), dense)


def register_pair(reference, moving, reference_features, moving_features):
    paired = sift_inliers(reference, moving, reference_features=reference_features.sift,
                          moving_features=moving_features.sift)
    if paired is None:
        return Match(np.eye(2, 3), 0, float('inf'), 0., False), {"reason": "no_sift"}
    src, dst, initial = paired
    def residual(p):
        matrix = _matrix_from_params(p)
        return (src @ matrix[:, :2].T + matrix[:, 2] - dst).ravel()
    initial = _matrix_from_params(_solve(residual, _params_from_matrix(initial), 70))
    matrix, info = refine_lines(reference_features.lines, moving_features.lines,
                                initial, anchors=(src, dst))
    before = float(np.median(np.linalg.norm(src @ initial[:, :2].T + initial[:, 2] - dst, axis=1)))
    error = float(np.median(np.linalg.norm(src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1)))
    if error > max(before + .5, before * 1.3):
        matrix, error = initial, before
        info.update(accepted=False, reason='line_refinement_conflicts_with_sift')
    overlap = _overlap(matrix, reference.shape, moving.shape)
    return Match(matrix, len(src), error, overlap, error <= 5 and overlap >= .15), info
