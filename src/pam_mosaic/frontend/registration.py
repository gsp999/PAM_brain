"""Stable SIFT and edge-distance registration with conservative recovery."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from .view import registration_view


@dataclass(frozen=True)
class Match:
    matrix: np.ndarray  # moving pixel coordinates -> reference pixel coordinates
    inliers: int
    median_error: float
    overlap: float
    valid: bool


def _matrix_from_params(p: np.ndarray) -> np.ndarray:
    angle, tx, ty = p
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty]], dtype=np.float64)


def _params_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.array([np.arctan2(matrix[1, 0], matrix[0, 0]), matrix[0, 2], matrix[1, 2]])


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _huber_residual(residual: np.ndarray, scale: float) -> np.ndarray:
    """Transform residuals so LM minimizes the exact Huber objective."""
    magnitude = np.abs(residual)
    return np.sign(residual) * np.where(
        magnitude <= scale, magnitude,
        np.sqrt(np.maximum(2 * scale * magnitude - scale * scale, 0)),
    )


def _solve(residual, start: np.ndarray, max_nfev: int) -> np.ndarray:
    # SciPy's LM accepts only linear loss. This transform preserves the
    # Huber objective: 0.5*sum(transformed_residual**2).
    fit = least_squares(lambda p: _huber_residual(residual(p), 2.0), start,
                        method="lm", max_nfev=max_nfev)
    return fit.x


def _overlap(matrix: np.ndarray, reference_shape: tuple[int, int],
             moving_shape: tuple[int, int]) -> float:
    h, w = reference_shape
    mask = np.ones(moving_shape, np.uint8)
    warped = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return float(np.count_nonzero(warped)) / float(min(h * w, mask.size))


def sift_inliers(reference: np.ndarray, moving: np.ndarray,
                 min_inliers: int = 12, *,
                 reference_features=None, moving_features=None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return moving/reference SIFT points and their RANSAC rigid initial pose."""
    if reference_features is None or moving_features is None:
        ref8, mov8 = registration_view(reference), registration_view(moving)
        sift = cv2.SIFT_create(nfeatures=1200)
        key_ref, desc_ref = sift.detectAndCompute(ref8, None)
        key_mov, desc_mov = sift.detectAndCompute(mov8, None)
    else:
        key_ref, desc_ref = reference_features
        key_mov, desc_mov = moving_features
    if desc_ref is None or desc_mov is None or len(desc_ref) < 2 or len(desc_mov) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_mov, desc_ref, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([key_mov[m.queryIdx].pt for m in good])
    dst = np.float32([key_ref[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=3000, confidence=0.995,
    )
    if matrix is None or mask is None:
        return None
    inlier = mask.ravel().astype(bool)
    if inlier.sum() < min_inliers:
        return None
    angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    rigid = _matrix_from_params(np.array([angle, matrix[0, 2], matrix[1, 2]]))
    return src[inlier], dst[inlier], rigid


def register(reference: np.ndarray, moving: np.ndarray, *, min_inliers: int = 12,
             max_points: int = 350, refine_geometry: bool = True,
             reference_features=None, moving_features=None) -> Match:
    """Estimate an SE(2) transform; reject weak or nonoverlapping matches."""
    ref8, mov8 = registration_view(reference), registration_view(moving)
    identity = np.eye(2, 3, dtype=np.float64)
    invalid = Match(identity, 0, float("inf"), 0.0, False)
    correspondence = sift_inliers(reference, moving, min_inliers,
                                 reference_features=reference_features,
                                 moving_features=moving_features)
    if correspondence is None:
        return invalid
    src, dst, matrix = correspondence
    point_residual = lambda p: (_transform(src, _matrix_from_params(p)) - dst).ravel()
    matrix = _matrix_from_params(_solve(point_residual, _params_from_matrix(matrix), 70))
    if refine_geometry:
        # A 2D analogue of point-to-line: moving vessel-edge points seek the
        # reference edge distance field. SIFT point residuals retain identity.
        edge_ref = cv2.Canny(ref8, 35, 90) > 0
        edge_mov = cv2.Canny(mov8, 35, 90) > 0
        ys, xs = np.nonzero(edge_mov)
        if edge_ref.sum() > 100 and len(xs) > 100:
            step = max(1, len(xs) // max_points)
            edge_points = np.column_stack((xs[::step], ys[::step]))[:max_points]
            distance = distance_transform_edt(~edge_ref).astype(np.float32)
            h, w = distance.shape

            def residual(p: np.ndarray) -> np.ndarray:
                transformed = _transform(edge_points, _matrix_from_params(p))
                x, y = transformed[:, 0], transformed[:, 1]
                values = map_coordinates(distance, [y, x], order=1, mode="constant", cval=20)
                outside = (x < 0) | (x >= w) | (y < 0) | (y >= h)
                values[outside] = 20
                points = (_transform(src, _matrix_from_params(p)) - dst).ravel()
                return np.concatenate((points, 0.3 * np.minimum(values, 20)))

            matrix = _matrix_from_params(_solve(residual, _params_from_matrix(matrix), 35))
    errors = np.linalg.norm(_transform(src, matrix) - dst, axis=1)
    median = float(np.median(errors))
    overlap = _overlap(matrix, reference.shape, moving.shape)
    valid = bool(median <= 5.0 and overlap >= 0.15)
    return Match(matrix, len(src), median, overlap, valid)


def _warped_correlation(reference: np.ndarray, moving: np.ndarray,
                        matrix: np.ndarray) -> tuple[float, float]:
    ref = registration_view(reference).astype(np.float32)
    mov = registration_view(moving).astype(np.float32)
    h, w = ref.shape
    warped = cv2.warpAffine(mov, matrix.astype(np.float32), (w, h))
    mask = cv2.warpAffine(np.ones_like(mov, np.uint8), matrix.astype(np.float32),
                          (w, h), flags=cv2.INTER_NEAREST) > 0
    overlap = float(mask.mean())
    if overlap < .3 or ref[mask].std() < 3 or warped[mask].std() < 3:
        return overlap, float("nan")
    return overlap, float(np.corrcoef(ref[mask], warped[mask])[0, 1])


def register_enhanced_adjacent(reference: np.ndarray, moving: np.ndarray) -> Match:
    """Recover a missed adjacent match using contrast-enhanced SIFT evidence."""
    invalid = Match(np.eye(2, 3), 0, float("inf"), 0.0, False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=1800)
    ref_features = sift.detectAndCompute(clahe.apply(registration_view(reference)), None)
    mov_features = sift.detectAndCompute(clahe.apply(registration_view(moving)), None)
    matched = sift_inliers(reference, moving, 15,
                           reference_features=ref_features, moving_features=mov_features)
    if matched is None:
        return invalid
    src, dst, _ = matched
    if len(src) < 20 or len(dst) < 20:
        return invalid
    spread = min(np.sqrt(max(0., np.linalg.eigvalsh(np.cov(points.T))[0]))
                 for points in (src, dst))
    if spread < 40:
        return invalid
    match = register(reference, moving, min_inliers=15, refine_geometry=False,
                     reference_features=ref_features, moving_features=mov_features)
    if not match.valid:
        return invalid
    overlap, correlation = _warped_correlation(reference, moving, match.matrix)
    angle = abs(float(np.degrees(np.arctan2(match.matrix[1, 0], match.matrix[0, 0]))))
    valid = bool(match.inliers >= 20 and match.median_error <= 2.5 and
                 overlap >= .4 and correlation >= .55 and angle <= 5)
    return Match(match.matrix, match.inliers, match.median_error, overlap, valid)


def register_weak_signal_adjacent(reference: np.ndarray, moving: np.ndarray) -> Match:
    """Translation-only fallback; never infer a pose from featureless frames."""
    invalid = Match(np.eye(2, 3), 0, float("inf"), 0.0, False)
    ref = registration_view(reference).astype(np.float32)
    mov = registration_view(moving).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(ref, mov)
    if not np.isfinite([dx, dy, response]).all() or response < .2 or np.hypot(dx, dy) > 50:
        return invalid
    matrix = np.array([[1., 0., -dx], [0., 1., -dy]], dtype=np.float64)
    overlap, correlation = _warped_correlation(reference, moving, matrix)
    if overlap < .8 or not np.isfinite(correlation) or correlation < .7:
        return invalid
    return Match(matrix, 0, float(1 - correlation), overlap, True)
