"""Preserved legacy registration and shared SIFT/LM primitives."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Match:
    matrix: np.ndarray  # moving pixel coordinates -> reference pixel coordinates
    inliers: int
    median_error: float
    overlap: float
    valid: bool


@dataclass(frozen=True)
class MotionModelDecision:
    match: Match
    model: str
    rigid_angle_deg: float | None
    inlier_spread_px: float | None
    inlier_hull_fraction: float | None
    rigid_point_error_px: float | None
    translation_point_error_px: float | None
    rigid_correlation: float | None
    translation_correlation: float | None
    closure_shift_px: float | None
    closure_angle_deg: float | None
    rotation_supported: bool


def registration_view(image: np.ndarray, lo: float = 0.02, hi: float = 0.25) -> np.ndarray:
    clipped = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return np.uint8(np.rint(255 * clipped))


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


def register_translation_fallback(reference: np.ndarray, moving: np.ndarray) -> Match:
    """Strict low-texture fallback for *adjacent* frames only.

    Phase correlation supplies a translation hypothesis. A high phase peak,
    substantial actual image correlation after warping, and a small step are
    all required; this must not be used as a generic loop-closure matcher.
    """
    ref = registration_view(reference).astype(np.float32)
    mov = registration_view(moving).astype(np.float32)
    identity = np.eye(2, 3, dtype=np.float64)
    invalid = Match(identity, 0, float("inf"), 0.0, False)
    (dx, dy), response = cv2.phaseCorrelate(ref, mov)
    if not np.isfinite([dx, dy, response]).all() or response < .5 or np.hypot(dx, dy) > 30:
        return invalid
    matrix = np.array([[1., 0., -dx], [0., 1., -dy]], dtype=np.float64)
    warped = cv2.warpAffine(mov, matrix, (ref.shape[1], ref.shape[0]))
    mask = cv2.warpAffine(np.ones_like(mov, np.uint8), matrix,
                          (ref.shape[1], ref.shape[0]), flags=cv2.INTER_NEAREST) > 0
    if mask.mean() < .7:
        return invalid
    a, b = ref[mask], warped[mask]
    if a.std() < 3 or b.std() < 3:
        return invalid
    correlation = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(correlation) or correlation < .55:
        return invalid
    return Match(matrix, 0, float(1 - correlation), float(mask.mean()), True)


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


def select_supported_motion(reference: np.ndarray, moving: np.ndarray) -> MotionModelDecision:
    """Use rotation only when registration evidence favors it over translation."""
    invalid = MotionModelDecision(
        Match(np.eye(2, 3), 0, float("inf"), 0.0, False), "invalid",
        None, None, None, None, None, None, None, None, None, False)
    correspondence = sift_inliers(reference, moving, 12)
    if correspondence is None:
        return invalid
    source, target, _ = correspondence
    rigid = register(reference, moving)
    if not rigid.valid:
        return invalid

    translation_xy = np.median(target - source, axis=0)
    translation_matrix = np.array(
        [[1.0, 0.0, translation_xy[0]], [0.0, 1.0, translation_xy[1]]],
        dtype=np.float64)
    rigid_error = float(np.median(np.linalg.norm(
        _transform(source, rigid.matrix) - target, axis=1)))
    translation_error = float(np.median(np.linalg.norm(
        _transform(source, translation_matrix) - target, axis=1)))
    eigenvalues = np.linalg.eigvalsh(np.cov(source.T)) if len(source) >= 3 else np.zeros(2)
    spread = float(np.sqrt(max(0.0, eigenvalues[0])))
    hull = cv2.convexHull(source.astype(np.float32)) if len(source) >= 3 else None
    hull_fraction = (float(cv2.contourArea(hull)) / float(reference.size)
                     if hull is not None else 0.0)
    rigid_overlap, rigid_correlation = _warped_correlation(
        reference, moving, rigid.matrix)
    translation_overlap, translation_correlation = _warped_correlation(
        reference, moving, translation_matrix)
    reverse = register(moving, reference, refine_geometry=False)
    closure_shift = closure_angle = None
    consistent = False
    if reverse.valid:
        closure = (np.vstack((rigid.matrix, [0.0, 0.0, 1.0])) @
                   np.vstack((reverse.matrix, [0.0, 0.0, 1.0])))
        closure_shift = float(np.hypot(closure[0, 2], closure[1, 2]))
        closure_angle = float(np.degrees(np.arctan2(closure[1, 0], closure[0, 0])))
        consistent = closure_shift <= 3.0 and abs(closure_angle) <= .5
    observable = spread >= 35.0 and hull_fraction >= .03
    fit_improved = (translation_error - rigid_error >= .2 and
                    rigid_error <= .85 * max(translation_error, 1e-6))
    correlation_supported = (np.isfinite(rigid_correlation) and
                             np.isfinite(translation_correlation) and
                             rigid_correlation >= translation_correlation - .01)
    rotation_supported = bool(observable and fit_improved and consistent and
                              correlation_supported)
    translation_valid = bool(
        translation_error <= 5.0 and translation_overlap >= .3 and
        np.isfinite(translation_correlation) and translation_correlation >= .55 and
        np.isfinite(rigid_correlation) and
        translation_correlation >= rigid_correlation - .005 and
        translation_error <= rigid_error + .1)
    if rotation_supported:
        selected, model = rigid, "rigid"
    elif translation_valid:
        selected = Match(translation_matrix, len(source), translation_error,
                         translation_overlap, True)
        model = "translation"
    else:
        # The registered rigid result still carries image evidence. Keep it as
        # an explicitly low-confidence constraint instead of inventing a
        # translation or breaking the graph solely because rotation is weakly
        # observable.
        selected, model = rigid, "rigid_unverified"
    angle = float(np.degrees(np.arctan2(rigid.matrix[1, 0], rigid.matrix[0, 0])))
    return MotionModelDecision(
        selected, model, angle, spread, hull_fraction, rigid_error,
        translation_error,
        float(rigid_correlation) if np.isfinite(rigid_correlation) else None,
        float(translation_correlation) if np.isfinite(translation_correlation) else None,
        closure_shift, closure_angle, rotation_supported)


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
