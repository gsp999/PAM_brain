"""PCA line elements and iteratively reassociated signed point-to-line LM."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial import cKDTree
from .legacy import registration_view, _solve
from ..poses import pose_to_matrix, matrix_to_pose


@dataclass
class Lines:
    points: np.ndarray
    normals: np.ndarray


def line_elements(points, maximum=2200):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 12:
        return Lines(np.empty((0, 2)), np.empty((0, 2)))
    # Voxel downsampling prevents repeated map observations dominating PCA.
    _, keep = np.unique(np.floor(points / 1.5), axis=0, return_index=True)
    points = points[np.sort(keep)]
    sampled = points[np.linspace(0, len(points)-1, min(maximum, len(points))).astype(int)]
    distance, ids = cKDTree(points).query(sampled, k=min(9, len(points)))
    neighborhoods = points[ids]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / centered.shape[1]
    eig, vectors = np.linalg.eigh(covariance)
    valid = (eig[:, 1] > 1) & (eig[:, 0] < .20 * eig[:, 1]) & (distance[:, -1] < 10)
    return Lines(sampled[valid], vectors[valid, :, 0])


def extract_lines(image):
    y, x = np.nonzero(cv2.Canny(registration_view(image), 35, 90))
    return line_elements(np.column_stack((x, y)))


def transform_lines(lines, matrix):
    return Lines(lines.points @ matrix[:2, :2].T + matrix[:2, 2],
                 lines.normals @ matrix[:2, :2].T)


def associate(reference, moving, matrix, radius=6):
    if len(reference.points) < 12 or len(moving.points) < 12:
        return None
    transformed = transform_lines(moving, matrix)
    distance, ids = cKDTree(reference.points).query(transformed.points)
    normals = reference.normals[ids]
    valid = (distance < radius) & (np.abs(np.sum(normals * transformed.normals, axis=1)) > .8)
    return moving.points[valid], reference.points[ids[valid]], normals[valid]


def refine_lines(reference, moving, initial, anchors=None, iterations=3):
    """No absolute-angle prior. Reject rank-deficient geometry, retain SIFT init."""
    matrix = np.asarray(initial)[:2].copy()
    info = {"accepted": False, "reason": "insufficient_lines", "line_count": 0}
    for _ in range(iterations):
        paired = associate(reference, moving, matrix)
        if paired is None or len(paired[0]) < 60:
            return np.asarray(initial)[:2].copy(), info
        src, dst, normal = paired
        start = matrix_to_pose(matrix) * [1, 1, 100]

        def residual(p):
            trial = pose_to_matrix(p / [1, 1, 100])
            xy = src @ trial[:2, :2].T + trial[:2, 2]
            line = np.sum((xy - dst) * normal, axis=1)
            if anchors is None:
                return line
            a, b = anchors
            point = (a @ trial[:2, :2].T + trial[:2, 2] - b).ravel()
            # Normalize counts, so dense edges cannot erase intensity evidence.
            return np.r_[line / np.sqrt(len(line)) * 10,
                         point / np.sqrt(len(a)) * 10]

        # Geometry-only normalized Hessian detects unobservable translation.
        rotated = src @ matrix[:, :2].T
        angular = np.sum(np.column_stack((-rotated[:, 1], rotated[:, 0])) * normal, axis=1) / 100
        jac = np.column_stack((normal, angular))
        eig = np.linalg.eigvalsh(jac.T @ jac / len(jac))
        info.update(line_count=len(src), eigenvalues=eig.tolist())
        if eig[0] < 1e-3 or eig[-1] / max(eig[0], 1e-12) > 1e4:
            info['reason'] = 'degenerate_lines'
            return np.asarray(initial)[:2].copy(), info
        fitted = _solve(residual, start, 60)
        matrix = pose_to_matrix(fitted / [1, 1, 100])[:2]
    info.update(accepted=True, reason='observable_lines',
                median_line_error=float(np.median(np.abs(np.sum(
                    (src @ matrix[:, :2].T + matrix[:, 2] - dst) * normal, axis=1)))))
    return matrix, info
