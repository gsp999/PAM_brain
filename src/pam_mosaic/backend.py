"""Conservative 2-D backend measurements built from independently matched views.

Unlike a prior-based pseudo-measurement, every new edge connects two observed
frames. Candidate poses are used only to retrieve spatial neighbors and gate
implausible matches; the measurement itself comes from the image pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares

from .poses import Edge, pose_to_matrix
from .registration import register, registration_view, sift_inliers


@dataclass(frozen=True)
class BackendConfig:
    keyframe_motion_px: float = 70.0
    keyframe_interval: int = 12
    search_radius_px: float = 430.0
    minimum_frame_gap: int = 8
    max_neighbors: int = 5
    minimum_inliers: int = 25
    minimum_spread_px: float = 55.0
    maximum_innovation_px: float = 18.0
    maximum_innovation_deg: float = 2.0


def select_keyframes(poses: np.ndarray, config: BackendConfig) -> list[int]:
    if not len(poses):
        return []
    selected = [0]
    for index in range(1, len(poses)):
        previous = selected[-1]
        if (index - previous >= config.keyframe_interval or
                np.linalg.norm(poses[index, :2] - poses[previous, :2]) >= config.keyframe_motion_px):
            selected.append(index)
    return selected


def _spread_ok(points: np.ndarray, threshold: float) -> bool:
    if len(points) < 3:
        return False
    eigen = np.linalg.eigvalsh(np.cov(points.T))
    return bool(np.sqrt(max(0.0, eigen[0])) >= threshold)


def _innovation(reference: np.ndarray, moving: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    predicted = np.linalg.inv(pose_to_matrix(reference)) @ pose_to_matrix(moving)
    delta = np.linalg.inv(np.vstack((matrix, [0, 0, 1]))) @ predicted
    return float(np.linalg.norm(delta[:2, 2])), abs(float(np.rad2deg(np.arctan2(delta[1, 0], delta[0, 0]))))


def _edge_distance(reference: np.ndarray, moving: np.ndarray, matrix: np.ndarray) -> float:
    """Median vessel-edge distance over points that actually overlap."""
    ref = cv2.Canny(registration_view(reference), 35, 90) > 0
    mov = cv2.Canny(registration_view(moving), 35, 90) > 0
    ys, xs = np.nonzero(mov)
    if ref.sum() < 100 or len(xs) < 100:
        return float("inf")
    step = max(1, len(xs) // 600)
    points = np.column_stack((xs[::step], ys[::step]))[:600]
    mapped = points @ matrix[:, :2].T + matrix[:, 2]
    x, y = mapped[:, 0], mapped[:, 1]
    inside = (x >= 0) & (x < ref.shape[1] - 1) & (y >= 0) & (y < ref.shape[0] - 1)
    if inside.sum() < 100:
        return float("inf")
    distance = distance_transform_edt(~ref)
    values = map_coordinates(distance, [y[inside], x[inside]], order=1)
    return float(np.median(values))


def _local_map_geometry(frames: list[np.ndarray], poses: np.ndarray, current: int,
                        history: list[int]) -> np.ndarray | None:
    """Scan-to-local-map edge alignment; return an observed relative transform.

    The map is built from multiple historical keyframes. A near-singular
    2-D alignment or a solution that does not improve the edge distance is
    rejected before it can constrain the pose graph.
    """
    if len(history) < 2:
        return None
    side = 900
    origin = poses[current, :2] - 200
    map_edges = np.zeros((side, side), dtype=bool)
    for index in history[-4:]:
        ys, xs = np.nonzero(cv2.Canny(registration_view(frames[index]), 35, 90))
        if len(xs) < 100:
            continue
        step = max(1, len(xs) // 3000)
        points = np.column_stack((xs[::step], ys[::step]))[:3000]
        transform = pose_to_matrix(poses[index])[:2]
        xy = np.rint(points @ transform[:, :2].T + transform[:, 2] - origin).astype(int)
        valid = (xy[:, 0] >= 0) & (xy[:, 0] < side) & (xy[:, 1] >= 0) & (xy[:, 1] < side)
        map_edges[xy[valid, 1], xy[valid, 0]] = True
    if map_edges.sum() < 300:
        return None
    ys, xs = np.nonzero(cv2.Canny(registration_view(frames[current]), 35, 90))
    if len(xs) < 300:
        return None
    step = max(1, len(xs) // 350)
    moving = np.column_stack((xs[::step], ys[::step]))[:350]
    field = distance_transform_edt(~map_edges).astype(np.float32)
    initial_transform = pose_to_matrix(poses[current])[:2]
    initial_xy = moving @ initial_transform[:, :2].T + initial_transform[:, 2] - origin
    initial_distance = map_coordinates(field, [initial_xy[:, 1], initial_xy[:, 0]],
                                       order=1, mode="constant", cval=100)
    # A local map covers only part of the current field of view. Points with
    # no plausible map support must not dominate the alignment objective.
    moving = moving[initial_distance < 20]
    if len(moving) < 80 or not _spread_ok(moving, 35):
        return None

    def residual(delta: np.ndarray) -> np.ndarray:
        trial = poses[current].copy()
        trial[:2] += delta[:2]
        trial[2] += delta[2] / 100.0
        transform = pose_to_matrix(trial)[:2]
        xy = moving @ transform[:, :2].T + transform[:, 2] - origin
        x, y = xy[:, 0], xy[:, 1]
        values = map_coordinates(field, [y, x], order=1, mode="constant", cval=20)
        outside = (x < 0) | (x >= side) | (y < 0) | (y >= side)
        values[outside] = 20
        return np.minimum(values, 20)

    before = float(np.median(residual(np.zeros(3))))
    if before > 7:
        return None
    fit = least_squares(residual, np.zeros(3), loss="huber", f_scale=3.0,
                        max_nfev=25)
    after = float(np.median(residual(fit.x)))
    information = fit.jac.T @ fit.jac
    eigen = np.linalg.eigvalsh(information)
    if (not fit.success or after >= before - .25 or after > 4 or
            np.linalg.norm(fit.x[:2]) > 8 or abs(fit.x[2]) > 1.5 or
            eigen[0] < 1 or eigen[-1] / eigen[0] > 1e5):
        return None
    corrected = poses[current].copy()
    corrected[:2] += fit.x[:2]
    corrected[2] += fit.x[2] / 100.0
    relative = np.linalg.inv(pose_to_matrix(poses[history[-1]])) @ pose_to_matrix(corrected)
    return relative[:2]


def build_factors(frames: list[np.ndarray], poses: np.ndarray,
                  config: BackendConfig = BackendConfig()) -> tuple[list[Edge], dict]:
    """Find cross-time image evidence; reject degenerate or contradictory edges."""
    if len(frames) != len(poses):
        raise ValueError("Frame and pose counts differ")
    keyframes = select_keyframes(poses, config)
    edges: list[Edge] = []
    diagnostics = {"keyframes": len(keyframes), "attempted": 0,
                   "intensity_accepted": 0, "geometry_accepted": 0}
    for current in keyframes:
        candidates = [prior for prior in keyframes if current - prior >= config.minimum_frame_gap and
                      np.linalg.norm(poses[current, :2] - poses[prior, :2]) < config.search_radius_px]
        candidates.sort(key=lambda prior: np.linalg.norm(poses[current, :2] - poses[prior, :2]))
        history = sorted(candidates[:4])
        geometry_map = _local_map_geometry(frames, poses, current, history)
        if geometry_map is not None:
            edges.append(Edge(history[-1], current, geometry_map, .15, "geometry"))
            diagnostics["geometry_accepted"] += 1
        for prior in candidates[:config.max_neighbors]:
            diagnostics["attempted"] += 1
            matched = sift_inliers(frames[prior], frames[current], config.minimum_inliers)
            if matched is None:
                continue
            moving, reference, _ = matched
            if not (_spread_ok(moving, config.minimum_spread_px) and
                    _spread_ok(reference, config.minimum_spread_px)):
                continue
            # Register without geometry to obtain an independent intensity edge.
            intensity = register(frames[prior], frames[current],
                                 min_inliers=config.minimum_inliers, refine_geometry=False)
            if not intensity.valid or intensity.overlap < .3:
                continue
            shift, rotation = _innovation(poses[prior], poses[current], intensity.matrix)
            if shift > config.maximum_innovation_px or rotation > config.maximum_innovation_deg:
                continue
            edges.append(Edge(prior, current, intensity.matrix, .35, "intensity"))
            diagnostics["intensity_accepted"] += 1
            # The geometry correction must measurably improve edge alignment
            # and remain consistent with the independent SIFT measurement.
            geometry = register(frames[prior], frames[current],
                                min_inliers=config.minimum_inliers, refine_geometry=True)
            if not geometry.valid:
                continue
            before = _edge_distance(frames[prior], frames[current], intensity.matrix)
            after = _edge_distance(frames[prior], frames[current], geometry.matrix)
            disagreement = np.linalg.norm(geometry.matrix[:, 2] - intensity.matrix[:, 2])
            if (np.isfinite(after) and after + .15 < before and after < 4.0 and
                    disagreement < 5.0):
                edges.append(Edge(prior, current, geometry.matrix, .2, "geometry"))
                diagnostics["geometry_accepted"] += 1
    return edges, diagnostics
