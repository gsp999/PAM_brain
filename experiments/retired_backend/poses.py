"""Global SE(2) pose graph with robust loop constraints."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class Edge:
    reference: int
    moving: int
    matrix: np.ndarray  # moving -> reference
    weight: float = 1.0
    kind: str = "frontend"
    sqrt_information: np.ndarray | None = None  # acts on [dx, dy, 100*dtheta]


def matrix_to_pose(m: np.ndarray) -> np.ndarray:
    return np.array([m[0, 2], m[1, 2], np.arctan2(m[1, 0], m[0, 0])])


def pose_to_matrix(p: np.ndarray) -> np.ndarray:
    x, y, a = p
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]], dtype=np.float64)


def initial_poses(count: int, edges: list[Edge]) -> np.ndarray:
    poses = np.zeros((count, 3), dtype=float)
    known = {0}
    while len(known) < count:
        changed = False
        for edge in edges:
            z = np.vstack((edge.matrix, [0, 0, 1]))
            if edge.reference in known and edge.moving not in known:
                poses[edge.moving] = matrix_to_pose(pose_to_matrix(poses[edge.reference]) @ z)
                known.add(edge.moving)
                changed = True
            elif edge.moving in known and edge.reference not in known:
                poses[edge.reference] = matrix_to_pose(pose_to_matrix(poses[edge.moving]) @ np.linalg.inv(z))
                known.add(edge.reference)
                changed = True
        if not changed:
            raise ValueError("Pose graph contains disconnected nodes")
    return poses


def optimize_poses(start: np.ndarray, edges: list[Edge]) -> np.ndarray:
    """Optimize relative SE(2) measurements; anchor node zero."""
    n = len(start)
    if n < 2 or not edges:
        return start.copy()
    return _optimize_lm(start, edges)


def _huber_cost(residual: np.ndarray, scale: float = 3.0) -> float:
    magnitude = np.abs(residual)
    return float(np.where(magnitude <= scale, 0.5 * residual**2,
                          scale * magnitude - 0.5 * scale**2).sum())


def _lm_residual_jacobian(poses: np.ndarray, edges: list[Edge]):
    """Exact SE(2) relative-pose residual and sparse analytic Jacobian."""
    rows, cols, values = [], [], []
    residual = np.empty(3 * len(edges), dtype=float)
    for k, edge in enumerate(edges):
        i, j = edge.reference, edge.moving
        xi, yi, ai = poses[i]
        xj, yj, aj = poses[j]
        zi = matrix_to_pose(edge.matrix)
        cz, sz = np.cos(zi[2]), np.sin(zi[2])
        ci, si = np.cos(ai), np.sin(ai)
        ri_t = np.array([[ci, si], [-si, ci]])
        dri_t = np.array([[-si, ci], [-ci, -si]])
        rz_t = np.array([[cz, sz], [-sz, cz]])
        delta = np.array([xj - xi, yj - yi])
        q = ri_t @ delta
        scale = edge.weight
        residual[3*k:3*k+2] = scale * (rz_t @ (q - zi[:2]))
        angle_error = (aj - ai - zi[2] + np.pi) % (2 * np.pi) - np.pi
        residual[3*k+2] = scale * 100 * angle_error
        translation_j = scale * rz_t @ ri_t
        angle_i = scale * rz_t @ (dri_t @ delta)
        ji = np.zeros((3, 3))
        jj = np.zeros((3, 3))
        ji[:2, :2] = -translation_j
        ji[:2, 2] = angle_i
        ji[2, 2] = -100 * scale
        jj[:2, :2] = translation_j
        jj[2, 2] = 100 * scale
        if edge.sqrt_information is not None:
            residual[3*k:3*k+3] = edge.sqrt_information @ residual[3*k:3*k+3]
            ji = edge.sqrt_information @ ji
            jj = edge.sqrt_information @ jj
        for node, block in ((i, ji), (j, jj)):
            if node == 0:
                continue
            for r in range(3):
                for c in range(3):
                    if block[r, c] != 0:
                        rows.append(3*k+r)
                        cols.append(3*(node-1)+c)
                        values.append(block[r, c])
    jacobian = coo_matrix((values, (rows, cols)),
                          shape=(3*len(edges), 3*(len(poses)-1))).tocsr()
    return residual, jacobian


def _optimize_lm(start: np.ndarray, edges: list[Edge], max_iterations: int = 40) -> np.ndarray:
    poses = start.copy()
    damping = 1e-2
    residual, jacobian = _lm_residual_jacobian(poses, edges)
    cost = _huber_cost(residual)
    for _ in range(max_iterations):
        magnitude = np.abs(residual)
        robust_weight = np.minimum(1.0, 3.0 / np.maximum(magnitude, 1e-12))
        weighted_j = jacobian.multiply(np.sqrt(robust_weight)[:, None])
        weighted_r = np.sqrt(robust_weight) * residual
        hessian = (weighted_j.T @ weighted_j).tocsr()
        gradient = weighted_j.T @ weighted_r
        diagonal = np.maximum(hessian.diagonal(), 1.0)
        accepted = False
        for _ in range(12):
            step = spsolve(hessian + damping * diags(diagonal), -gradient)
            if not np.isfinite(step).all():
                damping *= 10
                continue
            candidate = poses.copy()
            candidate[1:] += step.reshape(-1, 3)
            next_residual, next_jacobian = _lm_residual_jacobian(candidate, edges)
            next_cost = _huber_cost(next_residual)
            if next_cost < cost:
                improvement = cost - next_cost
                poses, residual, jacobian, cost = candidate, next_residual, next_jacobian, next_cost
                damping = max(damping / 2, 1e-9)
                accepted = True
                break
            damping *= 4
        if not accepted or improvement < 1e-7 * max(1.0, cost) or np.linalg.norm(step) < 1e-5:
            break
    return poses
