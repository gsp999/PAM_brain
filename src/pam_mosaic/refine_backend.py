"""Opt-in backend experiment with a measured no-regression fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backend import build_factors
from .dataset import index_dataset
from .fusion import fuse
from .poses import Edge, optimize_poses, pose_to_matrix
from .reconstruction import Calibration, reconstruct
from .registration import register


def _edge_error(poses: np.ndarray, edges: list[Edge]) -> float:
    if not edges:
        return float("inf")
    errors = []
    for edge in edges:
        estimate = np.linalg.inv(pose_to_matrix(poses[edge.reference])) @ pose_to_matrix(poses[edge.moving])
        delta = np.linalg.inv(np.vstack((edge.matrix, [0, 0, 1]))) @ estimate
        errors.append(np.hypot(np.linalg.norm(delta[:2, 2]),
                               100 * np.arctan2(delta[1, 0], delta[0, 0])))
    return float(np.median(errors))


def refine(source: Path, data: Path, output: Path,
           component_limit: int | None = None) -> dict:
    baseline = json.loads((source / "report.json").read_text())
    files = {record.index: record.path for record in index_dataset(data)}
    matrix = baseline.get("backward_rigid_matrix")
    calibration = Calibration(reverse_even_x=baseline["reverse_even_x"],
                              backward_matrix=np.asarray(matrix) if matrix is not None else None)
    valid_pairs = {(entry["reference"], entry["moving"])
                   for entry in baseline["registrations"] if entry["valid"]}
    selected = baseline["components"][:component_limit] if component_limit else baseline["components"]
    components = []
    for number, component in enumerate(selected, 1):
        indices = component["file_indices"]
        if len(indices) < 20:
            components.append({**component, "backend": {"decision": "baseline", "reason": "small component"}})
            continue
        frames = []
        for index in indices:
            forward, backward = reconstruct(files[index], calibration)
            if calibration.backward_matrix is not None:
                forward = np.where(backward != 0, (forward + backward) / 2, forward)
            frames.append(forward.astype(np.float32))
        poses = np.load(Path(component["path"]) / "poses_xytheta.npy")
        index_of = {file_index: position for position, file_index in enumerate(indices)}
        frontend = []
        for first, second in valid_pairs:
            if first not in index_of or second not in index_of:
                continue
            i, j = index_of[first], index_of[second]
            measured = register(frames[i], frames[j])
            if measured.valid:
                frontend.append(Edge(i, j, measured.matrix,
                                     min(2.0, measured.inliers / 30), "frontend"))
        factors, diagnostics = build_factors(frames, poses)
        pair_ids = sorted({(edge.reference, edge.moving) for edge in factors})
        heldout = {pair for offset, pair in enumerate(pair_ids) if offset % 3 == 0}
        train = [edge for edge in factors if (edge.reference, edge.moving) not in heldout]
        test = [edge for edge in factors if edge.kind == "intensity" and
                (edge.reference, edge.moving) in heldout]
        accepted = False
        if len(train) >= 2 and len(test) >= 2 and frontend:
            refined = optimize_poses(poses, frontend + train)
            old_test, new_test = _edge_error(poses, test), _edge_error(refined, test)
            old_front, new_front = _edge_error(poses, frontend), _edge_error(refined, frontend)
            maximum_move = float(np.linalg.norm(refined[:, :2] - poses[:, :2], axis=1).max())
            accepted = (new_test < old_test - .05 and
                        new_front <= old_front + .25 and maximum_move <= 18.0)
        else:
            refined = poses
            old_test = new_test = old_front = new_front = float("nan")
            maximum_move = 0.0
        diagnostics.update({"decision": "refined" if accepted else "baseline",
                            "training_factors": len(train), "heldout_factors": len(test),
                            "heldout_before": old_test if np.isfinite(old_test) else None,
                            "heldout_after": new_test if np.isfinite(new_test) else None,
                            "frontend_before": old_front if np.isfinite(old_front) else None,
                            "frontend_after": new_front if np.isfinite(new_front) else None,
                            "maximum_move_px": maximum_move})
        if accepted:
            destination = output / f"component_{number:03d}"
            size = fuse(frames, refined, destination)
            np.save(destination / "poses_xytheta.npy", refined)
            components.append({**component, "path": str(destination), "size_xy": size,
                               "backend": diagnostics})
        else:
            components.append({**component, "backend": diagnostics})
        print(f"backend {number}/{len(selected)}: {diagnostics['decision']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    report = {**baseline, "backend_method": "conservative cross-time 2D factors",
              "baseline_source": str(source), "components": components}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative 2D backend refinement")
    parser.add_argument("--source", type=Path, default=Path("outputs/lm_full"))
    parser.add_argument("--data", type=Path, default=Path("PAM_Skull_FYH_20260929"))
    parser.add_argument("--output", type=Path, default=Path("outputs/backend_candidate"))
    parser.add_argument("--component-limit", type=int, default=None)
    args = parser.parse_args()
    result = refine(args.source, args.data, args.output, args.component_limit)
    print(f"Refined {sum(c['backend']['decision'] == 'refined' for c in result['components'])} components")


if __name__ == "__main__":
    main()
