"""Run reconstruction, matching, pose graph, and mosaic fusion."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .dataset import Record
from .fusion import fuse
from .graph import candidate_pairs, connected_components
from .poses import Edge, initial_poses, optimize_poses
from .reconstruction import Calibration, choose_even_direction, reconstruct
from .registration import register


def run(records: list[Record], output: Path, *, reverse_even: bool | None = None,
        use_backward: bool = True, temporal_window: int = 5,
        appearance_neighbors: int = 6) -> dict:
    if not records:
        raise ValueError("No data files selected")
    output.mkdir(parents=True, exist_ok=True)
    if reverse_even is None:
        reverse_even = choose_even_direction([r.path for r in records[:min(5, len(records))]])
    calibration = Calibration(reverse_even_x=reverse_even)
    frames: list[np.ndarray] = []
    details: list[dict] = []
    # Calibrate from a few files, then decode the sequence once.
    calibration_pairs = [reconstruct(r.path, calibration) for r in records[:min(5, len(records))]]
    fixed_candidates = []
    if use_backward:
        for forward, backward in calibration_pairs:
            match = register(forward, backward)
            if match.valid:
                fixed_candidates.append(match.matrix)
        if fixed_candidates:
            fixed = np.median(np.stack(fixed_candidates), axis=0)
            angle = np.arctan2(fixed[1, 0], fixed[0, 0])
            fixed[:,:2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            calibration = Calibration(reverse_even_x=reverse_even, backward_matrix=fixed)
    del calibration_pairs
    for record in records:
        forward, backward = reconstruct(record.path, calibration)
        if use_backward and fixed_candidates:
            # The two directions measure the same area. Average after correction;
            # this also prevents duplicate poses for one acquisition position.
            valid_backward = backward != 0
            image = np.where(valid_backward, (forward + backward) / 2, forward)
            frames.append(image.astype(np.float32))
        else:
            frames.append(forward)

    edges: list[Edge] = []
    candidates = candidate_pairs(frames, temporal_window=temporal_window,
                                 appearance_neighbors=appearance_neighbors)
    for done, (reference, moving) in enumerate(candidates, 1):
        match = register(frames[reference], frames[moving])
        if match.valid:
            edges.append(Edge(reference, moving, match.matrix, min(2.0, match.inliers / 30)))
        details.append({"reference": records[reference].index, "moving": records[moving].index,
                        "inliers": match.inliers, "median_error": match.median_error if np.isfinite(match.median_error) else None,
                        "overlap": match.overlap, "valid": match.valid})
        if done % 50 == 0 or done == len(candidates):
            print(f"matched {done}/{len(candidates)} candidate pairs", flush=True)

    # Each reliably connected run is optimized and fused independently. Never
    # invent an alignment across interrupted or failed matches.
    components = []
    groups = connected_components(len(frames), [(e.reference, e.moving) for e in edges])
    for component_id, group in enumerate(groups, 1):
        local_index = {global_index: i for i, global_index in enumerate(group)}
        local_edges = [Edge(local_index[e.reference], local_index[e.moving], e.matrix, e.weight)
                       for e in edges if e.reference in local_index and e.moving in local_index]
        poses = initial_poses(len(group), local_edges)
        local_frames = [frames[i] for i in group]
        optimized = optimize_poses(poses, local_edges)
        component_dir = output / f"component_{component_id:03d}"
        size = fuse(local_frames, optimized, component_dir)
        np.save(component_dir / "poses_xytheta.npy", optimized)
        components.append({"first_file": records[group[0]].index, "last_file": records[group[-1]].index,
                           "file_indices": [records[i].index for i in group],
                           "timestamps": [records[i].timestamp.isoformat() if records[i].timestamp else None for i in group],
                           "frames": len(group), "size_xy": size, "path": str(component_dir)})
    report = {"solver": "lm",
              "reverse_even_x": reverse_even, "backward_calibration_matches": len(fixed_candidates),
              "backward_rigid_matrix": calibration.backward_matrix.tolist() if calibration.backward_matrix is not None else None,
              "components": components, "registrations": details}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
