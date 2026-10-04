"""Track the two MAP frames in each acquisition as separate observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import Record, index_dataset
from .fusion import fuse
from .graph import candidate_pairs, connected_components
from .poses import Edge, initial_poses, optimize_poses
from .reconstruction import (Calibration, calibrate_even_trajectory,
                             choose_even_direction, reconstruct)
from .registration import register, register_translation_fallback, registration_view


def _calibration(records: list[Record]) -> tuple[Calibration, dict]:
    paths = [record.path for record in records[:5]]
    reverse = choose_even_direction(paths)
    shift = calibrate_even_trajectory(paths, reverse)
    basic = Calibration(reverse_even_x=reverse, even_time_shift_samples=shift)
    candidates = []
    for path in paths:
        forward, backward = reconstruct(path, basic)
        match = register(forward, backward)
        if match.valid:
            angle = np.rad2deg(np.arctan2(match.matrix[1, 0], match.matrix[0, 0]))
            if np.linalg.norm(match.matrix[:, 2]) <= 20 and abs(angle) <= 5:
                candidates.append(match.matrix)
    if not candidates:
        raise ValueError("Cannot calibrate the fixed backward-frame transform")
    fixed = np.median(np.stack(candidates), axis=0)
    angle = np.arctan2(fixed[1, 0], fixed[0, 0])
    fixed[:, :2] = [[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]]
    calibration = Calibration(reverse_even_x=reverse,
                              backward_matrix=fixed,
                              even_time_shift_samples=shift)
    info = {"reverse_even_x": reverse,
            "even_time_shift_samples": shift.tolist(),
            "backward_rigid_matrix": fixed.tolist(),
            "backward_calibration_matches": len(candidates)}
    return calibration, info


def run(records: list[Record], output: Path, *, temporal_window: int = 3,
        appearance_neighbors: int = 2) -> dict:
    if not records:
        raise ValueError("No data files selected")
    calibration, calibration_info = _calibration(records)
    frames: list[np.ndarray] = []
    sources = []
    for record in records:
        forward, backward = reconstruct(record.path, calibration)
        frames.extend((forward, backward))
        sources.extend(({"file_index": record.index, "scan": record.scan,
                         "direction": "forward"},
                        {"file_index": record.index, "scan": record.scan,
                         "direction": "backward"}))
    pairs = candidate_pairs(frames, temporal_window=temporal_window,
                            appearance_neighbors=appearance_neighbors)
    sift = cv2.SIFT_create(nfeatures=1200)
    features = [sift.detectAndCompute(registration_view(frame), None)
                for frame in frames]
    edges: list[Edge] = []
    diagnostics = []
    for done, (reference, moving) in enumerate(pairs, 1):
        match = register(frames[reference], frames[moving],
                         reference_features=features[reference],
                         moving_features=features[moving])
        if (not match.valid and moving - reference <= 3 and
                sources[reference]["scan"] == sources[moving]["scan"]):
            match = register_translation_fallback(frames[reference], frames[moving])
        diagnostics.append({"reference": reference, "moving": moving,
                            "valid": match.valid, "inliers": match.inliers,
                            "median_error": match.median_error if np.isfinite(match.median_error) else None,
                            "overlap": match.overlap})
        if match.valid:
            weight = .25 if match.inliers == 0 else min(2., match.inliers / 30)
            edges.append(Edge(reference, moving, match.matrix, weight))
        if done % 100 == 0 or done == len(pairs):
            print(f"matched {done}/{len(pairs)} frame pairs", flush=True)
    components = []
    groups = connected_components(len(frames), [(e.reference, e.moving) for e in edges])
    for number, group in enumerate(groups, 1):
        local = {global_id: i for i, global_id in enumerate(group)}
        local_edges = [Edge(local[e.reference], local[e.moving], e.matrix, e.weight)
                       for e in edges if e.reference in local and e.moving in local]
        poses = optimize_poses(initial_poses(len(group), local_edges), local_edges)
        destination = output / f"component_{number:03d}"
        size = fuse([frames[i] for i in group], poses, destination)
        np.save(destination / "poses_xytheta.npy", poses)
        components.append({"frames": len(group),
                           "source_files": len({sources[i]["file_index"] for i in group}),
                           "frame_sources": [sources[i] for i in group],
                           "size_xy": size, "path": str(destination)})
    report = {"solver": "lm", "input_files": len(records),
              "frame_count": len(frames), "calibration": calibration_info,
              "components": components, "registrations": diagnostics}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Mosaic sequential forward/backward MAP frames")
    parser.add_argument("--data", type=Path, default=Path("PAM_Skull_FYH_20260929"))
    parser.add_argument("--output", type=Path, default=Path("outputs/two_frame_full"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--temporal-window", type=int, default=3)
    parser.add_argument("--appearance-neighbors", type=int, default=2)
    args = parser.parse_args()
    records = [record for record in index_dataset(args.data)
               if record.index >= args.start and (args.stop is None or record.index <= args.stop)]
    result = run(records, args.output, temporal_window=args.temporal_window,
                 appearance_neighbors=args.appearance_neighbors)
    print(f"Built {len(result['components'])} components from {result['frame_count']} MAP frames")


if __name__ == "__main__":
    main()
