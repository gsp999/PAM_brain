"""Run reconstruction, matching, pose graph, and mosaic fusion."""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from .dataset import Record
from .fusion import fuse
from .graph import candidate_pairs, connected_components
from .poses import Edge, initial_poses, optimize_poses
from .reconstruction import Calibration, choose_even_direction, reconstruct
from .registration import (register, register_enhanced_adjacent,
                           register_weak_signal_adjacent, registration_view)
from .roundtrip import calibrate as calibrate_roundtrip, fuse_if_aligned


def run(records: list[Record], output: Path, *, reverse_even: bool | None = None,
        use_backward: bool = True, temporal_window: int = 5,
        appearance_neighbors: int = 6, validated_roundtrip: bool = False,
        frontend_recovery: bool = False, suppress_speckles: bool = False) -> dict:
    if not records:
        raise ValueError("No data files selected")
    if frontend_recovery and not validated_roundtrip:
        raise ValueError("Frontend recovery requires validated round-trip frames")
    if suppress_speckles and not validated_roundtrip:
        raise ValueError("Speckle suppression requires validated round-trip frames")
    output.mkdir(parents=True, exist_ok=True)
    if validated_roundtrip:
        calibration = calibrate_roundtrip([r.path for r in records])
        reverse_even = calibration.reverse_even_x
    elif reverse_even is None:
        reverse_even = choose_even_direction([r.path for r in records[:min(5, len(records))]])
    if not validated_roundtrip:
        calibration = Calibration(reverse_even_x=reverse_even)
    frames: list[np.ndarray] = []
    details: list[dict] = []
    roundtrip_decisions: list[dict] = []
    speckle_changes: list[dict] = []
    # Calibrate from a few files, then decode the sequence once.
    calibration_pairs = [] if validated_roundtrip else [reconstruct(r.path, calibration) for r in records[:min(5, len(records))]]
    fixed_candidates = []
    if use_backward and not validated_roundtrip:
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
        if validated_roundtrip:
            image, decision = fuse_if_aligned(
                forward, backward, suppress_bright_points=suppress_speckles)
            roundtrip_decisions.append({"file_index": record.index, **decision.__dict__})
            if suppress_speckles:
                speckle_changes.append({"file_index": record.index,
                                        "changed_pixels": decision.changed_pixels,
                                        "selected": decision.selected})
            frames.append(image.astype(np.float32))
        elif use_backward and fixed_candidates:
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
        match_source = "filtered"
        if match.valid:
            edges.append(Edge(reference, moving, match.matrix, min(2.0, match.inliers / 30)))
        details.append({"reference": records[reference].index, "moving": records[moving].index,
                        "inliers": match.inliers, "median_error": match.median_error if np.isfinite(match.median_error) else None,
                        "overlap": match.overlap, "valid": match.valid,
                        "match_source": match_source})
        if done % 50 == 0 or done == len(candidates):
            print(f"matched {done}/{len(candidates)} candidate pairs", flush=True)

    recovery_counts = {"enhanced_sift": 0, "weak_signal_phase": 0}
    if frontend_recovery:
        original_groups = connected_components(len(frames), [(e.reference, e.moving) for e in edges])
        group_of = {index: group for group, members in enumerate(original_groups)
                    for index in members}
        keypoint_count: dict[int, int] = {}
        sift = cv2.SIFT_create(nfeatures=1200)
        for detail, (reference, moving) in zip(details, candidates):
            if (detail["valid"] or moving != reference + 1 or
                    group_of[reference] == group_of[moving] or
                    records[reference].scan != records[moving].scan):
                continue
            enhanced = register_enhanced_adjacent(frames[reference], frames[moving])
            method = "enhanced_sift"
            match = enhanced
            if not match.valid:
                for index in (reference, moving):
                    if index not in keypoint_count:
                        keypoint_count[index] = len(sift.detect(registration_view(frames[index]), None))
                weak = (min(np.percentile(frames[reference], 99),
                            np.percentile(frames[moving], 99)) < .08 or
                        min(keypoint_count[reference], keypoint_count[moving]) < 50)
                if weak:
                    match = register_weak_signal_adjacent(frames[reference], frames[moving])
                    method = "weak_signal_phase"
            if match.valid:
                weight = .2 if method == "weak_signal_phase" else .35
                edges.append(Edge(reference, moving, match.matrix, weight))
                detail.update({"valid": True, "recovery_method": method,
                               "inliers": match.inliers,
                               "median_error": match.median_error,
                               "overlap": match.overlap})
                recovery_counts[method] += 1
        print(f"recovered {sum(recovery_counts.values())} adjacent links", flush=True)

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
    report = {"solver": "lm", "validated_roundtrip": validated_roundtrip,
              "frontend_recovery": frontend_recovery,
              "frontend_recovery_counts": recovery_counts,
              "roundtrip_decisions": roundtrip_decisions,
              "roundtrip_accepted": sum(item["accepted"] for item in roundtrip_decisions),
              "suppress_speckles": suppress_speckles,
              "speckle_changes": speckle_changes,
              "reverse_even_x": reverse_even, "backward_calibration_matches": len(fixed_candidates),
              "backward_rigid_matrix": calibration.backward_matrix.tolist() if calibration.backward_matrix is not None else None,
              "components": components, "registrations": details}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
