"""Quality audit for newly recovered adjacent links and stitched tile seams."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .dataset import index_dataset
from .poses import matrix_to_pose, pose_to_matrix
from .reconstruction import reconstruct
from .registration import (Match, register, register_enhanced_adjacent,
                           register_weak_signal_adjacent, registration_view,
                           sift_inliers)
from .roundtrip import calibrate, fuse_if_aligned


def _homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.vstack((matrix, [0.0, 0.0, 1.0]))


def _pose_lookup(component_root: Path) -> dict[int, np.ndarray]:
    report = json.loads((component_root / "report.json").read_text())
    result: dict[int, np.ndarray] = {}
    for component in report["components"]:
        poses = np.load(Path(component["path"]) / "poses_xytheta.npy")
        for file_index, pose in zip(component["file_indices"], poses):
            result[int(file_index)] = pose
    return result


def _adjacent(report: dict) -> dict[tuple[int, int], dict]:
    return {(int(item["reference"]), int(item["moving"])): item
            for item in report["registrations"]
            if int(item["moving"]) == int(item["reference"]) + 1}


def _match_for_method(reference: np.ndarray, moving: np.ndarray, method: str) -> Match:
    if method == "enhanced_sift":
        return register_enhanced_adjacent(reference, moving)
    if method == "weak_signal_phase":
        return register_weak_signal_adjacent(reference, moving)
    return register(reference, moving)


def _inlier_distribution(reference: np.ndarray, moving: np.ndarray,
                         method: str) -> tuple[float | None, float | None]:
    if method == "weak_signal_phase":
        return None, None
    if method == "enhanced_sift":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        sift = cv2.SIFT_create(nfeatures=1800)
        ref_features = sift.detectAndCompute(clahe.apply(registration_view(reference)), None)
        mov_features = sift.detectAndCompute(clahe.apply(registration_view(moving)), None)
        correspondence = sift_inliers(reference, moving, 15,
                                      reference_features=ref_features,
                                      moving_features=mov_features)
    else:
        correspondence = sift_inliers(reference, moving, 12)
    if correspondence is None:
        return None, None
    source, target, _ = correspondence
    spreads = []
    coverages = []
    area = float(reference.shape[0] * reference.shape[1])
    for points in (source, target):
        if len(points) < 3:
            return 0.0, 0.0
        eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
        spreads.append(float(np.sqrt(max(0.0, eigenvalues[0]))))
        hull = cv2.convexHull(points.astype(np.float32))
        coverages.append(float(cv2.contourArea(hull)) / area)
    return min(spreads), min(coverages)


def _alignment_metrics(reference: np.ndarray, moving: np.ndarray,
                       matrix: np.ndarray, reference_valid: np.ndarray | None = None,
                       moving_valid: np.ndarray | None = None) -> dict:
    ref = registration_view(reference)
    mov = registration_view(moving)
    height, width = ref.shape
    if reference_valid is None:
        reference_valid = reference > 0
    if moving_valid is None:
        moving_valid = moving > 0
    warped = cv2.warpAffine(mov, matrix.astype(np.float32), (width, height),
                            flags=cv2.INTER_LINEAR, borderValue=0)
    warped_valid = cv2.warpAffine(moving_valid.astype(np.uint8),
                                  matrix.astype(np.float32), (width, height),
                                  flags=cv2.INTER_NEAREST, borderValue=0) > 0
    overlap_mask = reference_valid & warped_valid
    overlap_pixels = int(overlap_mask.sum())
    denominator = max(1, min(int(reference_valid.sum()), int(moving_valid.sum())))
    overlap = overlap_pixels / denominator
    correlation = None
    if overlap_pixels >= 100 and ref[overlap_mask].std() >= 3 and warped[overlap_mask].std() >= 3:
        value = float(np.corrcoef(ref[overlap_mask], warped[overlap_mask])[0, 1])
        correlation = value if np.isfinite(value) else None

    ref_edge = (cv2.Canny(ref, 35, 90) > 0) & overlap_mask
    mov_edge = (cv2.Canny(warped, 35, 90) > 0) & overlap_mask
    distances = []
    if ref_edge.any() and mov_edge.any():
        distance_to_ref = distance_transform_edt(~ref_edge)
        distance_to_mov = distance_transform_edt(~mov_edge)
        distances = np.concatenate((distance_to_ref[mov_edge], distance_to_mov[ref_edge]))
    return {
        "actual_overlap": float(overlap),
        "intensity_correlation": correlation,
        "edge_distance_median_px": float(np.median(distances)) if len(distances) else None,
        "edge_distance_p90_px": float(np.percentile(distances, 90)) if len(distances) else None,
        "overlap_pixels": overlap_pixels,
    }


def _risk(item: dict) -> tuple[str, list[str]]:
    """Return a conservative audit level without treating missing fine vessels as ghosts."""
    high_checks = (
        (not item.get("valid_on_recheck", False), "recheck_failed"),
        (not item.get("reverse_valid", False), "reverse_failed"),
        (item.get("intensity_correlation") is None or
         item["intensity_correlation"] < .4, "correlation_below_0.4"),
        (item.get("closure_shift_px") is not None and
         item["closure_shift_px"] > 10, "closure_shift_above_10"),
        (item.get("closure_angle_deg") is not None and
         abs(item["closure_angle_deg"]) > 1.5, "closure_angle_above_1.5"),
        (item.get("edge_distance_median_px") is None or
         item["edge_distance_median_px"] > 4, "edge_median_above_4"),
        (abs(item.get("optimized_angle_deg") or 0.0) > 3, "pose_rotation_above_3"),
        (item.get("inliers", 0) < 15 and
         (item.get("inlier_spread_px") is None or
          item["inlier_spread_px"] < 20), "very_weak_inliers"),
    )
    review_checks = (
        (item.get("method") == "weak_signal_phase", "weak_signal_phase"),
        (item.get("inliers", 0) < 20, "few_inliers"),
        (item.get("median_error_px") is None or
         item["median_error_px"] > 2.5, "point_error_above_2.5"),
        (item.get("inlier_spread_px") is not None and
         item["inlier_spread_px"] < 30, "inlier_spread_below_30"),
        (item.get("inlier_hull_fraction") is not None and
         item["inlier_hull_fraction"] < .03, "inlier_area_below_0.03"),
        (item.get("intensity_correlation") is None or
         item["intensity_correlation"] < .6, "correlation_below_0.6"),
        (item.get("closure_shift_px") is not None and
         item["closure_shift_px"] > 3, "closure_shift_above_3"),
        (item.get("closure_angle_deg") is not None and
         abs(item["closure_angle_deg"]) > .5, "closure_angle_above_0.5"),
        (item.get("edge_distance_median_px") is None or
         item["edge_distance_median_px"] > 2.5, "edge_median_above_2.5"),
        (abs(item.get("optimized_angle_deg") or 0.0) > 2,
         "pose_rotation_above_2"),
    )
    high = [label for failed, label in high_checks if failed]
    review = [label for failed, label in review_checks if failed]
    if high:
        return "high", high + [label for label in review if label not in high]
    if review:
        return "medium", review
    return "low", []


def audit_new_adjacent(data_root: Path, baseline_root: Path, current_root: Path,
                       output: Path) -> dict:
    baseline = json.loads((baseline_root / "report.json").read_text())
    current = json.loads((current_root / "report.json").read_text())
    baseline_adjacent = _adjacent(baseline)
    current_adjacent = _adjacent(current)
    baseline_valid = {key for key, value in baseline_adjacent.items() if value["valid"]}
    current_valid = {key for key, value in current_adjacent.items() if value["valid"]}
    gained = sorted(current_valid - baseline_valid)
    lost = sorted(baseline_valid - current_valid)

    records = index_dataset(data_root)
    record_by_index = {record.index: record for record in records}
    needed = sorted({index for pair in gained for index in pair})
    missing = [index for index in needed if index not in record_by_index]
    if missing:
        raise ValueError(f"Missing acquisition files: {missing[:10]}")
    calibration = calibrate([record.path for record in records])
    frames: dict[int, np.ndarray] = {}
    for number, index in enumerate(needed, 1):
        forward, backward = reconstruct(record_by_index[index].path, calibration)
        frames[index], _ = fuse_if_aligned(forward, backward, suppress_bright_points=True)
        if number % 25 == 0 or number == len(needed):
            print(f"reconstructed {number}/{len(needed)} audit frames", flush=True)

    optimized = _pose_lookup(current_root)
    audited = []
    for number, pair in enumerate(gained, 1):
        reference_index, moving_index = pair
        detail = current_adjacent[pair]
        method = detail.get("recovery_method", "ordinary")
        reference, moving = frames[reference_index], frames[moving_index]
        cv2.setRNGSeed(0)
        match = _match_for_method(reference, moving, method)
        cv2.setRNGSeed(0)
        reverse = _match_for_method(moving, reference, method)
        spread, hull = _inlier_distribution(reference, moving, method)
        closure_shift = closure_angle = None
        if match.valid and reverse.valid:
            closure = _homogeneous(match.matrix) @ _homogeneous(reverse.matrix)
            closure_pose = matrix_to_pose(closure)
            closure_shift = float(np.hypot(*closure_pose[:2]))
            closure_angle = float(np.degrees(closure_pose[2]))
        pose = None
        if reference_index in optimized and moving_index in optimized:
            relative = (np.linalg.inv(pose_to_matrix(optimized[reference_index])) @
                        pose_to_matrix(optimized[moving_index]))
            pose = matrix_to_pose(relative)
        metrics = _alignment_metrics(reference, moving, match.matrix)
        item = {
            "reference": reference_index,
            "moving": moving_index,
            "method": method,
            "valid_on_recheck": bool(match.valid),
            "inliers": int(match.inliers),
            "median_error_px": float(match.median_error) if np.isfinite(match.median_error) else None,
            "reported_overlap": float(match.overlap),
            "inlier_spread_px": spread,
            "inlier_hull_fraction": hull,
            "reverse_valid": bool(reverse.valid),
            "closure_shift_px": closure_shift,
            "closure_angle_deg": closure_angle,
            "optimized_shift_px": float(np.hypot(*pose[:2])) if pose is not None else None,
            "optimized_angle_deg": float(np.degrees(pose[2])) if pose is not None else None,
            **metrics,
        }
        item["risk_level"], item["risk_flags"] = _risk(item)
        audited.append(item)
        if number % 25 == 0 or number == len(gained):
            print(f"audited {number}/{len(gained)} gained adjacent links", flush=True)

    counts = {level: sum(item["risk_level"] == level for item in audited)
              for level in ("low", "medium", "high")}
    result = {
        "baseline_valid_adjacent": len(baseline_valid),
        "current_valid_adjacent": len(current_valid),
        "gained": len(gained),
        "lost": len(lost),
        "net_gain": len(current_valid) - len(baseline_valid),
        "lost_pairs": [list(pair) for pair in lost],
        "risk_counts": counts,
        "links": audited,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "new_adjacent_links.json").write_text(json.dumps(result, indent=2))
    if audited:
        columns = [key for key in audited[0] if key != "risk_flags"] + ["risk_flags"]
        for filename, rows in (
                ("new_adjacent_links.csv", audited),
                ("high_risk_adjacent_links.csv",
                 [item for item in audited if item["risk_level"] == "high"])):
            with (output / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for item in rows:
                    writer.writerow({**item, "risk_flags": ";".join(item["risk_flags"])})
    return result


def _seam_overlay(reference: np.ndarray, moving: np.ndarray, matrix: np.ndarray,
                  reference_valid: np.ndarray, moving_valid: np.ndarray) -> np.ndarray | None:
    ref = registration_view(reference)
    mov = registration_view(moving)
    height, width = ref.shape
    warped = cv2.warpAffine(mov, matrix.astype(np.float32), (width, height))
    warped_valid = cv2.warpAffine(moving_valid.astype(np.uint8), matrix.astype(np.float32),
                                  (width, height), flags=cv2.INTER_NEAREST) > 0
    overlap = reference_valid & warped_valid
    ys, xs = np.nonzero(overlap)
    if not len(xs):
        return None
    pad = 20
    x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + pad + 1)
    # Reference is red; aligned moving is cyan. Correctly aligned vessels appear white.
    overlay = np.dstack((warped, warped, ref))[y0:y1, x0:x1]
    scale = min(1.0, 1200.0 / max(overlay.shape[:2]))
    if scale < 1:
        overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return overlay


def audit_tile_seams(component_root: Path, stitched_root: Path, output: Path) -> dict:
    stitched = json.loads((stitched_root / "report.json").read_text())
    if len(stitched["groups"]) != 1:
        raise ValueError("Tile seam audit currently expects one stitched group")
    group = stitched["groups"][0]
    tile_paths = [Path(path) for path in group["tiles"]]
    images = [np.load(path / "mosaic_float64.npy", mmap_mode="r") for path in tile_paths]
    weights = [np.load(path / "weight_map.npy", mmap_mode="r") for path in tile_paths]
    poses = np.load(Path(group["path"]) / "tile_poses_xytheta.npy")
    valid_edges = [item for item in stitched["registrations"] if item["valid"]]
    output.mkdir(parents=True, exist_ok=True)
    seams = []
    for number, detail in enumerate(valid_edges, 1):
        reference_index, moving_index = int(detail["reference"]), int(detail["moving"])
        relative = (np.linalg.inv(pose_to_matrix(poses[reference_index])) @
                    pose_to_matrix(poses[moving_index]))
        matrix = relative[:2]
        optimized_metrics = _alignment_metrics(
            images[reference_index], images[moving_index], matrix,
            weights[reference_index] > 0, weights[moving_index] > 0)
        measured = register(images[reference_index], images[moving_index],
                            min_inliers=20, refine_geometry=False)
        measured_metrics = _alignment_metrics(
            images[reference_index], images[moving_index], measured.matrix,
            weights[reference_index] > 0, weights[moving_index] > 0)
        pose = matrix_to_pose(relative)
        delta = np.linalg.inv(_homogeneous(measured.matrix)) @ relative
        delta_pose = matrix_to_pose(delta)
        optimized_correlation = optimized_metrics["intensity_correlation"]
        measured_correlation = measured_metrics["intensity_correlation"]
        correlation_drop = ((measured_correlation - optimized_correlation)
                            if measured_correlation is not None and
                            optimized_correlation is not None else None)
        item = {
            "reference": reference_index,
            "moving": moving_index,
            "reference_path": str(tile_paths[reference_index]),
            "moving_path": str(tile_paths[moving_index]),
            "relative_shift_px": float(np.hypot(*pose[:2])),
            "relative_angle_deg": float(np.degrees(pose[2])),
            "optimization_change_shift_px": float(np.hypot(*delta_pose[:2])),
            "optimization_change_angle_deg": float(np.degrees(delta_pose[2])),
            "measured_correlation": measured_correlation,
            "optimized_correlation": optimized_correlation,
            "correlation_drop_after_optimization": correlation_drop,
            "measured_edge_distance_median_px": measured_metrics["edge_distance_median_px"],
            "optimized_edge_distance_median_px": optimized_metrics["edge_distance_median_px"],
            "actual_overlap": optimized_metrics["actual_overlap"],
            "overlap_pixels": optimized_metrics["overlap_pixels"],
        }
        high = bool(optimized_correlation is None or optimized_correlation < .6 or
                    correlation_drop is not None and correlation_drop > .2)
        review = bool(high or optimized_correlation < .7 or
                      correlation_drop is not None and correlation_drop > .1 or
                      optimized_metrics["edge_distance_median_px"] is None or
                      optimized_metrics["edge_distance_median_px"] > 3 or
                      abs(item["optimization_change_angle_deg"]) > .3)
        item["risk_level"] = "high" if high else "medium" if review else "low"
        seams.append(item)
        overlay = _seam_overlay(images[reference_index], images[moving_index], matrix,
                                weights[reference_index] > 0, weights[moving_index] > 0)
        if overlay is not None:
            cv2.imwrite(str(output / f"tile_seam_{reference_index}_{moving_index}.png"), overlay)
        print(f"audited tile seam {number}/{len(valid_edges)}", flush=True)
    result = {
        "selected_tiles": len(tile_paths),
        "valid_tile_edges": len(valid_edges),
        "risk_counts": {level: sum(item["risk_level"] == level for item in seams)
                        for level in ("low", "medium", "high")},
        "seams": seams,
    }
    (output / "tile_seams.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit recovered adjacent links and tile seams")
    parser.add_argument("--data", type=Path, default=Path("PAM_Skull_FYH_20260929"))
    parser.add_argument("--baseline", type=Path, default=Path("outputs/lm_full"))
    parser.add_argument("--current", type=Path, default=Path("outputs/frontend_only_full"))
    parser.add_argument("--stitched", type=Path, default=Path("outputs/frontend_only_stitched"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/frontend_only_stitched/audit"))
    parser.add_argument("--links-only", action="store_true")
    parser.add_argument("--seams-only", action="store_true")
    args = parser.parse_args()
    if args.links_only and args.seams_only:
        parser.error("choose at most one of --links-only and --seams-only")
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if not args.seams_only:
        links = audit_new_adjacent(args.data, args.baseline, args.current, args.output)
        summary["new_adjacent_links"] = {key: value for key, value in links.items()
                                         if key != "links"}
    if not args.links_only:
        seams = audit_tile_seams(args.current, args.stitched, args.output)
        summary["tile_seams"] = {key: value for key, value in seams.items()
                                 if key != "seams"}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
