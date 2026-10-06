"""Connect independently reconstructed image groups through measured overlap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from .fusion import fuse_tiles
from .graph import connected_components
from .poses import Edge, initial_poses, optimize_poses, pose_to_matrix
from .registration import register


def _consistent_pose_graph(count: int, edges: list[Edge], scores: dict[tuple[int, int], float],
                           *, maximum_cycle_shift: float = 5.0,
                           maximum_cycle_angle_deg: float = .2
                           ) -> tuple[list[Edge], dict[tuple[int, int], dict]]:
    """Keep a strong spanning tree and only cycle constraints consistent with it."""
    parent = list(range(count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    ordered = sorted(edges, key=lambda edge: scores[(edge.reference, edge.moving)], reverse=True)
    tree = []
    tree_keys = set()
    for edge in ordered:
        a, b = find(edge.reference), find(edge.moving)
        if a == b:
            continue
        parent[a] = b
        tree.append(edge)
        tree_keys.add((edge.reference, edge.moving))
    if len(tree) != count - 1:
        raise ValueError("Cannot build a connected tile spanning tree")
    poses = initial_poses(count, tree)
    selected = list(tree)
    audit = {}
    for edge in ordered:
        key = (edge.reference, edge.moving)
        if key in tree_keys:
            audit[key] = {"pose_graph_use": "tree", "cycle_shift_px": 0.0,
                          "cycle_angle_deg": 0.0}
            continue
        measurement = np.vstack((edge.matrix, [0.0, 0.0, 1.0]))
        predicted = (np.linalg.inv(pose_to_matrix(poses[edge.reference])) @
                     pose_to_matrix(poses[edge.moving]))
        innovation = np.linalg.inv(measurement) @ predicted
        shift = float(np.hypot(innovation[0, 2], innovation[1, 2]))
        angle = float(np.degrees(np.arctan2(innovation[1, 0], innovation[0, 0])))
        accepted = shift <= maximum_cycle_shift and abs(angle) <= maximum_cycle_angle_deg
        audit[key] = {"pose_graph_use": "consistent_loop" if accepted else "rejected_loop",
                      "cycle_shift_px": shift, "cycle_angle_deg": angle}
        if accepted:
            selected.append(edge)
    return selected, audit


def stitch(component_root: Path, output: Path, *, min_frames: int = 50,
           cycle_gating: bool = True) -> dict:
    source = json.loads((component_root / "report.json").read_text())
    selected = [c for c in source["components"] if c["frames"] >= min_frames]
    selected.sort(key=lambda c: -c["frames"])
    if not selected:
        raise ValueError("No components meet --min-frames")
    images = [np.load(Path(c["path"]) / "mosaic_float64.npy", mmap_mode="r") for c in selected]
    weights = [np.load(Path(c["path"]) / "weight_map.npy", mmap_mode="r") for c in selected]
    edges = []
    scores = {}
    diagnostics = []
    for a in range(len(selected)):
        for b in range(a+1, len(selected)):
            match = register(images[a], images[b], min_inliers=20,
                             refine_geometry=False)
            diagnostics.append({"reference": a, "moving": b, "inliers": match.inliers,
                                "median_error": match.median_error if np.isfinite(match.median_error) else None,
                                "overlap": match.overlap, "valid": match.valid,
                                "matrix": match.matrix.tolist() if match.valid else None})
            if match.valid:
                edge = Edge(a, b, match.matrix, min(2.0, match.inliers / 30))
                edges.append(edge)
                scores[(a, b)] = match.inliers / (1.0 + match.median_error)
    groups = connected_components(len(selected), [(e.reference, e.moving) for e in edges])
    result_groups = []
    for group_number, group in enumerate(groups, 1):
        local = {g: i for i, g in enumerate(group)}
        local_edges = [Edge(local[e.reference], local[e.moving], e.matrix, e.weight)
                       for e in edges if e.reference in local and e.moving in local]
        local_scores = {(local[e.reference], local[e.moving]): scores[(e.reference, e.moving)]
                        for e in edges if e.reference in local and e.moving in local}
        if cycle_gating and len(local_edges) >= len(group):
            optimized_edges, edge_audit = _consistent_pose_graph(
                len(group), local_edges, local_scores)
        else:
            optimized_edges = local_edges
            edge_audit = {(edge.reference, edge.moving): {
                "pose_graph_use": "unfiltered", "cycle_shift_px": None,
                "cycle_angle_deg": None} for edge in local_edges}
        poses = optimize_poses(initial_poses(len(group), optimized_edges), optimized_edges)
        for global_edge in diagnostics:
            if (not global_edge["valid"] or
                    global_edge["reference"] not in local or
                    global_edge["moving"] not in local):
                continue
            key = (local[global_edge["reference"]], local[global_edge["moving"]])
            global_edge.update(edge_audit[key])
        destination = output / f"mosaic_{group_number:03d}"
        size = fuse_tiles([images[i] for i in group], [weights[i] for i in group], poses, destination)
        np.save(destination / "tile_poses_xytheta.npy", poses)
        result_groups.append({"tiles": [selected[i]["path"] for i in group],
                              "source_files": len({source["file_index"] for i in group
                                                   for source in selected[i].get("frame_sources", [])})
                              if any("frame_sources" in selected[i] for i in group)
                              else sum(selected[i]["frames"] for i in group),
                              "size_xy": size, "path": str(destination)})
    output.mkdir(parents=True, exist_ok=True)
    report = {"solver": "lm", "cycle_gating": cycle_gating,
              "min_frames": min_frames, "selected_tiles": len(selected),
              "groups": result_groups, "registrations": diagnostics}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect local PAM mosaics")
    parser.add_argument("--components", type=Path, default=Path("outputs/frontend_only_full"))
    parser.add_argument("--output", type=Path, default=Path("outputs/frontend_only_stitched"))
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--cycle-gating", action=argparse.BooleanOptionalAction, default=True,
                        help="reject tile loop constraints inconsistent with the strongest spanning tree")
    args = parser.parse_args()
    report = stitch(args.components, args.output, min_frames=args.min_frames,
                    cycle_gating=args.cycle_gating)
    print(f"Built {len(report['groups'])} large mosaic(s) from {report['selected_tiles']} tiles")


if __name__ == "__main__":
    main()
