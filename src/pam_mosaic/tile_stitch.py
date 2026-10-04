"""Connect independently reconstructed image groups through measured overlap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from .fusion import fuse_tiles
from .graph import connected_components
from .poses import Edge, initial_poses, optimize_poses
from .registration import register


def stitch(component_root: Path, output: Path, *, min_frames: int = 50) -> dict:
    source = json.loads((component_root / "report.json").read_text())
    selected = [c for c in source["components"] if c["frames"] >= min_frames]
    selected.sort(key=lambda c: -c["frames"])
    if not selected:
        raise ValueError("No components meet --min-frames")
    images = [np.load(Path(c["path"]) / "mosaic_float64.npy", mmap_mode="r") for c in selected]
    weights = [np.load(Path(c["path"]) / "weight_map.npy", mmap_mode="r") for c in selected]
    edges = []
    diagnostics = []
    for a in range(len(selected)):
        for b in range(a+1, len(selected)):
            match = register(images[a], images[b], min_inliers=20,
                             refine_geometry=False)
            diagnostics.append({"reference": a, "moving": b, "inliers": match.inliers,
                                "median_error": match.median_error if np.isfinite(match.median_error) else None,
                                "overlap": match.overlap, "valid": match.valid})
            if match.valid:
                edges.append(Edge(a, b, match.matrix, min(2.0, match.inliers / 30)))
    groups = connected_components(len(selected), [(e.reference, e.moving) for e in edges])
    result_groups = []
    for group_number, group in enumerate(groups, 1):
        local = {g: i for i, g in enumerate(group)}
        local_edges = [Edge(local[e.reference], local[e.moving], e.matrix, e.weight)
                       for e in edges if e.reference in local and e.moving in local]
        poses = optimize_poses(initial_poses(len(group), local_edges), local_edges)
        destination = output / f"mosaic_{group_number:03d}"
        size = fuse_tiles([images[i] for i in group], [weights[i] for i in group], poses, destination)
        np.save(destination / "tile_poses_xytheta.npy", poses)
        result_groups.append({"tiles": [selected[i]["path"] for i in group],
                              "source_files": sum(selected[i]["frames"] for i in group),
                              "size_xy": size, "path": str(destination)})
    output.mkdir(parents=True, exist_ok=True)
    report = {"solver": "lm", "min_frames": min_frames, "selected_tiles": len(selected),
              "groups": result_groups, "registrations": diagnostics}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect local PAM mosaics")
    parser.add_argument("--components", type=Path, default=Path("outputs/lm_full"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lm_stitched"))
    parser.add_argument("--min-frames", type=int, default=50)
    args = parser.parse_args()
    report = stitch(args.components, args.output, min_frames=args.min_frames)
    print(f"Built {len(report['groups'])} large mosaic(s) from {report['selected_tiles']} tiles")


if __name__ == "__main__":
    main()
