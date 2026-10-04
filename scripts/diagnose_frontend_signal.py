"""Compare signal and feature support at passed/failed adjacent matches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pam_mosaic.dataset import index_dataset
from pam_mosaic.reconstruction import reconstruct
from pam_mosaic.registration import registration_view
from pam_mosaic.roundtrip import calibrate, fuse_if_aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("PAM_Skull_FYH_20260929"))
    parser.add_argument("--report", type=Path, default=Path("outputs/validated_roundtrip_full/report.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validated_roundtrip_full/signal_diagnostics.json"))
    args = parser.parse_args()
    records = index_dataset(args.data)
    calibration = calibrate([record.path for record in records])
    sift = cv2.SIFT_create(nfeatures=1200)
    metrics = {}
    for record in records:
        forward, backward = reconstruct(record.path, calibration)
        image, _ = fuse_if_aligned(forward, backward)
        view = registration_view(image)
        keypoints = sift.detect(view, None)
        metrics[record.index] = {
            "p95": float(np.percentile(image, 95)),
            "p99": float(np.percentile(image, 99)),
            "fraction_above_002": float(np.mean(image > .02)),
            "fraction_above_005": float(np.mean(image > .05)),
            "sift_keypoints": len(keypoints),
        }
    report = json.loads(args.report.read_text())
    adjacent = [entry for entry in report["registrations"]
                if entry["moving"] == entry["reference"] + 1]
    for entry in adjacent:
        left, right = metrics[entry["reference"]], metrics[entry["moving"]]
        entry["minimum_p99"] = min(left["p99"], right["p99"])
        entry["minimum_fraction_above_005"] = min(left["fraction_above_005"], right["fraction_above_005"])
        entry["minimum_keypoints"] = min(left["sift_keypoints"], right["sift_keypoints"])
    result = {"frame_metrics": metrics, "adjacent_pairs": adjacent}
    args.output.write_text(json.dumps(result, indent=2))
    for label, group in (("passed", [x for x in adjacent if x["valid"]]),
                         ("failed", [x for x in adjacent if not x["valid"]])):
        print(label, len(group), "median minimum p99", np.median([x["minimum_p99"] for x in group]),
              "median minimum keypoints", np.median([x["minimum_keypoints"] for x in group]),
              "median minimum bright fraction", np.median([x["minimum_fraction_above_005"] for x in group]))
    for i in (8, 9, 312, 313, 500, 501):
        print(i, metrics[i])


if __name__ == "__main__":
    main()
