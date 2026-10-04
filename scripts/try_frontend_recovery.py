"""Audit missed adjacent registrations without changing the production mosaic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pam_mosaic.dataset import index_dataset
from pam_mosaic.reconstruction import reconstruct
from pam_mosaic.registration import register, registration_view
from pam_mosaic.roundtrip import calibrate, fuse_if_aligned


def agreement(reference: np.ndarray, moving: np.ndarray, matrix: np.ndarray) -> tuple[float, float]:
    a, b = registration_view(reference), registration_view(moving)
    h, w = a.shape
    warped = cv2.warpAffine(b, matrix.astype(np.float32), (w, h))
    mask = cv2.warpAffine(np.ones_like(b, np.uint8), matrix.astype(np.float32),
                          (w, h), flags=cv2.INTER_NEAREST) > 0
    overlap = float(mask.mean())
    if overlap < .3 or a[mask].std() < 3 or warped[mask].std() < 3:
        return overlap, float("nan")
    return overlap, float(np.corrcoef(a[mask], warped[mask])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("PAM_Skull_FYH_20260929"))
    parser.add_argument("--report", type=Path, default=Path("outputs/validated_roundtrip_full/report.json"))
    parser.add_argument("--signal", type=Path, default=Path("outputs/validated_roundtrip_full/signal_diagnostics.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/validated_roundtrip_full/recovery_trial.json"))
    args = parser.parse_args()
    records = {r.index: r.path for r in index_dataset(args.data)}
    calibration = calibrate(list(records.values()))
    original = json.loads(args.report.read_text())
    signal = json.loads(args.signal.read_text())["frame_metrics"]
    failed = [e for e in original["registrations"] if not e["valid"] and
              e["moving"] == e["reference"] + 1]
    needed = {i for e in failed for i in (e["reference"], e["moving"])}
    frames = {i: fuse_if_aligned(*reconstruct(records[i], calibration))[0] for i in sorted(needed)}
    sift = cv2.SIFT_create(nfeatures=1800)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    features = {i: sift.detectAndCompute(clahe.apply(registration_view(frame)), None)
                for i, frame in frames.items()}
    results = []
    for number, pair in enumerate(failed, 1):
        i, j = pair["reference"], pair["moving"]
        a, b = frames[i], frames[j]
        weak = (min(signal[str(i)]["p99"], signal[str(j)]["p99"]) < .08 or
                min(signal[str(i)]["sift_keypoints"], signal[str(j)]["sift_keypoints"]) < 50)
        choice = None
        match = register(a, b, min_inliers=15, refine_geometry=False,
                         reference_features=features[i], moving_features=features[j])
        if match.valid:
            overlap, corr = agreement(a, b, match.matrix)
            angle = abs(np.degrees(np.arctan2(match.matrix[1, 0], match.matrix[0, 0])))
            if (match.inliers >= 20 and match.median_error <= 2.5 and
                    overlap >= .4 and corr >= .55 and angle <= 5):
                choice = {"method": "enhanced_sift", "matrix": match.matrix.tolist(),
                          "inliers": match.inliers, "error_px": match.median_error,
                          "overlap": overlap, "correlation": corr}
        if choice is None and weak:
            ref = registration_view(a).astype(np.float32)
            mov = registration_view(b).astype(np.float32)
            (dx, dy), response = cv2.phaseCorrelate(ref, mov)
            if np.isfinite([dx, dy, response]).all() and np.hypot(dx, dy) <= 50:
                matrix = np.array([[1., 0., -dx], [0., 1., -dy]])
                overlap, corr = agreement(a, b, matrix)
                if response >= .2 and overlap >= .8 and corr >= .7:
                    choice = {"method": "weak_signal_phase", "matrix": matrix.tolist(),
                              "phase_response": response, "overlap": overlap,
                              "correlation": corr}
        results.append({"reference": i, "moving": j, "weak_signal": weak,
                        "accepted": choice is not None, "measurement": choice})
        if number % 50 == 0:
            print(f"checked {number}/{len(failed)}", flush=True)
    args.output.write_text(json.dumps(results, indent=2))
    print("recovered", sum(x["accepted"] for x in results), "of", len(results))
    print("by method", {method: sum(x["accepted"] and x["measurement"]["method"] == method
                              for x in results) for method in ("enhanced_sift", "weak_signal_phase")})
    print("examples", [(x["reference"], x["moving"], x["measurement"]["method"])
                       for x in results if x["accepted"]][:20])


if __name__ == "__main__":
    main()
