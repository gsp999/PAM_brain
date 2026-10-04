"""Inspect one acquisition's forward/backward MAP fusion without mosaicking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pam_mosaic.reconstruction import Calibration, reconstruct
from pam_mosaic.registration import register, registration_view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    info = json.loads(args.calibration_report.read_text())["calibration"]
    calibration = Calibration(
        reverse_even_x=info["reverse_even_x"],
        even_time_shift_samples=np.asarray(info["even_time_shift_samples"]),
        backward_matrix=np.asarray(info["backward_rigid_matrix"]),
    )
    forward, backward = reconstruct(args.data, calibration)
    match = register(forward, backward)
    angle_deg = float(np.degrees(np.arctan2(match.matrix[1, 0], match.matrix[0, 0])))
    shift = float(np.linalg.norm(match.matrix[:, 2]))
    accepted = bool(match.valid and match.inliers >= 20 and
                    match.median_error <= 2 and shift <= 5 and abs(angle_deg) <= 1)

    aligned = cv2.warpAffine(backward, match.matrix.astype(np.float32),
                            (500, 500), flags=cv2.INTER_LINEAR) if accepted else backward
    mask = cv2.warpAffine(np.ones_like(backward, np.uint8),
                          match.matrix.astype(np.float32) if accepted else np.eye(2, 3, dtype=np.float32),
                          (500, 500), flags=cv2.INTER_NEAREST) > 0
    # The fixed backward calibration has invalid border pixels represented by zero.
    mask &= aligned > 0
    fused = forward.copy()
    if accepted:
        fused[mask] = (forward[mask] + aligned[mask]) / 2
    before_corr = float(np.corrcoef(forward[mask], backward[mask])[0, 1])
    after_corr = float(np.corrcoef(forward[mask], aligned[mask])[0, 1])

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "maps_float32.npz", forward=forward,
                        backward=backward, backward_aligned=aligned, fused=fused,
                        backward_valid=mask)
    panels = [registration_view(x) for x in (forward, backward, aligned, fused)]
    preview = np.concatenate(panels, axis=1)
    cv2.imwrite(str(args.output / "comparison.png"), preview)
    report = {"data": str(args.data), "accepted": accepted,
              "inliers": match.inliers, "median_error_px": match.median_error if np.isfinite(match.median_error) else None,
              "residual_shift_px": shift, "residual_angle_deg": angle_deg,
              "valid_backward_fraction": float(mask.mean()),
              "forward_backward_correlation_before": before_corr,
              "forward_backward_correlation_after": after_corr,
              "panels": ["forward", "backward_fixed", "backward_residual_aligned", "fused_or_forward"]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
