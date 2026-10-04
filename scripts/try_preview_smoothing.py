"""Non-destructive comparison of modest smoothing on recovered MAP images."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pam_mosaic.dataset import index_dataset
from pam_mosaic.denoise import remove_isolated_bright_points
from pam_mosaic.reconstruction import reconstruct
from pam_mosaic.registration import registration_view
from pam_mosaic.roundtrip import calibrate, fuse_if_aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("PAM_Skull_FYH_20260929"))
    parser.add_argument("--output", type=Path, default=Path("outputs/image_enhancement_trial"))
    args = parser.parse_args()
    records = index_dataset(args.data)
    paths = {record.index: record.path for record in records}
    calibration = calibrate([record.path for record in records])
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in (8, 500):
        image, _ = fuse_if_aligned(*reconstruct(paths[index], calibration))
        speckle, changed = remove_isolated_bright_points(image)
        bilateral = cv2.bilateralFilter(speckle, 5, .012, 1.5)
        gaussian = cv2.GaussianBlur(speckle, (3, 3), .7)
        variants = [image, speckle, bilateral, gaussian]
        row = np.hstack([registration_view(variant) for variant in variants])
        cv2.putText(row, str(index), (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        rows.append(row)
        np.savez_compressed(args.output / f"{index}_float32.npz",
                            original=image, isolated_only=speckle,
                            bilateral=bilateral, gaussian=gaussian)
        background = image < .06
        print(index, "removed", changed,
              "background_std", [float(x[background].std()) for x in variants])
    cv2.imwrite(str(args.output / "comparison.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
