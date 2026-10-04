"""Read each MATLAB file and recover its two bidirectional MAP frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import cv2


SIZE = 500
POINTS = SIZE * SIZE * 2


@dataclass(frozen=True)
class Calibration:
    reverse_even_x: bool = False
    # This rigid transform maps the backward frame into forward coordinates.
    backward_matrix: np.ndarray | None = None


def read_raw(path: Path) -> np.ndarray:
    if path.stat().st_size != 8 + POINTS * 8:
        raise ValueError(f"Unexpected .dat size: {path}")
    values = np.fromfile(path, dtype="<f8", count=POINTS, offset=8)
    if values.size != POINTS or not np.isfinite(values).all():
        raise ValueError(f"Incomplete or nonfinite data: {path}")
    return values.reshape((SIZE, 2 * SIZE), order="F")


def split_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.shape != (SIZE, 2 * SIZE):
        raise ValueError(f"Expected (500, 1000), got {raw.shape}")
    return raw[:, :SIZE].T.copy(), raw[:, SIZE:].T.copy()


def _resample_x(frame: np.ndarray) -> np.ndarray:
    # Matches MATLAB's cos sampling grid and uniform output grid.
    x_sine = np.cos(np.linspace(0, np.pi, SIZE))
    x_uniform = np.linspace(1, -1, SIZE)
    spline = CubicSpline(x_sine[::-1], frame[:, ::-1], axis=1)
    return spline(x_uniform).astype(np.float32)


def reconstruct(path: Path, calibration: Calibration = Calibration()) -> tuple[np.ndarray, np.ndarray]:
    forward, backward = split_raw(read_raw(path))
    frames = []
    for frame in (forward, backward):
        if calibration.reverse_even_x:
            frame[1::2] = frame[1::2, ::-1]
        frames.append(_resample_x(frame))
    frames[1] = frames[1][::-1].copy()
    if calibration.backward_matrix is not None:
        frames[1] = cv2.warpAffine(
            frames[1], calibration.backward_matrix.astype(np.float32),
            (SIZE, SIZE), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return frames[0], frames[1]


def choose_even_direction(paths: list[Path]) -> bool:
    """Choose the fixed even-line orientation from several files."""
    scores = [[], []]
    for path in paths:
        for frame in split_raw(read_raw(path)):
            for reverse, target in ((False, scores[0]), (True, scores[1])):
                even = frame[1:-1:2, ::-1] if reverse else frame[1:-1:2]
                odd = (frame[0:-2:2] + frame[2::2]) / 2
                a, b = odd[:, 50:450].ravel(), even[:, 50:450].ravel()
                target.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.median(scores[1])) > float(np.median(scores[0]))
