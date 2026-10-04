"""Read each MATLAB file and recover its two bidirectional MAP frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
import cv2


SIZE = 500
POINTS = SIZE * SIZE * 2


@dataclass(frozen=True)
class Calibration:
    reverse_even_x: bool = False
    # This rigid transform maps the backward frame into forward coordinates.
    backward_matrix: np.ndarray | None = None
    even_time_shift_samples: np.ndarray | None = None


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


def _resample_x(frame: np.ndarray, even_shift: np.ndarray | None = None) -> np.ndarray:
    # The MATLAB script uses distinct fixed odd/even trajectories when its
    # trajectory calibration is available.
    x_sine = np.cos(np.linspace(0, np.pi, SIZE))
    x_uniform = np.linspace(1, -1, SIZE)
    if even_shift is None:
        return CubicSpline(x_sine[::-1], frame[:, ::-1], axis=1)(x_uniform).astype(np.float32)
    shift = np.asarray(even_shift, dtype=float)
    if shift.shape != (SIZE,):
        raise ValueError("Even trajectory correction must have 500 samples")
    effective = np.arange(SIZE) + shift
    if np.any(np.diff(effective) <= 0) or effective[0] < 0 or effective[-1] > SIZE - 1:
        raise ValueError("Even trajectory must be strictly monotonic and in bounds")
    x_even = np.cos(effective * np.pi / (SIZE - 1))
    result = np.empty_like(frame, dtype=np.float32)
    result[0::2] = CubicSpline(x_sine[::-1], frame[0::2, ::-1], axis=1)(x_uniform)
    result[1::2] = CubicSpline(x_even[::-1], frame[1::2, ::-1], axis=1)(x_uniform)
    return result


def reconstruct(path: Path, calibration: Calibration = Calibration()) -> tuple[np.ndarray, np.ndarray]:
    forward, backward = split_raw(read_raw(path))
    frames = []
    for frame in (forward, backward):
        if calibration.reverse_even_x:
            frame[1::2] = frame[1::2, ::-1]
        frames.append(_resample_x(frame, calibration.even_time_shift_samples))
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


def calibrate_even_trajectory(paths: list[Path], reverse_even_x: bool) -> np.ndarray:
    """Estimate one fixed even-line time trajectory, following MATLAB's windows.

    Uses phase correlation in nine overlapping windows, robust medians and
    MAD rejection, PCHIP interpolation, and zero shifts at both turnaround
    points. Fails rather than silently applying an invalid trajectory.
    """
    centers = np.rint(np.linspace(.08, .92, 9) * (SIZE - 1)).astype(int)
    width = max(32, round(SIZE * .22))
    samples: list[list[float]] = [[] for _ in centers]
    for path in paths[:5]:
        for frame in split_raw(read_raw(path)):
            if reverse_even_x:
                frame[1::2] = frame[1::2, ::-1]
            odd, even = frame[0::2], frame[1::2]
            for slot, center in enumerate(centers):
                start = int(np.clip(center - width // 2, 0, SIZE - width))
                a = odd[:, start:start + width].astype(np.float32)
                b = even[:, start:start + width].astype(np.float32)
                a = (a - a.mean()) / max(float(a.std()), 1e-6)
                b = (b - b.mean()) / max(float(b.std()), 1e-6)
                (dx, _), response = cv2.phaseCorrelate(b, a)
                if np.isfinite(dx) and abs(dx) <= 10 and response > .05:
                    samples[slot].append(float(dx))
    valid_x, valid_shift = [0], [0.0]
    for center, values in zip(centers, samples):
        if not values:
            continue
        array = np.asarray(values)
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        if mad > 0:
            array = array[np.abs(array - median) <= 3 * 1.4826 * mad]
        if len(array):
            valid_x.append(int(center))
            valid_shift.append(float(np.median(array)))
    if len(valid_x) < 4:
        raise ValueError("Insufficient valid trajectory calibration windows")
    valid_x.append(SIZE - 1)
    valid_shift.append(0.0)
    shift = PchipInterpolator(valid_x, valid_shift)(np.arange(SIZE))
    shift[0] = shift[-1] = 0
    if np.any(np.abs(shift) > 10) or np.any(np.diff(np.arange(SIZE) + shift) <= 0):
        raise ValueError("Calibrated even trajectory is invalid")
    return shift.astype(np.float64)
