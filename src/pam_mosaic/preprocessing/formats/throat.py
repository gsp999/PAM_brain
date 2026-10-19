"""Endoscopic spiral reconstruction matching throat/rec.m (first sweep)."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from ..inputs import InputState

FORMAT_NAME = 'throat'
FILENAME_REGEX = r'houlong(?P<index>\d+)\.dat'
DISPLAY_RANGE = (0.0, 0.1)
SAMPLES = 142 * 256
AMPLIFICATION = 6


@lru_cache(maxsize=1)
def trajectory() -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 256 * np.pi + .5 * np.pi, SAMPLES * AMPLIFICATION)
    radius = theta / (256 * np.pi + .5 * np.pi)
    x = np.maximum(1, np.floor(radius * np.cos(theta) * 128 + 128)).astype(int) - 1
    y = np.maximum(1, np.floor(radius * np.sin(theta) * 128 + 128)).astype(int) - 1
    pixels = x * 256 + y
    # MATLAB's sequential assignments keep the last visit to each pixel.
    _, reverse_indices = np.unique(pixels[::-1], return_index=True)
    last = len(pixels) - 1 - reverse_indices
    return pixels[last], last


def read_frame(path: Path) -> np.ndarray:
    if path.stat().st_size != (2 * SAMPLES + 1) * 8:
        raise ValueError(f'Unexpected throat .dat size: {path}')
    raw = np.fromfile(path, dtype='<f8')[:-1]
    if not np.isfinite(raw).all():
        raise ValueError(f'Nonfinite throat data: {path}')
    raw[::142] = raw[1::142]
    values = np.interp(np.linspace(0, SAMPLES - 1, SAMPLES * AMPLIFICATION),
                       np.arange(SAMPLES), raw[:SAMPLES])
    pixels, last = trajectory()
    image = np.zeros((256, 256), dtype=np.float64)
    image.ravel()[pixels] = values[last]
    # medfilt2([2 2]): zero padding, window above/left, mean of middle two.
    padded = np.pad(image, ((1, 0), (1, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (2, 2))
    return np.median(windows, axis=(-2, -1)).astype(np.float32)


def prepare(paths: list[Path]) -> InputState:
    return InputState(cache_config={'reconstruction': 'rec_m_first_sweep_v1'},
                      report={'reconstruction': 'rec_m_first_sweep_v1',
                              'sweeps_per_file': 2, 'selected_sweep': 1,
                              'median_window': [2, 2], 'interpolation_factor': 6})


def options(suppress_speckles: bool, speckle_excess: float) -> dict:
    # rec.m already specifies median filtering; no extra PAM512 outlier filter.
    return {'speckle_parameters': None}


def load_frame(path: Path, state: InputState, configuration: dict) -> tuple[np.ndarray, dict]:
    return read_frame(path), {'selected': 'matlab_spiral_first_sweep', 'changed_pixels': 0}
