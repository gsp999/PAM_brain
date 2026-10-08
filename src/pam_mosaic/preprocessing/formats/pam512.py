"""Read PAM acquisitions exactly as PAMreadtoMsCamPA.m selects MAP1."""
from pathlib import Path
import struct
import numpy as np
from ..denoise import remove_isolated_bright_points
from ..inputs import InputState

FORMAT_NAME = 'pam512'
FILENAME_REGEX = r"(?P<index>\d+)\.dat"
DISPLAY_RANGE = (0.0, 0.06)


def read_frame(path: Path) -> np.ndarray:
    if path.stat().st_size != 8 + 512 * 1024 * 8:
        raise ValueError(f"Unexpected PAM512 .dat size: {path}")
    with path.open('rb') as source:
        dimensions = struct.unpack('<ii', source.read(8))
    if dimensions != (1024, 512):
        raise ValueError(f"Unexpected PAM512 header {dimensions}: {path}")
    values = np.fromfile(path, dtype='<f8', offset=8)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite PAM512 data: {path}")
    # MATLAB: reshape(bbb(2:end), [512,1024]); MAP(:,512:1024).
    # MATLAB indexing is one-based and inclusive, hence 513 output columns.
    return values.reshape((512, 1024), order='F')[:, 511:1024].astype(np.float32)


def prepare(paths: list[Path]) -> InputState:
    return InputState()


def options(suppress_speckles: bool, speckle_excess: float) -> dict:
    return {'speckle_parameters': dict(excess=speckle_excess, max_area=4,
                                      max_extent=4, median_size=3)
            if suppress_speckles else None}


def load_frame(path: Path, state: InputState, configuration: dict) -> tuple[np.ndarray, dict]:
    image = read_frame(path)
    changed = 0
    parameters = configuration['speckle_parameters']
    if parameters is not None:
        image, changed = remove_isolated_bright_points(image, **parameters)
    return image, {'accepted': False, 'selected': 'matlab_map1', 'changed_pixels': changed}
