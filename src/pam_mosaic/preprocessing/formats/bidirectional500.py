"""Adapter for the original 500x1000 bidirectional acquisitions."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ..dataset import DEFAULT_FILENAME_REGEX
from ..reconstruction import reconstruct
from ..roundtrip import calibrate, fuse_if_aligned
from ..inputs import InputState

FORMAT_NAME = 'bidirectional500'
FILENAME_REGEX = DEFAULT_FILENAME_REGEX
DISPLAY_RANGE = (0.0, .25)
UNCORRECTED_DISPLAY_RANGE = (.02, .25)


def prepare(paths: list[Path]) -> InputState:
    calibration = calibrate(paths)
    return InputState(context=calibration, cache_config={'calibration': asdict(calibration)},
                      report={'validated_roundtrip': True,
                              'reverse_even_x': calibration.reverse_even_x,
                              'backward_calibration_matches': 0,
                              'backward_rigid_matrix': calibration.backward_matrix.tolist()
                              if calibration.backward_matrix is not None else None})


def options(suppress_speckles: bool, speckle_excess: float) -> dict:
    return {'suppress_speckles': suppress_speckles, 'speckle_parameters': None}


def load_frame(path: Path, state: InputState, configuration: dict) -> tuple[np.ndarray, dict]:
    forward, backward = reconstruct(path, state.context)
    image, decision = fuse_if_aligned(forward, backward,
                                     suppress_bright_points=configuration['suppress_speckles'])
    return image, decision.__dict__
