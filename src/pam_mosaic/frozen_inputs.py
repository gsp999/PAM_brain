"""Freeze numerical inputs and historical candidate policy before comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .dataset import index_dataset
from .reconstruction import Calibration, calibrate_even_trajectory, reconstruct
from .roundtrip import fuse_if_aligned


def freeze(data, reference, destination):
    if destination.exists():
        raise ValueError('Input cache already exists; do not silently overwrite it')
    cv2.setNumThreads(1)
    old = json.loads((reference/'report.json').read_text())
    records = index_dataset(data)
    # Keep the recorded backward matrix exactly, instead of estimating it again.
    reverse = old['reverse_even_x']
    shift = calibrate_even_trajectory([r.path for r in records[:5]], reverse)
    calibration = Calibration(reverse_even_x=reverse,
        backward_matrix=np.asarray(old['backward_rigid_matrix']), even_time_shift_samples=shift)
    destination.mkdir(parents=True)
    frames = []
    for n, record in enumerate(records, 1):
        cv2.setRNGSeed(record.index)
        frame, decision = fuse_if_aligned(*reconstruct(record.path, calibration), suppress_bright_points=True)
        path = destination/f'{record.index:04d}.npy'
        np.save(path, frame)
        frames.append({'index': record.index, 'scan': record.scan, 'path': str(path.resolve()),
                       'sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                       'p99': float(np.percentile(frame, 99)), 'decision': decision.__dict__})
        if n % 100 == 0:
            print(f'frozen {n}/{len(records)}', flush=True)
    manifest = {'frames': frames,
        'reference_report_sha256': hashlib.sha256((reference/'report.json').read_bytes()).hexdigest(),
        'calibration': {'reverse_even_x': bool(reverse), 'backward_matrix': calibration.backward_matrix.tolist(),
                        'even_time_shift_samples': shift.tolist()},
        'candidates': [{'reference': r['reference'], 'moving': r['moving']} for r in old['registrations']],
        'legacy_components': [{'file_indices': c['file_indices'], 'path': c['path']} for c in old['components']],
        'note': 'Frozen comparison inputs, not a claim of byte-identical reconstruction of the old output. Old even-time calibration was not recorded.'}
    (destination/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(f'complete: {len(frames)} frozen frames', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('PAM_Skull_FYH_20260929'))
    parser.add_argument('--reference', type=Path, default=Path('outputs/moderate_denoise_full'))
    parser.add_argument('--output', type=Path, default=Path('outputs/controlled_v2/inputs'))
    args = parser.parse_args()
    freeze(args.data, args.reference, args.output)
