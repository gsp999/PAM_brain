"""Atomic, configuration-bound checkpoints for interrupted full-data runs."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import scipy


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


class Checkpoint:
    def __init__(self, output, records, configuration):
        source = Path(__file__).parent
        description = {'configuration': configuration,
            'inputs': [(r.index, str(r.path.resolve()), r.path.stat().st_size, r.path.stat().st_mtime_ns) for r in records],
            'sources': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob('*.py'))},
            'runtime': [np.__version__, scipy.__version__, cv2.__version__, cv2.getNumThreads()]}
        fingerprint = hashlib.sha256(json.dumps(description, sort_keys=True,
            default=lambda value: value.tolist()).encode()).hexdigest()
        self.root = output/'.run_cache'
        self.root.mkdir(exist_ok=True)
        marker = self.root/'manifest.json'
        if marker.exists() and json.loads(marker.read_text())['fingerprint'] != fingerprint:
            raise ValueError('Checkpoint belongs to different data/code/configuration; choose a new output directory')
        if not marker.exists():
            write_json(marker, {'fingerprint': fingerprint})
        self.frames = self.read('frames.json', [])

    def read(self, name, default):
        path = self.root/name
        return json.loads(path.read_text()) if path.exists() else default

    def frame(self, number):
        return np.load(self.root/f'frame_{number:04d}.npy'), self.frames[number]

    def save_frame(self, number, image, decision):
        np.save(self.root/f'frame_{number:04d}.npy', image)
        self.frames.append(decision)
        if len(self.frames)%100 == 0:
            self.flush_frames()

    def flush_frames(self):
        write_json(self.root/'frames.json', self.frames)

    def save_pairs(self, candidates, details):
        write_json(self.root/'pairs.json', {'candidates': candidates, 'details': details})
