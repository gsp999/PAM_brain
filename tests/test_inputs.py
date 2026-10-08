import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import numpy as np

from pam_mosaic.preprocessing.inputs import InputState, available_formats, get_format
from pam_mosaic.preprocessing.dataset import Record
from pam_mosaic.pipeline import run


class InputAdapterTests(unittest.TestCase):
    def test_builtin_discovery_and_unknown_format(self):
        self.assertEqual(set(available_formats()), {'bidirectional500', 'pam512'})
        self.assertIs(get_format('pam512'), available_formats()['pam512'])
        with self.assertRaisesRegex(ValueError, 'Unknown input format'):
            get_format('missing')

    def test_another_adapter_uses_shared_background_and_fusion(self):
        adapter = ModuleType('new_sensor')
        adapter.FORMAT_NAME = 'new_sensor'
        adapter.FILENAME_REGEX = r'(?P<index>\d+)\.dat'
        adapter.DISPLAY_RANGE = (0., .08)
        adapter.prepare = lambda paths: InputState(cache_config={'sensor': 'new'})
        adapter.options = lambda enabled, excess: {'speckle_parameters': None}
        frame = np.full((20, 20), -.01, dtype=np.float32)
        frame[8:11, 5:15] += .03
        adapter.load_frame = lambda path, state, options: (frame.copy(), {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'1.dat'; path.touch()
            with patch('pam_mosaic.preprocessing.inputs.available_formats', return_value={'new_sensor': adapter}):
                report = run([Record(1, 1, path)], root/'result', input_format='new_sensor')
            self.assertEqual(report['intensity_range'], [0., .08])
            self.assertTrue(report['background_correction'])
            self.assertFalse(report['validated_roundtrip'])
            self.assertAlmostEqual(report['frame_decisions'][0]['background_baseline'], -.01, places=6)
            cached = np.load(root/'result/.run_cache/frame_0000.npy')
            np.testing.assert_allclose(cached, frame+.01, atol=1e-7)
