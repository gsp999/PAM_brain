import tempfile
import unittest
from pathlib import Path
import numpy as np
from pam_mosaic.preprocessing.formats.throat import read_frame, SAMPLES, trajectory


class ThroatTests(unittest.TestCase):
    def test_reconstruction_matches_sequential_matlab_reference(self):
        rng = np.random.default_rng(18)
        raw = rng.normal(size=2 * SAMPLES + 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'houlong201.dat'
            raw.astype('<f8').tofile(path)
            actual = read_frame(path)
            v = raw[:-1].copy()
            v[::142] = v[1::142]
            theta = np.linspace(0, 256*np.pi+.5*np.pi, SAMPLES*6)
            radius = theta/(256*np.pi+.5*np.pi)
            x = np.maximum(1, np.floor(radius*np.cos(theta)*128+128)).astype(int)-1
            y = np.maximum(1, np.floor(radius*np.sin(theta)*128+128)).astype(int)-1
            interp = np.interp(np.linspace(1,SAMPLES,SAMPLES*6),np.arange(1,SAMPLES+1),v[:SAMPLES])
            image = np.zeros((256,256))
            for row, col, value in zip(x,y,interp):
                image[row,col] = value
            padded = np.pad(image,((1,0),(1,0)))
            expected = np.empty_like(image)
            for row in range(256):
                for col in range(256):
                    expected[row,col] = np.median(padded[row:row+2,col:col+2])
            np.testing.assert_allclose(actual,expected,atol=2e-7)
            raw[SAMPLES:] = 999
            raw.astype('<f8').tofile(path)
            np.testing.assert_array_equal(read_frame(path),actual)
            path.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'size'):
                read_frame(path)
