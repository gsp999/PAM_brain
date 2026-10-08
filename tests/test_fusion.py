import tempfile
import unittest
from pathlib import Path
import numpy as np

from pam_mosaic.backend.fusion import fuse_tiles


class FusionTests(unittest.TestCase):
    def test_unobserved_tile_pixels_do_not_darken_overlap(self):
        first = np.full((20, 20), .1, dtype=np.float32)
        second = np.zeros((20, 20), dtype=np.float32)
        second[:, 10:] = .2
        weights = [np.ones((20, 20), np.float32), np.pad(np.ones((20, 10), np.float32), ((0, 0), (10, 0)))]
        poses = np.zeros((2, 3))
        with tempfile.TemporaryDirectory() as directory:
            fuse_tiles([first, second], weights, poses, Path(directory))
            result = np.load(Path(directory) / "mosaic_float64.npy")
            self.assertAlmostEqual(float(result[5, 5]), .1, places=6)
            self.assertAlmostEqual(float(result[5, 15]), .15, places=6)


if __name__ == "__main__":
    unittest.main()
