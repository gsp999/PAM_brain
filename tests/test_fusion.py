import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2

from pam_mosaic.backend.fusion import fuse_tiles


class FusionTests(unittest.TestCase):
    def test_signed_window_does_not_light_unobserved_canvas(self):
        frames=[np.full((4,4),-.01,np.float32) for _ in range(2)]
        weights=[np.ones((4,4),np.float32) for _ in range(2)]
        poses=np.array([[0,0,0],[8,0,0]],dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            fuse_tiles(frames,weights,poses,output,display_lo=-.02,display_hi=.02)
            preview=cv2.imread(str(output/'mosaic_preview.png'))
            self.assertTrue(np.all(preview[:,4:8]==0))
            self.assertTrue(np.any(preview[:,:4]>0))
            np.testing.assert_allclose(np.load(output/'mosaic_float64.npy')[:,:4],-.01)

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
