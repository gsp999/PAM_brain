import unittest

import cv2
import numpy as np

from pam_mosaic.roundtrip import fuse_if_aligned


class RoundtripTests(unittest.TestCase):
    def test_misaligned_low_texture_keeps_forward(self):
        forward = np.zeros((500, 500), np.float32)
        backward = np.ones_like(forward) * .1
        result, decision = fuse_if_aligned(forward, backward)
        self.assertFalse(decision.accepted)
        self.assertTrue(np.array_equal(result, forward))

    def test_small_residual_can_fuse(self):
        rng = np.random.default_rng(4)
        forward = np.zeros((500, 500), np.float32)
        for _ in range(100):
            x, y = rng.integers(30, 470, size=2)
            cv2.circle(forward, (int(x), int(y)), 4, float(rng.uniform(.08, .22)), -1)
        backward = cv2.warpAffine(forward, np.float32([[1, 0, 2], [0, 1, -1]]),
                                  (500, 500))
        fused, decision = fuse_if_aligned(forward, backward)
        self.assertTrue(decision.accepted)
        self.assertEqual(fused.shape, forward.shape)


if __name__ == "__main__":
    unittest.main()
