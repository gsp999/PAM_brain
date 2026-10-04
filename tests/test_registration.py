import unittest
import cv2
import numpy as np

from pam_mosaic.registration import register


class RegistrationTests(unittest.TestCase):
    def test_lm_recovers_same_known_translation(self):
        rng = np.random.default_rng(11)
        image = np.zeros((300, 300), dtype=np.float32)
        for _ in range(120):
            x, y = rng.integers(20, 280, size=2)
            cv2.circle(image, (int(x), int(y)), 3, float(rng.uniform(.08, .25)), -1)
        moving = cv2.warpAffine(image, np.float32([[1, 0, 11], [0, 1, -7]]), (300, 300))
        result = register(image, moving)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.matrix[0, 2], -11, delta=2)
        self.assertAlmostEqual(result.matrix[1, 2], 7, delta=2)
        self.assertAlmostEqual(np.linalg.det(result.matrix[:, :2]), 1, delta=.01)


if __name__ == "__main__":
    unittest.main()
