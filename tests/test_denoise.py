import unittest

import numpy as np

from pam_mosaic.preprocessing.denoise import remove_isolated_bright_points


class DenoiseTests(unittest.TestCase):
    def test_isolated_bright_pixel_removed_but_thin_vessel_retained(self):
        image = np.full((60, 60), .04, np.float32)
        image[10, 10] = .20
        image[30, 15:45] = .20
        result, changed = remove_isolated_bright_points(image)
        self.assertEqual(changed, 1)
        self.assertAlmostEqual(float(result[10, 10]), .04, places=5)
        self.assertTrue(np.array_equal(result[30, 15:45], image[30, 15:45]))


if __name__ == "__main__":
    unittest.main()
