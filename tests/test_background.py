import unittest
import numpy as np
from pam_mosaic.preprocessing.background import correct_background


class BackgroundTests(unittest.TestCase):
    def test_baseline_offset_preserves_contrast_and_negative_noise(self):
        rng = np.random.default_rng(13)
        source = rng.normal(-.01, .0008, (200, 200)).astype(np.float32)
        source[80:85, 20:180] += .03
        before = source.copy()
        result, baseline = correct_background(source)
        self.assertAlmostEqual(baseline, -.01, delta=.0003)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_allclose(result - source, -baseline, atol=2e-9)
        self.assertTrue(np.any(result < 0))
        self.assertGreater(float(np.mean(result[80:85, 20:180])), .025)

    def test_constant_and_invalid_input(self):
        a, b = correct_background(np.full((5, 5), -.012, np.float32))
        np.testing.assert_array_equal(a, np.zeros_like(a))
        self.assertAlmostEqual(b, -.012, places=7)
        with self.assertRaises(ValueError):
            correct_background(np.array([[np.nan]]))
