import unittest
import numpy as np
from pam_mosaic.preprocessing.background import correct_background


class BackgroundTests(unittest.TestCase):
    def test_offset_and_signed_contrast(self):
        source = np.random.default_rng(13).normal(-.01, .0008, (200,200)).astype(np.float32)
        source[80:85,20:180] += .03
        before = source.copy()
        result, baseline = correct_background(source)
        self.assertAlmostEqual(baseline, -.01, delta=.0003)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_allclose(result-source, -baseline, atol=2e-9)
        self.assertTrue(np.any(result<0))
        other, _ = correct_background(source+.006)
        np.testing.assert_allclose(result, other, atol=1e-5)

    def test_constant_and_invalid_input(self):
        result, baseline = correct_background(np.full((5,5), -.012, np.float32))
        self.assertAlmostEqual(baseline, -.012, places=7)
        np.testing.assert_array_equal(result, np.zeros((5,5)))
        for value in [np.array([[np.nan]]), np.empty((0,0)), np.ones(4)]:
            with self.assertRaises(ValueError):
                correct_background(value)
