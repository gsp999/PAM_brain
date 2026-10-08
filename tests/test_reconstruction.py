import tempfile
import unittest
from pathlib import Path
import numpy as np

from pam_mosaic.preprocessing.reconstruction import read_raw, split_raw, _resample_x


class ReconstructionTests(unittest.TestCase):
    def test_matlab_column_order_and_bidirectional_split(self):
        raw = np.arange(500 * 1000, dtype="<f8").reshape(500, 1000, order="F")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.dat"
            with path.open("wb") as stream:
                stream.write(bytes(8))
                stream.write(raw.tobytes(order="F"))
            result = read_raw(path)
            self.assertTrue(np.array_equal(result, raw))
            forward, backward = split_raw(result)
            self.assertEqual(forward.shape, (500, 500))
            self.assertEqual(backward.shape, (500, 500))
            self.assertEqual(forward[1, 2], raw[2, 1])
            self.assertEqual(backward[1, 2], raw[2, 501])

    def test_zero_even_trajectory_matches_uncalibrated_resampling(self):
        frame = np.tile(np.linspace(0, 1, 500), (500, 1))
        self.assertTrue(np.allclose(_resample_x(frame),
                                    _resample_x(frame, np.zeros(500)), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
