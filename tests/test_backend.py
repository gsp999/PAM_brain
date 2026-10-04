import unittest

import numpy as np

from pam_mosaic.backend import BackendConfig, select_keyframes
from pam_mosaic.poses import Edge, optimize_poses


class BackendTests(unittest.TestCase):
    def test_keyframe_constraint_also_moves_regular_frame(self):
        start = np.array([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.]])
        relative = np.array([[1., 0., 10.], [0., 1., 0.]])
        corrected = np.array([[1., 0., 18.], [0., 1., 0.]])
        edges = [Edge(0, 1, relative), Edge(1, 2, relative),
                 Edge(0, 2, corrected, .8, "intensity")]
        result = optimize_poses(start, edges)
        self.assertLess(result[1, 0], start[1, 0])
        self.assertLess(result[2, 0], start[2, 0])

    def test_keyframe_selector_does_not_discard_regular_frames(self):
        poses = np.array([[float(i), 0., 0.] for i in range(6)])
        self.assertEqual(select_keyframes(poses, BackendConfig(keyframe_interval=3)), [0, 3])
        self.assertEqual(len(poses), 6)


if __name__ == "__main__":
    unittest.main()
