import unittest

import numpy as np

from pam_mosaic.poses import Edge
from pam_mosaic.tile_stitch import _consistent_pose_graph


def translation(x: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0]])


class TileStitchTests(unittest.TestCase):
    def test_inconsistent_loop_is_rejected(self):
        edges = [Edge(0, 1, translation(10)), Edge(1, 2, translation(10)),
                 Edge(0, 2, translation(30))]
        scores = {(0, 1): 10.0, (1, 2): 9.0, (0, 2): 1.0}
        selected, audit = _consistent_pose_graph(3, edges, scores)
        self.assertEqual(len(selected), 2)
        self.assertEqual(audit[(0, 2)]["pose_graph_use"], "rejected_loop")

    def test_consistent_loop_is_retained(self):
        edges = [Edge(0, 1, translation(10)), Edge(1, 2, translation(10)),
                 Edge(0, 2, translation(21))]
        scores = {(0, 1): 10.0, (1, 2): 9.0, (0, 2): 1.0}
        selected, audit = _consistent_pose_graph(3, edges, scores)
        self.assertEqual(len(selected), 3)
        self.assertEqual(audit[(0, 2)]["pose_graph_use"], "consistent_loop")


if __name__ == "__main__":
    unittest.main()
