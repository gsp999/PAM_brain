import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pam_mosaic.poses import Edge
from pam_mosaic.registration import Match
from pam_mosaic.tile_stitch import _consistent_pose_graph, stitch


def translation(x: float) -> np.ndarray:
    return np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0]])


class TileStitchTests(unittest.TestCase):
    def test_tile_refinement_is_enabled_by_default_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            components = []
            for index in range(2):
                component = root / f'component_{index}'
                component.mkdir()
                np.save(component / 'mosaic_float64.npy', np.ones((8, 8)))
                np.save(component / 'weight_map.npy', np.ones((8, 8)))
                components.append({'frames': 50, 'path': str(component)})
            (root / 'report.json').write_text(json.dumps({'components': components}))
            match = Match(translation(0), 25, 0.0, 1.0, True)
            with patch('pam_mosaic.tile_stitch.register', return_value=match) as mocked:
                report = stitch(root, root / 'with_edges')
                self.assertTrue(report['refine_geometry'])
                self.assertTrue(mocked.call_args.kwargs['refine_geometry'])
                report = stitch(root, root / 'without_edges', refine_geometry=False)
                self.assertFalse(report['refine_geometry'])
                self.assertFalse(mocked.call_args.kwargs['refine_geometry'])

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
