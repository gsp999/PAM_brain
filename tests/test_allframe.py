import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pam_mosaic.backend.allframe import parse_factors, refine_all_frames
from pam_mosaic.backend.poses import Edge, pose_to_matrix, _lm_residual_jacobian
from pam_mosaic.preprocessing.inputs import InputState
from pam_mosaic.preprocessing.dataset import Record
from pam_mosaic.pipeline import run


class AllFrameTests(unittest.TestCase):
    def test_world_augmented_keeps_frontend_with_log_residual(self):
        frames = [np.full((32, 32), .02, np.float32)]*3
        baseline = np.zeros((3, 3))
        original = [Edge(0, 1, np.eye(3)[:2]), Edge(1, 2, np.eye(3)[:2])]
        def measured(k, neighbors, points, poses, reference):
            return (np.eye(3), np.eye(3)), {'kind': 'geometry'}
        with patch('pam_mosaic.backend.allframe.geometry_measurement', side_effect=measured), \
             patch('pam_mosaic.backend.allframe.optimize_poses', return_value=baseline.copy()) as solve:
            _, audit = refine_all_frames(frames, baseline, original, factors='geometry')
        self.assertTrue(solve.call_args.kwargs['lie_log'])
        self.assertEqual(len(solve.call_args.args[1]), 5)
        self.assertIs(solve.call_args.args[1][0], original[0])
        self.assertIs(solve.call_args.args[1][1], original[1])
        self.assertEqual(audit['optimization_components'], [[0, 1, 2]])
        self.assertEqual(audit['fixed_baseline_anchors'], [0])

    def test_options_and_disabled_exact_identity(self):
        self.assertEqual(parse_factors('none'), ())
        self.assertEqual(parse_factors('intensity,geometry'), ('geometry', 'intensity'))
        with self.assertRaises(ValueError):
            parse_factors('geometery')
        baseline = np.zeros((2, 3))
        result, audit = refine_all_frames([np.zeros((32, 32))]*2, baseline, [], factors='none')
        np.testing.assert_array_equal(result, baseline)
        self.assertIsNot(result, baseline)
        self.assertEqual(audit['extra_factors'], {})

    def test_every_original_edge_kept_and_current_excluded_from_map(self):
        baseline = np.zeros((3, 3))
        frames = [np.full((32, 32), .02, np.float32)]*3
        original = [Edge(0, 1, pose_to_matrix([0, 0, 0])[:2]),
                    Edge(1, 2, pose_to_matrix([0, 0, 0])[:2])]
        def measured(k, neighbors, points, poses, reference):
            self.assertNotIn(k, neighbors)
            return (np.eye(3), np.eye(3)), {'kind': 'geometry', 'accepted': True}
        with patch('pam_mosaic.backend.allframe.geometry_measurement', side_effect=measured), \
             patch('pam_mosaic.backend.allframe.optimize_poses', return_value=baseline.copy()) as solve:
            result, audit = refine_all_frames(frames, baseline, original, factors='geometry')
        supplied = solve.call_args.args[1]
        self.assertIs(supplied[0], original[0])
        self.assertIs(supplied[1], original[1])
        self.assertEqual(len(supplied), 5)
        self.assertEqual(audit['extra_factors'], {'geometry': 3})
        self.assertEqual(result.shape, baseline.shape)
        json.dumps(audit, allow_nan=False)

    def test_sequence_gaps_not_compressed(self):
        frames = [np.full((32, 32), .02, np.float32)]*3
        with patch('pam_mosaic.backend.allframe.geometry_measurement') as measurement:
            result, audit = refine_all_frames(frames, np.zeros((3, 3)),
                                             [Edge(0, 1, np.eye(3)[:2]), Edge(1, 2, np.eye(3)[:2])],
                                             factors='geometry', sequence_indices=[0, 40, 80])
        measurement.assert_not_called()
        self.assertFalse(audit['extra_factors'])

    def test_blank_frames_no_spurious_constraints(self):
        frames = [np.full((32, 32), .02, np.float32)]*3
        original = [Edge(0, 1, np.eye(3)[:2]), Edge(1, 2, np.eye(3)[:2])]
        result, audit = refine_all_frames(frames, np.zeros((3, 3)), original, factors='all')
        self.assertFalse(audit['extra_factors'])
        self.assertTrue(np.isfinite(result).all())

    def test_information_jacobian(self):
        positions = np.array([[0., 0., 0.], [10., 3., .1]])
        edge = Edge(0, 1, pose_to_matrix([9., 2., .09])[:2], .35, 'geometry', np.diag([.3, .5, .8]))
        residual, jac = _lm_residual_jacobian(positions, [edge])
        for column in range(3):
            shifted = positions.copy()
            shifted[1, column] += 1e-6
            numeric = (_lm_residual_jacobian(shifted, [edge])[0]-residual)/1e-6
            np.testing.assert_allclose(jac[:, column].toarray().ravel(), numeric, atol=1e-5)

    def test_loops_are_validated_and_do_not_duplicate_original_pairs(self):
        frames = [np.full((100, 100), .1, np.float32)]*3
        poses = np.zeros((3, 3))
        original = [Edge(0, 1, np.eye(3)[:2]), Edge(1, 2, np.eye(3)[:2])]
        points = np.array([[10., 10.], [90., 10.], [10., 90.], [90., 90.]])
        match = SimpleNamespace(valid=True, inliers=40, median_error=1., matrix=np.eye(3)[:2])
        for correlation, expected in [(.8, {'loop': 1}), (.1, {})]:
            with patch('pam_mosaic.backend.allframe.appearance_vector', return_value=np.array([1., 0.])), \
                 patch('pam_mosaic.backend.allframe.register', return_value=match), \
                 patch('pam_mosaic.backend.allframe.sift_inliers', return_value=(points, points, np.eye(3)[:2])), \
                 patch('pam_mosaic.backend.allframe._warped_correlation', return_value=(.9, correlation)):
                _, audit = refine_all_frames(frames, poses, original, factors='loop', sequence_indices=[0, 40, 80])
            self.assertEqual(audit['extra_factors'], expected)
            if expected:
                self.assertEqual([(e['reference'], e['moving']) for e in audit['extra_edges']], [(0, 2)])

    def test_pipeline_factors_independent_of_background(self):
        adapter = ModuleType('test_sensor')
        adapter.DISPLAY_RANGE = (0., .25)
        adapter.UNCORRECTED_DISPLAY_RANGE = (.02, .25)
        adapter.prepare = lambda paths: InputState()
        adapter.options = lambda enabled, excess: {}
        frame = np.full((32, 32), .04, np.float32)
        adapter.load_frame = lambda path, state, options: (frame.copy(), {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root/'1.dat'
            raw.touch()
            with patch('pam_mosaic.pipeline.get_format', return_value=adapter), \
                 patch('pam_mosaic.pipeline.correct_background', side_effect=AssertionError('must stay OFF')):
                report = run([Record(1, 1, raw)], root/'result', input_format='test_sensor',
                             background_correction=False, backend_factors='all')
            self.assertFalse(report['background_correction'])
            self.assertEqual(report['backend_mode'], 'world_augmented_log_v2')
            self.assertEqual(report['intensity_range'], [.02, .25])
            np.testing.assert_array_equal(np.load(root/'result/.run_cache/frame_0000.npy'), frame)
            self.assertTrue((root/'result/component_001/backend_factors.json').exists())
            np.testing.assert_array_equal(np.load(root/'result/component_001/poses_baseline_xytheta.npy'),
                                          np.load(root/'result/component_001/poses_xytheta.npy'))


if __name__ == '__main__':
    unittest.main()
