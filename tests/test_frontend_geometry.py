import numpy as np
import unittest
from pam_mosaic.frontend.geometry import Lines, refine_lines
from pam_mosaic.poses import pose_to_matrix


def test_parallel_lines_are_degenerate():
    points = np.column_stack((np.arange(200), np.zeros(200)))
    lines = Lines(points, np.tile([0., 1.], (200, 1)))
    _, report = refine_lines(lines, lines, np.eye(2, 3))
    assert not report['accepted']
    assert report['reason'] == 'degenerate_lines'


def test_explicit_line_lm_recovers_known_rigid_transform():
    from pam_mosaic.frontend.geometry import transform_lines
    x = np.linspace(20, 400, 200)
    points = np.vstack((np.column_stack((x, np.full_like(x, 100))),
                        np.column_stack((np.full_like(x, 200), x))))
    normals = np.vstack((np.tile([0., 1.], (len(x), 1)), np.tile([1., 0.], (len(x), 1))))
    reference = Lines(points, normals)
    truth = pose_to_matrix([2., -1., .003])
    moving = transform_lines(reference, np.linalg.inv(truth))
    result, report = refine_lines(reference, moving, np.eye(2, 3))
    assert report['accepted']
    np.testing.assert_allclose(result, truth[:2], atol=.01)


def test_zero_appearance_neighbors_really_disables_retrieval():
    from pam_mosaic.graph import candidate_pairs
    frames = [np.ones((20, 20)) for _ in range(12)]
    assert all(b-a <= 5 for a, b in candidate_pairs(frames, appearance_neighbors=0))


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in globals().items() if name.startswith('test_'))
