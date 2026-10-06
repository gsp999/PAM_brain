import unittest
import numpy as np

from pam_mosaic.poses import Edge, _lm_residual_jacobian, optimize_poses, pose_to_matrix


class PoseTests(unittest.TestCase):
    def test_sparse_lm_jacobian_matches_finite_difference(self):
        poses = np.array([[0., 0., 0.], [8., -4., .07], [14., 3., -.04]])
        edges = [Edge(0, 1, pose_to_matrix([8, -4, .07])[:2]),
                 Edge(1, 2, (np.linalg.inv(pose_to_matrix(poses[1])) @ pose_to_matrix(poses[2]))[:2])]
        residual, jacobian = _lm_residual_jacobian(poses, edges)
        for column in range(6):
            perturbed = poses.copy()
            perturbed[1:].reshape(-1)[column] += 1e-6
            shifted, _ = _lm_residual_jacobian(perturbed, edges)
            numeric = (shifted - residual) / 1e-6
            np.testing.assert_allclose(jacobian[:, column].toarray().ravel(), numeric, atol=1e-3)

    def test_lm_reduces_loop_disagreement(self):
        truth = np.array([[0., 0., 0.], [10., 0., .02], [20., 1., .01]])
        edges = [Edge(0, 1, pose_to_matrix([10, 0, .02])[:2]),
                 Edge(1, 2, (np.linalg.inv(pose_to_matrix(truth[1])) @ pose_to_matrix(truth[2]))[:2]),
                 Edge(0, 2, pose_to_matrix([20, 1, .01])[:2])]
        start = truth.copy()
        start[2, :2] += [4, -3]
        result = optimize_poses(start, edges)
        np.testing.assert_allclose(result[2], truth[2], atol=1e-2)



if __name__ == "__main__":
    unittest.main()
