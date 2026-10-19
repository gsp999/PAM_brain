import unittest
import numpy as np

from pam_mosaic.backend.poses import Edge, _lm_residual_jacobian, optimize_poses, pose_to_matrix


class PoseTests(unittest.TestCase):
    def test_log_jacobian_at_nonzero_and_small_errors(self):
        from scipy.linalg import logm
        for error in (0., 1e-7, .7, -1.2):
            poses = np.array([[0., 0., 0.], [8., -4., .07], [14., 3., .14+error]])
            z = pose_to_matrix([3., 2., .07])
            edge = Edge(1, 2, z[:2], .35, 'geometry', np.diag([.3, .5, .8]))
            residual, jac = _lm_residual_jacobian(poses, [edge], lie_log=True)
            delta = np.linalg.inv(z) @ np.linalg.inv(pose_to_matrix(poses[1])) @ pose_to_matrix(poses[2])
            logarithm = logm(delta).real
            expected = edge.sqrt_information @ (.35*np.array([logarithm[0, 2], logarithm[1, 2], 100*logarithm[1, 0]]))
            np.testing.assert_allclose(residual, expected, atol=1e-9)
            for column in range(6):
                plus, minus = poses.copy(), poses.copy()
                plus[1:].reshape(-1)[column] += 1e-6
                minus[1:].reshape(-1)[column] -= 1e-6
                numeric = (_lm_residual_jacobian(plus, [edge], lie_log=True)[0] -
                           _lm_residual_jacobian(minus, [edge], lie_log=True)[0])/2e-6
                np.testing.assert_allclose(jac[:, column].toarray().ravel(), numeric, atol=1e-6)

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
