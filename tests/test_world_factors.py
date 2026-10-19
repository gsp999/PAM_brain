import unittest
from unittest.mock import patch
import numpy as np
from pam_mosaic.backend.factors import geometry_measurement, intensity_measurement, transform
from pam_mosaic.backend.poses import pose_to_matrix


class WorldFactorTests(unittest.TestCase):
    def test_geometry_world_fit_independent_of_reference(self):
        line = np.arange(0., 200., 1.5)
        source = np.concatenate([np.column_stack((line, np.full_like(line, y))) for y in (0., 60., 120.)] +
                                [np.column_stack((np.full_like(line, x), line)) for x in (20., 100., 180.)])
        truth = pose_to_matrix([320., -140., .4])
        poses = {0: pose_to_matrix([300., -100., .2]),
                 1: pose_to_matrix([310., -130., -.1]),
                 2: pose_to_matrix([320.5, -140.4, .401])}
        points = {i: transform(transform(source, truth), np.linalg.inv(poses[i])) for i in (0, 1)}
        points[2] = source
        fits = []
        for reference in (0, 1):
            measured, audit = geometry_measurement(2, [0, 1], points, poses, reference)
            self.assertTrue(audit['accepted'])
            z, info = measured
            fit = poses[reference] @ z
            np.testing.assert_allclose(fit, truth, atol=.1)
            fits.append((fit, info))
        for index in (0, 1):
            np.testing.assert_allclose(fits[0][index], fits[1][index], atol=1e-9)

    def test_world_fit_independent_of_reference_and_relative_conversion(self):
        rng = np.random.default_rng(24)
        src = rng.uniform(0, 200, (150, 2))
        truth = pose_to_matrix([320., -140., .4])
        poses = {0: pose_to_matrix([300., -100., .2]),
                 1: pose_to_matrix([310., -130., -.1]),
                 2: pose_to_matrix([320.5, -140.4, .401])}
        frames = [0, 1, 2]
        def matches(reference, moving, **kwargs):
            dst = transform(transform(src, truth), np.linalg.inv(poses[reference]))
            return src, dst, np.eye(3)[:2]
        estimates = []
        with patch('pam_mosaic.backend.factors.sift_inliers', side_effect=matches):
            for reference in (0, 1):
                measurement, audit, _ = intensity_measurement(2, [0, 1], frames, frames, poses, reference)
                self.assertTrue(audit['accepted'])
                z, info = measurement
                np.testing.assert_allclose(poses[reference] @ z, truth, atol=1e-6)
                estimates.append(info)
        np.testing.assert_allclose(*estimates, atol=1e-10)
