import unittest
import cv2
import numpy as np

from pam_mosaic.registration import (register, register_translation_fallback,
                                     register_enhanced_adjacent,
                                     register_weak_signal_adjacent,
                                     select_supported_motion)


class RegistrationTests(unittest.TestCase):
    def test_low_texture_adjacent_translation_fallback(self):
        image = np.zeros((500, 500), np.float32)
        cv2.circle(image, (240, 220), 90, .18, -1)
        image = cv2.GaussianBlur(image, (0, 0), 16)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, -3]]),
                               (500, 500))
        result = register_translation_fallback(image, moved)
        self.assertTrue(result.valid)
        self.assertTrue(np.allclose(result.matrix[:, 2], [-4, 3], atol=.5))
    def test_lm_recovers_same_known_translation(self):
        rng = np.random.default_rng(11)
        image = np.zeros((300, 300), dtype=np.float32)
        for _ in range(120):
            x, y = rng.integers(20, 280, size=2)
            cv2.circle(image, (int(x), int(y)), 3, float(rng.uniform(.08, .25)), -1)
        moving = cv2.warpAffine(image, np.float32([[1, 0, 11], [0, 1, -7]]), (300, 300))
        result = register(image, moving)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.matrix[0, 2], -11, delta=2)
        self.assertAlmostEqual(result.matrix[1, 2], 7, delta=2)
        self.assertAlmostEqual(np.linalg.det(result.matrix[:, :2]), 1, delta=.01)

    def test_recovery_does_not_match_blank_frames(self):
        blank = np.full((500, 500), .04, np.float32)
        self.assertFalse(register_enhanced_adjacent(blank, blank).valid)
        self.assertFalse(register_weak_signal_adjacent(blank, blank).valid)

    def test_motion_selection_keeps_a_supported_rotation(self):
        rng = np.random.default_rng(19)
        image = np.zeros((400, 400), np.float32)
        for _ in range(180):
            x, y = rng.integers(30, 370, size=2)
            cv2.circle(image, (int(x), int(y)), 3,
                       float(rng.uniform(.08, .25)), -1)
        matrix = cv2.getRotationMatrix2D((200, 200), 3.0, 1.0)
        matrix[:, 2] += [7, -5]
        moving = cv2.warpAffine(image, matrix, (400, 400))
        result = select_supported_motion(image, moving)
        self.assertTrue(result.match.valid)
        self.assertEqual(result.model, "rigid")

    def test_motion_selection_uses_translation_without_rotation_evidence(self):
        rng = np.random.default_rng(23)
        image = np.zeros((400, 400), np.float32)
        for _ in range(180):
            x, y = rng.integers(30, 370, size=2)
            cv2.circle(image, (int(x), int(y)), 3,
                       float(rng.uniform(.08, .25)), -1)
        moving = cv2.warpAffine(image, np.float32([[1, 0, 8], [0, 1, -6]]),
                                (400, 400))
        result = select_supported_motion(image, moving)
        self.assertTrue(result.match.valid)
        self.assertEqual(result.model, "translation")


if __name__ == "__main__":
    unittest.main()
