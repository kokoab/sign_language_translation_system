from types import SimpleNamespace
import unittest

import numpy as np

from active.v17.continuous_vision_v17 import CausalVisionFeatures


class ContinuousVisionTest(unittest.TestCase):
    def detection(self):
        return SimpleNamespace(body_xy=np.array([[.3, .4], [.7, .4], [.2, .6], [.8, .6]], np.float32),
                               body_confidence=np.ones(4, np.float32),
                               face_xy=np.zeros((15, 2), np.float32), face_confidence=np.zeros(15, np.float32))

    def test_missing_hands_still_advance_without_inventing_or_trimming(self):
        normalizer = CausalVisionFeatures()
        first = normalizer.add(self.detection(), {"left": None, "right": None}, 640, 480)
        self.assertEqual(first.shape, (61, 5))
        self.assertTrue(np.all(first[:42] == 0))
        self.assertEqual(normalizer.frames_seen, 1)
        hand = SimpleNamespace(xy=np.tile([.5, .3], (21, 1)), confidence=np.ones(21))
        second = normalizer.add(self.detection(), {"left": None, "right": hand}, 640, 480)
        self.assertTrue(np.all(second[:21] == 0))
        self.assertTrue(np.all(second[21:42, 3] == 1))
        self.assertAlmostEqual(float(second[21, 1]), -.1875, places=5)
        self.assertTrue(np.all(first[:42] == 0))  # later observation never changes emitted history

    def test_reset_discards_scale_history_and_dimension_change_requires_reset(self):
        normalizer = CausalVisionFeatures()
        normalizer.add(self.detection(), {"left": None, "right": None}, 640, 480)
        with self.assertRaises(ValueError):
            normalizer.add(self.detection(), {"left": None, "right": None}, 480, 640)
        normalizer.reset()
        normalizer.add(self.detection(), {"left": None, "right": None}, 480, 640)
        self.assertEqual(normalizer.frames_seen, 1)
