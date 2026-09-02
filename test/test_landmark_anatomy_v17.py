import unittest

import numpy as np

from active.v17.landmark_anatomy_v17 import (
    anatomy_coverage,
    build_anatomy_template,
    complete_landmark_anatomy,
)


class LandmarkAnatomyV17Tests(unittest.TestCase):
    def setUp(self):
        self.pool = np.zeros((2, 32, 61, 5), np.float32)
        self.pool[..., 3:] = 1
        self.pool[..., 0] = np.arange(61)[None, None]
        self.pool[..., 1] = np.arange(32)[None, :, None] / 32
        self.template = build_anatomy_template(self.pool)

    def test_completion_preserves_observations_without_inventing_second_hand(self):
        value = self.pool[0].copy()
        value[4:8, 5, :3] = 0
        value[4:8, 5, 3:] = 0
        value[:, 21:42] = 0
        completed, observed = complete_landmark_anatomy(value, self.template)
        self.assertTrue(np.array_equal(completed[observed, :3], value[observed, :3]))
        self.assertTrue((completed[:, 21:42] == 0).all())
        self.assertTrue((completed[..., 42:, 3] == 1).all())
        self.assertTrue((completed[4:8, 5, 3] == 1).all())
        self.assertTrue(np.isfinite(completed).all())

    def test_participating_hand_only_bridges_short_visibility_gap(self):
        value = self.pool[0].copy()
        value[:, 21:42] = 0
        value[10:12, :21] = 0
        value[20:25, :21] = 0
        completed, _ = complete_landmark_anatomy(value, self.template)
        self.assertTrue((completed[10:12, :21, 3] == 1).all())
        self.assertTrue((completed[20:25, :21] == 0).all())
        self.assertTrue((completed[:, 21:42] == 0).all())

    def test_coverage_rewards_complete_real_detection(self):
        sparse = self.pool[0].copy()
        sparse[:, 21:42] = 0
        sparse[:, 57:61] = 0
        self.assertGreater(anatomy_coverage(self.pool[0]), anatomy_coverage(sparse))


if __name__ == "__main__":
    unittest.main()
