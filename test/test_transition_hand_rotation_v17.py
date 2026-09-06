import unittest

import numpy as np

from active.v17.signing_voice_phrase_v17 import HAND_TREE_EDGES, stabilize_transition_hands


class TransitionHandRotationTest(unittest.TestCase):
    def test_opposite_hand_orientations_rotate_without_midpoint_snap(self):
        left = np.zeros((1, 61, 5), np.float32)
        left[:, :21, 3:] = 1
        for parent, child in HAND_TREE_EDGES:
            left[0, child, :2] = left[0, parent, :2] + [0., .05]
        right = left.copy(); right[:, :21, :2] *= -1
        transition = np.repeat(left, 5, axis=0)
        result = stabilize_transition_hands(transition, left, right)
        stream = np.concatenate((left, result, right))
        bones = stream[:, 1, :2] - stream[:, 0, :2]
        angles = np.unwrap(np.arctan2(bones[:, 1], bones[:, 0]))
        self.assertLess(float(np.abs(np.diff(angles)).max()), np.pi / 2)
        np.testing.assert_allclose(np.linalg.norm(bones, axis=-1), .05, atol=1e-6)
        self.assertTrue(np.all(result[:, 21:42] == 0))

    def test_uniform_hand_scale_proxy_is_not_articulated_finger_depth(self):
        left = np.zeros((1, 61, 5), np.float32)
        left[:, :21, 3:] = 1
        for parent, child in HAND_TREE_EDGES:
            left[0, child, :2] = left[0, parent, :2] + [.01, .05]
        right = left.copy(); right[:, :21, :2] *= 2
        transition = np.repeat(left, 5, axis=0)
        transition[:, :21, 2] = np.arange(5)[:, None] * .1
        result = stabilize_transition_hands(transition, left, right)
        np.testing.assert_allclose(result[:, :21, 2], np.broadcast_to(result[:, :1, 2], (5, 21)))
