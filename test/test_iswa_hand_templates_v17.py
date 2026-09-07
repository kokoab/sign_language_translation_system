import unittest

import numpy as np

from active.v17.avatar_rig_v17 import HAND_BONE_LENGTHS, HAND_EDGES
from scripts.build_iswa_hand_templates_v17 import normalize_template, metric_oriented_template


class IswaHandTemplatesV17Test(unittest.TestCase):
    def test_normalize_template_removes_rigid_pose_and_sets_metric_bones(self):
        source = np.zeros((21, 3), np.float32)
        source[0] = [1, 2, 3]
        for edge, (parent, child) in enumerate(HAND_EDGES):
            direction = np.array([.15 * ((child % 4) - 1), 1., .1 * (edge % 3)], np.float32)
            source[child] = source[parent] + direction / np.linalg.norm(direction) * (.01 + edge * .001)
        # Give the palm a non-degenerate index-to-pinky span.
        source[5] += [.03, 0, 0]
        source[17] += [-.03, 0, 0]
        result = normalize_template(source)
        np.testing.assert_allclose(result[0], 0, atol=1e-7)
        np.testing.assert_allclose(
            [np.linalg.norm(result[child] - result[parent]) for parent, child in HAND_EDGES],
            HAND_BONE_LENGTHS, atol=1e-6)
        self.assertGreater(result[5, 0], result[17, 0])
        self.assertGreater(result[9, 1], 0)
        with self.assertRaises(ValueError):
            normalize_template(np.zeros((21, 3), np.float32))

    def test_oriented_template_undoes_detection_rotation(self):
        source = np.zeros((21,3),np.float32)
        for index,(parent,child) in enumerate(HAND_EDGES):
            direction=np.array([.2*((child%4)-1),1,.15*(index%2)],np.float32)
            source[child]=source[parent]+direction/np.linalg.norm(direction)*(.02+index*.001)
        source[5] += [.03,0,0]; source[17] -= [.03,0,0]
        angle=np.deg2rad(90);transform=np.array([[np.cos(angle),np.sin(angle)],
                                                [-np.sin(angle),np.cos(angle)]])
        rotated=source.copy();rotated[:,:2]=source[:,:2]@transform.T
        np.testing.assert_allclose(metric_oriented_template(rotated,90),
                                   metric_oriented_template(source,0),atol=1e-6)


if __name__ == '__main__':
    unittest.main()
