import unittest

import numpy as np

from scripts.audit_cokely_stage1_compatibility_v17 import segment_score


class CokelyCompatibilityAuditTests(unittest.TestCase):
    def test_segment_score_uses_only_samples_inside_annotation(self):
        scores = np.asarray([[1.0, 0.0], [0.0, 3.0], [0.0, 5.0]])
        positions = np.asarray([0.0, 5.0, 10.0])
        np.testing.assert_allclose(
            segment_score(scores, positions, 4.0, 11.0), [0.0, 4.0]
        )


if __name__ == '__main__':
    unittest.main()
