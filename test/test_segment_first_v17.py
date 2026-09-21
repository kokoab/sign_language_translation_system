import unittest

import numpy as np

from active.v17.segment_first_v17 import (
    END,
    IGNORE,
    OUTSIDE,
    SIGNING,
    START,
    SegmentCandidate,
    frame_targets,
    merge_candidates,
)


class SegmentFirstV17Test(unittest.TestCase):
    def test_frame_targets_mask_questionable_annotations(self):
        times = np.linspace(0.0, 1.0, 11)
        labels = frame_targets(
            times,
            accepted=[(0.2, 0.6)],
            excluded=[(0.75, 0.9)],
            negative_eligible=True,
        )
        self.assertEqual(labels[0], OUTSIDE)
        self.assertEqual(labels[2], START)
        self.assertTrue(np.all(labels[3:6] == SIGNING))
        self.assertEqual(labels[6], END)
        self.assertTrue(np.all(labels[8:10] == IGNORE))

    def test_positive_only_source_never_creates_outside_targets(self):
        labels = frame_targets(
            np.linspace(0.0, 1.0, 11),
            accepted=[(0.2, 0.6)],
            excluded=[],
            negative_eligible=False,
        )
        self.assertNotIn(OUTSIDE, labels.tolist())

    def test_same_boundary_merges_but_real_repeat_survives(self):
        held = merge_candidates([
            SegmentCandidate(0.20, 0.80, 0.7),
            SegmentCandidate(0.23, 0.84, 0.9),
            SegmentCandidate(0.19, 0.81, 0.8),
        ])
        repeated = merge_candidates([
            SegmentCandidate(0.20, 0.80, 0.9),
            SegmentCandidate(1.10, 1.70, 0.8),
        ])
        self.assertEqual(len(held), 1)
        self.assertEqual(len(repeated), 2)
        self.assertAlmostEqual(held[0].end_seconds, 0.84)


if __name__ == "__main__":
    unittest.main()
