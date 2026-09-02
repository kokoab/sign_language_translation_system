import unittest

import numpy as np

from active.v17.train_stage_2_aligned_ctc_v17 import aligned_ctc_targets, output_bins
from active.v17.train_stage_2_v17 import collapse_ctc


class Stage2AlignedCTCV17Tests(unittest.TestCase):
    def test_tail_window_uses_its_actual_source_duration(self):
        bins = output_bins(41)
        self.assertEqual(len(bins), 16)
        self.assertEqual(bins[0], (0.0, 4.0))
        self.assertEqual(bins[8], (32.0, 33.125))
        self.assertEqual(bins[-1], (39.875, 41.0))

    def test_manual_sign_intervals_collapse_to_the_sequence(self):
        targets = aligned_ctc_targets(
            41,
            [(5.0, 15.0), (25.0, 36.0)],
            [10, 91],
        )
        self.assertEqual(collapse_ctc(targets), [10, 91])
        self.assertTrue(np.any(targets == 0))


if __name__ == "__main__":
    unittest.main()
