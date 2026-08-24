import unittest

import numpy as np

from scripts.extract_stage2_multimodal_v17 import sample_indices, window_ranges


class Stage2MultimodalExtractionTests(unittest.TestCase):
    def test_full_and_tail_windows(self):
        self.assertEqual(window_ranges(100), [(0, 32), (32, 64), (64, 96), (96, 100)])

    def test_tiny_tail_is_dropped(self):
        self.assertEqual(window_ranges(98), [(0, 32), (32, 64), (64, 96)])

    def test_short_valid_clip_is_one_window(self):
        self.assertEqual(window_ranges(4), [(0, 4)])
        self.assertEqual(window_ranges(3), [])

    def test_temporal_sampling_is_bounded_and_deterministic(self):
        value = sample_indices(7, 16)
        self.assertEqual(value.shape, (16,))
        self.assertEqual(int(value[0]), 0)
        self.assertEqual(int(value[-1]), 6)
        self.assertTrue(np.all(value[1:] >= value[:-1]))


if __name__ == "__main__":
    unittest.main()
