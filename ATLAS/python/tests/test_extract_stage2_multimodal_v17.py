import unittest

import numpy as np

from scripts.extract_stage2_multimodal_v17 import build_parser, sample_indices, window_ranges
from active.v17.schema_stage2_features_v17 import Stage2FeatureV17Config, schema_fingerprint


class Stage2MultimodalExtractionTests(unittest.TestCase):
    def test_candidate_verification_role_is_explicitly_selectable(self):
        args = build_parser().parse_args([
            '--role', 'candidate_verification', '--source', 'cokely_verified'
        ])
        self.assertEqual(args.role, 'candidate_verification')

    def test_full_and_tail_windows(self):
        self.assertEqual(window_ranges(100), [(0, 32), (32, 64), (64, 96), (96, 100)])

    def test_tiny_tail_is_dropped(self):
        self.assertEqual(window_ranges(98), [(0, 32), (32, 64), (64, 96)])

    def test_short_valid_clip_is_one_window(self):
        self.assertEqual(window_ranges(4), [(0, 4)])
        self.assertEqual(window_ranges(3), [])

    def test_overlapping_windows_cover_tail_without_partial_duplicates(self):
        self.assertEqual(
            window_ranges(50, window_stride=8),
            [(0, 32), (8, 40), (16, 48), (18, 50)],
        )

    def test_overlapping_short_clip_is_one_window(self):
        self.assertEqual(window_ranges(20, window_stride=8), [(0, 20)])

    def test_official_online_half_second_windows(self):
        self.assertEqual(
            window_ranges(35, window_size=16, window_stride=4),
            [(0, 16), (4, 20), (8, 24), (12, 28), (16, 32), (19, 35)],
        )

    def test_temporal_sampling_is_bounded_and_deterministic(self):
        value = sample_indices(7, 16)
        self.assertEqual(value.shape, (16,))
        self.assertEqual(int(value[0]), 0)
        self.assertEqual(int(value[-1]), 6)
        self.assertTrue(np.all(value[1:] >= value[:-1]))

    def test_overlap_has_distinct_schema_without_changing_locked_default(self):
        self.assertEqual(
            schema_fingerprint(Stage2FeatureV17Config()), "f2b206169c243a1d"
        )
        self.assertNotEqual(
            schema_fingerprint(Stage2FeatureV17Config(window_stride=8)),
            schema_fingerprint(Stage2FeatureV17Config()),
        )


if __name__ == "__main__":
    unittest.main()
