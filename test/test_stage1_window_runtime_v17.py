from __future__ import annotations

import unittest

import numpy as np

from active.v17.stage1_window_v17 import (
    NO_EMIT,
    Stage1WindowTranscript,
    select_time_window,
    window_end_times,
    window_sample_times,
)
from scripts.live_reel_stage1_v17 import parser, validate_args


class Stage1WindowRuntimeTests(unittest.TestCase):
    def test_finish_rechecks_partial_tail_and_separates_capture_gaps(self):
        from scripts.live_reel_stage1_v17 import final_stage1_window_decode
        class Classifier:
            def __init__(self):
                self.ends = []
            def classify_raw_window(self, raw, timestamps, end):
                self.ends.append(end)
                return dict(gloss='A', end_seconds=end)
        classifier = Classifier()
        times = np.concatenate((np.arange(0, .81, .05), np.arange(2, 2.81, .05)))
        retained = [(float(t), np.zeros((61, 5), np.float32)) for t in times]
        words, results = final_stage1_window_decode(classifier, retained)
        self.assertEqual(words, ['A', 'A'])
        self.assertAlmostEqual(classifier.ends[-1], times[-1])
        self.assertEqual(len(results), len(classifier.ends))

    def test_time_window_uses_timestamp_grid_and_keeps_32_frame_contract(self) -> None:
        timestamps = np.array([0.00, 0.10, 0.22, 0.41, 0.53])
        features = np.broadcast_to(
            timestamps[:, None, None], (len(timestamps), 61, 5)
        ).copy()
        features[..., 3] = 1
        selected = select_time_window(features, timestamps, 0.53, 0.53)
        self.assertEqual(selected.shape, (32, 61, 5))
        np.testing.assert_allclose(selected[:, 0, 0], np.linspace(0, 0.53, 32))

    def test_resampling_keeps_binary_presence_and_zeroes_missing_nodes(self):
        features = np.ones((4, 61, 5), np.float32)
        features[1:3, :, 3] = 0
        selected = select_time_window(features, [0, .1, .3, .53], .53)
        self.assertEqual(set(np.unique(selected[..., 3])), {0., 1.})
        self.assertFalse(selected[..., :3][selected[..., 3] == 0].any())

    def test_raw_window_normalization_cannot_see_later_frames(self):
        from active.v17 import stage1_window_v17 as module
        self.assertTrue(hasattr(module, 'normalize_time_window'))
        features = np.zeros((15, 61, 5), np.float32)
        features[:, 57, 0], features[:, 58, 0] = -.1, .1
        features[:, 57:, 3:] = 1
        timestamps = np.arange(15) * .05
        first, _ = module.normalize_time_window(features, timestamps, .53)
        features[11:, :, 0] = 100
        second, _ = module.normalize_time_window(features, timestamps, .53)
        np.testing.assert_array_equal(first, second)

    def test_time_window_rejects_bad_feature_or_timestamp_contracts(self) -> None:
        good = np.zeros((3, 61, 5), np.float32)
        for timestamps in ([0.0, 0.2, 0.1], [0.0, np.nan, 0.2]):
            with self.assertRaises(ValueError):
                select_time_window(good, timestamps, 0.2, 0.2)
        with self.assertRaises(ValueError):
            select_time_window(np.zeros((3, 60, 5)), [0.0, 0.1, 0.2], 0.2, 0.2)

    def test_schedule_skips_timestamp_gaps_and_finish_adds_partial_tail(self) -> None:
        timestamps = np.array([0.0, 0.1, 0.2, 0.3, 0.52, 0.65, 1.7, 1.8, 1.91])
        regular = window_end_times(timestamps)
        finished = window_end_times(timestamps, include_final=True)
        self.assertTrue(all(np.count_nonzero(
            (timestamps >= end - 0.53) & (timestamps <= end)
        ) >= 2 for end in regular))
        self.assertAlmostEqual(finished[-1], 1.91)
        self.assertNotAlmostEqual(regular[-1], 1.91)
        self.assertEqual(len(window_sample_times(timestamps, 1.91, 0.53)), 32)

    def test_two_agreements_emit_and_boundaries_permit_identical_repeats(self) -> None:
        transcript = Stage1WindowTranscript()
        outputs = []
        for index, label in enumerate(("HELLO", "HELLO", "HELLO", NO_EMIT, NO_EMIT,
                                       "HELLO", "HELLO")):
            outputs = transcript.update(label, index * 0.13)
        self.assertEqual(outputs, ["HELLO", "HELLO"])

        transcript.reset()
        for index, label in enumerate(("HELLO", "HELLO", "YOU", "HELLO", "HELLO")):
            outputs = transcript.update(label, index * 0.13)
        self.assertEqual(outputs, ["HELLO", "HELLO"])

        transcript.reset()
        for index in range(200):
            outputs = transcript.update("HELLO", index * .13)
        # A long hold stays one word; identical repeats without a predicted
        # separator are indistinguishable and merge under this baseline rule.
        self.assertEqual(outputs, ["HELLO"])

    def test_recent_predictions_can_revise_without_losing_long_prefix(self) -> None:
        transcript = Stage1WindowTranscript(revision_seconds=2.0)
        for index in range(24):
            transcript.update(f"SIGN_{index // 2}", index * 0.13)
        prefix = transcript.words[:4]
        transcript.update("CORRECTED", 3.12)
        words = transcript.update("CORRECTED", 3.25)
        self.assertEqual(words[:4], prefix)
        self.assertEqual(words[-1], "CORRECTED")
        transcript.reset()
        self.assertEqual(transcript.words, [])

    def test_revision_replaces_recent_words_and_refuses_old_predictions(self):
        from active.v17.stage1_window_v17 import WindowPrediction
        transcript = Stage1WindowTranscript()
        for i, word in enumerate(('A', 'A', NO_EMIT, 'B', 'B')):
            transcript.update(word, i * .13)
        self.assertTrue(hasattr(transcript, 'replace_recent'))
        words = transcript.replace_recent([WindowPrediction('C', .39), WindowPrediction('C', .52)])
        self.assertEqual(words, ['A', 'C'])
        transcript.update(NO_EMIT, 3.)
        with self.assertRaises(ValueError):
            transcript.replace_recent([WindowPrediction('D', .39)])


class Stage1WindowCliTests(unittest.TestCase):
    def test_ctc_remains_default_and_window_checkpoint_is_explicit(self) -> None:
        baseline = parser().parse_args([])
        self.assertEqual(baseline.transcript_backend, "ctc")
        self.assertIsNone(baseline.stage1_window_checkpoint)
        validate_args(baseline)
        selected = parser().parse_args(["--transcript-backend", "stage1-window"])
        with self.assertRaisesRegex(ValueError, "stage1-window-checkpoint"):
            validate_args(selected)


if __name__ == "__main__":
    unittest.main()
