import unittest

import numpy as np
import torch

from active.v17.model_stage2_v17 import ctc_sequence_log_probability as torch_ctc_score
from scripts.live_stage2_ctc_v17 import (
    ElapsedWindowBuffer,
    StablePrefixSpeaker,
    collapse_ctc_path,
    collapse_ctc_tokens,
    ctc_sequence_log_probability,
    parser,
    roll_ctc_prefix,
)


class LiveStage2CTCTests(unittest.TestCase):
    def test_numpy_ctc_score_matches_selected_model_rule(self):
        logits = np.random.default_rng(17).normal(size=(8, 6)).astype(np.float32)
        tokens = (2, 4)
        expected = float(torch_ctc_score(torch.from_numpy(logits), tokens))
        self.assertAlmostEqual(
            ctc_sequence_log_probability(logits, tokens), expected, places=5
        )

    def test_ctc_collapse_preserves_repeat_only_across_blank(self):
        logits = np.full((6, 4), -10.0, np.float32)
        logits[np.arange(6), [0, 2, 2, 0, 2, 3]] = 10
        self.assertEqual(collapse_ctc_tokens(logits, 6), (2, 2, 3))

    def test_ctc_path_returns_emission_positions(self):
        logits = np.full((6, 4), -10.0, np.float32)
        logits[np.arange(6), [0, 2, 2, 0, 2, 3]] = 10
        self.assertEqual(collapse_ctc_path(logits, 6), ((2, 2, 3), (1, 4, 5)))

    def test_rolling_context_locks_only_first_window_emissions(self):
        locked, hypothesis, positions = roll_ctc_prefix(
            ["HELLO"], ["HOW", "YOU", "GOOD"], [4, 9, 17]
        )
        self.assertEqual(locked, ["HELLO", "HOW"])
        self.assertEqual(hypothesis, ["YOU", "GOOD"])
        self.assertEqual(positions, [1, 9])

    def test_stable_prefix_waits_for_a_following_update(self):
        value = StablePrefixSpeaker()
        self.assertEqual(value.update(["HELLO"]), [])
        self.assertEqual(value.update(["HELLO", "HOW"]), ["HELLO"])
        self.assertEqual(value.update(["HELLO", "HOW", "YOU"]), ["HOW"])

    def test_elapsed_windows_follow_time_not_observation_count(self):
        class Frame:
            def __init__(self, seconds):
                self.seconds = seconds

        value = ElapsedWindowBuffer(period_seconds=1.0)
        emitted = None
        for seconds in np.arange(0.0, 1.2, 0.2):
            emitted = value.add(Frame(float(seconds))) or emitted
        self.assertEqual(len(emitted), 5)
        self.assertAlmostEqual(emitted[0].seconds, 0.0)
        self.assertAlmostEqual(emitted[-1].seconds, 0.8)

    def test_defaults_use_separate_stage2_path(self):
        args = parser().parse_args([])
        self.assertIn("live_stage2_ctc_v17", str(args.output_root))
        self.assertIn("Stage2FrozenEncoder", str(args.stage2_encoder))
        self.assertIn("Stage2SelectorPrimary", str(args.stage2_primary))
        self.assertFalse(args.finish_at_eof)


if __name__ == "__main__":
    unittest.main()
