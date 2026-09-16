import unittest

from scripts import live_stage2_ctc_v17 as stage2

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
    def test_continuous_reel_enables_verified_repair_with_explicit_rollback(self):
        from scripts.live_reel_continuous_v17 import parser as continuous_parser
        args = continuous_parser().parse_args([])
        self.assertIn('stage2_v17_transition_repair_v3', str(args.stage2_other_preservation))
        self.assertIsNone(continuous_parser().parse_args(['--no-stage2-other-preservation']).stage2_other_preservation)
        self.assertIsNone(stage2.parser().parse_args([]).stage2_other_preservation)

    def test_preservation_rejects_changed_frozen_encoder(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from active.v17.train_stage_2_other_ctc_v17 import directory_sha256
        config = dict(blend_weight=.1, blank_bias=.3, score_margin=0., minimum_tokens=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selector = root / 'model.pth'
            selector.write_bytes(b'accepted')
            package = root / 'package'
            package.mkdir()
            (package / 'model').write_bytes(b'expected')
            encoder = root / 'encoder'
            encoder.mkdir()
            (encoder / 'model').write_bytes(b'changed')
            args = SimpleNamespace(stage2_selector=selector, stage2_primary=package,
                                   stage2_specialist=package, stage2_encoder=encoder,
                                   image_encoder=package)
            payload = dict(accepted_checkpoint={'selector_config': config},
                           accepted_sha256=stage2.sha256(selector))
            for name in ('primary', 'specialist', 'encoder', 'image_encoder'):
                payload[f'runtime_{name}_sha256'] = directory_sha256(package)
            with self.assertRaisesRegex(ValueError, 'encoder'):
                stage2.validate_preservation_runtime(args, payload, config)

    def test_preservation_rejects_changed_selector_configuration(self):
        self.assertTrue(hasattr(stage2, 'validate_preservation_runtime'))
        config = dict(blend_weight=.1, blank_bias=.3, score_margin=0., minimum_tokens=2)
        payload = {'accepted_checkpoint': {'selector_config': config}}
        with self.assertRaisesRegex(ValueError, 'configuration'):
            stage2.validate_preservation_runtime(None, payload, dict(config, blank_bias=1.))

    def test_other_filter_keeps_repeat_boundaries_and_emission_positions(self):
        self.assertTrue(hasattr(stage2, 'supported_ctc_path'))
        self.assertEqual(stage2.supported_ctc_path((2, 101, 2), (1, 4, 8)), ((2, 2), (1, 8)))
        with self.assertRaises(ValueError):
            stage2.supported_ctc_path((102,), (0,))

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

    def test_rollover_continuation_is_not_a_second_sign(self):
        # The previous context ended inside token 2's ongoing run.
        logits = np.full((5, 4), -10., np.float32)
        logits[np.arange(5), [2, 2, 0, 2, 3]] = 10.
        actual = collapse_ctc_path(logits, 5, previous_token=2)
        self.assertEqual(actual, ((2, 3), (3, 4)))

    def test_rollover_preserves_a_repeat_after_blank_or_other(self):
        for previous, path, expected in [
            (0, [2, 2, 0], ((2,), (0,))),
            (2, [0, 2, 2], ((2,), (1,))),
            (2, [101, 2, 2], ((101, 2), (0, 1))),
        ]:
            with self.subTest(previous=previous, path=path):
                logits = np.full((3, 102), -10., np.float32)
                logits[np.arange(3), path] = 10.
                self.assertEqual(collapse_ctc_path(logits, 3, previous_token=previous), expected)

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
