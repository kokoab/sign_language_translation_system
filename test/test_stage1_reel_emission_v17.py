from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.model_reel_emission_v17 import (
    ReelEmissionHeadConfig,
    ReelEmissionHeadV17,
    ReelEmissionStage1V17,
)
from active.v17.train_stage_1_reel_emission_v17 import (
    refuse_protected,
    restore_source_frames,
    select_threshold,
)
from scripts.live_reel_emission_stage1_v17 import parser as live_parser
from scripts.live_reel_stage1_v17 import proposal_as_fast_verifier


class ReelEmissionTrainingTest(unittest.TestCase):
    def test_live_experiment_defaults_to_landmark_only_commit(self) -> None:
        self.assertTrue(live_parser().parse_args([]).landmark_only_commit)
        self.assertFalse(
            live_parser().parse_args(["--full-visual-verifier"])
            .landmark_only_commit
        )

    def test_fast_verifier_reuses_the_stable_proposal(self) -> None:
        proposal = {
            "gloss": "HELLO", "candidate_gloss": "HELLO",
            "accepted": True, "diagnostics": {},
        }
        verifier = proposal_as_fast_verifier(proposal)
        self.assertEqual(verifier["gloss"], "HELLO")
        self.assertEqual(verifier["mode"], "learned-emission-landmark-fast")
        self.assertFalse(verifier["diagnostics"]["full_visual_verifier_used"])
        self.assertNotIn("full_visual_verifier_used", proposal["diagnostics"])

    def test_restores_recorded_source_lengths(self) -> None:
        windows = np.zeros((2, 32, 61, 5), np.float32)
        restored = restore_source_frames(windows, np.asarray(((0, 32), (32, 41))))
        self.assertEqual(restored.shape, (41, 61, 5))

    def test_threshold_preserves_at_least_95_percent_complete(self) -> None:
        complete = np.linspace(0.001, 0.20, 100)
        waits = np.linspace(0.40, 0.99, 100)
        result = select_threshold(
            np.concatenate((complete, waits)),
            np.asarray([False] * 100 + [True] * 100),
        )
        self.assertGreaterEqual(result["complete_accept_rate"], 0.95)
        self.assertGreater(result["wait_recall"], 0.90)

    def test_temporal_wrapper_keeps_base_logits_exactly(self) -> None:
        config = Stage1V17Config(
            num_classes=100, dim=32, depth=1, heads=4,
            conv_kernel=3, use_pairwise=False,
        )
        base = SLTStage1V17(config).eval()
        head = ReelEmissionHeadV17(ReelEmissionHeadConfig(
            stage1_dim=32, num_glosses=100, hidden_dim=16, dropout=0.0,
        )).eval()
        wrapper = ReelEmissionStage1V17(base, head).eval()
        value = torch.zeros((2, 32, 61, 5))
        with torch.inference_mode():
            expected = base(value)
            actual = wrapper(value)
        self.assertEqual(actual.shape, (2, 101))
        self.assertTrue(torch.equal(expected, actual[:, :100]))

    def test_protected_paths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            refuse_protected((Path("data/test/clip.npz"),))
        with self.assertRaises(ValueError):
            refuse_protected((Path("data/external_evaluation_reserved/clip.npz"),))


if __name__ == "__main__":
    unittest.main()
