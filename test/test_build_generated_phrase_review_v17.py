import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.build_generated_phrase_review_v17 import (
    boundary_diagnostics,
    load_separated_landmark_artifact,
)


class GeneratedPhraseReviewV17Tests(unittest.TestCase):
    def test_smooth_observed_transition_passes_structural_diagnostics(self):
        features = np.zeros((12, 61, 5), np.float32)
        features[..., 0] = np.linspace(0, 1, 12)[:, None]
        features[..., 3:] = 1
        result = boundary_diagnostics(features, [
            {"kind": "gloss", "start": 0, "stop": 4},
            {"kind": "transition", "start": 4, "stop": 8},
            {"kind": "gloss", "start": 8, "stop": 12},
        ])
        self.assertEqual(result["handless_transition_frames"], 0)
        self.assertEqual(result["frames_with_incomplete_left_hand"], 0)
        self.assertEqual(result["frames_with_incomplete_right_hand"], 0)
        self.assertEqual(result["frames_with_incomplete_face"], 0)
        self.assertEqual(result["frames_with_incomplete_body"], 0)
        self.assertEqual(result["presence_changes_anywhere"], 0)
        self.assertEqual(result["presence_changes_at_transition_joins"], 0)
        self.assertEqual(result["nodes_present_only_inside_transition"], 0)
        self.assertAlmostEqual(result["join_velocity_p95_over_gloss_p95"], 1.0)

    def test_separated_artifact_preserves_real_observation_mask(self):
        rig = np.ones((4, 61, 3), np.float32)
        present = np.ones((4, 61), bool)
        present[:, 21:42] = False
        observation = rig.copy()
        observation[~present] = 0
        rig_presence = present.copy()
        rig[~rig_presence] = 0
        metadata = {
            "artifact_contract_version": 3,
            "training_eligible": False,
            "validation_eligible": False,
            "test_eligible": False,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "artifact.npz"
            np.savez_compressed(
                path,
                animation_rig_xyz=rig,
                animation_rig_presence=rig_presence,
                animation_rig_confidence=rig_presence.astype(np.float32),
                observation_xyz=observation,
                observation_presence=present,
                observation_confidence=present.astype(np.float32),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            loaded_rig, loaded_observation, _, diagnostics = (
                load_separated_landmark_artifact(path)
            )
        self.assertEqual(loaded_rig.shape, (4, 61, 5))
        self.assertEqual(loaded_observation.shape, (4, 61, 5))
        self.assertFalse(diagnostics["observation_is_fabricated_all_present"])
        self.assertEqual(diagnostics["rig_preserves_observed_xyz_max_error"], 0.0)
        self.assertTrue(diagnostics["hand_participation_preserved"])

    def test_legacy_ambiguous_landmarks_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.npz"
            np.savez_compressed(path, landmarks=np.zeros((4, 61, 5), np.float32))
            with self.assertRaisesRegex(ValueError, "ambiguous legacy"):
                load_separated_landmark_artifact(path)


if __name__ == "__main__":
    unittest.main()
