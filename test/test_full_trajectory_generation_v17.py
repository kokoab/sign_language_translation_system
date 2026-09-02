import unittest
import json
from pathlib import Path

import torch

from active.v17.model_full_trajectory_v17 import (
    FullTrajectoryGeneratorV17,
    FullTrajectoryV17Config,
    observation_from_prediction,
)
from active.v17.model_temporal_motion_tokenizer_v17 import (
    TemporalMotionTokenizerV17,
    TemporalMotionTokenizerV17Config,
)
from active.v17.model_temporal_code_prior_v17 import (
    TemporalCodePriorV17,
    TemporalCodePriorV17Config,
    apply_temporal_hand_mask,
)
from scripts.prepare_full_trajectory_manifest_v17 import RESERVED
from scripts.evaluate_full_trajectory_generator_v17 import motion_ratios
from scripts.generate_grounded_text_to_sign_v17 import (
    hand_participation,
    parse_phrases,
    resolve_text,
)
from scripts.audit_grounded_transition_scalability_v17 import sample_rows


class FullTrajectoryGenerationV17Tests(unittest.TestCase):
    def test_reserved_tokens_are_stable(self):
        self.assertEqual(RESERVED, ("<PAD>", "<BOS>", "<EOS>"))

    def test_manifest_keeps_literal_local_glosses_and_source_splits(self):
        manifest = json.loads(Path(
            "active/v17/full_trajectory_generation_manifest_v17.json"
        ).read_text())
        self.assertEqual(manifest["row_count"], 2095)
        self.assertEqual(manifest["source_role_counts"]["asllrp_other_ctc:validation"], 225)
        local_tokens = {
            token for row in manifest["rows"]
            if row["source"] == "local_phrase_full"
            for token in row["target_sequence"]
        }
        self.assertTrue({"FOOD", "LATE", "TEACHER", "MEET"} <= local_tokens)
        self.assertIn("never concatenate", manifest["trajectory_contract"])

    def test_model_emits_observation_not_render_rig(self):
        model = FullTrajectoryGeneratorV17(FullTrajectoryV17Config(
            vocabulary_size=12, source_families=2, frames=8,
            maximum_tokens=5, model_dim=32, heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_dim=64,
        ))
        tokens = torch.tensor([[1, 3, 4, 2, 0]])
        valid = tokens != 0
        prediction = model(tokens, valid, torch.tensor([0]))
        observation = observation_from_prediction(prediction)
        self.assertEqual(observation.shape, (1, 8, 61, 5))
        absent = observation[..., 3] == 0
        self.assertTrue((observation[..., :3][absent] == 0).all())
        self.assertTrue((observation[..., 4][absent] == 0).all())

    def test_motion_gate_compares_speed_acceleration_and_jerk(self):
        real = {"hand_motion": {
            name: {"p95": value}
            for name, value in (("speed", 2.0), ("acceleration", 4.0), ("jerk", 8.0))
        }}
        generated = {"A B": {"motion": {"hand_motion": {
            name: {"p95": value}
            for name, value in (("speed", 1.0), ("acceleration", 2.0), ("jerk", 4.0))
        }}}}
        self.assertEqual(motion_ratios(generated, real, "speed"), {"A B": 0.5})
        self.assertEqual(motion_ratios(generated, real, "acceleration"), {"A B": 0.5})
        self.assertEqual(motion_ratios(generated, real, "jerk"), {"A B": 0.5})

    def test_temporal_tokenizer_roundtrips_discrete_code_shape(self):
        features = torch.zeros(2, 16, 61, 5)
        features[:, :, 21:42, 3:] = 1
        for factor, steps in ((2, 8), (4, 4)):
            model = TemporalMotionTokenizerV17(TemporalMotionTokenizerV17Config(
                frames=16, hidden_dim=32, latent_dim=8, codebook_size=16,
                downsample_factor=factor,
            )).eval()
            output = model(features)
            decoded = model.decode_codes(output["codes"])
            self.assertEqual(tuple(output["codes"].shape), (2, steps))
            self.assertEqual(tuple(decoded["xyz"].shape), (2, 16, 61, 3))
            self.assertTrue(torch.isfinite(output["quantization_loss"]))

    def test_temporal_prior_and_hand_mask_do_not_force_second_hand(self):
        model = TemporalCodePriorV17(TemporalCodePriorV17Config(
            vocabulary_size=12, source_families=2, codebook_size=16,
            code_steps=4, maximum_tokens=5, model_dim=32, heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_dim=64,
        )).eval()
        tokens = torch.tensor([[1, 3, 4, 2, 0]])
        valid = tokens != 0
        target_codes = torch.tensor([[2, 4, 6, 8]])
        output = model(tokens, valid, torch.tensor([0]), target_codes)
        self.assertEqual(tuple(output["code_logits"].shape), (1, 4, 16))
        generated = model.generate(
            tokens, valid, torch.tensor([0]), sample=False, top_k=1
        )
        self.assertEqual(tuple(generated["codes"].shape), (1, 4))

        observation = torch.ones(1, 8, 61, 5)
        right_only = torch.tensor([[[-9.0, 9.0] for _ in range(4)]])
        masked = apply_temporal_hand_mask(observation, right_only, 2)
        self.assertTrue((masked[:, :, :21] == 0).all())
        self.assertTrue((masked[:, :, 21:42] == 1).all())

    def test_text_anchor_predicts_codes_and_hand_sides_without_forcing_both(self):
        model = TemporalCodePriorV17(TemporalCodePriorV17Config(
            vocabulary_size=12, source_families=2, codebook_size=16,
            code_steps=4, maximum_tokens=5, model_dim=32, heads=4,
            encoder_layers=1, decoder_layers=1, feedforward_dim=64,
            text_anchor_loss_weight=0.5,
        )).eval()
        tokens = torch.tensor([[1, 3, 4, 2, 0]])
        output = model(
            tokens, tokens != 0, torch.tensor([0]),
            torch.tensor([[2, 4, 6, 8]]),
        )
        self.assertEqual(tuple(output["text_code_logits"].shape), (1, 4, 16))
        self.assertEqual(tuple(output["text_side_logits"].shape), (1, 4, 2))
        generated = model.generate(
            tokens, tokens != 0, torch.tensor([0]), sample=False, top_k=1,
        )
        self.assertEqual(tuple(generated["side_logits"].shape), (1, 4, 2))

    def test_grounded_phrase_helpers_preserve_literal_hand_participation(self):
        features = torch.zeros(8, 61, 5).numpy()
        features[:, 21:42, 3:] = 1
        self.assertEqual(hand_participation(features), [False, True])
        self.assertEqual(
            parse_phrases(["good,morning", "tomorrow,school,go"]),
            [("GOOD", "MORNING"), ("TOMORROW", "SCHOOL", "GO")],
        )
        catalog = json.loads(Path(
            "active/v17/text_to_sign_phrase_catalog_v17.json"
        ).read_text())
        self.assertEqual(
            resolve_text("Hello, how are you?", catalog),
            ("HELLO", "HOW", "YOU"),
        )
        sampled = sample_rows(list(range(10)), 3, __import__("numpy").random.default_rng(7))
        self.assertEqual(sampled, [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
