import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from active.v17.model_signing_voice_v17 import (
    SigningVoiceGeneratorV17,
    SigningVoiceV17Config,
)
from active.v17.train_signing_voice_v17 import (
    VoicePairDataset,
    cross_gloss_style_loss,
    emitted_style_loss,
    rank_auc,
    signer_aware_style_scores,
)
from active.v17.model_transition_inpainter_v17 import interpolate_masked_context
from active.v17.signing_voice_phrase_v17 import (
    NovelVoiceRecipe,
    build_novel_voice_recipes,
    compose_phrase,
    normalize_style_mix,
    synthesize_boundary,
    synthesize_join,
    trim_observed_span,
    trim_transition_span,
    stabilize_transition_hands,
)
from active.v17.model_signing_voice_profile_v17 import (
    SigningVoiceProfileV17,
    apply_voice_profile,
    apply_voice_profile_to_trajectory,
    decode_profile,
    encode_profile,
    estimate_voice_profile,
    fit_profile_latent,
)


class _DummyTiming(nn.Module):
    class Config:
        minimum_span = 4

    config = Config()

    def forward(self, context):
        logits = torch.zeros(len(context), 9, device=context.device)
        logits[:, 2] = 1
        return logits


class _DummyMean(nn.Module):
    def forward(self, features, mask):
        return interpolate_masked_context(features, mask)


class _HallucinatingPresenceMean(nn.Module):
    def forward(self, features, mask):
        output = interpolate_masked_context(features, mask)
        output[..., 3:] = torch.where(
            mask[:, :, None, None], torch.ones_like(output[..., 3:]), output[..., 3:]
        )
        return output


class SigningVoiceV17Tests(unittest.TestCase):
    def test_zero_initialized_generator_preserves_content_prototype(self):
        torch.manual_seed(7)
        model = SigningVoiceGeneratorV17(SigningVoiceV17Config(
            dim=32, style_dim=8, encoder_depth=1, decoder_depth=1,
            heads=4, dropout=0.0,
        ))
        prototype = torch.randn(2, 32, 61, 5)
        prototype[..., 3] = 1
        prototype[..., 4] = 0.8
        reference = torch.randn(2, 32, 61, 5)
        reference[..., 3] = 1
        reference[..., 4] = 0.8
        generated, style = model(prototype, torch.tensor([2, 7]), reference)
        self.assertTrue(torch.equal(generated, prototype))
        self.assertEqual(tuple(style.shape), (2, 8))
        self.assertTrue(torch.allclose(style.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_style_reference_is_same_signer_and_different_gloss(self):
        landmarks = np.zeros((6, 32, 61, 5), np.float16)
        targets = np.asarray([0, 1, 2, 0, 2, 3])
        signers = np.asarray(["a", "a", "a", "b", "b", "b"])
        prototypes = torch.zeros(4, 32, 61, 5)
        dataset = VoicePairDataset(
            landmarks, targets, signers, np.arange(6), prototypes,
            {"a": 0, "b": 1}, seed=11, fixed=True,
        )
        for index in range(len(dataset)):
            row = dataset[index]
            target_index = row["target_index"]
            reference_index = row["reference_index"]
            self.assertEqual(signers[target_index], signers[reference_index])
            self.assertNotEqual(targets[target_index], targets[reference_index])

    def test_rank_auc_orders_same_voice_above_different_voice(self):
        self.assertEqual(
            rank_auc(np.asarray([0.8, 0.9]), np.asarray([0.1, 0.2])), 1.0
        )

    def test_cross_gloss_style_loss_rewards_voice_separation(self):
        voices = torch.tensor([0, 1])
        separated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertLess(
            float(cross_gloss_style_loss(separated, separated, voices)),
            float(cross_gloss_style_loss(collapsed, collapsed, voices)),
        )

    def test_emitted_style_loss_uses_only_same_gloss_negatives(self):
        voices = torch.tensor([0, 1, 2])
        targets = torch.tensor([4, 4, 9])
        desired = torch.eye(3)
        correct = emitted_style_loss(desired, desired, voices, targets)
        wrong = emitted_style_loss(desired.roll(1, 0), desired, voices, targets)
        self.assertLess(float(correct), float(wrong))

    def test_style_verification_negatives_are_always_different_signers(self):
        reference = np.eye(4, dtype=np.float32)
        target = reference.copy()
        signers = np.asarray(["a", "a", "b", "c"])
        positive, negative = signer_aware_style_scores(
            reference, target, signers, np.asarray([0, 1, 0, 1])
        )
        self.assertEqual(len(positive), 4)
        self.assertEqual(len(negative), 4)
        self.assertEqual(rank_auc(positive, negative), 1.0)

    def test_style_verification_can_score_generated_motion_embeddings(self):
        generated = np.asarray([[1.0, 0.0], [0.0, 1.0]], np.float32)
        real_targets = generated.copy()
        positive, negative = signer_aware_style_scores(
            generated, real_targets, np.asarray(["voice-a", "voice-b"]),
            np.asarray([3, 3]),
        )
        self.assertEqual(rank_auc(positive, negative), 1.0)

    def test_novel_voice_is_a_normalized_three_voice_mix(self):
        centroids = torch.eye(12)
        recipe = build_novel_voice_recipes(centroids)[0]
        style = normalize_style_mix(
            centroids, recipe.source_voice_indices, recipe.weights
        )
        self.assertEqual(len(recipe.source_voice_indices), 3)
        self.assertAlmostEqual(float(style.norm()), 1.0, places=6)
        self.assertLess(float((centroids @ style).max()), 1.0)

    def test_complete_phrase_inserts_generated_boundary_and_timeline(self):
        first = np.zeros((32, 61, 5), np.float32)
        second = np.zeros_like(first)
        first[..., 0] = np.linspace(-1, 0, 32)[:, None]
        second[..., 0] = np.linspace(1, 2, 32)[:, None]
        first[..., 3:] = 1
        second[..., 3:] = 1
        transition, span = synthesize_boundary(
            first, second, _DummyMean(), _DummyTiming()
        )
        self.assertEqual(span, 6)
        self.assertEqual(transition.shape, (6, 61, 5))
        recipe = NovelVoiceRecipe("test", (0, 1, 2), (0.5, 0.3, 0.2))
        phrase, timeline = compose_phrase(
            [first, second], [3, 7], 1.0, torch.full((100,), 20),
            _DummyMean(), _DummyTiming(),
        )
        self.assertEqual(phrase.shape, (46, 61, 5))
        self.assertEqual([row["kind"] for row in timeline], [
            "gloss", "transition", "gloss"
        ])

    def test_phrase_trims_unobserved_padding_before_transition(self):
        sign = np.zeros((32, 61, 5), np.float32)
        sign[2:30, ..., 3:] = 1
        trimmed = trim_transition_span(sign)
        self.assertEqual(len(trimmed), 28)
        phrase, timeline = compose_phrase(
            [sign, sign], [3, 7], 1.0, torch.full((100,), 20),
            _DummyMean(), _DummyTiming(),
        )
        transition = next(row for row in timeline if row["kind"] == "transition")
        self.assertTrue(
            (phrase[transition["start"]:transition["stop"], :42, 3] > 0).any(axis=1).all()
        )

    def test_trim_removes_partial_hand_edges_without_adding_a_second_hand(self):
        sign = np.zeros((10, 61, 5), np.float32)
        sign[:, 21, 3:] = 1
        sign[2:8, 21:42, 3:] = 1
        trimmed = trim_transition_span(sign)
        self.assertEqual(len(trimmed), 6)
        self.assertFalse((trimmed[:, :21, 3] > 0).any())
        self.assertTrue((trimmed[:, 21:42, 3] > 0).all())

    def test_transition_presence_is_anchored_to_observed_endpoints(self):
        sign = np.zeros((32, 61, 5), np.float32)
        sign[:, :42, 3:] = 1
        transition, _ = synthesize_boundary(
            sign, sign, _HallucinatingPresenceMean(), _DummyTiming()
        )
        self.assertTrue((transition[:, :42, 3] > 0).all())
        self.assertFalse((transition[:, 42:, 3] > 0).any())

    def test_transition_hand_stabilization_preserves_bones_without_second_hand(self):
        left = np.zeros((2, 61, 5), np.float32)
        right = np.zeros_like(left)
        tree = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
        )
        for features, direction in ((left, 1.0), (right, -1.0)):
            hand = features[:, 21:42]
            hand[..., 3:] = 1
            for parent, child in tree:
                hand[:, child, :3] = hand[:, parent, :3] + (direction * 0.1, 0.02, 0.0)
        collapsed = np.zeros((5, 61, 5), np.float32)
        collapsed[:, :21, 3:] = 1
        collapsed[:, 21:42, 3:] = 1
        fixed = stabilize_transition_hands(collapsed, left, right)
        self.assertFalse((fixed[:, :21, 3] > 0).any())
        self.assertTrue((fixed[:, 21:42, 3] > 0).all())
        for parent, child in tree:
            length = np.linalg.norm(
                fixed[:, 21 + child, :3] - fixed[:, 21 + parent, :3], axis=1
            )
            self.assertTrue((length > 0.09).all())

    def test_transition_completes_only_a_legitimately_participating_hand(self):
        left = np.zeros((2, 61, 5), np.float32)
        right = np.zeros_like(left)
        for features in (left, right):
            features[:, 21:42, 3:] = 1
            features[:, 21:42, 0] = np.arange(21, dtype=np.float32)
        partial = np.zeros((5, 61, 5), np.float32)
        partial[:, 21, 3:] = 1
        fixed = stabilize_transition_hands(partial, left, right)
        self.assertFalse((fixed[:, :21, 3] > 0).any())
        self.assertTrue((fixed[:, 21:42, 3] > 0).all())

    def test_join_can_skip_one_genuine_entry_frame_without_adding_a_hand(self):
        left = np.zeros((10, 61, 5), np.float32)
        right = np.zeros_like(left)
        transition = np.zeros((4, 61, 5), np.float32)
        for value in (left, right, transition):
            value[:, 21:42, 3:] = 1
        failed = {"speed": 1.0, "acceleration": 1.0, "jerk": 5.0}
        passed = {"speed": 1.0, "acceleration": 1.0, "jerk": 1.0}
        with patch(
            "active.v17.signing_voice_phrase_v17.synthesize_boundary",
            side_effect=((transition, 4), (transition, 4)),
        ), patch(
            "active.v17.signing_voice_phrase_v17._transition_motion_ratios",
            side_effect=(failed, passed),
        ):
            result, prepared_right, span, offset = synthesize_join(
                left, right, object(), _DummyTiming()
            )
        self.assertEqual((span, offset, len(prepared_right)), (4, 1, 9))
        self.assertFalse((result[:, :21, 3] > 0).any())

    def test_profile_estimation_removes_content_and_roundtrips_latent(self):
        prototypes = np.zeros((2, 32, 61, 5), np.float32)
        prototypes[..., 3:] = 1
        landmarks = prototypes.copy()
        landmarks[:, :, :21, 0] += 0.2
        profile = estimate_voice_profile(
            landmarks, np.asarray([0, 1]), np.asarray([0, 1]), prototypes
        )
        self.assertTrue(np.allclose(profile.node_offset[:21, 0], 0.2))
        profiles = [
            SigningVoiceProfileV17(
                profile.node_offset + index * 0.01, profile.frame_curve
            )
            for index in range(4)
        ]
        mean, components, _ = fit_profile_latent(profiles, 2)
        decoded = decode_profile(encode_profile(profiles[1], mean, components), mean, components)
        self.assertTrue(np.allclose(decoded.vector(), profiles[1].vector(), atol=1e-5))

    def test_profile_application_preserves_auxiliary_content_channels(self):
        prototype = np.zeros((32, 61, 5), np.float32)
        prototype[..., 3] = 1
        prototype[..., 4] = 0.8
        profile = SigningVoiceProfileV17(
            np.full((61, 3), 0.1, np.float32), np.zeros((32, 3), np.float32)
        )
        generated = apply_voice_profile(prototype, profile)
        self.assertTrue(np.array_equal(generated[..., 3:], prototype[..., 3:]))
        self.assertTrue(np.allclose(generated[..., :3], 0.1))

    def test_trajectory_profile_cannot_invent_second_hand(self):
        trajectory = np.zeros((128, 61, 5), np.float32)
        trajectory[:, 21:42, :3] = 0.2
        trajectory[:, 21:42, 3] = 1
        trajectory[:, 21:42, 4] = 0.8
        profile = SigningVoiceProfileV17(
            np.full((61, 3), 0.1, np.float32),
            np.linspace(0, 0.2, 32, dtype=np.float32)[:, None]
            * np.ones((1, 3), np.float32),
        )

        generated = apply_voice_profile_to_trajectory(
            trajectory, profile, curve_strength=1.0
        )

        self.assertEqual(generated.shape, trajectory.shape)
        self.assertTrue(np.array_equal(generated[..., 3:], trajectory[..., 3:]))
        self.assertTrue(np.array_equal(generated[:, :21, :3], np.zeros((128, 21, 3))))
        self.assertGreater(float(generated[:, 21:42, :3].mean()), 0.2)


if __name__ == "__main__":
    unittest.main()
