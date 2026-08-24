import unittest

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
)
from active.v17.model_signing_voice_profile_v17 import (
    SigningVoiceProfileV17,
    apply_voice_profile,
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


if __name__ == "__main__":
    unittest.main()
