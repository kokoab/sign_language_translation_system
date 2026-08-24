import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
)
from active.v17.model_transition_diffusion_v17 import (
    TransitionResidualDiffusionV17,
    TransitionResidualDiffusionV17Config,
    sinusoidal_timestep_embedding,
)
from active.v17.model_transition_span_v17 import (
    TransitionSpanPredictorV17,
    TransitionSpanV17Config,
    endpoint_only_context,
    kinematic_span,
)
from active.v17.train_transition_span_multicorpus_v17 import TransitionSpanDataset
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    discover_signers,
    landmark_tree_fingerprint,
    linear_interpolation,
    loss_terms,
    motion_distribution_loss,
)
from active.v17.train_transition_inpainter_multicorpus_v17 import (
    combined_selection_score,
    manifest_signers,
    passes_primary_floor,
    weighted_sampler,
)


class TransitionInpainterV17Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = TransitionInpainterV17Config(
            dim=24, depth=1, heads=4, dropout=0.0
        )
        self.features = torch.randn(
            2,
            self.config.frames,
            self.config.nodes,
            self.config.channels,
        )
        self.features[..., 3] = 1.0
        self.features[..., 4] = 0.8
        self.mask = torch.zeros(2, self.config.frames, dtype=torch.bool)
        self.mask[0, 8:15] = True
        self.mask[1, 17:27] = True

    def test_forward_preserves_visible_context_and_is_finite(self):
        model = TransitionInpainterV17(self.config).eval()
        with torch.inference_mode():
            output = model(self.features, self.mask)
            baseline = linear_interpolation(self.features, self.mask)
        self.assertEqual(output.shape, self.features.shape)
        visible = ~self.mask[:, :, None, None].expand_as(self.features)
        self.assertTrue(torch.equal(output[visible], self.features[visible]))
        self.assertTrue(torch.equal(output, baseline))
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(model.parameter_count, 0)

    def test_rejects_invalid_or_empty_masks(self):
        model = TransitionInpainterV17(self.config)
        with self.assertRaisesRegex(ValueError, "boolean"):
            model(self.features, self.mask.float())
        with self.assertRaisesRegex(ValueError, "non-empty"):
            model(self.features, torch.zeros_like(self.mask))

    def test_linear_baseline_interpolates_only_masked_interval(self):
        features = self.features.clone()
        features[0, 7] = 2.0
        features[0, 15] = 10.0
        interpolated = linear_interpolation(features, self.mask)
        visible = ~self.mask[:, :, None, None].expand_as(features)
        self.assertTrue(torch.equal(interpolated[visible], features[visible]))
        expected = torch.linspace(3.0, 9.0, 7)
        self.assertTrue(torch.allclose(interpolated[0, 8:15, 0, 0], expected))

    def test_loss_terms_are_finite(self):
        interpolated = linear_interpolation(self.features, self.mask)
        terms = loss_terms(interpolated, self.features, self.mask)
        self.assertEqual(
            set(terms), {"spatial", "auxiliary", "velocity", "acceleration"}
        )
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        distribution = motion_distribution_loss(
            interpolated, self.features, self.mask
        )
        self.assertTrue(torch.isfinite(distribution))
        self.assertGreaterEqual(float(distribution), 0.0)

    def test_dataset_preloads_valid_windows_and_fingerprints_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signer_root = root / "voice_a"
            signer_root.mkdir()
            path = signer_root / "clip.transition_landmarks_v17.npz"
            landmarks = np.zeros((3, 32, 61, 5), dtype=np.float16)
            landmarks[1, ..., 0] = 2.0
            np.savez_compressed(
                path,
                landmarks=landmarks,
                window_valid=np.asarray([True, False, True]),
                metadata_json=np.array(json.dumps({"signer_id": "source:voice_a"})),
            )
            self.assertEqual(discover_signers(root), {"source:voice_a"})
            dataset = TransitionWindowDataset(
                root, {"source:voice_a"}, seed=17, fixed_masks=True
            )
            self.assertEqual(len(dataset), 2)
            self.assertIsNotNone(dataset.preloaded_features)
            self.assertEqual(dataset[0]["features"].shape, (32, 61, 5))
            count, fingerprint = landmark_tree_fingerprint(root)
            self.assertEqual(count, 1)
            self.assertEqual(len(fingerprint), 64)

    def test_residual_diffusion_shapes_and_visible_zeroing(self):
        config = TransitionResidualDiffusionV17Config(
            dim=24, depth=1, heads=4, dropout=0.0, timesteps=5
        )
        model = TransitionResidualDiffusionV17(config).eval()
        residual = torch.randn(2, 32, 61, 3)
        timesteps = torch.tensor([1, 4], dtype=torch.long)
        with torch.inference_mode():
            predicted = model(self.features, self.mask, residual, timesteps)
            sample = model.sample_normalized_residual(
                self.features, self.mask, temperature=0.5,
                generator=torch.Generator().manual_seed(17),
            )
        self.assertEqual(predicted.shape, residual.shape)
        self.assertEqual(sample.shape, residual.shape)
        visible = ~self.mask[:, :, None, None].expand_as(residual)
        self.assertTrue(torch.equal(predicted[visible], torch.zeros_like(predicted[visible])))
        self.assertTrue(torch.equal(sample[visible], torch.zeros_like(sample[visible])))
        self.assertTrue(torch.isfinite(sample).all())
        self.assertEqual(sinusoidal_timestep_embedding(timesteps, 25).shape, (2, 25))

    def test_multicorpus_sampler_enforces_source_probability(self):
        sampler = weighted_sampler(3, 2, 0.25, 100, 17)
        self.assertAlmostEqual(float(sampler.weights[:3].sum()), 0.75)
        self.assertAlmostEqual(float(sampler.weights[3:].sum()), 0.25)
        self.assertEqual(sampler.num_samples, 100)
        self.assertAlmostEqual(
            combined_selection_score(
                {"relative_score_improvement": 0.2},
                {"relative_score_improvement": 0.4},
            ),
            0.3,
        )
        self.assertTrue(passes_primary_floor(
            {"relative_score_improvement": 0.20}, 0.20
        ))
        self.assertFalse(passes_primary_floor(
            {"relative_score_improvement": 0.19}, 0.20
        ))

    def test_multicorpus_manifest_voice_roles_are_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({
                "format": "continuous_unlabeled_transition_manifest_v17",
                "rows": [
                    {"signer_id": "voice_a", "role": "train"},
                    {"signer_id": "voice_b", "role": "validation"},
                ],
            }))
            self.assertEqual(manifest_signers(path), ({"voice_a"}, {"voice_b"}))

    def test_transition_span_model_and_context_ablation(self):
        config = TransitionSpanV17Config(dim=24, depth=1, heads=4, dropout=0.0)
        model = TransitionSpanPredictorV17(config).eval()
        context = torch.randn(2, 16, 61, 5)
        with torch.inference_mode():
            logits = model(context)
            ablated = endpoint_only_context(context)
        self.assertEqual(logits.shape, (2, 9))
        self.assertEqual(ablated.shape, context.shape)
        self.assertTrue(torch.equal(ablated[:, 0], context[:, 7]))
        self.assertTrue(torch.equal(ablated[:, 15], context[:, 8]))

    def test_kinematic_span_recovers_constant_speed_gap(self):
        context = torch.zeros(1, 16, 61, 5)
        times = list(range(8)) + list(range(14, 22))
        for index, value in enumerate(times):
            context[:, index, :, 0] = value * 0.01
            context[:, index, :, 3] = 1.0
        self.assertEqual(int(kinematic_span(context)[0]), 6)

    def test_transition_span_dataset_hides_gap_width(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signer_root = root / "voice_a"
            signer_root.mkdir()
            landmarks = np.zeros((1, 32, 61, 5), dtype=np.float16)
            landmarks[..., 3] = 1.0
            np.savez_compressed(
                signer_root / "clip.transition_landmarks_v17.npz",
                landmarks=landmarks,
                window_valid=np.asarray([True]),
                metadata_json=np.array(json.dumps({"signer_id": "source:voice_a"})),
            )
            dataset = TransitionSpanDataset(root, {"source:voice_a"})
            self.assertEqual(len(dataset), 9)
            self.assertEqual(dataset[0]["context"].shape, (16, 61, 5))
            self.assertEqual(dataset[0]["target_span"], 4)
            self.assertEqual(dataset[8]["target_span"], 12)


if __name__ == "__main__":
    unittest.main()
