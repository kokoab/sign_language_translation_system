from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from active.v17.model_v17 import (
    SLTStage1V17,
    Stage1V17Config,
    ArticulatedPoseEmbeddingV17,
    KeypointTemporalGateV17,
    PartWiseTemporalEncoderV17,
    ScoreMixingMultiheadAttentionV17,
    masked_bone_features,
    canonicalize_camera_roll_v17,
    masked_hand_angle_features,
    masked_hand_bone_geometry,
    static_hand_frame_weights,
    articulated_bone_distance,
    wrist_relative_hand_features,
    anatomical_adjacency_v17,
    masked_temporal_features,
)
from active.v17.evaluate_model_soup_v17 import blend_state_dicts
from active.v17.evaluate_multimodal_ensemble_v17 import (
    normalized_item_id,
    per_sample_zscore,
    probabilities,
)
from active.v17.schema_v17 import V17Config, schema_fingerprint
from active.v17.schema_mediapipe_v17 import (
    MediaPipeV17Config,
    schema_fingerprint as mediapipe_schema_fingerprint,
)
from active.v17.pretrain_masked_pose_v17 import (
    anatomical_span_mask,
    reconstruction_loss,
)
from active.v17.train_stage_1_v17 import (
    Citizen100V17Dataset,
    ExponentialMovingAverage,
    LocalReviewSupplementV17Dataset,
    LocalValidationV17Dataset,
    MIRROR_NODE_INDEX,
    SemLexSupplementV17Dataset,
    augment_v17,
    class_source_balanced_weights,
    mirror_v17,
    mask_mouth_nodes_v17,
    parse_source_probabilities,
    partmix_cross_entropy,
    partmix_hands_v17,
    part_auxiliary_loss,
    rotate_camera_roll_v17,
    initialize_flat_graph_residual,
    initialize_exact_stage1_finetune,
    initialize_articulated_pose_embedding,
    initialize_masked_pose_encoder,
    load_phonology_supervision,
    phonology_auxiliary_loss,
    supervised_contrastive_loss,
    extractor_schema_fingerprint,
)


def write_archive(
    path: Path, value: float = 0.0, fingerprint: str | None = None
) -> None:
    features = np.zeros((32, 61, 5), dtype=np.float16)
    features[:, 0, 0] = value
    features[:, 0, 3:] = 1.0
    metadata = {
        "schema_fingerprint": fingerprint or schema_fingerprint(V17Config())
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        metadata_json=np.array(json.dumps(metadata)),
        diagnostics_json=np.array("{}"),
        schema_json=np.array("{}"),
    )


class V17Stage1DatasetTest(unittest.TestCase):
    def test_local_mouth_mask_preserves_facial_reference_geometry(self):
        features = torch.ones(2, 32, 61, 5)
        masked = mask_mouth_nodes_v17(features)
        self.assertTrue(torch.equal(masked[..., :49, :], features[..., :49, :]))
        self.assertTrue(torch.equal(masked[..., 53:, :], features[..., 53:, :]))
        self.assertEqual(int(torch.count_nonzero(masked[..., 49:53, :])), 0)
        self.assertGreater(int(torch.count_nonzero(masked[..., 42:49, :])), 0)
        self.assertGreater(int(torch.count_nonzero(masked[..., 53:57, :])), 0)
        self.assertTrue(torch.equal(features, torch.ones_like(features)))

    def test_partwise_temporal_projection_is_feature_isolated(self):
        torch.manual_seed(19)
        encoder = PartWiseTemporalEncoderV17(
            64, 4, 15, 0.0, 0.0, 1, use_pairwise=True, node_channels=11
        ).eval()
        derived = torch.randn(2, 5, 61, 11)
        pairwise = torch.randn(2, 5, 66)
        changed = derived.clone()
        changed[:, :, 21:42] += 3.0
        baseline_parts = encoder.project_parts(derived, pairwise)
        changed_parts = encoder.project_parts(changed, pairwise)
        for name in ("left_hand", "face", "body"):
            torch.testing.assert_close(
                baseline_parts[name], changed_parts[name], rtol=0.0, atol=0.0
            )
        self.assertFalse(torch.equal(
            baseline_parts["right_hand"], changed_parts["right_hand"]
        ))

    def test_partwise_global_model_is_finite_and_mobile_sized(self):
        config = Stage1V17Config(
            num_classes=100, dim=256, depth=4, heads=8,
            temporal_encoder="partwise_global", part_depth=1,
            dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )
        model = SLTStage1V17(config).eval()
        features = torch.zeros(2, 32, 61, 5)
        features[:, :, :42, 3:] = 1.0
        with torch.inference_mode():
            logits = model(features)
        self.assertEqual(tuple(logits.shape), (2, 100))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertLess(model.parameter_count, 8_000_000)

    def test_part_auxiliary_heads_are_isolated_training_only_decoders(self):
        config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            temporal_encoder="partwise_global", part_depth=1,
            use_part_auxiliary=True,
            dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )
        model = SLTStage1V17(config).eval()
        features = torch.zeros(3, 8, 61, 5)
        features[:, :, :42, 3:] = 1.0
        logits, embedding, auxiliary, part_valid = model.forward_part_auxiliary(features)
        ordinary_logits = model(features)
        torch.testing.assert_close(logits, ordinary_logits, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(logits.shape), (3, 7))
        self.assertEqual(tuple(embedding.shape), (3, 64))
        self.assertEqual(set(auxiliary), {"left_hand", "right_hand", "face", "body"})
        self.assertEqual(set(part_valid), set(auxiliary))
        self.assertTrue(part_valid["left_hand"].all())
        self.assertTrue(part_valid["right_hand"].all())
        self.assertFalse(part_valid["face"].any())
        self.assertFalse(part_valid["body"].any())
        self.assertTrue(all(tuple(value.shape) == (3, 7) for value in auxiliary.values()))
        loss = part_auxiliary_loss(auxiliary, torch.tensor([0, 1, 2]), part_valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(
            model.part_auxiliary_heads[name][-1].weight.grad is not None
            for name in ("left_hand", "right_hand")
        ))
        self.assertTrue(all(
            model.part_auxiliary_heads[name][-1].weight.grad is None
            for name in ("face", "body")
        ))

    def test_part_auxiliary_heads_require_partwise_encoder(self):
        with self.assertRaisesRegex(ValueError, "require partwise_global"):
            Stage1V17Config(use_part_auxiliary=True).validate()

    def test_keypoint_temporal_gate_starts_as_exact_identity_and_trains(self):
        torch.manual_seed(29)
        gate = KeypointTemporalGateV17()
        raw = torch.randn(2, 7, 61, 5)
        raw[..., 3:] = 1.0
        derived = masked_temporal_features(raw)
        model_input = torch.randn(2, 7, 61, 13, requires_grad=True)
        output = gate(model_input, derived)
        torch.testing.assert_close(output, model_input, rtol=0.0, atol=0.0)
        output.square().mean().backward()
        self.assertIsNotNone(gate.output.weight.grad)
        self.assertGreater(float(gate.output.weight.grad.abs().sum()), 0.0)

    def test_keypoint_temporal_gate_is_small_finite_and_partwise_only(self):
        with self.assertRaisesRegex(ValueError, "requires partwise_global"):
            Stage1V17Config(use_keypoint_temporal_gate=True).validate()
        common = dict(
            num_classes=7, dim=64, depth=1, heads=4,
            temporal_encoder="partwise_global", part_depth=1,
            dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )
        baseline = SLTStage1V17(Stage1V17Config(**common)).eval()
        gated = SLTStage1V17(Stage1V17Config(
            **common, use_keypoint_temporal_gate=True
        )).eval()
        self.assertEqual(gated.parameter_count - baseline.parameter_count, 732)
        features = torch.zeros(2, 32, 61, 5)
        features[:, :, :42, 3:] = 1.0
        with torch.inference_mode():
            logits = gated(features)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_articulated_pose_inputs_and_distance_are_missing_aware(self):
        features = torch.zeros(1, 2, 61, 5)
        features[..., :42, 3:] = 1.0
        for offset in (0, 21):
            features[..., offset + 0, 0] = 5.0
            features[..., offset + 1, 0] = 6.0
            features[..., offset + 2, 0] = 7.0
        relative = wrist_relative_hand_features(features)
        shifted = features.clone()
        shifted[..., :42, :3] += torch.tensor([3.0, -2.0, 0.5])
        torch.testing.assert_close(
            wrist_relative_hand_features(shifted), relative, rtol=0.0, atol=0.0
        )
        geometry = masked_hand_bone_geometry(features)
        self.assertEqual(tuple(geometry.shape), (1, 2, 40, 5))
        torch.testing.assert_close(
            articulated_bone_distance(geometry, geometry), torch.zeros(1, 2)
        )
        rotated = features.clone()
        rotated[..., 1, :3] = torch.tensor([5.0, 1.0, 0.0])
        changed = articulated_bone_distance(
            geometry, masked_hand_bone_geometry(rotated)
        )
        self.assertGreater(float(changed.max()), 0.0)
        missing = torch.zeros_like(features)
        no_overlap = articulated_bone_distance(
            geometry, masked_hand_bone_geometry(missing)
        )
        torch.testing.assert_close(no_overlap, torch.zeros_like(no_overlap))

    def test_articulated_pose_branch_is_finite_and_partwise_only(self):
        with self.assertRaisesRegex(ValueError, "requires partwise_global"):
            Stage1V17Config(use_articulated_pose_embedding=True).validate()
        config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            temporal_encoder="partwise_global", part_depth=1,
            use_articulated_pose_embedding=True,
            dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )
        model = SLTStage1V17(config).eval()
        self.assertIsInstance(
            model.articulated_pose_embedding, ArticulatedPoseEmbeddingV17
        )
        features = torch.zeros(2, 32, 61, 5)
        features[:, :, :42, 3:] = 1.0
        with torch.inference_mode():
            logits = model(features)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_static_hand_frame_selection_is_missing_aware_and_controlled(self):
        features = torch.zeros(2, 5, 61, 5)
        features[0, :, :21, 3] = 1.0
        features[0, :, :21, 4] = torch.tensor(
            [0.2, 0.9, 0.7, 0.8, 0.6]
        ).view(5, 1)
        # Frame 2 has high quality but large motion; frame 3 is equally reliable
        # and static. Quality-only and low-motion selection must therefore differ.
        features[0, 0, :21, 0] = -1.0
        features[0, 2, :21, 0] = 2.0
        features[0, 3, :21, 0] = 2.0
        quality = static_hand_frame_weights(features, "quality", top_k=1)
        low_motion = static_hand_frame_weights(features, "low_motion", top_k=1)
        self.assertEqual(int(quality[0, :, 0].argmax()), 1)
        self.assertEqual(int(low_motion[0, :, 0].argmax()), 3)
        torch.testing.assert_close(quality[0, :, 0].sum(), torch.tensor(1.0))
        torch.testing.assert_close(low_motion[0, :, 0].sum(), torch.tensor(1.0))
        torch.testing.assert_close(quality[:, :, 1], torch.zeros(2, 5))
        torch.testing.assert_close(quality[1, :, 0], torch.zeros(5))

    def test_static_hand_token_is_small_partwise_and_exact_identity_at_init(self):
        with self.assertRaisesRegex(ValueError, "requires partwise_global"):
            Stage1V17Config(static_hand_token="quality").validate()
        with self.assertRaisesRegex(ValueError, "none, quality, or low_motion"):
            Stage1V17Config(static_hand_token="unknown").validate()
        common = dict(
            num_classes=7, dim=64, depth=1, heads=4,
            temporal_encoder="partwise_global", part_depth=1,
            dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )
        baseline = SLTStage1V17(Stage1V17Config(**common)).eval()
        static = SLTStage1V17(Stage1V17Config(
            **common, static_hand_token="low_motion"
        )).eval()
        shared = baseline.state_dict()
        static.load_state_dict(
            {**static.state_dict(), **{key: value for key, value in shared.items()}},
            strict=True,
        )
        features = torch.randn(2, 8, 61, 5)
        features[..., 3:] = 1.0
        with torch.inference_mode():
            expected = baseline(features)
            actual = static(features)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(static.parameter_count - baseline.parameter_count, 2241)
        self.assertEqual(float(static.static_hand_residual_scale.detach()), 0.0)

    def test_attention_score_mixer_starts_as_ordinary_attention(self):
        torch.manual_seed(31)
        attention = ScoreMixingMultiheadAttentionV17(64, 4, 0.0).eval()
        value = torch.randn(2, 8, 64)
        with torch.inference_mode():
            expected, _ = attention.base(
                value, value, value, need_weights=False
            )
            actual = attention(value)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(float(attention.score_mixer.weight.detach().abs().sum()), 0.0)

    def test_attention_score_mixing_is_tiny_and_partwise(self):
        common = dict(
            num_classes=7, dim=64, depth=1, heads=4,
            temporal_encoder="partwise_global", part_depth=1,
            dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )
        baseline = SLTStage1V17(Stage1V17Config(**common)).eval()
        mixed = SLTStage1V17(Stage1V17Config(
            **common, use_attention_score_mixing=True
        )).eval()
        self.assertEqual(mixed.parameter_count - baseline.parameter_count, 180)
        features = torch.zeros(2, 8, 61, 5)
        features[..., :42, 3:] = 1.0
        with torch.inference_mode():
            logits = mixed(features)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_articulated_pose_pretrain_loader_is_strict_and_branch_only(self):
        config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            temporal_encoder="partwise_global", part_depth=1,
            use_articulated_pose_embedding=True,
        )
        source = SLTStage1V17(config)
        target = SLTStage1V17(config)
        before_fusion = {
            key: value.detach().clone()
            for key, value in target.articulated_pose_fusion.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citizen = root / "citizen.json"
            supplement = root / "semlex.json"
            citizen.write_text("{}")
            supplement.write_text("{}")
            import hashlib
            checkpoint_path = root / "geometry.pth"
            torch.save(
                {
                    "format": "slt_v17_articulated_pose_pretrain",
                    "model_state_dict": source.articulated_pose_embedding.state_dict(),
                    "manifest_sha256": hashlib.sha256(citizen.read_bytes()).hexdigest(),
                    "supplement_manifest_sha256": hashlib.sha256(
                        supplement.read_bytes()
                    ).hexdigest(),
                    "schema_fingerprint": "schema",
                    "epochs": 2,
                    "triplets": 32,
                    "objective": "test",
                },
                checkpoint_path,
            )
            info = initialize_articulated_pose_embedding(
                target, checkpoint_path, citizen, supplement, "schema"
            )
        for key, value in source.articulated_pose_embedding.state_dict().items():
            torch.testing.assert_close(
                target.articulated_pose_embedding.state_dict()[key], value
            )
        for key, value in before_fusion.items():
            torch.testing.assert_close(
                target.articulated_pose_fusion.state_dict()[key], value
            )
        self.assertEqual(info["triplets"], 32)

    def test_masked_pose_spans_cover_every_part_and_loss_is_finite(self):
        features = torch.zeros(3, 8, 61, 5)
        features[..., 3:] = 1.0
        mask = anatomical_span_mask(
            features,
            mask_ratio=0.35,
            span_length=3,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(tuple(mask.shape), (3, 8, 61))
        for start, end in ((0, 21), (21, 42), (42, 57), (57, 61)):
            self.assertTrue(mask[:, :, start:end].any(dim=(1, 2)).all())
            self.assertTrue(torch.equal(
                mask[:, :, start], mask[:, :, start:end].all(dim=-1)
            ))
        prediction = torch.zeros_like(features, requires_grad=True)
        loss, pieces = reconstruction_loss(prediction, features, mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(pieces), {"xyz", "confidence", "presence"})
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_masked_pose_loader_is_strict_and_encoder_only(self):
        config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            temporal_encoder="partwise_global", part_depth=1,
        )
        source = SLTStage1V17(config)
        target = SLTStage1V17(config)
        prefixes = ("part_temporal_encoder.", "position", "blocks.")
        encoder_state = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
            if key.startswith(prefixes)
        }
        classifier_before = {
            key: value.detach().clone()
            for key, value in target.classifier.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            citizen = root / "citizen.json"
            supplement = root / "semlex.json"
            citizen.write_text("{}")
            supplement.write_text("{}")
            import hashlib
            checkpoint_path = root / "masked_pose.pth"
            torch.save(
                {
                    "format": "slt_v17_masked_pose_pretrain",
                    "encoder_state_dict": encoder_state,
                    "model_config": config.to_dict(),
                    "manifest_sha256": hashlib.sha256(citizen.read_bytes()).hexdigest(),
                    "supplement_manifest_sha256": hashlib.sha256(
                        supplement.read_bytes()
                    ).hexdigest(),
                    "schema_fingerprint": "schema",
                    "epochs": 2,
                    "objective": "test",
                },
                checkpoint_path,
            )
            info = initialize_masked_pose_encoder(
                target, checkpoint_path, citizen, supplement, "schema"
            )
        for key, value in encoder_state.items():
            torch.testing.assert_close(target.state_dict()[key], value)
        for key, value in classifier_before.items():
            torch.testing.assert_close(target.classifier.state_dict()[key], value)
        self.assertEqual(info["loaded_encoder_keys"], len(encoder_state))

    def test_bone_features_are_directed_missing_aware_and_temporal(self):
        features = torch.zeros(1, 3, 61, 5)
        features[..., 3:] = 1.0
        features[:, :, 7, 0] = 1.0
        features[:, 0, 8, 0] = 3.0
        features[:, 1:, 8, 0] = 4.0
        bones = masked_bone_features(features)
        self.assertEqual(tuple(bones.shape), (1, 3, 61, 6))
        torch.testing.assert_close(bones[0, 0, 8, :3], torch.tensor([2.0, 0.0, 0.0]))
        torch.testing.assert_close(bones[0, 1, 8, 3:], torch.tensor([1.0, 0.0, 0.0]))
        self.assertEqual(float(bones[0, 0, 42].abs().sum()), 0.0)
        missing = features.clone()
        missing[:, :, 7] = 0.0
        self.assertEqual(float(masked_bone_features(missing)[:, :, 8].abs().sum()), 0.0)

    def test_bone_feature_model_preserves_public_input_contract(self):
        model = SLTStage1V17(Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            use_bone_features=True, dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )).eval()
        features = torch.zeros(2, 32, 61, 5)
        features[:, :, :42, 3:] = 1.0
        with torch.inference_mode():
            logits = model(features)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_hand_angle_features_are_geometric_and_missing_aware(self):
        features = torch.zeros(1, 2, 61, 5)
        features[..., 3:] = 1.0
        features[..., 0, 0] = 0.0
        features[..., 1, 0] = 1.0
        features[..., 2, 0] = 2.0
        angles = masked_hand_angle_features(features)
        self.assertEqual(tuple(angles.shape), (1, 2, 61, 1))
        torch.testing.assert_close(angles[..., 1, 0], torch.full((1, 2), -1.0))
        self.assertEqual(float(angles[..., 42:, :].abs().sum()), 0.0)
        missing = features.clone()
        missing[..., 0, :] = 0.0
        self.assertEqual(float(masked_hand_angle_features(missing)[..., 1, :].abs().sum()), 0.0)
        torch.manual_seed(23)
        random_features = torch.randn(2, 4, 61, 5)
        random_features[..., 3:] = 1.0
        mirrored = mirror_v17(random_features)
        expected = masked_hand_angle_features(random_features).index_select(
            2, torch.tensor(MIRROR_NODE_INDEX)
        )
        torch.testing.assert_close(
            masked_hand_angle_features(mirrored), expected, rtol=1e-5, atol=1e-6
        )

    def test_hand_angle_model_preserves_public_input_contract(self):
        model = SLTStage1V17(Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            use_hand_angle_features=True,
            dropout=0.0, head_dropout=0.0, drop_path=0.0,
        )).eval()
        features = torch.zeros(2, 32, 61, 5)
        features[:, :, :42, 3:] = 1.0
        with torch.inference_mode():
            logits = model(features)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_partmix_replaces_exactly_one_whole_hand_and_preserves_contract(self):
        torch.manual_seed(11)
        features = torch.zeros(4, 3, 61, 5)
        for sample in range(4):
            features[sample, :, :, :3] = float(sample + 1)
            features[sample, :, :, 3:] = 1.0
        targets = torch.tensor([2, 4, 6, 8])
        mixed, primary, donor, weight = partmix_hands_v17(features, targets, 1.0)
        torch.testing.assert_close(primary, targets)
        torch.testing.assert_close(weight, torch.full((4,), 0.5))
        self.assertTrue(torch.all(donor != primary))
        for sample in range(4):
            changed_left = not torch.equal(mixed[sample, :, :21], features[sample, :, :21])
            changed_right = not torch.equal(mixed[sample, :, 21:42], features[sample, :, 21:42])
            self.assertNotEqual(changed_left, changed_right)
            torch.testing.assert_close(mixed[sample, :, 42:], features[sample, :, 42:])
        self.assertTrue(torch.all((mixed[..., 3] == 0.0) | (mixed[..., 3] == 1.0)))
        torch.testing.assert_close(mixed[..., :3] * (1.0 - mixed[..., 3:4]), torch.zeros_like(mixed[..., :3]))

    def test_partmix_zero_is_identity_and_loss_matches_manual_mix(self):
        features = torch.randn(3, 2, 61, 5)
        targets = torch.tensor([0, 1, 2])
        same, primary, donor, weight = partmix_hands_v17(features, targets, 0.0)
        self.assertIs(same, features)
        torch.testing.assert_close(primary, targets)
        torch.testing.assert_close(donor, targets)
        torch.testing.assert_close(weight, torch.ones(3))

        logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
        primary = torch.tensor([0, 1])
        donor = torch.tensor([1, 0])
        mix_weight = torch.tensor([0.5, 1.0])
        actual = partmix_cross_entropy(
            logits, primary, donor, mix_weight, label_smoothing=0.0
        )
        first = 0.5 * (
            torch.nn.functional.cross_entropy(logits[:1], primary[:1])
            + torch.nn.functional.cross_entropy(logits[:1], donor[:1])
        )
        second = torch.nn.functional.cross_entropy(logits[1:], primary[1:])
        torch.testing.assert_close(actual, (first + second) / 2.0)

    def test_flat_graph_residual_warm_start_is_exact_at_zero_gate(self):
        flat_config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            spatial_encoder="flat", graph_node_dim=32,
            graph_layers=1, graph_heads=4,
        )
        hybrid_config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4,
            spatial_encoder="flat_graph_residual", graph_node_dim=32,
            graph_layers=1, graph_heads=4,
        )
        flat = SLTStage1V17(flat_config).eval()
        hybrid = SLTStage1V17(hybrid_config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            import hashlib
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            checkpoint = root / "flat.pth"
            torch.save(
                {
                    "format": "slt_stage1_v17",
                    "epoch": 3,
                    "model_config": flat_config.to_dict(),
                    "model_state_dict": flat.state_dict(),
                    "manifest_sha256": manifest_hash,
                    "schema_fingerprint": "schema",
                    "validation_metrics": {"top1": 90.0},
                },
                checkpoint,
            )
            info = initialize_flat_graph_residual(
                hybrid, checkpoint, manifest, "schema"
            )
        features = torch.randn(2, 32, 61, 5)
        features[..., 3:] = 1.0
        with torch.inference_mode():
            flat_logits = flat(features)
            hybrid_logits = hybrid(features)
        self.assertTrue(torch.equal(flat_logits, hybrid_logits))
        self.assertEqual(float(hybrid.graph_residual_scale.detach()), 0.0)
        self.assertTrue(info["zero_initialized_residual"])

    def test_exact_replay_finetune_loader_restores_selected_state_strictly(self):
        config = Stage1V17Config(num_classes=7, dim=64, depth=1, heads=4)
        source = SLTStage1V17(config).eval()
        restored = SLTStage1V17(config).eval()
        label_to_index = {f"CLASS_{index}": index for index in range(7)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            import hashlib
            checkpoint = root / "selected.pth"
            torch.save(
                {
                    "format": "slt_stage1_v17",
                    "epoch": 9,
                    "model_config": config.to_dict(),
                    "model_state_dict": source.state_dict(),
                    "manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "schema_fingerprint": "schema",
                    "label_to_index": label_to_index,
                    "validation_metrics": {"top1": 91.0},
                    "training_data_provenance": {
                        "citizen_test_accessed": False,
                        "semlex_test_accessed": False,
                    },
                },
                checkpoint,
            )
            info = initialize_exact_stage1_finetune(
                restored, checkpoint, manifest, "schema", label_to_index
            )
        for expected, actual in zip(source.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(info["strict_state_dict"])

    def test_real_phonology_targets_cover_the_frozen_vocabulary(self):
        path = Path("active/v17/citizen100_phonology.json")
        manifest = Path("active/v17/citizen100_manifest.json")
        if not path.exists():
            self.skipTest("frozen phonology targets are unavailable")
        classes = json.loads(manifest.read_text())["classes"]
        label_to_index = {
            item["canonical_label"]: int(item["class_index"]) for item in classes
        }
        supervision = load_phonology_supervision(path, manifest, label_to_index)
        self.assertEqual(len(supervision["head_sizes"]), 10)
        self.assertEqual(
            dict(supervision["head_sizes"])["handshape"], 30
        )
        self.assertEqual(
            len(supervision["target_maps"]["minor_location"]), 100
        )

    def test_phonology_auxiliary_loss_ignores_missing_class_targets(self):
        logits = {
            "shape": torch.tensor([[4.0, 0.0], [0.0, 4.0]], requires_grad=True),
            "location": torch.tensor([[3.0, 0.0], [3.0, 0.0]], requires_grad=True),
        }
        maps = {
            "shape": torch.tensor([0, 1]),
            "location": torch.tensor([0, -100]),
        }
        loss = phonology_auxiliary_loss(logits, maps, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits["shape"].grad)

    def test_graph_encoder_is_anatomical_missing_aware_and_finite(self):
        adjacency = anatomical_adjacency_v17()
        self.assertEqual(tuple(adjacency.shape), (61, 61))
        self.assertTrue(torch.allclose(adjacency, adjacency.T))
        self.assertGreater(float(adjacency[0, 1]), 0.0)
        self.assertEqual(float(adjacency[0, 41]), 0.0)

        config = Stage1V17Config(
            num_classes=7,
            dim=64,
            depth=1,
            heads=4,
            spatial_encoder="graph_parts",
            graph_node_dim=32,
            graph_layers=1,
            graph_heads=4,
            phonology_head_sizes=(("shape", 5), ("location", 3)),
        )
        model = SLTStage1V17(config).eval()
        features = torch.zeros(2, 32, 61, 5)
        features[0, :, :21, 3:] = 1.0
        features[0, :, :21, :3] = torch.randn(32, 21, 3) * 0.1
        logits, embedding, auxiliary = model.forward_multitask(features)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(tuple(embedding.shape), (2, 64))
        self.assertEqual(tuple(auxiliary["shape"].shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all())

    def test_class_source_weights_equalize_both_expected_margins(self):
        targets = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2])
        sources = [
            "citizen", "citizen", "semlex",
            "citizen", "semlex", "semlex", "semlex",
            "citizen", "citizen",
        ]
        weights, summary = class_source_balanced_weights(targets, sources, 3)
        self.assertEqual(summary["mode"], "class_source_balanced_replacement")
        for target in range(3):
            self.assertAlmostEqual(float(weights[targets == target].sum()), 1.0 / 3.0)
        for source in ("citizen", "semlex"):
            total = sum(
                float(weights[index])
                for index, value in enumerate(sources)
                if value == source
            )
            self.assertAlmostEqual(total, 0.5)

    def test_class_source_weights_accept_explicit_source_margins(self):
        targets = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2])
        sources = [
            "citizen", "citizen", "semlex",
            "citizen", "semlex", "semlex", "local",
            "citizen", "local",
        ]
        requested = {"citizen": 0.45, "semlex": 0.45, "local": 0.10}
        weights, summary = class_source_balanced_weights(
            targets, sources, 3, requested
        )
        for source, expected in requested.items():
            total = sum(
                float(weights[index])
                for index, value in enumerate(sources)
                if value == source
            )
            self.assertAlmostEqual(total, expected)
            self.assertAlmostEqual(
                summary["expected_source_probabilities"][source], expected
            )
        self.assertEqual(
            parse_source_probabilities("citizen=.45,semlex=.45,local=.10"),
            requested,
        )

    def test_local_review_loader_filters_to_explicit_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "local"
            write_archive(feature_root / "A/approved.v17.npz", 1.0)
            write_archive(feature_root / "A/rejected.v17.npz", 2.0)
            manifest = {
                "split_eligibility": "train_only_after_human_review",
                "videos": [
                    {
                        "canonical_label": "A",
                        "raw_path": "raw/A/approved.mp4",
                        "feature_path": "elsewhere/A/approved.v17.npz",
                        "consensus_tier": "tier_a_dual_top1",
                    },
                    {
                        "canonical_label": "A",
                        "raw_path": "raw/A/rejected.mp4",
                        "feature_path": "elsewhere/A/rejected.v17.npz",
                        "consensus_tier": "tier_b_dual_top5_one_top1",
                    },
                ],
            }
            manifest_path = root / "local.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset = LocalReviewSupplementV17Dataset(
                feature_root, manifest_path, {"A": 0, "B": 1}
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.files[0].name, "approved.v17.npz")

    def test_local_validation_loader_enforces_explicit_overlap_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "local_val"
            write_archive(feature_root / "A/example.v17.npz", 1.0)
            manifest = {
                "format": "slt_v17_local_deep_clean_final_v1",
                "split": "val",
                "split_eligibility": "validation_nonsigner_disjoint_user_approved",
                "signer_disjoint": False,
                "signer_overlap_user_approved": True,
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "extraction_complete": True,
                "extractor_schema_fingerprint": schema_fingerprint(V17Config()),
                "selected_clips": 1,
                "videos": [
                    {
                        "canonical_label": "A",
                        "item_id": "example",
                        "feature_path": "landmarks/val/A/example.v17.npz",
                        "local_split": "val",
                        "training_eligible": False,
                        "validation_eligible": True,
                    }
                ],
            }
            manifest_path = root / "local_val.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset = LocalValidationV17Dataset(
                feature_root, manifest_path, {"A": 0, "B": 1}
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.files[0].name, "example.v17.npz")

            manifest["signer_overlap_user_approved"] = False
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "validation contract"):
                LocalValidationV17Dataset(
                    feature_root, manifest_path, {"A": 0, "B": 1}
                )

    def test_finalized_local_train_loader_requires_extraction_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "local_train"
            write_archive(feature_root / "A/example.v17.npz", 1.0)
            manifest = {
                "format": "slt_v17_local_deep_clean_final_v1",
                "split": "train",
                "split_eligibility": "train_only_after_human_review",
                "signer_disjoint": False,
                "signer_overlap_user_approved": True,
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "extraction_complete": True,
                "extractor_schema_fingerprint": schema_fingerprint(V17Config()),
                "selected_clips": 1,
                "videos": [
                    {
                        "canonical_label": "A",
                        "raw_path": "raw/train/A/example.mp4",
                        "feature_path": "landmarks/train/A/example.v17.npz",
                        "consensus_tier": "owner_approved_v16_deep_clean",
                        "local_split": "train",
                        "training_eligible": True,
                        "validation_eligible": False,
                    }
                ],
            }
            manifest_path = root / "local_train.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset = LocalReviewSupplementV17Dataset(
                feature_root,
                manifest_path,
                {"A": 0, "B": 1},
                allowed_tiers=("owner_approved_v16_deep_clean",),
            )
            self.assertEqual(len(dataset), 1)

            manifest["extraction_complete"] = False
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "training extraction contract"):
                LocalReviewSupplementV17Dataset(
                    feature_root,
                    manifest_path,
                    {"A": 0, "B": 1},
                    allowed_tiers=("owner_approved_v16_deep_clean",),
                )

    def test_official_exact_supplement_enforces_variant_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "local"
            write_archive(feature_root / "A/exact.v17.npz", 1.0)
            row = {
                "canonical_label": "A",
                "raw_path": "raw/A/exact.mp4",
                "feature_path": "landmarks/A/exact.v17.npz",
                "consensus_tier": "official_asllex_signbank_exact",
                "training_eligible": True,
                "citizen_asl_lex_code": "asllex-code",
                "asllex_entry_id": "asllex-entry",
                "signbank_annotation_id": "EXACT-VARIANT",
                "asllrp_entry_variant": "EXACT-VARIANT",
                "asllrp_occurrence": "EXACT-VARIANT+",
            }
            manifest = {
                "format": "slt_v17_asllrp_asllex_exact_supplement",
                "split_eligibility": "train_only_official_asllex_signbank_cross_reference",
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "videos": [row],
            }
            manifest_path = root / "official.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset = LocalReviewSupplementV17Dataset(
                feature_root,
                manifest_path,
                {"A": 0},
                allowed_tiers=("official_asllex_signbank_exact",),
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.files[0].name, "exact.v17.npz")

            manifest["videos"][0]["asllrp_occurrence"] = "WRONG"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "variant contract"):
                LocalReviewSupplementV17Dataset(
                    feature_root,
                    manifest_path,
                    {"A": 0},
                    allowed_tiers=("official_asllex_signbank_exact",),
                )

            manifest["format"] = "slt_v17_asllvd_asllex_exact_supplement"
            manifest["videos"][0].update(
                {"variant_gloss": "EXACT-VARIANT+", "signer": "Signer A"}
            )
            manifest_path.write_text(json.dumps(manifest))
            dataset = LocalReviewSupplementV17Dataset(
                feature_root,
                manifest_path,
                {"A": 0},
                allowed_tiers=("official_asllex_signbank_exact",),
            )
            self.assertEqual(len(dataset), 1)

    def test_semlex_supplement_is_manifest_locked_and_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "semlex"
            write_archive(feature_root / "A/example.v17.npz", 1.0)
            manifest = {
                "split": "train_only",
                "selected_clips": 1,
                "videos": [
                    {
                        "canonical_label": "A",
                        "semlex_video_id": "example",
                        "semlex_split": "train",
                    }
                ],
            }
            manifest_path = root / "semlex.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset = SemLexSupplementV17Dataset(
                feature_root, manifest_path, {"A": 0, "B": 1}
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(int(dataset[0][1]), 0)
            self.assertEqual(tuple(dataset[0][0].shape), (32, 61, 5))

            manifest["videos"][0]["semlex_split"] = "test"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "non-train"):
                SemLexSupplementV17Dataset(
                    feature_root, manifest_path, {"A": 0, "B": 1}
                )

    def test_mediapipe_schema_is_explicitly_accepted_without_mixing_apple(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"classes": [{"canonical_label": "A", "class_index": 0}]})
            )
            config = MediaPipeV17Config(
                minimum_hand_detection_confidence=0.50,
                minimum_hand_presence_confidence=0.50,
                minimum_hand_tracking_confidence=0.50,
            )
            fingerprint = mediapipe_schema_fingerprint(config)
            self.assertEqual(fingerprint, extractor_schema_fingerprint("mediapipe_t50"))
            write_archive(
                root / "features/train/A/example.v17.npz",
                fingerprint=fingerprint,
            )
            dataset = Citizen100V17Dataset(
                root / "features", "train", manifest_path,
                expected_schema=fingerprint,
            )
            self.assertEqual(len(dataset), 1)
            with self.assertRaisesRegex(ValueError, "schema fingerprint mismatch"):
                Citizen100V17Dataset(root / "features", "train", manifest_path)

    def test_official_directory_split_and_rejection_ledger_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "classes": [
                    {"canonical_label": "A", "class_index": 0},
                    {"canonical_label": "B", "class_index": 1},
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            write_archive(root / "features/train/A/keep-A.v17.npz", 1.0)
            write_archive(root / "features/train/A/drop-A.v17.npz", 2.0)
            write_archive(root / "features/train/B/keep-B.v17.npz", 3.0)
            rejection_path = root / "rejections.csv"
            with rejection_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("split", "canonical_label", "video"),
                )
                writer.writeheader()
                writer.writerow(
                    {"split": "train", "canonical_label": "A", "video": "drop-A.mp4"}
                )
            dataset = Citizen100V17Dataset(
                root / "features", "train", manifest_path, rejection_path
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.targets.tolist(), [0, 1])
            self.assertEqual(tuple(dataset[0][0].shape), (32, 61, 5))

    def test_real_citizen_counts_respect_both_reviewed_rejections(self):
        root = Path("data/local/citizen100_v17/landmarks")
        if not root.exists():
            self.skipTest("Citizen100 v17 features are unavailable")
        common = {
            "root": root,
            "manifest_path": Path("active/v17/citizen100_manifest.json"),
            "rejection_path": Path("data/local/citizen100_v17/rejections.csv"),
            "cache": False,
        }
        self.assertEqual(len(Citizen100V17Dataset(split="train", **common)), 1475)
        self.assertEqual(len(Citizen100V17Dataset(split="val", **common)), 378)
        self.assertEqual(len(Citizen100V17Dataset(split="test", **common)), 1247)

    def test_real_semlex_balanced_selection_has_expected_count(self):
        root = Path(
            "data/local/semlex_citizen100_train_audit/balanced_landmarks_v17"
        )
        manifest_path = Path(
            "data/local/semlex_citizen100_train_audit/balanced_train_candidates.json"
        )
        citizen_manifest = Path("active/v17/citizen100_manifest.json")
        if not root.exists() or not manifest_path.exists():
            self.skipTest("balanced SemLex v17 features are unavailable")
        classes = json.loads(citizen_manifest.read_text())["classes"]
        label_to_index = {
            item["canonical_label"]: int(item["class_index"]) for item in classes
        }
        dataset = SemLexSupplementV17Dataset(
            root, manifest_path, label_to_index, cache=False
        )
        self.assertEqual(len(dataset), 1058)
        self.assertNotIn(
            label_to_index["TAKE"], set(dataset.targets.tolist())
        )

    def test_real_semlex_full_clean_selection_has_expected_count(self):
        root = Path(
            "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"
        )
        manifest_path = Path(
            "data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json"
        )
        citizen_manifest = Path("active/v17/citizen100_manifest.json")
        if not root.exists() or not manifest_path.exists():
            self.skipTest("full-clean SemLex v17 features are unavailable")
        classes = json.loads(citizen_manifest.read_text())["classes"]
        label_to_index = {
            item["canonical_label"]: int(item["class_index"]) for item in classes
        }
        dataset = SemLexSupplementV17Dataset(
            root, manifest_path, label_to_index, cache=False
        )
        self.assertEqual(len(dataset), 1388)
        self.assertNotIn(label_to_index["TAKE"], set(dataset.targets.tolist()))


class V17Stage1FeatureTest(unittest.TestCase):
    def test_derivatives_do_not_cross_missing_observations(self):
        value = torch.zeros(1, 4, 61, 5)
        value[0, 0, 0, (0, 3, 4)] = torch.tensor((1.0, 1.0, 1.0))
        value[0, 2, 0, (0, 3, 4)] = torch.tensor((3.0, 1.0, 1.0))
        value[0, 3, 0, (0, 3, 4)] = torch.tensor((4.0, 1.0, 1.0))
        derived = masked_temporal_features(value)
        self.assertEqual(tuple(derived.shape), (1, 4, 61, 11))
        self.assertEqual(derived[0, :3, 0, 5].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(float(derived[0, 3, 0, 5]), 1.0)
        self.assertEqual(float(derived[0, 3, 0, 8]), 0.0)

    def test_mirror_is_an_involution_for_hands_face_and_body(self):
        self.assertEqual(len(MIRROR_NODE_INDEX), 61)
        value = torch.arange(2 * 32 * 61 * 5, dtype=torch.float32).reshape(2, 32, 61, 5)
        restored = mirror_v17(mirror_v17(value))
        torch.testing.assert_close(restored, value, rtol=0.0, atol=0.0)
        mirrored = mirror_v17(value)
        torch.testing.assert_close(mirrored[:, :, 0, 1:], value[:, :, 21, 1:])
        torch.testing.assert_close(mirrored[:, :, 42, 1:], value[:, :, 43, 1:])
        torch.testing.assert_close(mirrored[:, :, 57, 1:], value[:, :, 58, 1:])

    def test_augmentation_preserves_binary_missing_contract(self):
        torch.manual_seed(17)
        value = torch.zeros(8, 32, 61, 5)
        value[:, :, :42, :3] = 0.25
        value[:, :, :42, 3:] = 1.0
        augmented = augment_v17(value)
        presence = augmented[..., 3]
        self.assertTrue(torch.all((presence == 0) | (presence == 1)))
        missing = presence == 0
        self.assertEqual(float(augmented[..., :3][missing].abs().max()), 0.0)
        self.assertEqual(float(augmented[..., 4][missing].abs().max()), 0.0)

    def test_arbitrary_camera_roll_is_invertible_and_missing_safe(self):
        torch.manual_seed(1701)
        value = torch.randn(4, 9, 61, 5)
        value[..., 3] = (torch.rand(4, 9, 61) > 0.2).float()
        value[..., 4] = torch.rand(4, 9, 61) * value[..., 3]
        value[..., :3] *= value[..., 3:4]
        angles = torch.tensor([17.0, 37.0, 123.0, 180.0]) * torch.pi / 180.0
        rotated = rotate_camera_roll_v17(value, angles)
        restored = rotate_camera_roll_v17(rotated, -angles)
        torch.testing.assert_close(restored, value, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(rotated[..., 2:], value[..., 2:], rtol=0.0, atol=0.0)
        missing = rotated[..., 3] == 0
        self.assertEqual(float(rotated[..., :3][missing].abs().max()), 0.0)

    def test_full_roll_augmentation_accepts_the_entire_circle(self):
        value = torch.zeros(8, 32, 61, 5)
        value[:, :, :42, 0] = 0.25
        value[:, :, :42, 3:] = 1.0
        torch.manual_seed(91)
        augmented = augment_v17(
            value,
            full_roll_probability=1.0,
            maximum_roll_degrees=180.0,
            mild_roll_degrees=0.0,
        )
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertTrue((augmented[..., 1].abs().sum(dim=(1, 2)) > 0).all())

    def test_camera_roll_canonicalization_is_continuously_invariant(self):
        torch.manual_seed(51)
        value = torch.randn(4, 12, 61, 5)
        value[..., 3:] = 1.0
        value[..., 57, :2] = torch.tensor([-0.5, 0.08])
        value[..., 58, :2] = torch.tensor([0.5, -0.08])
        angles = torch.tensor([17.0, 37.0, 123.0, 180.0]) * torch.pi / 180.0
        expected = canonicalize_camera_roll_v17(value)
        rotated = rotate_camera_roll_v17(value, angles)
        actual = canonicalize_camera_roll_v17(rotated)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)

    def test_camera_roll_canonicalization_falls_back_to_eyes_or_identity(self):
        value = torch.zeros(2, 6, 61, 5)
        value[0, :, 42, (0, 1, 3, 4)] = torch.tensor([-0.4, 0.2, 1.0, 0.8])
        value[0, :, 43, (0, 1, 3, 4)] = torch.tensor([0.4, -0.2, 1.0, 0.8])
        value[0, :, :42, 3:] = 1.0
        value[0, :, :42, 0] = 0.3
        canonical = canonicalize_camera_roll_v17(value)
        self.assertTrue(torch.isfinite(canonical).all())
        torch.testing.assert_close(canonical[1], value[1], rtol=0.0, atol=0.0)

    def test_supervised_contrastive_loss_handles_positive_and_empty_batches(self):
        embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
        with_positive = supervised_contrastive_loss(
            embeddings, torch.tensor([0, 0, 1])
        )
        self.assertTrue(torch.isfinite(with_positive))
        self.assertGreater(float(with_positive), 0.0)
        no_positive = supervised_contrastive_loss(
            embeddings, torch.tensor([0, 1, 2])
        )
        self.assertEqual(float(no_positive), 0.0)

    def test_model_consumes_only_the_five_channel_archive_contract(self):
        config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4, drop_path=0.0
        )
        model = SLTStage1V17(config).eval()
        value = torch.zeros(3, 32, 61, 5)
        value[:, :, :42, 3:] = 1.0
        logits, embeddings = model(value, return_embeddings=True)
        self.assertEqual(tuple(logits.shape), (3, 7))
        self.assertEqual(tuple(embeddings.shape), (3, 64))
        self.assertTrue(torch.isfinite(logits).all())

    def test_face_modality_cannot_observe_hand_nodes(self):
        torch.manual_seed(4)
        config = Stage1V17Config(
            num_classes=7, dim=64, depth=1, heads=4, drop_path=0.0,
            dropout=0.0, head_dropout=0.0, use_pairwise=False,
            input_modality="face",
        )
        model = SLTStage1V17(config).eval()
        baseline = torch.zeros(2, 32, 61, 5)
        baseline[:, :, 42:57, 3:] = 1.0
        changed_hands = baseline.clone()
        changed_hands[:, :, :42, :3] = torch.randn(2, 32, 42, 3)
        changed_hands[:, :, :42, 3:] = 1.0
        with torch.no_grad():
            expected = model(baseline)
            actual = model(changed_hands)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_ema_warm_start_is_not_dominated_by_random_initialization(self):
        layer = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            layer.weight.zero_()
        ema = ExponentialMovingAverage(layer, decay=0.999)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        ema.update(layer)
        self.assertGreater(float(ema.shadow["weight"]), 0.8)

    def test_model_soup_blends_floating_state_and_checks_bounds(self):
        first = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2)}
        second = {"weight": torch.tensor([5.0, 7.0]), "count": torch.tensor(2)}
        blended = blend_state_dicts(first, second, 0.25)
        torch.testing.assert_close(blended["weight"], torch.tensor([4.0, 6.0]))
        self.assertEqual(int(blended["count"]), 2)
        with self.assertRaises(ValueError):
            blend_state_dicts(first, second, 1.1)

    def test_multimodal_ensemble_normalizes_ids_and_probabilities(self):
        self.assertEqual(
            normalized_item_id("val/GOOD/clip.mouth_rgb_v17.npz"),
            "GOOD/clip",
        )
        self.assertEqual(
            normalized_item_id("val/GOOD/clip.visual_speech_v17.npz"),
            "GOOD/clip",
        )
        self.assertEqual(
            normalized_item_id("val/GOOD/clip.hand_mobileclip2_v17.npz"),
            "GOOD/clip",
        )
        self.assertEqual(normalized_item_id("GOOD/clip.v17.npz"), "GOOD/clip")
        scores = probabilities(np.asarray([[1000.0, 999.0]], dtype=np.float32))
        self.assertTrue(np.isfinite(scores).all())
        self.assertAlmostEqual(float(scores.sum()), 1.0)
        standardized = per_sample_zscore(
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        )
        self.assertAlmostEqual(float(standardized.mean()), 0.0)
        self.assertAlmostEqual(float(standardized.std()), 1.0)


if __name__ == "__main__":
    unittest.main()
