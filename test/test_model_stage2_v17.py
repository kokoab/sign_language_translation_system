import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from active.v17.model_hand_mobileclip2_v17 import HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17
from active.v17.model_stage2_v17 import (
    ctc_sequence_log_probability,
    ctc_exact_match_mask,
    FROZEN_TEMPORAL_FEATURE_DIM,
    Stage2ContextAdapterV17,
    Stage2DualHeadV17,
    Stage2TemporalHeadV17,
    Stage2V17Config,
    warm_start_dual_stage2,
)
from active.v17.export_stage2_coreml_v17 import ManualMaskedMHA
from active.v17.train_stage_2_v17 import (
    SyntheticCompositionDataset,
    compose_signer_voice,
    resample_temporal,
)


class Stage2ModelTests(unittest.TestCase):
    def test_manual_masked_attention_matches_torch(self):
        torch.manual_seed(17)
        source = torch.nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True).eval()
        manual = ManualMaskedMHA(source).eval()
        value = torch.randn(2, 7, 32)
        mask = torch.tensor([
            [False, False, False, False, True, True, True],
            [False, False, False, False, False, False, True],
        ])
        with torch.inference_mode():
            expected, _ = source(value, value, value, key_padding_mask=mask, need_weights=False)
            actual, _ = manual(value, value, value, key_padding_mask=mask, need_weights=False)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    def test_ctc_exact_match_mask_requires_exact_collapsed_sequence(self):
        predictions = torch.tensor([
            [4, 4, 0, 9, 9, 0],
            [4, 0, 4, 9, 0, 0],
            [4, 9, 7, 0, 0, 0],
        ])
        logits = torch.full((3, 6, 12), -10.0)
        logits.scatter_(2, predictions.unsqueeze(-1), 10.0)
        matched = ctc_exact_match_mask(logits, torch.tensor([6, 6, 4]), (4, 9))
        self.assertEqual(matched.tolist(), [True, False, False])

    def test_ctc_sequence_probability_matches_torch_reference(self):
        torch.manual_seed(13)
        logits = torch.randn(9, 7)
        ctc_tokens = (2, 4, 3)
        actual = ctc_sequence_log_probability(logits, ctc_tokens)
        expected = -torch.nn.functional.ctc_loss(
            logits.log_softmax(dim=-1).unsqueeze(1),
            torch.tensor(ctc_tokens),
            torch.tensor([len(logits)]),
            torch.tensor([len(ctc_tokens)]),
            reduction="sum",
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_signer_voice_composition_restores_timing_and_repackages_windows(self):
        tokens = np.zeros((2, 32, FROZEN_TEMPORAL_FEATURE_DIM), np.float32)
        tokens[0, :, 0] = np.linspace(0, 1, 32)
        tokens[1, :, 0] = np.linspace(2, 3, 32)
        value = compose_signer_voice(
            tokens, [12, 14], np.zeros(FROZEN_TEMPORAL_FEATURE_DIM, np.float32),
            np.ones(FROZEN_TEMPORAL_FEATURE_DIM, np.float32),
            context_frames=5, bridge_frames=2, max_trim_frames=3,
            minimum_keep_frames=4,
        )
        self.assertEqual(value.shape[1:], (32, FROZEN_TEMPORAL_FEATURE_DIM))
        self.assertIn(value.shape[0], (1, 2))
        self.assertTrue(np.isfinite(value).all())
        resized = resample_temporal(tokens[0], 7)
        self.assertEqual(resized.shape, (7, FROZEN_TEMPORAL_FEATURE_DIM))
        self.assertAlmostEqual(float(resized[0, 0]), 0.0)
        self.assertAlmostEqual(float(resized[-1, 0]), 1.0)

    def test_hand_frame_api_preserves_pooled_features(self):
        torch.manual_seed(17)
        model = HandMobileCLIP2Stage1V17(HandMobileCLIP2Stage1Config(dropout=0.0, head_dropout=0.0))
        model.eval()
        embeddings = torch.randn(2, 16, 3, 512)
        valid = torch.ones(2, 16, 3, dtype=torch.bool)
        boxes = torch.rand(2, 16, 3, 4)
        frames, frame_valid = model.encode_frames(embeddings, valid, boxes)
        scores = model.frame_attention(frames).squeeze(-1).masked_fill(~frame_valid, -1e4)
        weights = torch.softmax(scores, dim=1) * frame_valid
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        expected = (frames * weights.unsqueeze(-1)).sum(dim=1)
        self.assertTrue(torch.allclose(expected, model.forward_features(embeddings, valid, boxes)))

    def test_ctc_head_shapes_and_lengths(self):
        model = Stage2TemporalHeadV17()
        value = torch.randn(3, 5, 32, FROZEN_TEMPORAL_FEATURE_DIM)
        mask = torch.tensor([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
        ], dtype=torch.bool)
        logits, lengths = model(value, mask)
        self.assertEqual(tuple(logits.shape), (3, 40, 101))
        self.assertEqual(lengths.tolist(), [40, 24, 8])
        self.assertEqual(model.config.blank_index, 0)

    def test_dual_head_warm_start_extends_positions_and_preserves_locked_logits(self):
        torch.manual_seed(19)
        source = Stage2TemporalHeadV17(Stage2V17Config(max_windows=8))
        checkpoint = {
            "format": "slt_stage2_ctc_v17",
            "model_state_dict": source.state_dict(),
        }
        dual = Stage2DualHeadV17(Stage2V17Config(max_windows=40), auxiliary_num_classes=448)
        warm_start_dual_stage2(dual, checkpoint)
        self.assertTrue(torch.equal(dual.locked.position[:, :64], source.position))
        self.assertTrue(torch.equal(dual.locked.ctc_head.weight, source.ctc_head.weight))
        self.assertTrue(torch.equal(
            dual.auxiliary_ctc_head.weight[1:101], source.ctc_head.weight[1:101]
        ))
        value = torch.randn(2, 10, 32, FROZEN_TEMPORAL_FEATURE_DIM)
        mask = torch.ones(2, 10, dtype=torch.bool)
        locked, locked_lengths = dual.forward_locked(value, mask)
        auxiliary, auxiliary_lengths = dual.forward_auxiliary(value, mask)
        self.assertEqual(tuple(locked.shape), (2, 80, 101))
        self.assertEqual(tuple(auxiliary.shape), (2, 80, 449))
        self.assertEqual(locked_lengths.tolist(), auxiliary_lengths.tolist())

    def test_context_adapter_changes_only_selected_ctc_classes(self):
        torch.manual_seed(23)
        base = Stage2TemporalHeadV17(Stage2V17Config(dropout=0.0)).eval()
        summary_dim = FROZEN_TEMPORAL_FEATURE_DIM * 4
        class_indices = torch.tensor([9, 25, 50, 86])
        adapted = Stage2ContextAdapterV17(
            base,
            feature_mode="mean_std_max_delta",
            scaler_mean=torch.zeros(summary_dim),
            scaler_scale=torch.ones(summary_dim),
            coefficients=torch.randn(len(class_indices), summary_dim),
            intercept=torch.zeros(len(class_indices)),
            class_indices=class_indices,
            target_class_indices=(9, 86),
            weight=0.5,
        ).eval()
        value = torch.randn(2, 3, 32, FROZEN_TEMPORAL_FEATURE_DIM)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        with torch.inference_mode():
            base_logits, base_lengths = base(value, mask)
            adapted_logits, adapted_lengths = adapted(value, mask)
        difference = adapted_logits - base_logits
        untouched = [index for index in range(101) if index not in {10, 87}]
        self.assertTrue(torch.equal(difference[..., untouched], torch.zeros_like(
            difference[..., untouched]
        )))
        self.assertGreater(float(difference[..., [10, 87]].abs().sum()), 0.0)
        self.assertTrue(torch.equal(base_lengths, adapted_lengths))
        self.assertTrue(torch.equal(difference[1, 16:], torch.zeros_like(difference[1, 16:])))

    def test_mixed_synthetic_dataset_enforces_target_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "pool.npz"
            metadata = {"source_split": "citizen_asllrp_train_only_replay"}
            np.savez_compressed(
                pool,
                frozen_features=np.zeros((2, 32, FROZEN_TEMPORAL_FEATURE_DIM), np.float16),
                target_indices=np.asarray([4, 9], np.int64),
                source_codes=np.asarray([0, 1], np.uint8),
                metadata_json=np.array(json.dumps(metadata)),
            )
            digest = hashlib.sha256(pool.read_bytes()).hexdigest()
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "pool_sha256": digest,
                "rows": [{
                    "sequence_id": "asllrp_0",
                    "source": "synthetic_asllrp_contextual_train",
                    "target_indices": [9],
                    "pool_indices": [1],
                    "leading_padding_frames": 7,
                }],
            }))
            dataset = SyntheticCompositionDataset(pool, plan)
            sample = dataset[0]
            self.assertEqual(sample.source, "synthetic_asllrp_contextual_train")
            self.assertEqual(sample.targets.tolist(), [10])
            self.assertEqual(sample.features.shape, (2, 32, FROZEN_TEMPORAL_FEATURE_DIM))
            self.assertTrue(np.all(sample.features[0, :7] == 0))

            invalid = json.loads(plan.read_text())
            invalid["rows"][0]["source"] = "synthetic_citizen_train"
            plan.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "pool source mismatch"):
                SyntheticCompositionDataset(pool, plan)[0]

    def test_multivoice_style_uses_target_neutral_with_asllrp_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pool = root / "pool.npz"
            features = np.zeros((3, 32, FROZEN_TEMPORAL_FEATURE_DIM), np.float16)
            features[0, :, 0] = np.linspace(0.0, 1.0, 32)
            features[1, :, 0] = np.linspace(2.0, 3.0, 32)
            features[2, :, 0] = 5.0
            metadata = {
                "source_split": "citizen_semlex_asllrp_train_only_replay",
                "transition_scale_source_code": 1,
                "source_code_map": {"0": "citizen", "1": "asllrp", "2": "semlex"},
            }
            np.savez_compressed(
                pool,
                frozen_features=features,
                target_indices=np.asarray([4, 9, 12], np.int64),
                source_codes=np.asarray([1, 1, 2], np.uint8),
                metadata_json=np.array(json.dumps(metadata)),
            )
            plan = root / "plan.json"
            plan.write_text(json.dumps({
                "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
                "signer_pool_indices": {"semlex:7": [2]},
                "rows": [{
                    "sequence_id": "transferred_0",
                    "source": "synthetic_multivoice_train",
                    "pool_source_code": 1,
                    "target_indices": [4, 9],
                    "pool_indices": [0, 1],
                    "signer_voice_synthesis": {
                        "signer_id": "semlex:7",
                        "style_source_code": 2,
                        "content_pool_source_code": 1,
                        "token_duration_frames": [12, 16],
                        "context_frames": 3,
                        "bridge_frames": 2,
                        "max_trim_frames": 3,
                        "minimum_keep_frames": 4,
                    },
                }],
            }))
            sample = SyntheticCompositionDataset(pool, plan)[0]
            self.assertEqual(sample.source, "synthetic_multivoice_train")
            self.assertEqual(sample.targets.tolist(), [5, 10])
            self.assertEqual(sample.features.shape[1:], (32, FROZEN_TEMPORAL_FEATURE_DIM))
            self.assertTrue(np.isfinite(sample.features).all())


if __name__ == "__main__":
    unittest.main()
