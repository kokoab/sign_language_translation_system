import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from active.v17.extract_hand_rgb_v17 import (
    boxes_overlap,
    crop_square,
    decode_packed_crops,
    pack_crops,
    union_box,
    video_frame_count,
)
from active.v17.extract_mobileclip2_v17 import landmark_reference_indices
from active.v17.fine_tune_hand_mobileclip2_v17 import temporal_shift
from active.v17.extract_hand_rgb_supplement_v17 import selection_items
from active.v17.extract_hand_rgb_semlex_val_v17 import validation_items
from active.v17.model_hand_mobileclip2_v17 import (
    HandMobileCLIP2Stage1Config,
    HandMobileCLIP2Stage1V17,
)
from active.v17.schema_hand_rgb_v17 import HandRGBV17Config, schema_fingerprint
from active.v17.train_feature_fusion_v17 import GatedFeatureResidual
from active.v17.evaluate_multimodal_ensemble_v17 import normalized_item_id
from active.v17.local_multimodal_audit_v17 import local_audit_items
from active.v17.train_stage_1_hand_mobileclip2_v17 import (
    checkpoint_improves_citizen_then_local_tie,
    initialize_exact_hand_finetune,
    source_balanced_weights,
)
from active.v17.train_stage_1_hand_spatial_mobileclip2_v17 import (
    SpatialTemporalMobileCLIP2V17,
)


class HandRGBV17Test(unittest.TestCase):
    def test_hand_replay_checkpoint_selection_keeps_citizen_primary(self):
        self.assertTrue(
            checkpoint_improves_citizen_then_local_tie(81.0, 20.0, 80.0, 90.0)
        )
        self.assertTrue(
            checkpoint_improves_citizen_then_local_tie(80.0, 91.0, 80.0, 90.0)
        )
        self.assertFalse(
            checkpoint_improves_citizen_then_local_tie(79.0, 99.0, 80.0, 20.0)
        )
        self.assertFalse(
            checkpoint_improves_citizen_then_local_tie(80.0, 89.0, 80.0, 90.0)
        )

    def test_semlex_multimodal_item_ids_normalize_across_modalities(self):
        landmark = (
            "data/local/semlex_citizen100_val_audit/landmarks_v17/GOOD/abc.v17.npz"
        )
        hand = "semlex_val/GOOD/abc"
        self.assertEqual(normalized_item_id(landmark), "GOOD/abc")
        self.assertEqual(normalized_item_id(hand), "GOOD/abc")

    def test_local_deep_clean_item_ids_normalize_across_modalities(self):
        landmark = "GOOD/abc"
        hand = "local_deep_clean_val/GOOD/abc.hand_mobileclip2_v17.npz"
        self.assertEqual(normalized_item_id(hand), landmark)

    def test_local_audit_resolver_rejects_training_eligible_manifest(self):
        manifest = {
            "training_eligible": True,
            "split_eligibility": "train_only_after_exact_variant_review",
            "selected_clips": 1021,
            "videos": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen 1,021"):
                local_audit_items(path)

    def test_zero_gated_spatial_residual_is_exact_compact_path(self):
        head = HandMobileCLIP2Stage1V17(
            HandMobileCLIP2Stage1Config(
                num_classes=5, dim=64, depth=1, heads=4,
                dropout=0.0, head_dropout=0.0, drop_path=0.0,
            )
        ).eval()
        model = SpatialTemporalMobileCLIP2V17(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1), head,
            residual_shift=True,
        ).eval()
        maps = torch.randn(2, 16, 3, 512, 8, 8)
        valid = torch.ones(2, 16, 3, dtype=torch.bool)
        boxes = torch.rand(2, 16, 3, 4)
        compact = torch.nn.functional.normalize(
            maps.mean(dim=(-1, -2)), dim=-1
        )
        with torch.inference_mode():
            expected = head(compact, valid, boxes)
            actual = model(maps, valid, boxes)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_source_balanced_weights_preserve_requested_margins(self):
        class FakeDataset:
            def __init__(self, targets):
                self.targets = torch.tensor(targets)
        datasets = [FakeDataset([0, 0, 1]), FakeDataset([0, 1, 1, 2])]
        weights = source_balanced_weights(datasets, [0.6, 0.4])
        self.assertAlmostEqual(float(weights[:3].sum()), 0.6)
        self.assertAlmostEqual(float(weights[3:].sum()), 0.4)

    def test_exact_hand_replay_loader_restores_selected_state_strictly(self):
        config = HandMobileCLIP2Stage1Config(
            num_classes=5, dim=64, depth=1, heads=4
        )
        source = HandMobileCLIP2Stage1V17(config)
        restored = HandMobileCLIP2Stage1V17(config)
        label_to_index = {f"CLASS_{index}": index for index in range(5)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            import hashlib
            checkpoint = root / "selected.pth"
            torch.save(
                {
                    "format": "slt_stage1_hand_mobileclip2_v17",
                    "epoch": 4,
                    "model_config": config.to_dict(),
                    "model_state_dict": source.state_dict(),
                    "manifest_sha256": hashlib.sha256(
                        manifest.read_bytes()
                    ).hexdigest(),
                    "schema_fingerprint": "schema",
                    "label_to_index": label_to_index,
                    "validation_metrics": {"top1": 80.0},
                    "test_evaluated": False,
                    "training_data_provenance": {"test_evaluated": False},
                },
                checkpoint,
            )
            info = initialize_exact_hand_finetune(
                restored, checkpoint, manifest, "schema", label_to_index
            )
        for expected, actual in zip(source.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(info["strict_state_dict"])

    def test_frame_count_falls_back_when_container_metadata_is_invalid(self):
        capture = unittest.mock.MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = -9.2e18
        capture.read.side_effect = [(True, object()), (True, object()), (False, None)]
        with patch("active.v17.extract_hand_rgb_v17.cv2.VideoCapture", return_value=capture):
            self.assertEqual(video_frame_count(Path("broken-count.webm")), 2)
        capture.release.assert_called_once()

    def test_landmark_reference_reconstructs_overreported_container_sample(self):
        metadata = {
            "reported_frame_count": 156,
            "decoded_frame_count": 151,
            "source_frames_before_hand_trim": 93,
        }
        reference = landmark_reference_indices(metadata)
        self.assertEqual(len(reference), 93)
        self.assertEqual(int(reference[0]), 0)
        self.assertLess(int(reference[-1]), 151)

    def test_landmark_reference_uses_decoded_count_for_short_overreport(self):
        metadata = {
            "reported_frame_count": 72,
            "decoded_frame_count": 47,
            "source_frames_before_hand_trim": 47,
        }
        np.testing.assert_array_equal(
            landmark_reference_indices(metadata), np.arange(47)
        )

    def test_supplement_selection_keeps_only_local_tier_a(self):
        rows = [
            {
                "canonical_label": "GOOD", "consensus_tier": tier,
                "raw_path": f"raw/{index}.mp4",
                "feature_path": f"features/{index}.v17.npz",
            }
            for index, tier in enumerate(
                ("tier_a_dual_top1", "tier_b_dual_top5_one_top1")
            )
        ]
        manifest = {
            "split_eligibility":
                "train_only_after_ASL_fluent_exact_variant_review",
            "videos": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(Path, "is_file", return_value=True):
                items, _ = selection_items(path, "local_tier_a")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].item_id, "0")

    def test_deep_clean_selection_enforces_finalized_split_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.mp4"
            feature = root / "feature.v17.npz"
            raw.touch()
            feature.touch()
            manifest = {
                "format": "slt_v17_local_deep_clean_final_v1",
                "split": "train",
                "split_eligibility": "train_only_after_human_review",
                "extraction_complete": True,
                "selected_classes": 94,
                "selected_clips": 1,
                "signer_disjoint": False,
                "signer_overlap_user_approved": True,
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "videos": [{
                    "canonical_label": "HELLO",
                    "item_id": "hello-one",
                    "raw_path": str(raw),
                    "feature_path": str(feature),
                    "local_split": "train",
                    "training_eligible": True,
                    "validation_eligible": False,
                }],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            items, _ = selection_items(path, "local_deep_clean")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].item_id, "hello-one")
            manifest["videos"][0]["validation_eligible"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "split contract"):
                selection_items(path, "local_deep_clean")

    def test_semlex_validation_selection_rejects_training_eligible_manifest(self):
        manifest = {
            "split": "val", "training_eligible": True,
            "split_eligibility": "evaluation_only_never_training",
            "selected_clips": 0, "videos": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation-only"):
                validation_items(path)

    def test_crop_uses_real_reflected_pixels_and_fixed_shape(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[:, :60, 1] = 200
        crop = crop_square(frame, np.asarray([-20, 10, 60, 90]), 64)
        self.assertEqual(crop.shape, (64, 64, 3))
        self.assertGreater(int(crop[..., 1].max()), 0)

    def test_pack_round_trip_preserves_missing_contract(self):
        red = np.zeros((32, 32, 3), dtype=np.uint8)
        red[..., 2] = 255
        crops = [[red, None, red], [None, red, red]]
        blob, offsets = pack_crops(crops, quality=95)
        decoded = decode_packed_crops(blob, offsets, 32)
        self.assertEqual(decoded.shape, (2, 3, 32, 32, 3))
        self.assertTrue((decoded[0, 1] == 0).all())
        self.assertEqual(offsets[0, 1].tolist(), [-1, 0])
        self.assertGreater(float(decoded[0, 0, ..., 0].mean()), 240.0)

    def test_union_and_overlap_geometry(self):
        first = np.asarray([10, 10, 30, 30], dtype=np.float32)
        second = np.asarray([25, 10, 45, 30], dtype=np.float32)
        self.assertTrue(boxes_overlap(first, second))
        union = union_box([first, second], 100, 80, 1.0)
        self.assertAlmostEqual(float(union[0]), 10.0)
        self.assertAlmostEqual(float(union[2]), 45.0)
        self.assertFalse(boxes_overlap(first, np.asarray([60, 60, 80, 80], dtype=np.float32)))

    def test_schema_is_deterministic(self):
        first = schema_fingerprint(HandRGBV17Config())
        second = schema_fingerprint(HandRGBV17Config())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_temporal_shift_respects_view_and_missing_masks(self):
        value = torch.zeros(1 * 3 * 2, 8, 1, 1)
        value[:, :, 0, 0] = torch.arange(6).float().unsqueeze(1)
        valid = torch.tensor([[[True, True], [True, False], [True, True]]])
        shifted = temporal_shift(value, valid, fold_div=4).reshape(1, 3, 2, 8)
        self.assertTrue((shifted[0, 1, 1] == 0).all())
        self.assertEqual(tuple(shifted.shape), (1, 3, 2, 8))

    def test_view_temporal_model_handles_explicit_missing_views(self):
        model = HandMobileCLIP2Stage1V17(
            HandMobileCLIP2Stage1Config(num_classes=5, dim=64, depth=1, heads=4)
        )
        embeddings = torch.randn(2, 16, 3, 512)
        valid = torch.ones(2, 16, 3, dtype=torch.bool)
        valid[:, 0, :] = False
        embeddings[:, 0, :] = 0
        boxes = torch.rand(2, 16, 3, 4)
        logits = model(embeddings, valid, boxes)
        self.assertEqual(tuple(logits.shape), (2, 5))
        logits.sum().backward()

    def test_fusion_starts_as_exact_landmark_baseline(self):
        model = GatedFeatureResidual(dim=32, classes=5).eval()
        landmark = torch.randn(4, 32)
        hand = torch.randn(4, 32)
        base_logits = torch.randn(4, 5)
        with torch.no_grad():
            fused_logits = model(landmark, hand, base_logits)
        torch.testing.assert_close(fused_logits, base_logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
