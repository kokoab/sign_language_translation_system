import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from active.v17.extract_mobileclip2_v17 import (
    letterbox_rgb,
    reference_indices,
    temporal_sample_indices,
)
from active.v17.model_mobileclip2_v17 import (
    MobileCLIP2Stage1Config,
    MobileCLIP2Stage1V17,
)
from active.v17.schema_mobileclip2_v17 import MobileCLIP2V17Config, schema_fingerprint
from active.v17.train_stage_1_mobileclip2_v17 import MobileCLIP2CitizenDataset


class MobileCLIP2V17Test(unittest.TestCase):
    def test_real_citizen_rgb_counts_and_finite_contract(self):
        root = Path("data/local/citizen100_v17/mobileclip2_s0")
        if not root.exists():
            self.skipTest("MobileCLIP2 Citizen features are unavailable")
        common = {
            "root": root,
            "manifest_path": Path("active/v17/citizen100_manifest.json"),
            "rejection_path": Path("data/local/citizen100_v17/rejections.csv"),
            "cache": False,
        }
        train = MobileCLIP2CitizenDataset(split="train", **common)
        validation = MobileCLIP2CitizenDataset(split="val", **common)
        self.assertEqual(len(train), 1475)
        self.assertEqual(len(validation), 378)
        sample, _ = validation[0]
        self.assertEqual(tuple(sample.shape), (16, 512))
        self.assertTrue(torch.isfinite(sample).all())

    def test_reference_and_trim_sampling_match_v17_contract(self):
        np.testing.assert_array_equal(reference_indices(4), np.arange(4))
        reference = reference_indices(200, 96)
        self.assertEqual(len(reference), 96)
        selected = temporal_sample_indices(200, 10, 30, 16, 96)
        self.assertEqual(selected.shape, (16,))
        self.assertGreaterEqual(selected[0], reference[10])
        self.assertLessEqual(selected[-1], reference[29])
        self.assertTrue(np.all(selected[1:] >= selected[:-1]))

    def test_letterbox_preserves_portrait_and_landscape_geometry(self):
        landscape = np.full((100, 200, 3), 255, dtype=np.uint8)
        portrait = cv2.rotate(landscape, cv2.ROTATE_90_CLOCKWISE)
        output_landscape = letterbox_rgb(landscape, 256)
        output_portrait = letterbox_rgb(portrait, 256)
        self.assertEqual(output_landscape.shape, (256, 256, 3))
        self.assertEqual(output_portrait.shape, (256, 256, 3))
        self.assertTrue((output_landscape[:60] == 0).all())
        self.assertTrue((output_portrait[:, :60] == 0).all())

    def test_temporal_head_forward_backward(self):
        config = MobileCLIP2Stage1Config(num_classes=7, dim=64, depth=1, heads=4)
        model = MobileCLIP2Stage1V17(config)
        value = torch.randn(2, 16, 512)
        logits = model(value)
        self.assertEqual(tuple(logits.shape), (2, 7))
        logits.sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_dataset_rejects_test_and_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"classes": [{"canonical_label": "A", "class_index": 0}]}))
            with self.assertRaisesRegex(ValueError, "test split is sealed"):
                MobileCLIP2CitizenDataset(root, "test", manifest)
            feature_root = root / "features" / "train" / "A"
            feature_root.mkdir(parents=True)
            np.savez_compressed(
                feature_root / "a.mobileclip2_v17.npz",
                embeddings=np.zeros((16, 512), dtype=np.float16),
                metadata_json=np.array(json.dumps({"schema_fingerprint": "wrong"})),
            )
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                MobileCLIP2CitizenDataset(
                    root / "features", "train", manifest,
                    expected_schema=schema_fingerprint(MobileCLIP2V17Config()),
                )


if __name__ == "__main__":
    unittest.main()
