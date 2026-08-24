from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from active.v17.evaluate_local_validation_v17 import classification_metrics
from active.v17.schema_v17 import V17Config, schema_fingerprint
from scripts.finalize_local_deep_clean_v17 import finalize_manifest
from scripts.prepare_local_deep_clean_v17 import (
    content_group_splits,
    resolve_current_label,
)


class LocalDeepCleanPreparationTest(unittest.TestCase):
    def test_finalizer_keeps_valid_archives_and_records_missing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            landmark_root = root / "landmarks"
            (landmark_root / "HELLO").mkdir(parents=True)
            features = np.zeros((32, 61, 5), dtype=np.float16)
            np.savez_compressed(
                landmark_root / "HELLO" / "HELLO.v17.npz",
                features=features,
                metadata_json=np.array(
                    json.dumps(
                        {"schema_fingerprint": schema_fingerprint(V17Config())}
                    )
                ),
            )
            manifest = {
                "format": "slt_v17_local_deep_clean_v1",
                "split": "train",
                "selected_clips": 2,
                "selected_classes": 1,
                "class_counts": {"HELLO": 2},
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "videos": [
                    {"canonical_label": "HELLO", "item_id": "HELLO"},
                    {"canonical_label": "HELLO", "item_id": "MISSING"},
                ],
            }
            manifest_path = root / "train_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            finalized, rejected = finalize_manifest(manifest_path, landmark_root)
            self.assertEqual(finalized["selected_clips"], 1)
            self.assertEqual(finalized["selected_classes"], 1)
            self.assertEqual(finalized["extraction_rejections"], 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                rejected[0]["extraction_rejection_reason"], "missing_v17_archive"
            )

    def test_local_validation_metrics_report_present_class_macro(self):
        logits = np.asarray(
            [
                [3.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 2.0, 1.0, -1.0, -2.0],
                [0.0, 2.0, 1.0, -1.0, -2.0],
            ],
            dtype=np.float32,
        )
        metrics = classification_metrics(logits, np.asarray([0, 1, 2]), 5)
        self.assertEqual(metrics["top1_correct"], 2)
        self.assertEqual(metrics["top5_correct"], 3)
        self.assertEqual(metrics["classes_present"], 3)

    def test_traceable_merged_classes_resolve_by_source_prefix(self):
        current = {"EAT", "MAKE", "SAME", "HOME"}
        cases = (
            ("EAT_abcd.npy", "EAT_FOOD", "EAT"),
            ("MAKE_abcd.npy", "MAKE_CREATE", "MAKE"),
            ("ALSO_SAME_abcd.npy", "ALSO", "SAME"),
            ("HOUSE_abcd.npy", "HOUSE", "HOME"),
        )
        for filename, old_label, expected in cases:
            with self.subTest(filename=filename):
                label, lineage = resolve_current_label(
                    filename, old_label, current
                )
                self.assertEqual(label, expected)
                self.assertEqual(lineage, "traceable_v16_source_folder_alias")

    def test_untraceable_semantic_neighbor_is_not_merged(self):
        label, lineage = resolve_current_label(
            "SEARCH_abcd.npy", "SEARCH", {"FIND"}
        )
        self.assertIsNone(label)
        self.assertEqual(lineage, "not_in_current_vocabulary")

    def test_duplicate_connected_rows_remain_in_one_split(self):
        rows = [
            {
                "canonical_label": "HELLO",
                "raw_sha256": "raw-a",
                "source_feature_sha256": "feature-a",
                "item_id": "one",
            },
            {
                "canonical_label": "HELLO",
                "raw_sha256": "raw-b",
                "source_feature_sha256": "feature-a",
                "item_id": "two",
            },
        ]
        admitted, quarantined = content_group_splits(rows, 1701)
        self.assertFalse(quarantined)
        self.assertEqual({row["local_split"] for row in admitted}, {admitted[0]["local_split"]})
        self.assertEqual({row["duplicate_group_size"] for row in admitted}, {2})

    def test_cross_label_duplicate_is_quarantined(self):
        rows = [
            {
                "canonical_label": "HELLO",
                "raw_sha256": "same-raw",
                "source_feature_sha256": "feature-a",
            },
            {
                "canonical_label": "GOODBYE",
                "raw_sha256": "same-raw",
                "source_feature_sha256": "feature-b",
            },
        ]
        admitted, quarantined = content_group_splits(rows, 1701)
        self.assertFalse(admitted)
        self.assertEqual(len(quarantined), 2)
        self.assertEqual(
            {row["quarantine_reason"] for row in quarantined},
            {"duplicate_content_label_conflict"},
        )


if __name__ == "__main__":
    unittest.main()
