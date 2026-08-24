import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_stage2_training_manifest_v17 import validate


def row(item, role, digest, parent=None):
    value = {
        "source": "fixture",
        "role": role,
        "source_item_id": item,
        "video_sha256": digest,
        "target_sequence": ["HELLO"],
    }
    if parent:
        value["parent_video_sha256"] = parent
    return value


class Stage2TrainingManifestTests(unittest.TestCase):
    def test_disjoint_rows_validate(self):
        result = validate([
            row("a", "train", "1"),
            row("b", "validation", "2"),
            row("c", "external_evaluation_reserved", "3"),
        ])
        self.assertEqual(result["active_video_hash_overlap"], 0)

    def test_video_hash_leakage_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "video hash leakage"):
            validate([row("a", "train", "1"), row("b", "validation", "1")])

    def test_parent_utterance_leakage_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "parent utterance leakage"):
            validate([
                row("a", "train", "1", "parent"),
                row("b", "validation", "2", "parent"),
            ])

    def test_duplicate_item_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate source_item_id"):
            validate([row("a", "train", "1"), row("a", "train", "2")])


if __name__ == "__main__":
    unittest.main()
