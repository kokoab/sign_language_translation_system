import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_stage2_transition_adapt_v17 import (
    EXPECTED_ENCODER_SHA256,
    EXPECTED_QUEUE_SHA256,
    frame_interval_times,
    load_vocab,
    participant_split,
    validate_encoder_sha256,
    validate_no_overlap,
    validate_queue,
    validate_target_index,
    directory_digest,
    validate_pool,
    validate_pool_pair,
    validate_identity_ledger,
)


class TransitionAdaptPreparationTests(unittest.TestCase):
    def test_participant_split_is_hash_deterministic(self):
        participants = ["P12", "P4", "P0", "P10", "P11", "P13"]
        validation, train = participant_split(participants)
        ordered = sorted(participants, key=lambda value: hashlib.sha256(
            f"1701:{value}".encode("utf-8")).hexdigest())
        self.assertEqual(validation, ordered[:4])
        self.assertEqual(train, ordered[4:])

    def test_inclusive_interval_uses_timestamp_bounds_at_30fps(self):
        timing = frame_interval_times(30, 59, 29.97)
        self.assertEqual(timing["start_seconds"], 30 / 29.97)
        self.assertEqual(timing["end_seconds_exclusive"], 60 / 29.97)
        self.assertEqual(timing["sample_rate_fps"], 30.0)
        self.assertEqual(timing["geometry_transform"], "none")

    def test_queue_hash_and_eligibility_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queue.csv"
            fields = ["participant", "canonical_label", "video_path", "video_sha256",
                      "verified_start_frame", "verified_end_frame", "training_eligible"]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({name: value for name, value in zip(fields,
                    ["P0", "HELLO", "x.mp4", "a" * 64, "1", "2", "True"])})
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = validate_queue(path, digest, expected_count=1, expected_participants=1)
            self.assertEqual(rows[0]["canonical_label"], "HELLO")
            with self.assertRaisesRegex(ValueError, "sha256"):
                validate_queue(path, EXPECTED_QUEUE_SHA256, expected_count=1, expected_participants=1)

    def test_overlap_and_encoder_mismatch_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_no_overlap([
                {"role": "train", "parent_video_sha256": "parent", "interval": [1, 4]},
                {"role": "validation", "parent_video_sha256": "parent", "interval": [3, 6]},
            ])
        with self.assertRaisesRegex(ValueError, "encoder"):
            validate_encoder_sha256("278a9933" + "0" * 56)
        self.assertEqual(validate_encoder_sha256(EXPECTED_ENCODER_SHA256), EXPECTED_ENCODER_SHA256)

    def test_same_split_context_intervals_are_not_cross_split_leakage(self):
        validate_no_overlap([
            {"role": "train", "parent_video_sha256": "parent", "interval": [1, 4]},
            {"role": "train", "parent_video_sha256": "parent", "interval": [3, 6]},
        ])

    def test_label_index_contract(self):
        self.assertEqual(validate_target_index(0), 0)
        self.assertEqual(validate_target_index(99), 99)
        with self.assertRaises(ValueError):
            validate_target_index(-1)
        with self.assertRaises(ValueError):
            validate_target_index(100)

    def test_stem_manifest_uses_zero_based_stored_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocabulary.json"
            path.write_text(json.dumps({"classes": [
                {"canonical_label": f"SIGN_{index}"} for index in range(100)
            ]}))
            vocabulary = load_vocab(path)
        self.assertEqual(vocabulary["SIGN_0"], 0)
        self.assertEqual(vocabulary["SIGN_99"], 99)

    def test_directory_digest_is_non_null_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_text("one")
            first = directory_digest(root)
            (root / "a").write_text("two")
            self.assertNotEqual(first, directory_digest(root))

    def test_pool_rejects_wrong_role_or_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.npz"
            import numpy as np
            np.savez(path, target_indices=np.array([0, 99]), item_ids=np.array(["A/a", "B/b"]), metadata_json=np.array(
                '{"source_split":"citizen_official_train_only","stage1_checkpoint_sha256":"bad"}'
            ))
            with self.assertRaisesRegex(ValueError, "encoder"):
                validate_pool(path, "citizen_official_train_only", 100)

    def test_pool_pair_rejects_shared_item_ids(self):
        with self.assertRaisesRegex(ValueError, "item leakage"):
            validate_pool_pair({"item_ids": ["HELLO/a"], "role": "train"}, {"item_ids": ["HELLO/a"], "role": "validation"})

    def test_pool_requires_one_nonempty_item_id_per_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.npz"
            import numpy as np
            np.savez(path, target_indices=np.array([0]), metadata_json=np.array(
                '{"source_split":"citizen_official_train_only","stage1_checkpoint_sha256":"1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b","class_counts":{"HELLO":1}}'
            ))
            with self.assertRaisesRegex(ValueError, "item IDs"):
                validate_pool(path, "citizen_official_train_only", 1, {"HELLO": 1})

    def test_identity_ledger_rejects_cross_source_parent_collision(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_identity_ledger([
                {"namespace": "parent_video_sha256", "identity": "same", "role": "train", "source": "stem"},
                {"namespace": "parent_video_sha256", "identity": "same", "role": "validation", "source": "asllrp"},
            ])


if __name__ == "__main__":
    unittest.main()
