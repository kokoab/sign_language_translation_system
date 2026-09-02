import unittest
from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.audit_real_motion_reference_v17 import Summary
from scripts.evaluate_transition_real_sources_v17 import fixed_mask
from scripts.prepare_local_phrase_motion_manifest_v17 import run as prepare_local_motion
from active.v17.train_transition_inpainter_v17 import TransitionWindowDataset


class RealMotionReferenceAuditTest(unittest.TestCase):
    def test_presence_and_motion_summary(self):
        value = np.zeros((4, 61, 5), dtype=np.float32)
        value[:, :21, 3:] = 1.0
        value[1:, :21, 0] = np.arange(1, 4)[:, None]
        summary = Summary()
        summary.add(value, count_archive=True)
        result = summary.result()
        self.assertEqual(result["archives"], 1)
        self.assertEqual(result["frames"], 4)
        self.assertEqual(result["presence_frame_fractions"]["left_complete"], 1.0)
        self.assertEqual(result["presence_frame_fractions"]["right_any"], 0.0)
        self.assertEqual(result["hand_motion"]["speed"]["p50"], 1.0)
        self.assertEqual(result["presence_change_step_fraction"], 0.0)

    def test_fixed_mask_is_bounded_and_deterministic(self):
        first = fixed_mask(3)
        self.assertTrue(np.array_equal(first, fixed_mask(3)))
        self.assertGreaterEqual(int(first.sum()), 4)
        self.assertLessEqual(int(first.sum()), 12)
        indices = np.flatnonzero(first)
        self.assertGreaterEqual(int(indices[0]), 3)
        self.assertLessEqual(int(indices[-1]), 28)

    def test_local_motion_manifest_keeps_every_fifth_for_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "audit.csv"
            rows = []
            for index in range(20):
                video = root / f"clip_{index}.mp4"
                video.write_bytes(str(index).encode())
                rows.append({
                    "phrase": "HELLO_HOW_YOU",
                    "path": str(video),
                    "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                    "duration_seconds": "3.0",
                })
            with csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            output = root / "manifest.json"
            payload = prepare_local_motion(Namespace(audit_csv=csv_path, output=output))
            self.assertEqual(payload["row_count"], 20)
            self.assertEqual(sum(row["role"] == "validation" for row in payload["rows"]), 4)
            self.assertTrue(all(not row["target_sequence"] for row in payload["rows"]))

    def test_transition_dataset_reads_stage2_validity_and_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "nested"
            root.mkdir(parents=True)
            value = np.ones((2, 32, 61, 5), dtype=np.float16)
            np.savez_compressed(
                root / "sample.npz",
                landmarks=value,
                landmark_window_valid=np.array([True, False]),
                metadata_json=np.array(json.dumps({
                    "role": "train", "source": "sample", "signer_id": None,
                })),
            )
            dataset = TransitionWindowDataset(
                root.parent, None, seed=1, fixed_masks=True,
                all_archives=True, roles={"train"}, sources={"sample"},
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(dataset[0]["features"].shape), (32, 61, 5))


if __name__ == "__main__":
    unittest.main()
