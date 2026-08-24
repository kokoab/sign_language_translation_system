import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.build_portrait_iphone_eval_v17 import (
    DEFAULT_CANDIDATES,
    LEDGER_FIELDS,
    REVIEW_FIELDS,
    PortraitPackError,
    audit_pack,
    build_pack,
    init_review,
    probe_video_full_decode,
    validate_candidates,
)
from scripts.import_portrait_iphone_captures_v17 import import_captures


class PortraitIPhoneEvalV17Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.json"
        self.phonology = self.root / "phonology.json"
        classes = [
            {
                "class_index": index,
                "canonical_label": f"LABEL_{index:03d}",
                "citizen_raw_gloss": f"RAW_{index:03d}",
                "citizen_asl_lex_code": f"A_00_{index:03d}",
            }
            for index in range(100)
        ]
        self.manifest.write_text(
            json.dumps({"class_count": 100, "classes": classes}), encoding="utf-8"
        )
        self.phonology.write_text(
            json.dumps(
                {
                    "class_count": 100,
                    "classes": [
                        {
                            "class_index": index,
                            "canonical_label": f"LABEL_{index:03d}",
                            "citizen_asl_lex_code": f"A_00_{index:03d}",
                            "asllex_entry_id": f"entry_{index:03d}",
                        }
                        for index in range(100)
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.review = self.root / "review.csv"
        init_review(self.manifest, self.phonology, self.review)

    def tearDown(self):
        self.temp.cleanup()

    def approve_review(self):
        with self.review.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            row["review_status"] = "approved"
            row["reviewer_id"] = "reviewer_a"
            row["reviewed_utc"] = "2026-08-13T10:00:00+08:00"
        with self.review.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def build(self, name="pack"):
        self.approve_review()
        pack = self.root / name
        build_pack(
            self.manifest,
            self.phonology,
            self.review,
            pack,
            ["S01", "S02", "S03", "S04", "S05"],
        )
        return pack

    def make_phone_export(self, pack, *, row_index=0, payload=b"portrait-video"):
        with (pack / "capture_ledger.csv").open(encoding="utf-8", newline="") as handle:
            row = list(csv.DictReader(handle))[row_index]
        export = self.root / f"export_{row_index}"
        video = export / row["video_path"]
        video.parent.mkdir(parents=True)
        video.write_bytes(payload)
        import hashlib

        row.update(
            {
                "performed_gloss": (
                    "non-target wave"
                    if row["canonical_label"] == "UNKNOWN"
                    else row["expected_raw_gloss"]
                ),
                "prompt_hidden_before_capture": "true",
                "video_sha256": hashlib.sha256(payload).hexdigest(),
                "recorded_utc": "2026-08-13T10:15:00Z",
                "device_model": "iPhone17,1",
                "ios_version": "26.0",
                "camera": "front",
                "width": "1080",
                "height": "1920",
                "fps": "30.000000",
                "orientation": "portrait",
                "mirrored": "true",
                "lighting": "indoor_even",
                "background": "ordinary_room",
                "objective_qc_status": "pending",
                "objective_qc_reason": "",
            }
        )
        updates = export / "capture_updates.csv"
        with updates.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        return export, updates, row

    def test_review_template_pins_all_exact_variants_and_links_only(self):
        with self.review.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, list(REVIEW_FIELDS))
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(row["review_status"] == "pending" for row in rows))
        self.assertEqual(rows[7]["citizen_raw_gloss"], "RAW_007")
        self.assertEqual(
            rows[7]["asllex_reference_url"],
            "https://asl-lex.org/visualization/?sign=entry_007",
        )

    def test_capture_pack_refuses_pending_review(self):
        with self.assertRaisesRegex(PortraitPackError, "Capture is blocked"):
            build_pack(
                self.manifest,
                self.phonology,
                self.review,
                self.root / "pack",
                ["S01", "S02", "S03", "S04", "S05"],
            )

    def test_capture_pack_has_1000_targets_100_oov_and_distinct_orders(self):
        pack = self.build()
        with (pack / "capture_ledger.csv").open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, list(LEDGER_FIELDS))
        self.assertEqual(len(rows), 1100)
        self.assertEqual(sum(row["canonical_label"] == "UNKNOWN" for row in rows), 100)
        first = [
            row["class_index"]
            for row in rows
            if row["signer_id"] == "S01" and row["repetition"] == "1"
        ]
        second = [
            row["class_index"]
            for row in rows
            if row["signer_id"] == "S01" and row["repetition"] == "2"
        ]
        self.assertEqual(set(first), {str(index) for index in range(100)})
        self.assertEqual(set(second), set(first))
        self.assertNotEqual(first, second)
        result = audit_pack(
            pack, self.manifest, self.phonology, self.review, phase="setup"
        )
        self.assertTrue(result["pass"], result["errors"])
        self.assertFalse(result["ready_for_first_inference"])

    def test_setup_audit_detects_pinned_variant_tampering(self):
        pack = self.build()
        ledger = pack / "capture_ledger.csv"
        with ledger.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["expected_raw_gloss"] = "WRONG_VARIANT"
        with ledger.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        result = audit_pack(
            pack, self.manifest, self.phonology, self.review, phase="setup"
        )
        self.assertFalse(result["pass"])
        self.assertTrue(any("pinned target" in error for error in result["errors"]))
        self.assertTrue(any("changed after pack creation" in error for error in result["errors"]))

    def test_pre_inference_audit_rejects_pending_capture(self):
        pack = self.build()
        result = audit_pack(
            pack, self.manifest, self.phonology, self.review, phase="pre-inference"
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["ready_for_first_inference"])
        self.assertTrue(any("unresolved pending" in error for error in result["errors"]))

    def test_pack_requires_five_unique_pseudonyms(self):
        self.approve_review()
        with self.assertRaisesRegex(PortraitPackError, "five unique"):
            build_pack(
                self.manifest,
                self.phonology,
                self.review,
                self.root / "pack",
                ["S01", "S02", "S03", "S04"],
            )

    def test_real_frozen_candidates_pin_teacher_and_compact_evidence(self):
        payload = validate_candidates(DEFAULT_CANDIDATES)
        teacher = payload["fusions"]["four_stream_teacher_30_15_35_20"]
        self.assertEqual(teacher["citizen_validation_correct"], 370)
        self.assertEqual(teacher["semlex_validation_correct"], 882)
        self.assertFalse(teacher["recalibration_allowed"])

    def test_frozen_candidate_validator_rejects_weight_change(self):
        payload = json.loads(DEFAULT_CANDIDATES.read_text(encoding="utf-8"))
        payload["repository_root"] = str(DEFAULT_CANDIDATES.parents[2])
        payload["fusions"]["four_stream_teacher_30_15_35_20"]["members"][
            "teacher_landmark_flat"
        ] = 0.31
        changed = self.root / "changed_candidates.json"
        changed.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(PortraitPackError, "weight changed"):
            validate_candidates(changed)

    def test_full_decode_probe_uses_rotation_and_video_stream_only(self):
        metadata = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "width": 1920,
                            "height": 1080,
                            "avg_frame_rate": "30000/1001",
                            "nb_frames": "30",
                            "side_data_list": [{"rotation": 90}],
                        }
                    ]
                }
            ),
            stderr="",
        )
        decoded = SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch(
            "scripts.build_portrait_iphone_eval_v17.shutil.which",
            side_effect=lambda value: f"/usr/bin/{value}",
        ), patch(
            "scripts.build_portrait_iphone_eval_v17.subprocess.run",
            side_effect=[metadata, decoded],
        ) as runner:
            result = probe_video_full_decode(Path("clip.mov"))
        self.assertEqual((result["oriented_width"], result["oriented_height"]), (1080, 1920))
        self.assertAlmostEqual(result["fps"], 29.97002997)
        self.assertFalse(result["audio_accessed"])
        self.assertIn("0:v:0", runner.call_args_list[1].args[0])

    def test_full_decode_probe_rejects_corrupt_video(self):
        metadata = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"streams": [{"width": 1080, "height": 1920, "avg_frame_rate": "30/1"}]}
            ),
            stderr="",
        )
        decoded = SimpleNamespace(returncode=1, stdout="", stderr="corrupt packet")
        with patch(
            "scripts.build_portrait_iphone_eval_v17.shutil.which",
            side_effect=lambda value: f"/usr/bin/{value}",
        ), patch(
            "scripts.build_portrait_iphone_eval_v17.subprocess.run",
            side_effect=[metadata, decoded],
        ):
            with self.assertRaisesRegex(PortraitPackError, "corrupt packet"):
                probe_video_full_decode(Path("clip.mov"))

    def test_phone_import_replaces_placeholder_and_is_idempotent(self):
        pack = self.build()
        export, updates, update_row = self.make_phone_export(pack)
        report = self.root / "import.json"
        first = import_captures(
            pack,
            export,
            updates,
            report,
            manifest_path=self.manifest,
            phonology_path=self.phonology,
            review_path=self.review,
        )
        self.assertEqual(first["new_ledger_rows"], 1)
        self.assertEqual(first["new_video_copies"], 1)
        with (pack / "capture_ledger.csv").open(encoding="utf-8", newline="") as handle:
            imported = {
                row["capture_id"]: row for row in csv.DictReader(handle)
            }[update_row["capture_id"]]
        self.assertEqual(imported, update_row)
        self.assertEqual((pack / update_row["video_path"]).read_bytes(), b"portrait-video")

        second = import_captures(
            pack,
            export,
            updates,
            report,
            manifest_path=self.manifest,
            phonology_path=self.phonology,
            review_path=self.review,
        )
        self.assertEqual(second["new_ledger_rows"], 0)
        self.assertEqual(second["new_video_copies"], 0)
        self.assertEqual(second["idempotent_rows"], 1)

    def test_phone_import_rejects_changed_pinned_variant_before_copy(self):
        pack = self.build()
        export, updates, row = self.make_phone_export(pack)
        row["expected_raw_gloss"] = "WRONG_VARIANT"
        with updates.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        with self.assertRaisesRegex(PortraitPackError, "changed frozen field"):
            import_captures(
                pack,
                export,
                updates,
                self.root / "import.json",
                manifest_path=self.manifest,
                phonology_path=self.phonology,
                review_path=self.review,
            )
        self.assertFalse((pack / row["video_path"]).exists())

    def test_phone_import_rejects_hash_mismatch_before_ledger_change(self):
        pack = self.build()
        original_ledger = (pack / "capture_ledger.csv").read_bytes()
        export, updates, row = self.make_phone_export(pack)
        row["video_sha256"] = "0" * 64
        with updates.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)
        with self.assertRaisesRegex(PortraitPackError, "hash-mismatched"):
            import_captures(
                pack,
                export,
                updates,
                self.root / "import.json",
                manifest_path=self.manifest,
                phonology_path=self.phonology,
                review_path=self.review,
            )
        self.assertEqual((pack / "capture_ledger.csv").read_bytes(), original_ledger)


if __name__ == "__main__":
    unittest.main()
