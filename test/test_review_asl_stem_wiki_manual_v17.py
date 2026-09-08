import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.review_asl_stem_wiki_manual_v17 import INDEX_HTML, ReviewStore


FIELDS = [
    "participant", "explicit_subject_id", "signer_status", "raw_gloss",
    "canonical_label", "citizen_asl_lex_code", "filename", "video_path",
    "video_sha256", "manual_token_index", "proposed_frame",
    "proposed_start_frame", "proposed_end_frame", "pseudo_confidence",
    "pseudo_candidate_index", "pseudo_alignment_matches",
    "signer_quality_decision", "variant_decision", "verified_start_frame",
    "verified_end_frame", "reviewer_notes", "signer_quality_verified",
    "variant_verified", "boundary_verified", "training_eligible",
]


class ReviewAslStemWikiManualV17Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / "source.mp4"
        self.source.write_bytes(b"source")
        self.train = root / "raw" / "train"
        (self.train / "WATER").mkdir(parents=True)
        (self.train / "WATER" / "reference.mp4").write_bytes(b"reference")
        self.queue = root / "queue.csv"

    def tearDown(self):
        self.temp.cleanup()

    def write_row(self, signer_status="subject_mapping_unresolved"):
        row = {field: "" for field in FIELDS}
        row.update({
            "participant": "P12", "signer_status": signer_status,
            "raw_gloss": "WATER", "canonical_label": "WATER",
            "citizen_asl_lex_code": "A_02_031", "filename": "source.mp4",
            "video_path": str(self.source), "video_sha256": "abc",
            "proposed_frame": "42", "signer_quality_verified": "False",
            "variant_verified": "False", "boundary_verified": "False",
            "training_eligible": "False",
        })
        with self.queue.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(row)

    def store(self, signer_status="subject_mapping_unresolved"):
        self.write_row(signer_status)
        return ReviewStore(
            self.queue, self.train,
            probe=lambda path: {"frames": 300, "fps": 30.0, "duration": 10.0},
        )

    def test_loads_locked_gloss_and_train_only_citizen_references(self):
        row = self.store().public_rows()[0]
        self.assertEqual(row["expected_gloss"], "WATER")
        self.assertEqual(row["citizen_asl_lex_code"], "A_02_031")
        self.assertEqual(len(row["reference_videos"]), 1)
        self.assertIn("/raw/train/WATER/", row["reference_videos"][0]["path"])

    def test_approved_review_with_short_valid_bounds_becomes_eligible(self):
        store = self.store()
        saved = store.apply(0, {
            "signer_quality_decision": "yes", "variant_decision": "yes",
            "verified_start_frame": 40, "verified_end_frame": 80,
            "reviewer_notes": "matches the locked reference",
        })
        self.assertTrue(saved["signer_quality_verified"])
        self.assertTrue(saved["variant_verified"])
        self.assertTrue(saved["boundary_verified"])
        self.assertTrue(saved["training_eligible"])
        with self.queue.open(newline="") as handle:
            persisted = next(csv.DictReader(handle))
        self.assertEqual(persisted["training_eligible"], "True")

    def test_rejects_invalid_or_long_boundaries(self):
        store = self.store()
        with self.assertRaisesRegex(ValueError, "256 frames"):
            store.apply(0, {
                "signer_quality_decision": "yes", "variant_decision": "yes",
                "verified_start_frame": 1, "verified_end_frame": 299,
                "reviewer_notes": "",
            })

    def test_source_excluded_l2_row_cannot_become_eligible(self):
        saved = self.store("l2_excluded").apply(0, {
            "signer_quality_decision": "yes", "variant_decision": "yes",
            "verified_start_frame": 40, "verified_end_frame": 80,
            "reviewer_notes": "",
        })
        self.assertFalse(saved["signer_quality_verified"])
        self.assertFalse(saved["training_eligible"])

    def test_page_exposes_review_and_timeline_controls(self):
        for label in (
            "Expected gloss", "Citizen training reference", "Set start",
            "Set end", "Loop selection", "Save &amp; next", "Playback speed",
            "Use automatic bounds", "Automatic model proposal",
            "Automatic: high evidence",
        ):
            self.assertIn(label, INDEX_HTML)

    def test_save_next_uses_visible_order_from_before_pending_row_disappears(self):
        self.assertIn("const before=[...visible],at=before.indexOf(current)", INDEX_HTML)
        self.assertIn("current=before[at+1]??visible.find(i=>i>current)??visible[0]??current", INDEX_HTML)

    def test_loads_matching_automatic_annotation_without_changing_review(self):
        self.write_row()
        automatic = self.queue.parent / "automatic.json"
        automatic.write_text(json.dumps({"complete": True, "annotations": [{
            "queue_index": 0, "participant": "P12", "filename": "source.mp4",
            "raw_gloss": "WATER", "citizen_asl_lex_code": "A_02_031",
            "start_frame": 40, "end_frame": 80, "confidence": .72,
            "confidence_tier": "review", "full_top1_matches": True,
            "ctc_agrees": True, "full_top3": [{"gloss": "WATER", "probability": .8}],
        }]}))
        row = ReviewStore(
            self.queue, self.train,
            probe=lambda path: {"frames": 300, "fps": 30.0, "duration": 10.0},
            auto_annotations=automatic,
        ).public_rows()[0]
        self.assertEqual(row["automatic_annotation"]["start_frame"], 40)
        self.assertEqual(row["signer_quality_decision"], "")
        self.assertEqual(row["verified_start_frame"], "")

    def test_rejects_automatic_annotation_for_a_different_queue_identity(self):
        self.write_row()
        automatic = self.queue.parent / "automatic.json"
        automatic.write_text(json.dumps({"complete": True, "annotations": [{
            "queue_index": 0, "participant": "P99", "filename": "source.mp4",
            "raw_gloss": "WATER", "citizen_asl_lex_code": "A_02_031",
        }]}))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            ReviewStore(self.queue, self.train, auto_annotations=automatic)


if __name__ == "__main__":
    unittest.main()
