import unittest

from scripts.audit_asl_stem_wiki_manual_v17 import (
    align_manual_to_pseudo,
    build_admission_rows,
    build_review_queue,
    explicit_subject_map,
    tokenize_gloss,
)


class AuditAslStemWikiManualV17Test(unittest.TestCase):
    def test_tokenize_gloss_preserves_parenthesized_classifier_description(self):
        self.assertEqual(
            tokenize_gloss("CL:1(2h)(those values from 0 to 1) WATER fs-TERM"),
            ["CL:1(2h)(those values from 0 to 1)", "WATER", "fs-TERM"],
        )

    def test_explicit_subject_map_uses_only_source_notes(self):
        rows = [
            {
                "filename": "p13.mp4",
                "Gloss Notes": "Subject #2 Overall Review - clear classifiers.",
            },
            {"filename": "p13-other.mp4", "Gloss Notes": "Not Subject 17"},
            {"filename": "p12.mp4", "Gloss Notes": "ordinary note"},
        ]
        metadata = {
            "p13.mp4": {"participant": "P13"},
            "p13-other.mp4": {"participant": "P13"},
            "p12.mp4": {"participant": "P12"},
        }
        self.assertEqual(explicit_subject_map(rows, metadata), {"P13": 2})

    def test_alignment_prefers_candidate_with_more_ordered_manual_matches(self):
        candidates = [
            [["GIVE", 5, 0.9, "in-domain"], ["WATER", 30, 0.8, "in-domain"]],
            [["WATER", 10, 0.7, "in-domain"], ["TERM", [15, 20], 0.6, "fs"], ["GIVE", 30, 0.9, "in-domain"]],
        ]
        aligned = align_manual_to_pseudo(["WATER", "fs-TERM", "GIVE"], candidates)
        self.assertEqual(aligned["candidate_index"], 1)
        self.assertEqual(aligned["manual_to_pseudo"], {0: 0, 1: 1, 2: 2})

    def test_review_queue_never_admits_unverified_signer_or_variant(self):
        downloaded = [
            {
                "filename": "clip.mp4",
                "participant": "P28",
                "path": "/tmp/clip.mp4",
                "sha256": "abc",
                "gloss_tokens": ["WATER", "fs-TERM", "GIVE"],
            }
        ]
        targets = {
            "WATER": {"canonical_label": "WATER", "citizen_asl_lex_code": "WATER_CODE"},
            "GIVE": {"canonical_label": "GIVE", "citizen_asl_lex_code": "GIVE_CODE"},
        }
        pseudo = {
            "clip": {
                "candidates": [
                    [["WATER", 10, 0.7, "in-domain"], ["TERM", [15, 20], 0.6, "fs"], ["GIVE", 30, 0.9, "in-domain"]]
                ]
            }
        }
        queue = build_review_queue(downloaded, targets, pseudo, {"P28": 7})
        self.assertEqual([row["raw_gloss"] for row in queue], ["GIVE", "WATER"])
        self.assertTrue(all(row["signer_status"] == "l2_excluded" for row in queue))
        self.assertTrue(all(not row["training_eligible"] for row in queue))
        self.assertEqual(queue[1]["proposed_frame"], 10)

    def test_admission_manifest_records_every_failed_gate(self):
        downloaded = [{
            "filename": "clip.mp4", "participant": "P28", "path": "/tmp/clip.mp4",
            "sha256": "abc", "gloss_tokens": ["WATER"],
        }]
        rows = build_admission_rows(downloaded, {"P28": 7})
        self.assertEqual(rows[0]["signer_status"], "l2_excluded")
        self.assertFalse(rows[0]["training_eligible"])
        self.assertEqual(
            rows[0]["rejection_reasons"],
            ["signer_quality_not_admitted", "exact_variant_unverified", "boundaries_unverified"],
        )


if __name__ == "__main__":
    unittest.main()
