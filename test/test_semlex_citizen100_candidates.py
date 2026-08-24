import unittest

from scripts.prepare_semlex_citizen100_candidates import (
    build_selection,
    choose_distinct_signers,
    normalize_asllex_label,
)


class SemLexCitizen100CandidatesTests(unittest.TestCase):
    def test_label_normalization_preserves_variant_number(self):
        self.assertEqual(normalize_asllex_label("WHAT-2"), "what_2")
        self.assertNotEqual(normalize_asllex_label("WHAT-2"), normalize_asllex_label("WHAT-1"))

    def test_distinct_signer_cap(self):
        rows = [
            {"video_id": "a", "signer_id": "1", "duration": "1600"},
            {"video_id": "b", "signer_id": "1", "duration": "1500"},
            {"video_id": "c", "signer_id": "2", "duration": "1700"},
        ]
        selected = choose_distinct_signers(rows, 5)
        self.assertEqual({row["signer_id"] for row in selected}, {"1", "2"})
        self.assertEqual(len(selected), 2)

    def test_zero_cap_keeps_every_distinct_signer(self):
        rows = [
            {"video_id": str(index), "signer_id": str(index), "duration": "1600"}
            for index in range(3)
        ]
        self.assertEqual(len(choose_distinct_signers(rows, 0)), 3)

    def test_build_selection_is_train_exact_asllex_only(self):
        manifest = {
            "classes": [
                {
                    "canonical_label": "WHAT",
                    "citizen_raw_gloss": "WHAT1",
                    "citizen_asl_lex_code": "D_02_094",
                }
            ]
        }
        asllex = [{"Code": "D_02_094", "EntryID": "what_2"}]
        semlex = [
            {"split": "train", "label_type": "asllex", "label": "what_2", "duration": "1200", "video_id": "ok", "signer_id": "7"},
            {"split": "train", "label_type": "asllex", "label": "what_1", "duration": "1200", "video_id": "wrong_variant", "signer_id": "8"},
            {"split": "test", "label_type": "asllex", "label": "what_2", "duration": "1200", "video_id": "test", "signer_id": "9"},
            {"split": "train", "label_type": "freetext", "label": "what_2", "duration": "1200", "video_id": "free", "signer_id": "10"},
        ]
        videos, classes = build_selection(manifest, asllex, semlex, 5)
        self.assertEqual([row["semlex_video_id"] for row in videos], ["ok"])
        self.assertEqual(classes[0]["selected_clips"], 1)
        self.assertFalse(videos[0]["training_eligible"])

    def test_validation_selection_is_exact_and_evaluation_only(self):
        manifest = {
            "classes": [
                {
                    "canonical_label": "WHAT",
                    "citizen_raw_gloss": "WHAT1",
                    "citizen_asl_lex_code": "D_02_094",
                }
            ]
        }
        asllex = [{"Code": "D_02_094", "EntryID": "what_2"}]
        semlex = [
            {
                "split": "val",
                "label_type": "asllex",
                "label": "what_2",
                "duration": "1200",
                "video_id": "valid",
                "signer_id": "7",
            },
            {
                "split": "train",
                "label_type": "asllex",
                "label": "what_2",
                "duration": "1200",
                "video_id": "train",
                "signer_id": "8",
            },
        ]
        videos, _ = build_selection(manifest, asllex, semlex, 0, split="val")
        self.assertEqual([row["semlex_video_id"] for row in videos], ["valid"])
        self.assertEqual(videos[0]["semlex_split"], "val")
        self.assertEqual(videos[0]["archive_member"], "./val/valid.webm")


if __name__ == "__main__":
    unittest.main()
