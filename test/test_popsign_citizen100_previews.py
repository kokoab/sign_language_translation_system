import unittest

from scripts.download_popsign_citizen100_previews import (
    choose_participant_distinct,
    overlapping_classes,
)


class PopSignCitizen100PreviewTest(unittest.TestCase):
    def test_overlap_preserves_exact_citizen_variant(self):
        manifest = {
            "classes": [
                {
                    "class_index": 0,
                    "canonical_label": "GOODBYE",
                    "citizen_raw_gloss": "BYE",
                    "citizen_asl_lex_code": "E_01_058",
                },
                {
                    "class_index": 1,
                    "canonical_label": "WRITE",
                    "citizen_raw_gloss": "WRITE",
                    "citizen_asl_lex_code": "A_00_001",
                },
            ]
        }
        metadata = {"signs": {"bye": {}, "read": {}}}
        self.assertEqual(
            overlapping_classes(manifest, metadata),
            [
                {
                    "canonical_label": "GOODBYE",
                    "citizen_raw_gloss": "BYE",
                    "citizen_asl_lex_code": "E_01_058",
                    "popsign_gloss": "bye",
                }
            ],
        )

    def test_selection_is_deterministic_and_participant_distinct(self):
        records = {
            "2-sign.mp4": {"participant": "b", "original_name": "b-2.mp4"},
            "1-sign.mp4": {"participant": "a", "original_name": "a-2.mp4"},
            "0-sign.mp4": {"participant": "a", "original_name": "a-1.mp4"},
        }
        selected = choose_participant_distinct(records, 2)
        self.assertEqual([name for name, _ in selected], ["0-sign.mp4", "2-sign.mp4"])
        self.assertEqual([row["participant"] for _, row in selected], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
