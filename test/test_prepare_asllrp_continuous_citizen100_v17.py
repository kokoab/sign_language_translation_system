import csv
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_asllrp_continuous_citizen100_v17 import (
    build_stage2_contiguous_spans,
    occurrence_matches,
    read_sentence_csv,
    signer_id,
)


FIELDS = [
    "Video ID number",
    "Main entry gloss label",
    "Entry/variant gloss label",
    "Occurrence label",
    "Start frame of the sign video",
    "End frame of the sign video",
    "Start frame of the containing utterance",
    "End frame of the containing utterance",
    "Dominant start handshape",
    "Non-dominant start handshape",
    "Dominant end handshape",
    "Non-dominant end handshape",
    "Sign video filename",
    "Utterance video filename",
    "Source collection",
    "Utterance number",
    "Master video filename",
    "Sign type",
    "Class label",
    "Hidden",
]


class PrepareAsllrpContinuousCitizen100V17Test(unittest.TestCase):
    def test_occurrence_contract_only_allows_trailing_repetition(self):
        self.assertTrue(occurrence_matches("FATHER", "FATHER"))
        self.assertTrue(occurrence_matches("FATHER", "FATHER++"))
        self.assertFalse(occurrence_matches("FATHER", "FATHERwg"))
        self.assertFalse(occurrence_matches("", ""))
        self.assertFalse(occurrence_matches("NO", "#NO"))

    def test_participant_derivation_is_source_specific(self):
        self.assertEqual(signer_id("asllrp", "Rachel_2012-02-14_sc66"), "RACHEL")
        self.assertEqual(
            signer_id("asllrp", "6-Ben-Control-of-Sound"),
            "BENJAMIN_JAMES_BAHAN",
        )
        self.assertEqual(signer_id("rit", "RIT-P02-s02"), "RIT_P02")

    def test_malformed_numeric_row_is_rejected_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            good = {
                "Video ID number": "1",
                "Main entry gloss label": "WATER",
                "Entry/variant gloss label": "WATER",
                "Occurrence label": "WATER",
                "Start frame of the sign video": "10",
                "End frame of the sign video": "20",
                "Start frame of the containing utterance": "1",
                "End frame of the containing utterance": "30",
                "Sign video filename": "sign_1.mp4",
                "Utterance video filename": "utterance_1.mp4",
                "Source collection": "Cory_1",
                "Utterance number": "1",
                "Master video filename": "master.mp4",
                "Sign type": "Lexical Signs",
                "Class label": "WATER",
                "Hidden": "F",
            }
            bad = dict(good)
            bad["Video ID number"] = "2"
            bad["Start frame of the sign video"] = "not-a-frame"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(good)
                writer.writerow(bad)
            rows, rejected = read_sentence_csv(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Video ID number"], "1")
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["line_number"], 3)

    def test_non_target_annotation_breaks_contiguous_phrase(self):
        def row(start, variant, sign_type="Lexical Signs"):
            return {
                "Entry/variant gloss label": variant,
                "Occurrence label": variant,
                "Start frame of the sign video": str(start),
                "End frame of the sign video": str(start + 4),
                "Start frame of the containing utterance": "0",
                "End frame of the containing utterance": "99",
                "Utterance video filename": "u.mp4",
                "Source collection": "Cory_1",
                "Sign type": sign_type,
                "Hidden": "F",
            }

        targets = [
            {"canonical_label": "WATER", "signbank_annotation_id": "WATER"},
            {"canonical_label": "COLD", "signbank_annotation_id": "COLD"},
        ]
        rows = [
            row(10, "WATER"),
            row(20, "WAVE", "Gestures"),
            row(30, "COLD"),
            row(40, "WATER"),
        ]
        spans = build_stage2_contiguous_spans("asllrp", rows, targets)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["target_sequence"], ["COLD", "WATER"])
        self.assertEqual(spans[0]["crop_start_frame_local"], 25)
        self.assertEqual(spans[0]["crop_end_frame_local"], 49)


if __name__ == "__main__":
    unittest.main()
