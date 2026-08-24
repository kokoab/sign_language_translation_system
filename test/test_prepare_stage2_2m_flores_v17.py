import unittest

from scripts.prepare_stage2_2m_flores_v17 import normalize_gloss


class PrepareTwoMFloresStage2Tests(unittest.TestCase):
    def setUp(self):
        self.locked = {"THANKYOU": "THANKYOU", "MORNING": "MORNING"}

    def test_locked_tokens_and_annotation_categories_are_stable(self):
        self.assertEqual(
            normalize_gloss(
                "THANK-YOU MORNING #JAS W-A-T-E-R 9 IX+++ POSS-1P cl:FLIGHT-IN-OUT",
                self.locked,
            ),
            ["THANKYOU", "MORNING", "__FS__", "__FS__", "__NUM__", "__IX__", "__IX__", "__CL__"],
        )

    def test_tokens_are_not_deleted_when_punctuation_is_removed(self):
        self.assertEqual(
            normalize_gloss("PUNISH //WHAT\\ EUROPE,", self.locked),
            ["PUNISH", "WHAT", "EUROPE"],
        )


if __name__ == "__main__":
    unittest.main()
