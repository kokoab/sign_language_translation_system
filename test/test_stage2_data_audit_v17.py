import unittest

from scripts.acquire_ncslgr_stage2_v17 import parse_annotation, participant_id
from scripts.audit_stage2_phrase_sources_v17 import audit_phrase_vocabulary, normalized_token


class Stage2DataAuditV17Test(unittest.TestCase):
    def test_local_phrase_vocabulary_is_fail_closed(self):
        result = audit_phrase_vocabulary(
            {"GOOD_MORNING": 60, "I_WANT_FOOD": 60, "SORRY_I_LATE": 140},
            ["GOOD", "MORNING", "I", "WANT", "EAT", "SORRY"],
        )
        self.assertEqual(result["strict_usable_videos"], 60)
        self.assertEqual(result["usable_if_review_alias_approved"], 120)

    def test_signstream_annotation_keeps_gloss_order(self):
        parsed = parse_annotation(
            "Start frame:\t100\n"
            "End frame:\t900\n"
            "Participant:\tSigner One\n"
            "main gloss\tGOOD\t120\t260\tMORNING+\t300\t600\n"
            "\tYOU\t620\t800\n"
            "non-dominant hand gloss\n"
            "English translation\tGood morning to you.\n"
        )
        self.assertEqual(parsed["main_glosses"], ["GOOD", "MORNING+", "YOU"])
        self.assertEqual(parsed["participant"], "Signer One")
        self.assertEqual(parsed["start_frame"], 100)
        self.assertEqual(parsed["end_frame"], 900)

    def test_participant_name_variants_normalize(self):
        self.assertEqual(
            participant_id("Norma Bowers-Tourangeau"),
            participant_id("Norma Bowers Tourangeau"),
        )

    def test_label_normalization_is_punctuation_insensitive(self):
        self.assertEqual(normalized_token("THANK-YOU"), "THANKYOU")


if __name__ == "__main__":
    unittest.main()
