import unittest

from scripts.audit_2m_flores_stage2_v17 import normalized_gloss_tokens


class TwoMFloresAuditTests(unittest.TestCase):
    def test_fingerspelling_is_not_misread_as_lexical_sign(self):
        self.assertEqual(
            normalized_gloss_tokens("WATER W-A-T-E-R, THANK-YOU IX-1P"),
            ["WATER", "THANKYOU", "IX1P"],
        )

    def test_punctuation_is_normalized(self):
        self.assertEqual(normalized_gloss_tokens("HELLO, HOW YOU?"), ["HELLO", "HOW", "YOU"])


if __name__ == "__main__":
    unittest.main()
