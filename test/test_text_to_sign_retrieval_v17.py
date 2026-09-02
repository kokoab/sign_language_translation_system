import unittest

from scripts.render_text_to_sign_retrieval_v17 import normalize_phrase, resolve_item


class TextToSignRetrievalV17Tests(unittest.TestCase):
    def test_normalizes_plain_text_and_compound_thankyou(self):
        self.assertEqual(normalize_phrase("hello, how you?"), "HELLO_HOW_YOU")
        self.assertEqual(normalize_phrase("thank you friend"), "THANKYOU_FRIEND")

    def test_unknown_phrase_fails_closed(self):
        report = {"items": [{"phrase": "HELLO_HOW_YOU"}]}
        self.assertEqual(
            resolve_item(report, "hello how you")["phrase"], "HELLO_HOW_YOU"
        )
        with self.assertRaisesRegex(ValueError, "unsupported exact phrase"):
            resolve_item(report, "hello friend")


if __name__ == "__main__":
    unittest.main()
