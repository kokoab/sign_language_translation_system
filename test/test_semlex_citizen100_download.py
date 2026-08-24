import unittest

from scripts.download_semlex_citizen100_candidates import (
    normalized_member,
    parse_content_range,
)


class SemLexCitizen100DownloadTests(unittest.TestCase):
    def test_content_range_parser(self):
        self.assertEqual(parse_content_range("bytes 0-63/100"), (0, 63, 100))
        with self.assertRaises(ValueError):
            parse_content_range("0-63/100")

    def test_member_normalization(self):
        self.assertEqual(normalized_member("./train/id.webm"), "train/id.webm")
        self.assertEqual(normalized_member("train/id.webm"), "train/id.webm")


if __name__ == "__main__":
    unittest.main()
