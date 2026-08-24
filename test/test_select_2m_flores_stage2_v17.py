import unittest

from scripts.select_2m_flores_stage2_v17 import select_rows


class TwoMFloresSelectionTests(unittest.TestCase):
    def test_minimum_byte_multicover_selection(self):
        rows = [
            {"id": 1, "video_url": "https://x/data/dev/a.MOV", "matched_locked_labels": ["A"]},
            {"id": 2, "video_url": "https://x/data/dev/b.MOV", "matched_locked_labels": ["B"]},
            {"id": 3, "video_url": "https://x/data/dev/c.MOV", "matched_locked_labels": ["A", "B"]},
        ]
        files = {
            "data/dev/a.MOV": {"source_bytes": 40, "source_sha256": "a" * 64},
            "data/dev/b.MOV": {"source_bytes": 40, "source_sha256": "b" * 64},
            "data/dev/c.MOV": {"source_bytes": 50, "source_sha256": "c" * 64},
        }
        selected, targets = select_rows(rows, {"A": 2, "B": 2}, files, quota=1)
        self.assertEqual([row["id"] for row in selected], [3])
        self.assertEqual(targets, {"A": 1, "B": 1})

    def test_rare_label_target_is_capped_at_availability(self):
        rows = [
            {"id": 1, "video_url": "https://x/data/dev/a.MOV", "matched_locked_labels": ["RARE"]},
        ]
        files = {
            "data/dev/a.MOV": {"source_bytes": 10, "source_sha256": "a" * 64},
        }
        selected, targets = select_rows(rows, {"RARE": 1}, files, quota=5)
        self.assertEqual(len(selected), 1)
        self.assertEqual(targets, {"RARE": 1})


if __name__ == "__main__":
    unittest.main()
