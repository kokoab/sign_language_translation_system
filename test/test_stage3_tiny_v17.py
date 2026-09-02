import unittest
from pathlib import Path
import tempfile

from active.v17.evaluate_stage3_tiny_v17 import write_jsonl
from active.v17.train_stage3_tiny_v17 import (
    controlled_long_rows,
    load_pairs,
    normalized,
)


class Stage3TinyDataTests(unittest.TestCase):
    def test_controlled_long_rows_cover_validation_test_and_long_buffers(self):
        rows = controlled_long_rows(17032)
        self.assertTrue(rows)
        self.assertTrue(all(row.length >= 5 for row in rows))
        self.assertTrue(any(row.length >= 9 for row in rows))
        self.assertEqual({row.split for row in rows}, {"train", "validation", "test"})

    def test_complete_data_has_no_gloss_split_leakage(self):
        rows, plan = load_pairs(17032)
        splits = {}
        for row in rows:
            splits.setdefault(row.gloss, set()).add(row.split)
        self.assertTrue(all(len(value) == 1 for value in splits.values()))
        self.assertEqual(plan["split_gloss_overlap"], 0)
        self.assertGreaterEqual(plan["locked100_legacy_long_test_rows"], 1)
        self.assertGreater(plan["controlled_long_by_split"]["validation"], 0)
        self.assertGreater(plan["controlled_long_by_split"]["test"], 0)

    def test_normalized_comparison_ignores_only_surface_punctuation(self):
        self.assertEqual(normalized("Hello, friend!"), "hello friend")
        self.assertNotEqual(normalized("Hello, friend!"), normalized("Hello!"))

    def test_jsonl_writer_preserves_long_unicode_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl(path, [{"gloss": "HELLO HOW YOU TODAY", "text": "I'm fine."}])
            self.assertIn("I'm fine.", path.read_text())


if __name__ == "__main__":
    unittest.main()
