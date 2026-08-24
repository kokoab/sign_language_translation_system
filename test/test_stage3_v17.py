import unittest

from active.v17.evaluate_stage3_reference_gloss_v17 import load_rows
from active.v17.train_stage3_reference_replay_v17 import (
    DEFAULT_ALL_FLORES,
    DEFAULT_NCSLGR,
    DEFAULT_SELECTED_FLORES,
    DEFAULT_SYNTHETIC,
    load_data,
    stable_key,
)
from scripts.acquire_2m_flores_dev_metadata_v17 import compact_row


class Stage3V17DataContractTests(unittest.TestCase):
    def test_metadata_compaction_drops_video_payload(self):
        row = compact_row(
            {
                "id": 7,
                "video": {"path": "data/dev/dev_7_0.mov", "src": "https://example/video.mov", "bytes": b"x"},
                "gloss": "HELLO",
                "sentence": "Hello.",
            }
        )
        self.assertNotIn("video", row)
        self.assertEqual(row["video_path"], "data/dev/dev_7_0.mov")
        self.assertEqual(row["video_url"], "https://example/video.mov")

    def test_reference_rows_are_present_without_reserved_split(self):
        rows = load_rows(DEFAULT_NCSLGR, DEFAULT_SELECTED_FLORES)
        self.assertEqual(len(rows), 321)
        self.assertEqual(sum(row["source"] == "ncslgr" for row in rows), 166)
        self.assertEqual(sum(row["source"] == "2m_flores_dev" for row in rows), 155)

    def test_replay_plan_is_exact_and_leak_free(self):
        train, validation, plan = load_data(
            DEFAULT_ALL_FLORES,
            DEFAULT_SELECTED_FLORES,
            DEFAULT_NCSLGR,
            DEFAULT_SYNTHETIC,
            17031,
        )
        self.assertEqual(plan["flores_train_rows"], 844)
        self.assertEqual(plan["ncslgr_train_rows"], 166)
        self.assertEqual(plan["synthetic_replay_rows"], 1010)
        self.assertEqual(plan["validation_rows"], 155)
        train_ids = {row["id"] for row in train if row["source"] == "2m_flores_dev_train"}
        validation_ids = {row["id"] for row in validation}
        self.assertFalse(train_ids & validation_ids)

    def test_synthetic_replay_order_is_deterministic(self):
        row = {"gloss": "HELLO", "text": "Hello."}
        self.assertEqual(stable_key(row, 17031), stable_key(dict(row), 17031))
        self.assertNotEqual(stable_key(row, 17031), stable_key(row, 17032))


if __name__ == "__main__":
    unittest.main()
