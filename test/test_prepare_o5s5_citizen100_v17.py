import unittest

from scripts.prepare_o5s5_citizen100_v17 import deduplicate_hand_events


class O5S5PreparationTests(unittest.TestCase):
    def test_deduplicates_cross_hand_copies_one_to_one(self):
        events = [
            {"tier": "LeftHand_IDg", "value": "HELPpa", "start_ms": 100, "end_ms": 300},
            {"tier": "RightHand_IDg", "value": "HELPpa", "start_ms": 110, "end_ms": 290},
            {"tier": "LeftHand_IDg", "value": "HELPpa", "start_ms": 400, "end_ms": 600},
            {"tier": "RightHand_IDg", "value": "HELPpa", "start_ms": 410, "end_ms": 590},
            {"tier": "RightHand_IDg", "value": "GOix", "start_ms": 650, "end_ms": 800},
        ]
        result = deduplicate_hand_events(events)
        self.assertEqual(3, len(result))
        self.assertEqual(["HELPpa", "HELPpa", "GOix"], [row["id_gloss"] for row in result])
        self.assertEqual([2, 2, 1], [row["tier_event_count"] for row in result])


if __name__ == "__main__":
    unittest.main()
