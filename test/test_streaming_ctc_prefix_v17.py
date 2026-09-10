"""Tests for conservative online CTC prefix confirmation."""

import unittest

from active.v17.streaming_ctc_prefix_v17 import StreamingCTCPrefix


class StreamingCTCPrefixTest(unittest.TestCase):
    def test_held_same_evidence_is_not_confirmation(self):
        state = StreamingCTCPrefix()
        first = state.update(["HELLO"], [3], evidence_end=20)
        held = state.update(["HELLO"], [3], evidence_end=20)
        self.assertEqual(first["committed"], [])
        self.assertEqual(held["committed"], [])
        self.assertEqual(held["new_commits"], [])
        self.assertEqual(held["provisional"], ["HELLO"])

    def test_earlier_common_prefix_commits_with_lookahead(self):
        state = StreamingCTCPrefix()
        state.update(["HELLO", "YOU"], [3, 10], evidence_end=16)
        result = state.update(["HELLO", "YOU", "HELP"], [3, 10, 18], evidence_end=20)
        self.assertEqual(result, {
            "committed": ["HELLO", "YOU"],
            "provisional": ["HELP"],
            "conflict": False,
            "new_commits": ["HELLO", "YOU"],
        })

    def test_adjacent_repeated_signs_remain_separate_positions(self):
        state = StreamingCTCPrefix()
        state.update(["I", "I"], [4, 9], evidence_end=16)
        result = state.update(["I", "I"], [4, 9], evidence_end=20)
        self.assertEqual(result["committed"], ["I", "I"])
        self.assertEqual(result["new_commits"], ["I", "I"])

    def test_conflict_does_not_rewrite_commit_and_reset_clears_state(self):
        state = StreamingCTCPrefix()
        state.update(["HELLO", "YOU"], [2, 10], evidence_end=16)
        state.update(["HELLO", "YOU", "HELP"], [2, 10, 18], evidence_end=20)
        conflict = state.update(["BYE", "YOU", "HELP"], [2, 10, 18], evidence_end=24)
        self.assertTrue(conflict["conflict"])
        self.assertEqual(conflict["committed"], ["HELLO", "YOU"])
        self.assertEqual(conflict["new_commits"], [])
        self.assertEqual(conflict["provisional"], [])
        replayed = state.update(
            ["HELLO", "YOU", "HELP", "MORE"], [2, 10, 18, 22], evidence_end=24
        )
        self.assertEqual(replayed["new_commits"], [])
        self.assertEqual(replayed["provisional"], ["HELP"])
        state.reset()
        reset = state.update(["BYE"], [2], evidence_end=12)
        self.assertEqual(reset["committed"], [])
        self.assertEqual(reset["provisional"], ["BYE"])

    def test_invalid_positions_are_rejected(self):
        state = StreamingCTCPrefix()
        for labels, positions, end in ((["A"], [0], 0), (["A"], [5], 5),
                                       (["A", "B"], [4, 4], 8), ([""], [1], 2),
                                       (["A"], [-1], 2), (["A"], [], 2)):
            with self.subTest(labels=labels, positions=positions, end=end):
                with self.assertRaises(ValueError):
                    state.update(labels, positions, end)


if __name__ == "__main__":
    unittest.main()
