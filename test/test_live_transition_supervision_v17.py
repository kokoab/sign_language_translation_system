import unittest

from active.v17.live_transition_supervision_v17 import (
    _prefix_targets, blank_positions_for_windows, ctc_bin_intervals, interior_gaps,
    known_core_positions_for_windows,
)


class LiveTransitionSupervisionTests(unittest.TestCase):
    def test_only_full_bins_inside_guarded_interior_gap_are_blank(self):
        intervals = [
            {"start_seconds": 0.00, "end_seconds": 0.20},
            {"start_seconds": 0.60, "end_seconds": 0.80},
        ]
        positions, gaps = blank_positions_for_windows(
            [{"start": 0.0, "end": 0.8, "accepted": True}], intervals, 0.10
        )
        self.assertEqual(gaps, [(0.30000000000000004, 0.5)])
        self.assertEqual(positions, [3, 4])  # [0.3, 0.4] and [0.4, 0.5] fit the gap.

    def test_rejected_windows_do_not_consume_ctc_positions(self):
        intervals = [
            {"start_seconds": 0.00, "end_seconds": 0.10},
            {"start_seconds": 0.90, "end_seconds": 1.00},
        ]
        positions, _ = blank_positions_for_windows([
            {"start": 0.0, "end": 0.5, "accepted": False},
            {"start": 0.5, "end": 1.0, "accepted": True},
        ], intervals, 0.10)
        self.assertEqual(positions, list(range(4)))

    def test_bin_intervals_and_edge_gaps_are_not_transition_targets(self):
        self.assertEqual(len(ctc_bin_intervals({"start": 0.0, "end": 0.8})), 8)
        self.assertEqual(ctc_bin_intervals({"start": 0.0, "end": 0.8})[0], (0.0, 0.1))
        self.assertEqual(interior_gaps([{"start_seconds": 0.2, "end_seconds": 0.5}], 0.1), [])

    def test_prefix_collapses_adjacent_other_and_keeps_ctc_feasible(self):
        events = [
            {"start_seconds": 0.0, "end_seconds": 0.1, "ctc_index": 101},
            {"start_seconds": 0.1, "end_seconds": 0.2, "ctc_index": 101},
            {"start_seconds": 0.3, "end_seconds": 0.4, "ctc_index": 7},
        ]
        targets = _prefix_targets(
            [{"start": 0.0, "end": 0.25, "accepted": True}, {"start": 0.25, "end": 0.8, "accepted": True}],
            events, [101, 7], True,
        )
        self.assertEqual(targets["1"], [101])
        self.assertEqual(targets["2"], [101, 7])
        self.assertTrue(all(len(value) + sum(a == b for a, b in zip(value, value[1:])) <= int(key) * 8 for key, value in targets.items()))

    def test_known_cores_exclude_other_and_annotation_overlap(self):
        intervals = [
            {"start_seconds": 0.0, "end_seconds": 0.5, "ctc_index": 5},
            {"start_seconds": 0.25, "end_seconds": 0.3, "ctc_index": 101},
            {"start_seconds": 0.5, "end_seconds": 0.8, "ctc_index": 9},
        ]
        positions = known_core_positions_for_windows(
            [{"start": 0.0, "end": 0.8, "accepted": True}], intervals,
        )
        self.assertEqual(positions, {"1": 5, "3": 5, "6": 9})


if __name__ == "__main__":
    unittest.main()
