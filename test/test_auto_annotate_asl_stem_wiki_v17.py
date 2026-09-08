from pathlib import Path
import sys
import unittest

from scripts.auto_annotate_asl_stem_wiki_v17 import (
    assign_confidence_tier,
    boundary_iou,
    calibrate_high_threshold,
    candidate_spans,
    ensure_repo_imports,
    requested_frame,
    select_annotation,
)


class AutoAnnotateAslStemWikiV17Test(unittest.TestCase):
    def test_candidate_spans_stay_inside_five_second_search_range(self):
        spans = candidate_spans(frame_count=900, fps=30.0, anchor=450)
        self.assertTrue(spans)
        self.assertTrue(all(300 <= start < end <= 601 for start, end in spans))
        self.assertIn((438, 462), spans)

    def test_selector_prefers_full_model_ctc_agreement_over_coarse_peak(self):
        candidates = [
            {
                "start_frame": 90, "end_frame": 114,
                "coarse_target_probability": .98, "full_target_probability": .35,
                "full_top1_matches": False, "ctc_agrees": False,
                "boundary_stability": .4, "motion_score": .7,
            },
            {
                "start_frame": 100, "end_frame": 124,
                "coarse_target_probability": .75, "full_target_probability": .82,
                "full_top1_matches": True, "ctc_agrees": True,
                "boundary_stability": .9, "motion_score": .8,
            },
        ]
        selected = select_annotation(candidates)
        self.assertEqual((selected["start_frame"], selected["end_frame"]), (100, 124))
        self.assertGreater(selected["confidence"], .8)

    def test_calibration_places_high_threshold_above_every_review_failure(self):
        threshold = calibrate_high_threshold([
            {"confidence": .91, "review_success": True},
            {"confidence": .88, "review_success": True},
            {"confidence": .85, "review_success": True},
            {"confidence": .62, "review_success": False},
            {"confidence": .70, "review_success": False},
        ])
        self.assertEqual(threshold, .71)
        self.assertEqual(assign_confidence_tier({"confidence": .88, "has_target_evidence": True}, threshold), "high")
        self.assertEqual(assign_confidence_tier({"confidence": .70, "has_target_evidence": True}, threshold), "review")
        self.assertEqual(assign_confidence_tier({"confidence": .99, "has_target_evidence": False}, threshold), "abstain")

    def test_calibration_refuses_high_tier_when_fewer_than_three_successes_survive(self):
        self.assertIsNone(calibrate_high_threshold([
            {"confidence": .91, "review_success": True},
            {"confidence": .89, "review_success": True},
            {"confidence": .88, "review_success": False},
        ]))

    def test_boundary_iou_uses_inclusive_review_bounds(self):
        self.assertAlmostEqual(boundary_iou(100, 124, 110, 129), 15 / 30)

    def test_disjoint_search_ranges_keep_exact_source_frame_numbers(self):
        ranges = [(2, 5), (8, 10)]
        self.assertEqual(
            [index for index in range(12) if requested_frame(index, ranges)],
            [2, 3, 4, 8, 9],
        )

    def test_direct_entry_adds_repository_root_to_import_path(self):
        root = Path(__file__).resolve().parents[1]
        original = list(sys.path)
        try:
            sys.path[:] = [value for value in sys.path if Path(value or ".").resolve() != root]
            self.assertEqual(ensure_repo_imports(), root)
            self.assertEqual(Path(sys.path[0]), root)
        finally:
            sys.path[:] = original


if __name__ == "__main__":
    unittest.main()
