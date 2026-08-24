from __future__ import annotations

import unittest

from scripts.select_local_citizen100_consensus import classify_candidate


class LocalCitizen100ConsensusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.provenance = {
            "lexical_tier": "canonical_and_pinned_raw_text_equal"
        }
        self.diagnostics = {
            "observed_hand_frame_fraction": 0.90,
            "face_presence_fraction": 0.75,
        }

    @staticmethod
    def prediction(top1: bool, top5: bool) -> dict[str, str]:
        return {"top1_hit": str(top1), "top5_hit": str(top5)}

    def test_dual_top1_is_tier_a(self):
        prediction = self.prediction(True, True)
        self.assertEqual(
            classify_candidate(
                self.provenance, prediction, prediction, self.diagnostics
            ),
            "tier_a_dual_top1",
        )

    def test_one_top1_with_dual_top5_is_tier_b(self):
        self.assertEqual(
            classify_candidate(
                self.provenance,
                self.prediction(False, True),
                self.prediction(True, True),
                self.diagnostics,
            ),
            "tier_b_dual_top5_one_top1",
        )

    def test_dual_top5_without_top1_is_tier_c(self):
        prediction = self.prediction(False, True)
        self.assertEqual(
            classify_candidate(
                self.provenance, prediction, prediction, self.diagnostics
            ),
            "tier_c_dual_top5_only",
        )

    def test_quality_failure_is_quarantined_before_model_agreement(self):
        diagnostics = dict(self.diagnostics, observed_hand_frame_fraction=0.79)
        prediction = self.prediction(True, True)
        self.assertEqual(
            classify_candidate(
                self.provenance, prediction, prediction, diagnostics
            ),
            "quarantine_extraction_quality",
        )


if __name__ == "__main__":
    unittest.main()
