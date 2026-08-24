import unittest

from scripts.evaluate_local_citizen100_quality_audit import aggregate_classes


def record(top1, top5):
    return {
        "true_label": "HELP",
        "citizen_raw_gloss": "HELP",
        "citizen_asl_lex_code": "D_01_042",
        "lexical_tier": "canonical_and_pinned_raw_text_equal",
        "top1_hit": top1,
        "top5_hit": top5,
        "true_probability": 0.5,
    }


class LocalCitizen100QualityAuditTest(unittest.TestCase):
    def test_consistent_requires_three_clips_and_rates(self):
        output = aggregate_classes(
            [record(True, True), record(True, True), record(False, True), record(False, False)]
        )
        self.assertEqual(output[0]["triage"], "model_consistent_manual_variant_review_required")
        self.assertFalse(output[0]["training_approved"])

    def test_top5_half_is_ambiguous(self):
        output = aggregate_classes([record(False, True), record(False, False)])
        self.assertEqual(output[0]["triage"], "ambiguous_manual_variant_review_required")

    def test_low_top5_is_high_risk(self):
        output = aggregate_classes([record(False, False), record(False, False), record(False, True)])
        self.assertEqual(output[0]["triage"], "high_risk_manual_variant_review_required")


if __name__ == "__main__":
    unittest.main()
