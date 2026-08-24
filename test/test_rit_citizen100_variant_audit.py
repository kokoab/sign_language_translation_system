import unittest

from scripts.evaluate_rit_citizen100_variant_audit import aggregate_classes


def record(participant, top1, top5, probability=0.5):
    return {
        "true_label": "HELP",
        "participant": participant,
        "citizen_raw_gloss": "HELP",
        "citizen_asl_lex_code": "D_01_042",
        "match_tier": "pinned_raw_gloss_exact",
        "top1_hit": top1,
        "top5_hit": top5,
        "true_probability": probability,
    }


class RITCitizen100VariantAuditTest(unittest.TestCase):
    def test_consistent_requires_two_participants_and_rates(self):
        output = aggregate_classes(
            [record("P01", True, True), record("P02", False, True)]
        )
        self.assertEqual(
            output[0]["triage"],
            "model_consistent_manual_variant_review_required",
        )
        self.assertFalse(output[0]["training_approved"])

    def test_single_participant_remains_ambiguous(self):
        output = aggregate_classes([record("P01", True, True)])
        self.assertEqual(
            output[0]["triage"], "ambiguous_manual_variant_review_required"
        )

    def test_low_top5_is_high_risk(self):
        output = aggregate_classes(
            [record("P01", False, False), record("P02", False, True)]
        )
        self.assertEqual(
            output[0]["triage"], "ambiguous_manual_variant_review_required"
        )
        output = aggregate_classes(
            [record("P01", False, False), record("P02", False, False)]
        )
        self.assertEqual(
            output[0]["triage"], "high_risk_manual_variant_review_required"
        )


if __name__ == "__main__":
    unittest.main()
