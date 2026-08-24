import unittest

from scripts.evaluate_msasl_citizen100_gap_audit import aggregate_classes


def record(signer, top1, top5):
    return {
        "true_label": "WE",
        "msasl_signer_id": signer,
        "citizen_raw_gloss": "WE",
        "citizen_asl_lex_code": "X",
        "top1_hit": top1,
        "top5_hit": top5,
        "true_probability": 0.5,
    }


class MSASLCitizen100GapAuditTest(unittest.TestCase):
    def test_consistent_requires_two_signers_and_rates(self):
        output = aggregate_classes([record("1", True, True), record("2", True, True)])
        self.assertEqual(output[0]["triage"], "model_consistent_manual_variant_review_required")
        self.assertFalse(output[0]["training_approved"])

    def test_one_signer_stays_ambiguous(self):
        output = aggregate_classes([record("1", True, True)])
        self.assertEqual(output[0]["triage"], "ambiguous_manual_variant_review_required")

    def test_low_top5_is_high_risk(self):
        output = aggregate_classes([record("1", False, False), record("2", False, False)])
        self.assertEqual(output[0]["triage"], "high_risk_manual_variant_review_required")


if __name__ == "__main__":
    unittest.main()
