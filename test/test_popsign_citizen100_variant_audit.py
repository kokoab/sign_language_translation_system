import unittest

from scripts.evaluate_popsign_citizen100_variant_audit import aggregate_classes


class PopSignCitizen100VariantAuditTest(unittest.TestCase):
    def test_aggregation_never_auto_approves_training(self):
        rows = [
            {
                "true_label": "HELLO",
                "citizen_raw_gloss": "HELLO",
                "citizen_asl_lex_code": "D_02_055",
                "popsign_gloss": "hello",
                "top1_hit": hit,
                "top5_hit": True,
                "true_probability": 0.8,
            }
            for hit in (True, True, False)
        ]
        result = aggregate_classes(rows)
        self.assertEqual(result[0]["triage"], "model_consistent_manual_review_required")
        self.assertFalse(result[0]["training_approved"])


if __name__ == "__main__":
    unittest.main()
