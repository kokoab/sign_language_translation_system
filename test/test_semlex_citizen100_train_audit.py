import unittest

from scripts.evaluate_semlex_citizen100_train_audit import aggregate_classes


class SemLexCitizen100TrainAuditTests(unittest.TestCase):
    def test_class_aggregation_counts_official_signers(self):
        rows = [
            {
                "true_label": "HELLO",
                "semlex_signer_id": signer,
                "top1_hit": top1,
                "top5_hit": top5,
                "true_probability": probability,
                "citizen_raw_gloss": "HELLO",
                "citizen_asl_lex_code": "D_02_055",
                "asllex_entry_id": "hello",
            }
            for signer, top1, top5, probability in [
                ("1", True, True, 0.8),
                ("2", True, True, 0.7),
                ("3", False, True, 0.2),
            ]
        ]
        result = aggregate_classes(rows)[0]
        self.assertEqual(result["signers"], 3)
        self.assertEqual(result["triage"], "model_consistent")
        self.assertFalse(result["training_approved"])


if __name__ == "__main__":
    unittest.main()
