import unittest
from scripts.evaluate_annotation_cores_v17 import core_metrics


class CoreMetricsTests(unittest.TestCase):
    def test_blank_and_mixed_emissions_do_not_count_as_correct_unknown(self):
        rows = [dict(expected=101, predicted=p) for p in [[], [101], [101, 3], [3]]]
        rows += [dict(expected=3, predicted=p) for p in [[3], [101], [3, 101], []]]
        metrics = core_metrics(rows)
        self.assertEqual(metrics['oov']['exact'], 1)
        self.assertEqual(metrics['oov']['other_emitted'], 2)
        self.assertEqual(metrics['oov']['false_known'], 2)
        self.assertEqual(metrics['oov']['blank_only'], 1)
        self.assertEqual(metrics['known']['exact'], 1)
        self.assertEqual(metrics['known']['other_without_expected'], 1)
        self.assertEqual(metrics['known']['other_emitted'], 2)
        self.assertEqual(metrics['known']['blank_only'], 1)


if __name__ == '__main__':
    unittest.main()
