import unittest

import numpy as np

from scripts.fit_stage2_context_adapter_v17 import temporal_summary


class Stage2ContextAdapterTests(unittest.TestCase):
    def test_temporal_summary_dimensions_and_delta(self):
        value = np.arange(4 * 3, dtype=np.float32).reshape(1, 4, 3)
        self.assertEqual(temporal_summary(value, "mean").shape, (3,))
        self.assertEqual(temporal_summary(value, "mean_std_max").shape, (9,))
        summary = temporal_summary(value, "mean_std_max_delta")
        self.assertEqual(summary.shape, (12,))
        np.testing.assert_allclose(summary[-3:], np.zeros(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
