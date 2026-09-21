import unittest
import numpy as np
from scripts.recover_phrase_tails_v17 import recovery_ranges

class TailRecoveryTests(unittest.TestCase):
    def test_rebalances_only_last_window_and_covers_each_source_frame(self):
        for n in (33, 34, 35, 65, 98, 163):
            old = np.array([(i, i + 32) for i in range(0, n - n % 32, 32)])
            new = recovery_ranges(old, n)
            np.testing.assert_array_equal(new[:-2], old[:-1])
            self.assertEqual(new[-1, 1], n)
            self.assertEqual(new[0, 0], 0)
            self.assertTrue(np.all(new[1:, 0] == new[:-1, 1]))
            self.assertTrue(np.all((new[:, 1]-new[:, 0] >= 4) & (new[:, 1]-new[:, 0] <= 32)))
        for old, n in (([[0, 32]], 32), ([[0, 32]], 36), ([[0, 32], [33, 65]], 66)):
            with self.assertRaises(ValueError):
                recovery_ranges(np.array(old), n)
