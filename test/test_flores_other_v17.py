"""Small runnable checks for the Flores mapping and weighting contract."""
import unittest
from scripts.train_flores_other_v17 import map_targets, ctc_min_steps, device_setup
from active.v17.train_unified_streaming_aligned_grounded_v17 import supplement_weight

class FloresOtherTests(unittest.TestCase):
    def test_cpu_device_override(self):
        self.assertEqual(str(device_setup("cpu")), "cpu")

    def test_mapping(self):
        labels = {'HELLO': 0, 'TIME': 1}
        self.assertEqual(map_targets('HELLO strange unknown TIME, TIME', labels), [1, 101, 2, 2])
        self.assertEqual(map_targets('TIME2 TIME+ #TIME T-I-M-E', labels), [101])
        self.assertEqual(ctc_min_steps([1, 101, 2, 2]), 5)
        self.assertEqual(supplement_weight(1000, 20, .1), 5)
        with self.assertRaises(ValueError):
            supplement_weight(1000, 0, .1)

if __name__ == '__main__':
    unittest.main()
