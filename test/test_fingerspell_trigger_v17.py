"""FINGERSPELL-sign trigger: portable JSON trees, streaming firing rules (no video)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from active.v17.fingerspell_trigger_v17 import MODEL, N_HAND, FingerspellTrigger, TriggerModel, window_features


class StubModel:
    """Probability follows a script so the streaming rules can be checked without the trees."""
    threshold, smooth = .5, 1

    def __init__(self, probs):
        self.probs = list(probs)

    def probability(self, row):
        return self.probs.pop(0) if self.probs else 0.


class TriggerRules(unittest.TestCase):
    frame = np.zeros((61, 5), np.float32)

    def run_probs(self, probs):
        trig = FingerspellTrigger(StubModel(probs))
        fires = []
        for i in range(20 + 2 * (len(probs) - 1) + 1):
            f = trig.push(self.frame, i / 20)
            if f:
                fires.append(f)
        return fires

    def test_fires_once_per_sign(self):
        fires = self.run_probs([0, .9, .9, .9, 0])
        self.assertEqual(len(fires), 1)
        self.assertAlmostEqual(fires[0][1], 21 / 20)      # second window ends at frame 21

    def test_second_sign_needs_drop_and_refractory(self):
        on = [.9] * 3
        self.assertEqual(len(self.run_probs(on + [0] * 3 + on)), 1)       # dropped, but < 1.5 s later
        self.assertEqual(len(self.run_probs(on + [0] * 20 + on)), 2)      # dropped and 2 s later


class TriggerModelFile(unittest.TestCase):
    @unittest.skipUnless(MODEL.exists(), 'trigger model not exported')
    def test_json_trees_and_features(self):
        model = TriggerModel()
        rng = np.random.default_rng(0)
        raw = np.zeros((40, 61, 5), np.float32)
        raw[:, :, :2] = rng.normal(0, .1, (40, 61, 2)); raw[:, :, 4] = .9
        rows = window_features(raw)
        self.assertEqual(rows.shape, (11, N_HAND + 4))
        for row in rows:
            self.assertTrue(0 <= model.probability(row) <= 1)
        self.assertLess(model.probability(np.zeros(N_HAND + 4)), model.threshold)


if __name__ == '__main__':
    unittest.main()
