import unittest

import torch

from scripts.train_transition_anchors_v17 import anchor_loss


class TransitionAnchorTest(unittest.TestCase):
    def test_ignores_unlabeled_and_supports_blank_and_known(self):
        logits = torch.zeros(2, 4, 101, requires_grad=True)
        loss, counts = anchor_loss(logits, ["a", "b"], {
            "a": [{"indices": [0, 1], "target": 0}, {"indices": [2], "target": 7}],
        })
        self.assertEqual(counts, {"blank": 1, "known": 1})
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
