import unittest
import torch

from scripts.train_local_familiar_ctc_v17 import collapse, ctc_loss, known, paired_plan


class LocalFamiliarCTCTest(unittest.TestCase):
    def test_paired_plan_covers_singles_and_cycles_phrases(self):
        plan = paired_plan(["a", "b", "c"], ["p", "q"], 7, 1)
        self.assertEqual(set(plan["singles"]), {"a", "b", "c"})
        self.assertEqual(len(plan["phrases"]), 3)

    def test_ctc_is_target_normalized_and_collapse_keeps_repeat_across_blank(self):
        logits = torch.zeros(2, 3, 102, requires_grad=True)
        loss = ctc_loss(logits, [{"targets": (1,), "weight": 1.0}, {"targets": (2, 3), "weight": .5}], [3, 3])
        self.assertTrue(torch.isfinite(loss)); loss.backward(); self.assertGreater(logits.grad.abs().sum(), 0)
        self.assertEqual(collapse([0, 1, 1, 0, 1]), [1, 1])
        self.assertEqual(known([1, 101, 2]), [1, 2])


if __name__ == "__main__": unittest.main()
