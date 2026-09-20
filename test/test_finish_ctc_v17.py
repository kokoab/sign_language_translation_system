"""Contracts for the bounded Finish-time CTC experiment."""
import unittest

import torch

from active.v17.finish_ctc_v17 import FinishCTCHead, ctc_loss, decode, error_rate


class FinishCTCTest(unittest.TestCase):
    def test_head_keeps_locked_output_contract_and_uses_future_context(self):
        torch.manual_seed(7)
        model = FinishCTCHead(input_dim=108, stage1_dim=8, hidden_dim=6).eval()
        values = torch.randn(2, 7, 108)
        lengths = [7, 4]
        with torch.inference_mode():
            logits = model(values, lengths)
            prefix = model(values[:1, :4], [4])
        self.assertEqual(tuple(logits.shape), (2, 7, 102))
        self.assertFalse(torch.equal(logits[0, :4], prefix[0]))

    def test_ctc_accepts_verified_empty_background_and_repeated_glosses(self):
        logits = torch.randn(2, 5, 102, requires_grad=True)
        loss = ctc_loss(logits, [(), (1, 1)], [5, 5])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        with self.assertRaises(ValueError):
            ctc_loss(torch.randn(1, 1, 102), [(1, 1)], [1])

    def test_decode_never_exposes_blank_or_unknown_as_glosses(self):
        self.assertEqual(decode([0, 1, 1, 101, 1, 0, 100]), [1, 1, 100])

    def test_error_rate_excludes_matches(self):
        operations = {"match": 80, "substitution": 5, "deletion": 10, "insertion": 3}
        self.assertEqual(error_rate(operations, 100), 18.0)


if __name__ == "__main__":
    unittest.main()
