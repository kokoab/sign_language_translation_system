"""Tests for the trainable live Stage-2 OTHER adaptation wrapper."""

import tempfile
import unittest
from pathlib import Path

import torch

from active.v17 import model_stage2_v17 as models
from active.v17.model_stage2_live_adapt_v17 import (
    Stage2LiveAdaptedCTCV17,
    load_stage2_live_adapted,
    make_stage2_live_adapted_checkpoint,
)


def small_preservation_model():
    config = models.Stage2V17Config(dim=16, heads=4, depth=1, dropout=0.)
    adapter_config = dict(feature_mode="mean", target_class_indices=[0], weight=1.)
    primary = models.Stage2ContextAdapterV17(
        models.Stage2TemporalHeadV17(config), **adapter_config,
        scaler_mean=torch.zeros(612), scaler_scale=torch.ones(612),
        coefficients=torch.zeros(1, 612), intercept=torch.zeros(1),
        class_indices=torch.tensor([0]),
    )
    accepted = models.Stage2GeneralCTCSelectorV17(
        primary, models.Stage2TemporalHeadV17(config),
        blend_weight=.1, blank_bias=.3, score_margin=0., minimum_tokens=2,
    )
    evidence = models.Stage2TemporalHeadV17(models.Stage2V17Config(
        **{**config.to_dict(), "num_classes": 101},
    ))
    return models.Stage2OtherPreservingCTCV17(
        accepted, evidence, torch.nn.Linear(16, 1), margin=1.,
    )


class Stage2LiveAdaptedCTCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = small_preservation_model()
        self.model = Stage2LiveAdaptedCTCV17(self.base)
        self.features = torch.randn(2, 2, 32, 612)
        self.mask = torch.tensor([[True, True], [True, False]])

    def test_initial_known_conditionals_and_argmax_match_accepted(self):
        with torch.no_grad():
            known, lengths = self.base.accepted(self.features, self.mask)
            actual, actual_lengths = self.model(self.features, self.mask)
        torch.testing.assert_close(actual[..., :101].softmax(-1), known.softmax(-1))
        self.assertTrue(torch.equal(actual_lengths, lengths))
        self.assertTrue(torch.equal(actual.argmax(-1), known.argmax(-1)))

    def test_ctc_backward_reaches_trainable_parts_but_not_evidence(self):
        self.model.train()
        logits, lengths = self.model(self.features, self.mask)
        loss = torch.nn.CTCLoss(blank=0)(
            logits.log_softmax(-1).transpose(0, 1), torch.tensor([1, 2]),
            lengths, torch.tensor([1, 1]),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None for p in self.model.accepted.parameters()))
        self.assertIsNotNone(self.model.other_head.weight.grad)
        self.assertIsNotNone(self.model.other_shift.grad)
        self.assertTrue(all(p.grad is None for p in self.model.evidence.parameters()))
        self.assertFalse(self.model.accepted.training)
        self.assertFalse(self.model.evidence.training)

    def test_package_reload_is_exact_and_carries_input_contract(self):
        contract = {"feature": "frozen-v17", "windows": 8}
        payload = make_stage2_live_adapted_checkpoint(self.model, input_contract=contract)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapted.pth"
            torch.save(payload, path)
            loaded, checkpoint = load_stage2_live_adapted(path)
            with torch.no_grad():
                expected = self.model(self.features, self.mask)
                actual = loaded(self.features, self.mask)
        torch.testing.assert_close(actual[0], expected[0])
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(checkpoint["input_contract"], contract)


if __name__ == "__main__":
    unittest.main()
