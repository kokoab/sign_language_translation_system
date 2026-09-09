import unittest
import tempfile
from pathlib import Path

import torch

from active.v17 import model_stage2_v17 as models


class OtherPreservationTests(unittest.TestCase):
    def test_package_reload_preserves_real_model_predictions(self):
        config = models.Stage2V17Config(dim=16, heads=4, depth=1, dropout=0.)
        base = models.Stage2TemporalHeadV17(config)
        adapter_config = dict(feature_mode='mean', target_class_indices=[0], weight=1.)
        primary = models.Stage2ContextAdapterV17(
            base, **adapter_config, scaler_mean=torch.zeros(612),
            scaler_scale=torch.ones(612), coefficients=torch.zeros(1, 612),
            intercept=torch.zeros(1), class_indices=torch.tensor([0]),
        )
        selector_config = dict(blend_weight=.1, blank_bias=.3, score_margin=0., minimum_tokens=2)
        accepted = models.Stage2GeneralCTCSelectorV17(
            primary, models.Stage2TemporalHeadV17(config), **selector_config,
        )
        evidence = models.Stage2TemporalHeadV17(models.Stage2V17Config(
            **{**config.to_dict(), 'num_classes': 101},
        ))
        head = torch.nn.Linear(16, 1)
        model = models.Stage2OtherPreservingCTCV17(accepted, evidence, head, 1.)
        payload = dict(
            format='slt_stage2_other_preserving_ctc_v17', margin=1.,
            accepted_checkpoint=dict(
                format='slt_stage2_general_ctc_selector_v17', model_config=config.to_dict(),
                model_state_dict=accepted.state_dict(), primary_context_adapter_config=adapter_config,
                selector_config=selector_config,
            ),
            evidence_checkpoint=models.make_stage2_checkpoint(evidence, evidence.state_dict()),
            other_head_state_dict=head.state_dict(),
        )
        features = torch.randn(2, 2, 32, 612)
        mask = torch.tensor([[True, True], [True, False]])
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path = Path(directory) / 'model.pth'
            torch.save(payload, path)
            loaded, _ = models.load_stage2_model_v17(path)
            expected, lengths = model(features, mask)
            actual, actual_lengths = loaded(features, mask)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.equal(lengths, actual_lengths))
            self.assertTrue(torch.isfinite(actual[..., :101]).all())

    def test_preservation_rejects_whole_runs_without_splitting_or_touching_padding(self):
        self.assertTrue(hasattr(models, 'preserve_ctc_emission_runs'))
        path = torch.tensor([[1, 1, 0, 1, 0, 2, 2, 2]])
        old = torch.full((1, 8, 3), -8.)
        old.scatter_(-1, path.unsqueeze(-1), 8.)
        odds = torch.tensor([[[-5.], [3.], [9.], [-5.], [9.], [3.], [3.], [9.]]])
        actual = models.preserve_ctc_emission_runs(old, odds, torch.tensor([7]), 1.)
        self.assertEqual(actual.argmax(-1).tolist(), [[1, 1, 0, 1, 0, 3, 3, 2]])
        torch.testing.assert_close(actual[..., :3].softmax(-1), old.softmax(-1))
        with self.assertRaises(ValueError):
            models.preserve_ctc_emission_runs(old, odds, torch.tensor([9]), 1.)

    def test_other_extension_preserves_old_conditional_and_trains_rejection(self):
        self.assertTrue(hasattr(models, 'preserve_known_ctc_logits'))
        old = torch.tensor([[[0., 3., -1.], [4., 0., 1.]]])
        other = torch.tensor([[[-5.], [5.]]], requires_grad=True)
        combined = models.preserve_known_ctc_logits(old, other)
        torch.testing.assert_close(combined[..., :3].softmax(-1), old.softmax(-1))
        self.assertEqual(combined.argmax(-1).tolist(), [[1, 3]])
        loss = -combined.log_softmax(-1)[0, 1, 0]
        loss.backward()
        self.assertGreater(float(other.grad[0, 1, 0]), 0.)
        self.assertFalse(old.requires_grad)


if __name__ == '__main__':
    unittest.main()
