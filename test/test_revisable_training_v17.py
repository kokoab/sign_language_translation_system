import unittest
import torch
from active.v17.train_stage2_live_adapt_v17 import transition_blank_loss, prefix_ctc_loss, known_core_loss


class TransitionTrainingTests(unittest.TestCase):
    def test_revision_metric_counts_replacements_and_deletions_not_appends(self):
        from scripts.evaluate_revisable_transcription_v17 import revision_size
        self.assertEqual(revision_size([1], [1, 2]), 0)
        self.assertEqual(revision_size([1, 2], [1, 3]), 1)
        self.assertEqual(revision_size([1, 2], []), 2)

    def test_positive_core_loss_opposes_silent_or_other_predictions(self):
        logits = torch.zeros(1, 8, 102, requires_grad=True)
        batch = {'item_ids': ['a'], 'sources': ['asllrp_contiguous']}
        supervision = {'a': {'role': 'train', 'known_core_positions': {'2': 7}}}
        loss = known_core_loss(logits, batch, supervision)
        loss.backward()
        self.assertLess(logits.grad[0, 2, 7], 0)
        self.assertGreater(logits.grad[0, 2, 0], 0)
        self.assertGreater(logits.grad[0, 2, 101], 0)
        self.assertFalse(logits.grad[0, :2].any())

    def test_prefix_training_hides_future_features_and_uses_partial_target(self):
        batch = {'item_ids': ['a'], 'sources': ['asllrp_contiguous'],
                 'features': torch.ones(1, 3, 32, 2), 'window_mask': torch.ones(1, 3, dtype=torch.bool)}
        supervision = {'a': {'role': 'train', 'prefix_targets': {'1': [1]}}}
        def model(features, masks):
            self.assertFalse(features[:, 1:].any())
            self.assertEqual(masks.tolist(), [[True, False, False]])
            return torch.zeros(1, 24, 3, requires_grad=True), torch.tensor([8])
        def criterion(logits, targets, lengths, target_lengths):
            self.assertEqual(targets.tolist(), [1])
            self.assertEqual(target_lengths.tolist(), [1])
            return logits.sum() * 0 + 1
        self.assertEqual(prefix_ctc_loss(model, batch, supervision, criterion, 0), 1)
        supervision['a']['role'] = 'validation'
        with self.assertRaises(ValueError):
            prefix_ctc_loss(model, batch, supervision, criterion, 0)

    def test_only_annotated_training_gap_positions_receive_blank_gradient(self):
        logits = torch.zeros(2, 8, 102, requires_grad=True)
        batch = {'item_ids': ['a', 'b'], 'sources': ['asllrp_contiguous', 'replay:asllrp_contiguous']}
        supervision = {'a': {'role': 'train', 'blank_positions': [2, 3]},
                       'b': {'role': 'validation', 'blank_positions': [0]}}
        loss = transition_blank_loss(logits, batch, supervision)
        loss.backward()
        self.assertLess(logits.grad[0, 2, 0], 0)
        self.assertEqual(logits.grad[0, 0].abs().sum(), 0)
        self.assertEqual(logits.grad[1].abs().sum(), 0)
        self.assertGreater(logits.grad[0, 2, 101], 0)
        batch['sources'][1] = 'asllrp_contiguous'
        with self.assertRaises(ValueError):
            transition_blank_loss(logits, batch, supervision)


if __name__ == '__main__':
    unittest.main()
