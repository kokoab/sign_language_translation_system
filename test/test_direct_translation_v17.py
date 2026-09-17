"""Small real-network checks; no downloads or dataset access."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

from active.v17.direct_translation_v17 import DirectTranslation, epoch_batches


class SmallEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(dim=8)
        self.projection = nn.Linear(5, 8)
        self.classifier = nn.Linear(8, 3)

    def encode(self, x):
        values = self.projection(x.mean(2))
        return values, torch.ones(values.shape[:2], dtype=torch.bool)

    def forward(self, x):
        return self.classifier(self.encode(x)[0].mean(1))


class DirectTranslationChecks(unittest.TestCase):
    def test_sequence_masks_gradients_and_target_free_generation(self):
        torch.manual_seed(4)
        text = T5ForConditionalGeneration(T5Config(vocab_size=24, d_model=16,
            d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2,
            d_kv=8, dropout_rate=0, decoder_start_token_id=0, eos_token_id=1))
        model = DirectTranslation(SmallEncoder(), text, torch.tensor([[2, 3]]))
        windows = [torch.randn(2,32,61,5), torch.randn(1,32,61,5)]
        valid = [torch.tensor([True,False]), torch.tensor([True])]
        inputs = model.visual_inputs(windows, valid)
        self.assertEqual(inputs['inputs_embeds'].shape, (2,258,16))
        self.assertEqual(inputs['attention_mask'].sum(1).tolist(), [34,34])
        self.assertTrue(torch.equal(inputs['attention_mask'][:,34:], torch.zeros(2,224,dtype=torch.long)))
        labels = torch.tensor([[4,5,1],[6,1,-100]])
        loss = model.translation_loss(windows, valid, labels)
        unpadded_loss=model.text(**{k:v[:,:66] for k,v in inputs.items()},labels=labels).loss
        self.assertTrue(torch.allclose(loss,unpadded_loss,atol=1e-6), 'padding must not change the training target/objective')
        loss.backward()
        for module in [model.base.projection, model.projection, model.text.decoder]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None),0)
        logits = model.base(windows[0]);nn.functional.cross_entropy(logits, torch.tensor([0,1])).backward()
        self.assertGreater(model.base.classifier.weight.grad.abs().sum().item(),0)
        model.eval()
        with torch.no_grad():
            predicted=model.translate(windows,valid,max_new_tokens=3,num_beams=1)
        self.assertEqual(predicted.shape[0],2)
        with self.assertRaises(ValueError):
            model.visual_inputs(windows, [torch.zeros(2,dtype=torch.bool),valid[1]])

    def test_full_coverage_and_no_duplicate_replay(self):
        batches = epoch_batches(9, 31, seed=3)
        self.assertEqual(sorted(i for pair in batches for i in pair[0]),list(range(9)))
        self.assertEqual(sorted(i for pair in batches for i in pair[1]),list(range(31)))
        self.assertEqual(batches,epoch_batches(9,31,seed=3))
        self.assertNotEqual(batches,epoch_batches(9,31,seed=4))


if __name__ == '__main__':
    unittest.main()
