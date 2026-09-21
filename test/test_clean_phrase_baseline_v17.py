import unittest
from scripts.train_clean_phrase_baseline_v17 import summarize

class CleanBaselineTests(unittest.TestCase):
    def test_blank_deletions_other_and_source_balancing(self):
        rows=[dict(source='local_phrases',expected=[1,2],predicted=[]),
              dict(source='local_phrases',expected=[1],predicted=[1,101]),
              dict(source='asllrp_contiguous',expected=[3],predicted=[3])]
        m=summarize(rows)
        self.assertEqual(m['overall']['blank_only'],1)
        self.assertEqual(m['overall']['deletions'],2)
        self.assertEqual(m['overall']['exact'],1)
        self.assertEqual(m['overall']['other_emitted'],1)
        self.assertAlmostEqual(m['selection_score'],1/3)
        self.assertAlmostEqual(m['overall']['known_wer'],.5)

    def test_mps_head_receives_cpu_ctc_gradients(self):
        import torch
        from active.v17.model_unified_streaming_ctc_v17 import UnifiedStreamingCTCHeadV17, UnifiedStreamingCTCConfig
        if not torch.backends.mps.is_available(): self.skipTest('MPS unavailable')
        model=UnifiedStreamingCTCHeadV17(UnifiedStreamingCTCConfig(stage1_dim=8,num_glosses=3,hidden_dim=16,blocks=1)).to('mps')
        logits=model(torch.randn(2,5,11,device='mps'))
        loss=torch.nn.CTCLoss()(logits.cpu().log_softmax(-1).transpose(0,1),torch.tensor([1,2,1,2]),torch.tensor([5,5]),torch.tensor([2,2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.parameters()))
