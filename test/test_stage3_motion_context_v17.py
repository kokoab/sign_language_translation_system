import unittest

import torch


class MotionContextTest(unittest.TestCase):
    def test_runtime_proposal_evidence_tracks_final_tail_once_and_resets(self):
        from pathlib import Path
        import numpy as np
        from active.v17.continuous_runtime_v17 import ContinuousRecognizer
        checkpoint = Path('artifacts/models/continuous_evidence_v17_causal_v2/best_model.pth')
        context = Path('artifacts/models/stage3_motion_context_v17_causal_v2/model.pth')
        if not checkpoint.exists() or not context.exists():
            self.skipTest('optional research checkpoints unavailable')
        runtime = ContinuousRecognizer(checkpoint, context_checkpoint=context, context_proposals=True)
        for _ in range(5):runtime.add(np.zeros((61,5),np.float32))
        runtime.finish()
        self.assertEqual(len(runtime.context_log_probs),2)
        self.assertEqual(len(runtime.context_evidence),2)
        runtime.finish()
        self.assertEqual(len(runtime.context_log_probs),2)
        runtime.reset()
        self.assertEqual(runtime.context_log_probs,[])
        self.assertEqual(runtime.context_evidence,[])

    def test_joint_correction_can_recover_a_missing_beam_candidate(self):
        from active.v17.stage3_motion_context_v17 import rerank_motion_candidates
        class Scorer:
            num_glosses = 4
            def propose_candidates(self, evidence):
                return [(2,), (2, 2, 2)]
            def score_candidates(self, evidence, candidates):
                return torch.tensor([-10. if c == (1,) else 0. for c in candidates])
        probabilities = torch.tensor([[.1, .6, .3, 0., 0., 0.], [.8, .1, .1, 0., 0., 0.]])
        result = rerank_motion_candidates(Scorer(), torch.zeros(2, 12), [((1,), -1.)], 1.,
                                          log_probabilities=probabilities.log())
        self.assertEqual(result[0][0], (2,))
        self.assertEqual(result[-1][1], -float('inf'))

    def test_exact_ctc_scores_repeated_tokens_and_rejects_impossible_paths(self):
        from active.v17.stage3_motion_context_v17 import ctc_candidate_scores
        probabilities = torch.tensor([[.1, .9], [.8, .2], [.1, .9]])
        score = ctc_candidate_scores(probabilities.log(), [(), (1,), (1, 1), (1, 1, 1)])
        self.assertAlmostEqual(float(score[2].exp()), .9 * .8 * .9, places=6)
        self.assertAlmostEqual(float(score[0].exp()), .1 * .8 * .1, places=6)
        self.assertTrue(torch.isneginf(score[3]))

    def test_motion_control_uses_different_labels_within_same_source(self):
        from active.v17.stage3_motion_context_v17 import mismatched_motion_indices
        rows = [dict(source=source, targets=target, evidence=torch.zeros(length, 12))
                for source in ("local", "isolated")
                for target, length in (((1,), 5), ((1,), 7), ((2,), 6), ((3,), 8))]
        indices = mismatched_motion_indices(rows)
        for row, index in zip(rows, indices):
            self.assertEqual(row["source"], rows[index]["source"])
            self.assertNotEqual(row["targets"], rows[index]["targets"])

    def test_motion_decoder_can_propose_beyond_recognition_beam(self):
        model = self.model()
        with torch.no_grad():
            model.output[-1].weight.zero_()
            model.output[-1].bias.fill_(-20)
            model.output[-1].bias[3] = 0
            model.output[-1].bias[model.eos_index] = -1
        candidates = model.propose_candidates(torch.zeros(4, 12), beam_width=2, max_tokens=2)
        self.assertTrue(any(3 in sequence for sequence in candidates))
        self.assertTrue(all(len(sequence) <= 2 for sequence in candidates))
        self.assertTrue(all(0 not in sequence and model.eos_index not in sequence for sequence in candidates))

    def test_reranking_preserves_empty_and_unknown_hypotheses(self):
        from active.v17.stage3_motion_context_v17 import rerank_motion_candidates
        class Scorer:
            num_glosses = 4
            def score_candidates(self, evidence, candidates):
                return torch.tensor([-20., 0.])
        evidence = torch.zeros(3, 12)
        for beam in ([((), -1.), ((1,), -2.)], [((5,), -1.), ((1,), -2.)]):
            self.assertEqual(rerank_motion_candidates(Scorer(), evidence, beam, 1.), beam)
            self.assertEqual(rerank_motion_candidates(Scorer(), evidence, beam, 1.,
                log_probabilities=torch.zeros(3,6).log_softmax(-1)), beam)
        beam = [((1,), -1.), ((2,), -2.)]
        self.assertEqual(rerank_motion_candidates(Scorer(), evidence, beam, 1.)[0][0], (2,))

    def model(self):
        from active.v17 import stage3_motion_context_v17 as context
        return context.MotionContextScorer(evidence_dim=12, hidden_dim=16, num_glosses=4, dropout=0).eval()

    def test_padding_cannot_change_scores_but_valid_motion_can(self):
        model = self.model()
        evidence = torch.randn(2, 7, 12)
        lengths = torch.tensor([3, 7])
        tokens = torch.tensor([[0, 1, 2], [0, 2, 3]])
        padded = evidence.clone(); padded[0, 3:] = 999
        with torch.no_grad():
            first = model(evidence, lengths, tokens)
            second = model(padded, lengths, tokens)
            changed = model(-evidence, lengths, tokens)
        torch.testing.assert_close(first, second)
        self.assertGreater(float((first - changed).abs().max()), 1e-5)

    def test_candidate_future_cannot_change_earlier_word_probability(self):
        model = self.model()
        evidence = torch.randn(1, 8, 12)
        tokens = torch.tensor([[0, 1, 2, 3]])
        changed = tokens.clone(); changed[:, 2:] = 4
        with torch.no_grad():
            first = model(evidence, torch.tensor([8]), tokens)
            second = model(evidence, torch.tensor([8]), changed)
        torch.testing.assert_close(first[:, :2], second[:, :2])


if __name__ == "__main__":
    unittest.main()
