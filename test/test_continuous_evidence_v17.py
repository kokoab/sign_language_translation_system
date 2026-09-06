import unittest

import numpy as np
import torch

from active.v17.continuous_evidence_v17 import (
    ContinuousConfig, ContinuousEvidenceModel, annotated_gaps, interval_targets,
    observation_ends, observation_windows,
)


class ContinuousEvidenceTest(unittest.TestCase):
    def test_true_gaps_exclude_every_lexical_interval_including_other(self):
        intervals = [(4, 12, 1), (10, 16, 101), (22, 26, 2)]
        self.assertEqual(list(annotated_gaps(32, intervals)), [(0, 4), (16, 22), (26, 32)])
        targets = interval_targets([4, 8, 12, 16, 20, 24, 28, 32], intervals, 4)
        self.assertEqual(targets.tolist(), [0, 1, -100, 101, 0, -100, -100, 0])

    def test_observation_never_reads_future_and_finish_flushes_only_the_tail(self):
        frames = np.zeros((25, 61, 5), np.float32)
        frames[..., 3:] = 1
        frames[..., 0] = np.arange(25)[:, None]
        changed = frames.copy()
        changed[12:, :, 0] = 999
        np.testing.assert_array_equal(
            observation_windows(frames, 12, (4, 8, 16)),
            observation_windows(changed, 12, (4, 8, 16)),
        )
        self.assertEqual(observation_ends(25, 4, final=False), [4, 8, 12, 16, 20, 24])
        self.assertEqual(observation_ends(25, 4), [4, 8, 12, 16, 20, 24, 25])

    def test_model_prefix_is_causal_and_scales_share_gloss_contract(self):
        config = ContinuousConfig(windows=(4, 8), stage1_dim=8, num_glosses=3,
                                  hidden_dim=12, blocks=2, dropout=0)
        model = ContinuousEvidenceModel(config).eval()
        x = torch.randn(2, 12, config.input_dim)
        with torch.inference_mode():
            full = model(x)
            prefix = model(x[:, :7])
        torch.testing.assert_close(full[:, :7], prefix)
        expected = x.reshape(2, 12, 2, 11)[..., 8:].mean(-2)
        torch.testing.assert_close(full[..., 1:4], expected)
        self.assertEqual(tuple(full.shape), (2, 12, 5))


if __name__ == "__main__":
    unittest.main()
