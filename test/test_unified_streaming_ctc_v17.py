import unittest

import numpy as np
import torch

from active.v17.model_unified_streaming_ctc_v17 import (
    UnifiedStreamingCTCConfig,
    UnifiedStreamingCTCHeadV17,
)
from active.v17.train_unified_streaming_ctc_v17 import rolling_windows


class UnifiedStreamingCTCTest(unittest.TestCase):
    def test_short_rolling_window_preserves_more_observation_steps(self):
        frames = np.zeros((20, 2, 5), dtype=np.float32)
        frames[..., 3] = 1.0
        short = rolling_windows(frames, stride=4, window_frames=8)
        long = rolling_windows(frames, stride=4, window_frames=32)
        self.assertEqual(short.shape, (4, 32, 2, 5))
        self.assertEqual(long.shape, (3, 32, 2, 5))

    def test_shape_and_output_contract(self):
        config = UnifiedStreamingCTCConfig(
            stage1_dim=16, num_glosses=7, hidden_dim=12, blocks=2, dropout=0.0
        )
        output = UnifiedStreamingCTCHeadV17(config)(
            torch.randn(3, 11, config.input_dim)
        )
        self.assertEqual(tuple(output.shape), (3, 11, 9))
        self.assertEqual(config.other_index, 8)

    def test_future_does_not_change_prefix(self):
        config = UnifiedStreamingCTCConfig(
            stage1_dim=16, num_glosses=7, hidden_dim=12, blocks=2, dropout=0.0
        )
        model = UnifiedStreamingCTCHeadV17(config).eval()
        first = torch.randn(1, 13, config.input_dim)
        second = first.clone()
        second[:, 8:] = torch.randn_like(second[:, 8:])
        with torch.inference_mode():
            first_output, second_output = model(first), model(second)
        torch.testing.assert_close(first_output[:, :8], second_output[:, :8])

    def test_initial_gloss_path_preserves_stage1_logits(self):
        config = UnifiedStreamingCTCConfig(
            stage1_dim=16, num_glosses=7, hidden_dim=12, blocks=1, dropout=0.0
        )
        model = UnifiedStreamingCTCHeadV17(config).eval()
        evidence = torch.randn(2, 5, config.input_dim)
        with torch.inference_mode():
            output = model(evidence)
        torch.testing.assert_close(output[..., 1:-1], evidence[..., 16:])


if __name__ == "__main__":
    unittest.main()
