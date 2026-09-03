import unittest

import torch

from active.v17.model_streaming_tcn_ctc_v17 import (
    StreamingLandmarkTCNCTCV17,
    StreamingTCNConfig,
)
from active.v17.model_streaming_stage1_head_v17 import (
    StreamingStage1CTCHeadV17,
    StreamingStage1HeadConfig,
)


class StreamingTCNV17Test(unittest.TestCase):
    def test_shape_and_receptive_field(self):
        config = StreamingTCNConfig(hidden_dim=32, group_dim=8, blocks=4, dropout=0.0)
        model = StreamingLandmarkTCNCTCV17(config).eval()
        output = model(torch.randn(2, 17, 61, 5))
        self.assertEqual(tuple(output.shape), (2, 17, 101))
        self.assertEqual(config.receptive_field_frames, 31)

    def test_incremental_matches_batched_causal_forward(self):
        torch.manual_seed(17)
        model = StreamingLandmarkTCNCTCV17(
            StreamingTCNConfig(hidden_dim=32, group_dim=8, blocks=3, dropout=0.0)
        ).eval()
        features = torch.randn(1, 23, 61, 5)
        expected = model(features)
        state = None
        actual = []
        for frame in features.unbind(dim=1):
            logits, state = model.stream_step(frame, state)
            actual.append(logits)
        actual = torch.stack(actual, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_hands_only_ignores_face_and_body(self):
        torch.manual_seed(18)
        model = StreamingLandmarkTCNCTCV17(
            StreamingTCNConfig(
                hidden_dim=32, group_dim=8, blocks=2, dropout=0.0,
                use_face_body=False,
            )
        ).eval()
        first = torch.randn(1, 8, 61, 5)
        second = first.clone()
        second[:, :, 42:] = torch.randn_like(second[:, :, 42:]) * 100
        torch.testing.assert_close(model(first), model(second))

    def test_stage1_head_streaming_matches_batched(self):
        torch.manual_seed(19)
        model = StreamingStage1CTCHeadV17(
            StreamingStage1HeadConfig(hidden_dim=32, blocks=3, dropout=0.0)
        ).eval()
        evidence = torch.randn(1, 15, 100)
        expected = model(evidence)
        state = None
        actual = []
        for step in evidence.unbind(dim=1):
            logits, state = model.stream_step(step, state)
            actual.append(logits)
        torch.testing.assert_close(
            torch.stack(actual, dim=1), expected, rtol=1e-5, atol=1e-5
        )

    def test_stage1_head_starts_as_gloss_residual(self):
        model = StreamingStage1CTCHeadV17(
            StreamingStage1HeadConfig(hidden_dim=16, blocks=1, dropout=0.0)
        ).eval()
        evidence = torch.randn(2, 5, 100)
        torch.testing.assert_close(model(evidence)[..., 1:], evidence)


if __name__ == "__main__":
    unittest.main()
