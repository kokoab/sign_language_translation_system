import unittest

import torch

from active.v17.model_hand_mobileclip2_v17 import (
    HandMobileCLIP2Stage1Config,
    HandMobileCLIP2Stage1V17,
)
from active.v17.model_unified_multimodal_v17 import (
    UnifiedFusionHeadV17,
    UnifiedMultimodalStage1V17,
    UnifiedMultimodalV17Config,
    per_sample_zscore,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_unified_multimodal_student_v17 import selection_key


class UnifiedMultimodalV17Test(unittest.TestCase):
    def test_zero_residual_starts_as_fixed_75_25_fusion(self):
        head = UnifiedFusionHeadV17(UnifiedMultimodalV17Config(dropout=0.0)).eval()
        landmark_features = torch.randn(3, 256)
        hand_features = torch.randn(3, 256)
        landmark_logits = torch.randn(3, 100)
        hand_logits = torch.randn(3, 100)
        expected = 0.75 * per_sample_zscore(landmark_logits) + 0.25 * per_sample_zscore(hand_logits)
        with torch.inference_mode():
            actual = head(landmark_features, hand_features, landmark_logits, hand_logits)
        torch.testing.assert_close(actual, expected)

    def test_small_unified_model_has_four_runtime_inputs(self):
        landmark = SLTStage1V17(Stage1V17Config(dim=32, depth=1, heads=4))
        hand = HandMobileCLIP2Stage1V17(
            HandMobileCLIP2Stage1Config(dim=32, depth=1, heads=4)
        )
        head = UnifiedFusionHeadV17(
            UnifiedMultimodalV17Config(feature_dim=32, hidden_dim=64, dropout=0.0)
        )
        model = UnifiedMultimodalStage1V17(landmark, hand, head).eval()
        with torch.inference_mode():
            output = model(
                torch.zeros(1, 32, 61, 5),
                torch.zeros(1, 16, 3, 512),
                torch.zeros(1, 16, 3, dtype=torch.bool),
                torch.zeros(1, 16, 3, 4),
            )
        self.assertEqual(tuple(output.shape), (1, 100))

    def test_selection_is_equal_domain_mean_then_fixed_tiebreakers(self):
        value = selection_key({
            "citizen": {"top1": 96.0, "top1_correct": 363},
            "semlex": {"top1": 89.0, "top1_correct": 870},
            "local": {"top1": 97.0, "top1_correct": 2810},
        })
        self.assertAlmostEqual(value[0], 94.0)
        self.assertEqual(value[1:], (363, 870, 2810))


if __name__ == "__main__":
    unittest.main()
