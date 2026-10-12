import unittest

import torch

from active.v17.model_stage1_family_v17 import (
    FLAT_TRANSFORMER,
    FlatTransformerStage1Config,
    FlatTransformerStage1V17,
    build_stage1_landmark_model,
)
from active.v17.model_unified_multimodal_v17 import UnifiedFusionHeadV17
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from scripts.benchmark_stage1_families_v17 import TransformerBaseline


def features(batch: int, *, empty_rows: tuple[int, ...] = ()) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    value = torch.randn(batch, 32, 61, 5, generator=generator)
    value[..., 3] = (torch.rand(batch, 32, 61, generator=generator) > 0.3).float()
    for row in empty_rows:
        value[row, :, :42, 3] = 0.0
    return value


class FamilyTransformerAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.reference = TransformerBaseline(100).eval()
        self.adapter = FlatTransformerStage1V17(FlatTransformerStage1Config()).eval()
        self.adapter.load_state_dict(self.reference.state_dict(), strict=True)

    def test_state_dict_keys_match_reference(self):
        self.assertEqual(list(self.adapter.state_dict()), list(self.reference.state_dict()))

    def test_logits_identical_including_empty_activity(self):
        value = features(6, empty_rows=(2,))
        # nn.TransformerEncoder takes its fused fast path only without autograd, so each
        # grad mode is compared with the reference in that same mode.
        with torch.inference_mode():
            expected = self.reference(value)
            logits, pooled = self.adapter(value, return_embeddings=True)
        self.assertTrue(torch.equal(logits, expected))
        self.assertTrue(torch.equal(self.adapter(value).detach(), self.reference(value).detach()))
        self.assertEqual(tuple(pooled.shape), (6, 256))
        self.assertTrue(torch.equal(pooled[2], torch.zeros(256)))

    def test_embedding_fits_unified_fusion_head(self):
        value = features(3)
        with torch.inference_mode():
            logits, pooled = self.adapter(value, return_embeddings=True)
            fused = UnifiedFusionHeadV17()(pooled, torch.randn(3, 256), logits, torch.randn(3, 100))
        self.assertEqual(tuple(fused.shape), (3, 100))

    def test_factory_dispatch_and_legacy_rejection(self):
        config = FlatTransformerStage1Config().to_dict()
        self.assertEqual(config["architecture"], FLAT_TRANSFORMER)
        self.assertIsInstance(build_stage1_landmark_model(config), FlatTransformerStage1V17)
        squeeze = Stage1V17Config(num_classes=100, temporal_encoder="partwise_global", part_depth=1)
        self.assertIsInstance(build_stage1_landmark_model(squeeze.to_dict()), SLTStage1V17)
        with self.assertRaises(TypeError):
            Stage1V17Config(**config)


if __name__ == "__main__":
    unittest.main()
