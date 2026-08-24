from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from active.v17.model_stage2_v17 import FROZEN_TEMPORAL_FEATURE_DIM
from active.v17.train_stage_2_2m_flores_v17 import collate_long, metrics_key
from active.v17.pretrain_stage2_temporal_v17 import (
    apply_token_mask,
    contiguous_token_mask,
    stable_fold,
    temporal_pretraining_loss,
)
from active.v17.model_stage2_v17 import Stage2TemporalHeadV17, Stage2V17Config
from active.v17.train_stage_2_general_selector_distill_v17 import (
    apply_index_weight_boost,
    gradients_are_finite,
    load_student_initialization,
    source_masses_for_synthetic_ratio,
)
from active.v17.train_stage_2_v17 import Sample


class TrainTwoMFloresStage2Tests(unittest.TestCase):
    def test_long_collate_preserves_forty_windows(self):
        sample = Sample(
            features=np.zeros((40, 32, FROZEN_TEMPORAL_FEATURE_DIM), np.float16),
            targets=np.asarray([1, 2, 3], np.int64),
            source="two_m_flores_asl",
            item_id="row",
            target_sequence=("A", "B", "C"),
        )
        batch = collate_long([sample], maximum_windows=40)
        self.assertEqual(tuple(batch["features"].shape), (1, 40, 32, FROZEN_TEMPORAL_FEATURE_DIM))
        self.assertEqual(batch["target_lengths"].tolist(), [3])
        self.assertEqual(batch["locked_partial_targets"].tolist(), [1, 2, 3])
        self.assertEqual(batch["locked_partial_target_lengths"].tolist(), [3])

    def test_selection_prioritizes_contextual_wer(self):
        phrase = {
            "local_phrases": {"wer": 0.05},
            "asllrp_contiguous": {"wer": 0.50},
        }
        contextual = {"asllrp_segmented_validation": {"wer": 0.19}}
        self.assertEqual(metrics_key(phrase, contextual), (-0.19, -0.50, -0.05))

    def test_temporal_mask_is_contiguous_bounded_and_zeroes_only_selected_frames(self):
        windows = torch.tensor([[True, True], [True, False]])
        mask = contiguous_token_mask(
            windows, tokens_per_window=8, ratio=0.25, span_tokens=3,
            generator=torch.Generator().manual_seed(17),
        )
        self.assertEqual(tuple(mask.shape), (2, 16))
        self.assertGreater(int(mask[0].sum()), 0)
        self.assertLess(int(mask[0].sum()), 16)
        self.assertEqual(int(mask[1, 8:].sum()), 0)
        features = torch.ones(2, 2, 32, 5)
        masked = apply_token_mask(features, mask, tokens_per_window=8)
        frame_mask = mask.repeat_interleave(4, dim=1)
        flat = masked.flatten(1, 2)
        self.assertEqual(int(torch.count_nonzero(flat[frame_mask])), 0)
        self.assertGreater(int(torch.count_nonzero(flat[~frame_mask])), 0)

    def test_temporal_objective_is_finite(self):
        teacher = torch.randn(2, 8, 16)
        student = teacher + 0.1 * torch.randn_like(teacher)
        predicted = student + 0.1 * torch.randn_like(student)
        valid = torch.ones(2, 8, dtype=torch.bool)
        masked = torch.zeros_like(valid)
        masked[:, 2:4] = True
        loss, pieces = temporal_pretraining_loss(
            student, predicted, teacher, masked, valid,
            temperature=0.1, contrastive_weight=0.1, visible_weight=0.25,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(pieces), {
            "reconstruction", "cosine", "contrastive", "visible_preservation"
        })

    def test_synthetic_ratio_preserves_declared_mass(self):
        masses = source_masses_for_synthetic_ratio(0.2)
        self.assertAlmostEqual(sum(masses.values()), 1.0)
        self.assertAlmostEqual(
            masses["synthetic_citizen_train"]
            + masses["synthetic_multivoice_train"]
            + masses["synthetic_balanced_multivoice_train"],
            0.2,
        )
        self.assertEqual(stable_fold("same", 5), stable_fold("same", 5))

    def test_temporal_initialization_is_strict_and_keeps_ctc_frozen(self):
        model = Stage2TemporalHeadV17(Stage2V17Config())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrain.pth"
            torch.save({
                "format": "slt_stage2_temporal_pretrain_v17",
                "model_config": model.config.to_dict(),
                "model_state_dict": model.state_dict(),
                "ctc_head_trained": False,
                "source_split": "2m_flores_dev_train_only",
            }, path)
            loaded, payload = load_student_initialization(path, model)
        self.assertEqual(loaded.config.to_dict(), model.config.to_dict())
        self.assertEqual(payload["source_split"], "2m_flores_dev_train_only")

    def test_nonfinite_gradient_gate_fails_closed(self):
        clean = torch.nn.Parameter(torch.tensor([1.0]))
        corrupt = torch.nn.Parameter(torch.tensor([2.0]))
        clean.grad = torch.tensor([0.5])
        corrupt.grad = torch.tensor([float("nan")])
        self.assertFalse(gradients_are_finite([clean, corrupt]))
        corrupt.grad = None
        self.assertTrue(gradients_are_finite([clean, corrupt]))

    def test_selector_sampling_boost_changes_only_declared_rows(self):
        weights = torch.ones(5, dtype=torch.double)
        boosted = apply_index_weight_boost(weights, [1, 3], 8.0)
        self.assertEqual(boosted.tolist(), [1.0, 8.0, 1.0, 8.0, 1.0])
        self.assertEqual(weights.tolist(), [1.0] * 5)


if __name__ == "__main__":
    unittest.main()
