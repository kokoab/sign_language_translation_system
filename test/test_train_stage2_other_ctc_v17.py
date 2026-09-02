from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from active.v17.train_stage_2_other_ctc_v17 import (
    collapse_ctc,
    distillation_loss,
    resolve_initialization,
    selection_key,
)
from active.v17.model_stage2_v17 import Stage2TemporalHeadV17, Stage2V17Config
from active.v17.train_stage_2_v17 import sha256


class TrainStage2OtherCTCV17Tests(unittest.TestCase):
    def test_collapse_keeps_other_as_class_100(self):
        sequence = np.asarray([0, 1, 1, 0, 101, 101, 0, 15], dtype=np.int64)
        self.assertEqual(collapse_ctc(sequence), [0, 100, 14])

    def test_distillation_ignores_natural_rows_and_padding(self):
        teacher = torch.randn(2, 4, 101)
        student = torch.cat((teacher.clone(), torch.randn(2, 4, 1)), dim=-1)
        loss = distillation_loss(
            student, teacher, torch.tensor([2, 4]),
            torch.tensor([True, False]), temperature=2.0,
        )
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        student[0, 0, 1] += 2.0
        changed = distillation_loss(
            student, teacher, torch.tensor([2, 4]),
            torch.tensor([True, False]), temperature=2.0,
        )
        self.assertGreater(float(changed), 0.0)

    def test_selection_requires_no_regression_on_legacy_gates(self):
        def domain(edits, tokens):
            return {"edits": edits, "tokens": tokens, "wer": edits / tokens}

        safe = {
            "target_only": {
                "local_phrases": domain(7, 259),
                "asllrp_contiguous": domain(11, 24),
                "asllrp_other_ctc": domain(100, 284),
            },
            "full": {"asllrp_other_ctc": domain(200, 682)},
        }
        regressed = {
            "target_only": {
                "local_phrases": domain(8, 259),
                "asllrp_contiguous": domain(10, 24),
                "asllrp_other_ctc": domain(1, 284),
            },
            "full": {"asllrp_other_ctc": domain(1, 682)},
        }
        self.assertGreater(selection_key(safe), selection_key(regressed))

    def test_temporal_initialization_keeps_original_teacher(self):
        model = Stage2TemporalHeadV17(Stage2V17Config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.pth"
            temporal = root / "temporal.pth"
            torch.save({
                "format": "slt_stage2_ctc_v17",
                "model_config": model.config.to_dict(),
                "model_state_dict": model.state_dict(),
            }, base)
            changed = {
                key: value.clone() for key, value in model.state_dict().items()
            }
            changed["input_projection.0.bias"] += 1.0
            torch.save({
                "format": "slt_stage2_temporal_pretrain_v17",
                "model_config": model.config.to_dict(),
                "model_state_dict": changed,
                "ctc_head_trained": False,
                "base_checkpoint": base.as_posix(),
                "base_checkpoint_sha256": sha256(base),
                "source_split": "2m_flores_dev_train_only",
            }, temporal)
            initialization, teacher, provenance = resolve_initialization(
                temporal, temporal_mix=0.25
            )
        self.assertEqual(initialization["format"], "slt_stage2_ctc_v17")
        self.assertFalse(torch.equal(
            initialization["model_state_dict"]["input_projection.0.bias"],
            teacher["model_state_dict"]["input_projection.0.bias"],
        ))
        self.assertTrue(torch.allclose(
            initialization["model_state_dict"]["input_projection.0.bias"],
            teacher["model_state_dict"]["input_projection.0.bias"] + 0.25,
        ))
        self.assertEqual(provenance["source_split"], "2m_flores_dev_train_only")
        self.assertEqual(provenance["temporal_mix"], 0.25)


if __name__ == "__main__":
    unittest.main()
