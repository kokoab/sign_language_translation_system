import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from active.v17.train_stage1_window_v17 import (
    NO_EMIT_INDEX,
    audit_existing_supervision,
    _validate_seed,
    load_context_samples,
    sample_weights,
    window_objective,
)


class Stage1WindowTrainingTests(unittest.TestCase):
    def test_training_refuses_unfinished_baseline_freeze(self):
        from active.v17 import train_stage1_window_v17 as trainer
        self.assertTrue(hasattr(trainer, 'verify_development_freeze'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, supervision, frozen = root/'base', root/'supervision', root/'freeze.json'
            base.write_text('base'); supervision.write_text('supervision')
            frozen.write_text(json.dumps({'status': 'pending'}))
            with self.assertRaisesRegex(ValueError, 'baselines'):
                trainer.verify_development_freeze(frozen, base, supervision)

    def test_epoch_sampler_has_exact_bounded_category_counts(self):
        from active.v17 import train_stage1_window_v17 as trainer
        self.assertTrue(hasattr(trainer, 'epoch_indices'))
        rows = [('replay', 'citizen', 0), ('replay', 'semlex', 1),
                ('context', 'asllrp', 2), ('background', 'asllrp', 100)]
        chosen = trainer.epoch_indices(rows, torch.Generator().manual_seed(17111))
        self.assertEqual(len(chosen), 3000)
        self.assertEqual(sum(rows[i][0] == 'replay' for i in chosen), 1500)
        self.assertEqual(sum(rows[i][0] == 'context' for i in chosen), 900)
        self.assertEqual(sum(rows[i][0] == 'background' for i in chosen), 600)

    def test_confirmation_seed_requires_first_seed_eligibility(self):
        _validate_seed(17111, None)
        with self.assertRaisesRegex(ValueError, "seed 17112"):
            _validate_seed(17112, None)

    def test_existing_supervision_audit_requires_disjoint_checked_roles(self):
        supervision = {
            "format": "slt_stage2_live_transition_supervision_v17", "version": 1,
            "items": {
                "train": {"role": "train", "annotation_status": "available",
                          "sign_intervals": [{"start_seconds": 0.0, "end_seconds": 0.2, "label": "A"},
                                             {"start_seconds": 1.2, "end_seconds": 1.4, "label": "__OTHER__"}]},
                "val": {"role": "validation", "annotation_status": "available",
                        "sign_intervals": [{"start_seconds": 0.0, "end_seconds": 0.2, "label": "B"},
                                           {"start_seconds": 1.2, "end_seconds": 1.4, "label": "A"}]},
            },
        }
        frozen = {"rows": [
            {"source_item_id": "train", "signer_id": "S1"},
            {"source_item_id": "val", "signer_id": "S2"},
        ]}
        audit = audit_existing_supervision(supervision, frozen)
        self.assertTrue(audit["train_validation_signer_disjoint"])
        self.assertEqual(3, audit["train"]["potential_full_0_53_second_background_windows"])
        self.assertEqual(3, audit["validation"]["potential_full_0_53_second_background_windows"])

    def _manifest(self, root: Path, *, complete=True, gap=True) -> Path:
        timestamps = np.arange(0.0, 2.01, 0.05)
        features = np.zeros((len(timestamps), 61, 5), dtype=np.float32)
        features[:, :, 0] = timestamps[:, None]
        features[:, 0, 3:] = 1.0
        archive = root / "sequence.npz"
        np.savez_compressed(
            archive, raw_features=features, timestamps_seconds=timestamps,
            raw_format=np.array("apple_vision_isotropic_xy_confidence_v1"),
        )
        second_start = 1.40 if gap else 0.70
        payload = {
            "format": "slt_stage1_window_supervision_v17",
            "version": 1,
            "rows": [{
                "role": "train",
                "source": "fixture",
                "signer_id": "S1",
                "source_item_id": "item-1",
                "archive_path": str(archive),
                "all_signs_annotated": complete,
                "intervals": [
                    {"start_seconds": 0.30, "end_seconds": 0.50, "label": "A"},
                    {"start_seconds": second_start, "end_seconds": 1.60,
                     "label": "B"},
                ],
            }],
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_preparation_uses_verified_intervals_and_exact_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            samples, audit = load_context_samples(
                self._manifest(Path(directory)), {"A": 0, "B": 1}, "train"
            )

        positives = [row for row in samples if row.category == "context"]
        backgrounds = [row for row in samples if row.category == "background"]
        self.assertTrue(backgrounds)
        self.assertTrue(any(row.schedule == "trailing-0.53/0.13" for row in positives))
        self.assertEqual({0, 1}, {row.target for row in positives})
        self.assertTrue(all(row.foreground.any() for row in positives))
        self.assertTrue(all(not row.foreground.any() for row in backgrounds))
        self.assertEqual(2, audit["distinct_signs"])
        self.assertEqual(1, audit["distinct_sign_pairs"])
        self.assertEqual(1, audit["distinct_signers"])

    def test_preparation_fails_closed_without_complete_background_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory), complete=False)
            with self.assertRaisesRegex(ValueError, "verified background"):
                load_context_samples(manifest, {"A": 0, "B": 1}, "train")

    def test_incomplete_rows_supply_positives_but_never_background(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            payload = json.loads(manifest.read_text())
            payload["rows"][0]["source"] = "complete"
            positive_only = dict(payload["rows"][0])
            positive_only.update(source="positive-only", source_item_id="item-2", all_signs_annotated=False)
            payload["rows"].append(positive_only)
            manifest.write_text(json.dumps(payload))
            samples, audit = load_context_samples(manifest, {"A": 0, "B": 1}, "train")
        self.assertEqual({"complete", "positive-only"}, {row.source for row in samples if row.category == "context"})
        self.assertEqual({"complete"}, {row.source for row in samples if row.category == "background"})
        self.assertEqual(1, audit["positive_only_rows"])

    def test_overlapping_all_sign_annotations_exclude_only_ambiguous_target(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory))
            payload = json.loads(manifest.read_text())
            payload["rows"][0]["intervals"].append(
                {"start_seconds": 0.38, "end_seconds": 0.42, "label": "__OTHER__"}
            )
            manifest.write_text(json.dumps(payload))
            samples, audit = load_context_samples(manifest, {"A": 0, "B": 1}, "train")
        self.assertNotIn(0, {row.target for row in samples if row.category == "context"})
        self.assertIn("ambiguous_center_annotation", audit["rejected_windows"])

    def test_preparation_fails_when_required_category_is_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._manifest(Path(directory), gap=False)
            with self.assertRaisesRegex(ValueError, "verified background"):
                load_context_samples(manifest, {"A": 0, "B": 1}, "train")

    def test_recipe_weights_and_losses_apply_to_the_intended_rows(self):
        rows = [
            ("replay", "citizen", 0),
            ("replay", "semlex", 0),
            ("context", "asllrp", 1),
            ("background", "asllrp", NO_EMIT_INDEX),
        ]
        weights = sample_weights(rows)
        self.assertAlmostEqual(0.50, float(weights[:2].sum() / weights.sum()))
        self.assertAlmostEqual(0.30, float(weights[2] / weights.sum()))
        self.assertAlmostEqual(0.20, float(weights[3] / weights.sum()))

        window = torch.zeros((3, 101), requires_grad=True)
        foreground = torch.zeros((3, 100), requires_grad=True)
        teacher = torch.zeros((3, 100))
        losses = window_objective(
            window, foreground, teacher,
            torch.tensor([0, 1, NO_EMIT_INDEX]),
            ("replay", "context", "background"),
        )
        self.assertAlmostEqual(
            float(losses["total"].detach()),
            float((losses["window"] + 0.5 * losses["foreground"] + losses["replay"]).detach()),
        )
        losses["total"].backward()
        self.assertGreater(float(window.grad.abs().sum()), 0.0)
        self.assertGreater(float(foreground.grad[1].abs().sum()), 0.0)
        self.assertEqual(0.0, float(foreground.grad[[0, 2]].abs().sum()))

        no_optional_rows = window_objective(
            torch.zeros((1, 101), requires_grad=True),
            torch.zeros((1, 100), requires_grad=True),
            torch.zeros((1, 100)), torch.tensor([NO_EMIT_INDEX]),
            ("background",),
        )
        self.assertTrue(torch.isfinite(no_optional_rows["total"]))


if __name__ == "__main__":
    unittest.main()
