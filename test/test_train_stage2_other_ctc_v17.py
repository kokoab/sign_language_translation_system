from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from active.v17.train_stage_2_other_ctc_v17 import (
    BASELINE_EXPECTATIONS,
    CombinedDataset,
    build_transition_student,
    candidate_key,
    collapse_ctc,
    configure_trainable,
    ctc_lengths_are_feasible,
    distillation_loss,
    eligibility,
    parser,
    replay_distillation_mask,
    resolve_initialization,
    selection_key,
    transition_optimizer,
    transition_defaults,
    transition_masses,
    transition_sampling_weights,
    validate_frozen_inputs,
    validate_transition_manifest,
)
from active.v17.model_stage2_v17 import Stage2TemporalHeadV17, Stage2V17Config
from active.v17.train_stage_2_v17 import Sample, sha256


class TrainStage2OtherCTCV17Tests(unittest.TestCase):
    def test_legacy_parser_defaults_are_unchanged(self):
        args = parser().parse_args([])
        self.assertIsNone(args.experiment_manifest)
        self.assertEqual(args.epochs, 40)
        self.assertEqual(args.head_epochs, 5)
        self.assertEqual(args.lr, 3e-4)
        self.assertEqual(args.distill_weight, 0.35)

    def test_transition_defaults_are_fixed(self):
        args = transition_defaults(parser().parse_args([
            "--experiment-manifest", "manifest.json", "--arm", "matched",
        ]))
        self.assertEqual((args.epochs, args.patience, args.samples_per_epoch), (20, 5, 1800))
        self.assertEqual((args.batch_size, args.head_epochs), (16, 2))
        self.assertEqual((args.lr, args.backbone_lr), (1e-4, 5e-6))
        self.assertEqual((args.distill_weight, args.temperature), (1.0, 2.0))
        self.assertEqual(args.seeds, (1701, 1702))

    def test_collapse_keeps_other_as_class_100(self):
        sequence = np.asarray([0, 1, 1, 0, 101, 101, 0, 15], dtype=np.int64)
        self.assertEqual(collapse_ctc(sequence), [0, 100, 14])

    def test_distillation_ignores_natural_rows_and_padding(self):
        teacher = torch.randn(2, 4, 101)
        student = torch.cat((teacher.clone(), torch.full((2, 4, 1), -100.0)), dim=-1)
        loss = distillation_loss(
            student, teacher, torch.tensor([2, 4]),
            torch.tensor([True, False]), temperature=2.0,
        )
        self.assertLess(abs(float(loss)), 1e-6)
        student[0, 0, 1] += 2.0
        changed = distillation_loss(
            student, teacher, torch.tensor([2, 4]),
            torch.tensor([True, False]), temperature=2.0,
        )
        self.assertGreater(float(changed), 0.0)

    def test_distillation_penalizes_other_competing_with_replay_classes(self):
        teacher = torch.zeros(1, 1, 101)
        student = torch.cat((teacher.clone(), torch.full((1, 1, 1), -100.0)), dim=-1)
        protected = distillation_loss(
            student, teacher, torch.tensor([1]), torch.tensor([True]), temperature=2.0,
        )
        student[..., 101] = 100.0
        displaced = distillation_loss(
            student, teacher, torch.tensor([1]), torch.tensor([True]), temperature=2.0,
        )
        self.assertLess(float(protected), 1e-6)
        self.assertGreater(float(displaced), 1.0)

    def test_transition_distillation_is_replay_only(self):
        mask = replay_distillation_mask([
            "local_phrases", "asllrp_other_ctc", "isolated_citizen_train",
            "asl_stem_wiki_verified_interval", "asllrp_contiguous",
        ])
        self.assertEqual(mask.tolist(), [True, False, True, False, True])

    def test_projection_only_for_first_two_epochs(self):
        model = Stage2TemporalHeadV17(Stage2V17Config(dim=16, heads=4, depth=1))
        configure_trainable(model, epoch=1)
        self.assertTrue(all(p.requires_grad for p in model.input_projection.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.ctc_head.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.sequence.parameters()))
        configure_trainable(model, epoch=3)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_transition_optimizer_makes_a_finite_update(self):
        model = Stage2TemporalHeadV17(Stage2V17Config(dim=16, heads=4, depth=1, dropout=0.0))
        configure_trainable(model, epoch=1)
        optimizer = transition_optimizer(model, 1e-4, 5e-6, .02)
        before = model.ctc_head.weight.detach().clone()
        logits, lengths = model(
            torch.randn(1, 1, 32, 612), torch.ones(1, 1, dtype=torch.bool)
        )
        loss = torch.nn.CTCLoss(blank=0)(
            logits.log_softmax(-1).transpose(0, 1),
            torch.tensor([1]), lengths, torch.tensor([1]),
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(model.ctc_head.weight).all())
        self.assertFalse(torch.equal(before, model.ctc_head.weight))

    def test_transition_sampling_balances_class_then_participant(self):
        def sample(item, target, source):
            return Sample(np.zeros((1, 32, 612), np.float32), np.asarray([target]), source, item, (str(target),))
        stem = type("Rows", (), {"__len__": lambda self: len(self.samples), "samples": [
            sample("stem:P1:1", 1, "asl_stem_wiki_verified_interval"),
            sample("stem:P1:2", 1, "asl_stem_wiki_verified_interval"),
            sample("stem:P2:3", 1, "asl_stem_wiki_verified_interval"),
            sample("stem:P1:4", 2, "asl_stem_wiki_verified_interval"),
        ]})()
        weights, counts = transition_sampling_weights(
            CombinedDataset([stem]), {"asl_stem_wiki_verified_interval": 1.0}
        )
        self.assertEqual(counts["asl_stem_wiki_verified_interval"], 4)
        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3]), 0.5)
        self.assertAlmostEqual(float(weights[0] + weights[1]), float(weights[2]))

    def test_arm_masses_are_exact(self):
        self.assertEqual(transition_masses("with_stem"), {
            "asllrp_other_ctc": .40, "local_phrases": .20,
            "asllrp_contiguous": .10, "isolated_citizen_train": .20,
            "asl_stem_wiki_verified_interval": .10,
        })
        self.assertEqual(transition_masses("no_stem")["isolated_citizen_train"], .30)

    def test_manifest_validation_fails_closed_on_hash_or_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "input.bin"
            item.write_bytes(b"pinned")
            manifest = root / "manifest.json"
            manifest.write_text(__import__("json").dumps({
                "format": "slt_v17_stage2_transition_adapt", "version": 1,
                "encoder": {"sha256": "1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b"},
                "vocabulary": {"blank_index": 0, "locked_gloss_indices": "1-100", "other_index": 101},
                "inputs": [{"path": str(item), "sha256": sha256(item), "source_role": "locked vocabulary"}],
                "training_participants": ["P1"], "validation_participants": ["P2"],
            }))
            validated = validate_transition_manifest(
                manifest, required_roles={"locked vocabulary"}, expected_sha256=sha256(manifest)
            )
            self.assertEqual(validated["vocabulary"]["other_index"], 101)
            item.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                validate_transition_manifest(
                    manifest, required_roles={"locked vocabulary"}, expected_sha256=sha256(manifest)
                )

    def test_manifest_bytes_are_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                validate_transition_manifest(path, expected_sha256="0" * 64)

    def test_stem_validation_checks_indices_identity_and_role_membership(self):
        from active.v17.train_stage_2_other_ctc_v17 import _validate_stem_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [f"SIGN_{index}" for index in range(100)]
            labels[86] = "YEAR"
            vocabulary = root / "vocabulary.json"
            vocabulary.write_text(__import__("json").dumps({
                "classes": [{"canonical_label": label} for label in labels],
            }))
            archive = root / "row.stage2_frozen_v17.npz"
            manifest = {
                "vocabulary": {"path": str(vocabulary)},
                "training_participants": ["P1"], "validation_participants": ["P2"],
                "rows": [{"source_item_id": "stem:P1:000", "role": "train",
                          "participant": "P1", "signer_id": "P1",
                          "target_sequence": ["YEAR"], "target_indices": [86]}],
            }
            np.savez_compressed(
                archive, target_indices=np.asarray([86]),
                metadata_json=np.asarray(__import__("json").dumps({
                    "source_item_id": "stem:P1:000", "role": "train",
                    "target_sequence": ["YEAR"], "signer_id": "P1",
                })),
            )
            _validate_stem_manifest(root, manifest)
            np.savez_compressed(
                archive, target_indices=np.asarray([85]),
                metadata_json=np.asarray(__import__("json").dumps({
                    "source_item_id": "stem:P1:000", "role": "train",
                    "target_sequence": ["YEAR"], "signer_id": "P1",
                })),
            )
            with self.assertRaisesRegex(ValueError, "STEM"):
                _validate_stem_manifest(root, manifest)

    def test_stem_validation_rejects_semantically_wrong_matching_indices(self):
        from active.v17.train_stage_2_other_ctc_v17 import _validate_stem_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [f"SIGN_{index}" for index in range(100)]
            labels[86] = "YEAR"
            vocabulary = root / "vocabulary.json"
            vocabulary.write_text(__import__("json").dumps({
                "classes": [{"canonical_label": label} for label in labels],
            }))
            manifest = {
                "vocabulary": {"path": str(vocabulary)},
                "training_participants": ["P1"], "validation_participants": ["P2"],
                "rows": [{"source_item_id": "stem:P1:000", "role": "train",
                          "participant": "P1", "signer_id": "P1",
                          "target_sequence": ["YEAR"], "target_indices": [87]}],
            }
            np.savez_compressed(
                root / "row.stage2_frozen_v17.npz", target_indices=np.asarray([87]),
                metadata_json=np.asarray(__import__("json").dumps({
                    "source_item_id": "stem:P1:000", "role": "train",
                    "target_sequence": ["YEAR"], "signer_id": "P1",
                })),
            )
            with self.assertRaisesRegex(ValueError, "semantic"):
                _validate_stem_manifest(root, manifest)

    def test_frozen_inputs_sidecar_pins_both_semantic_roots(self):
        from active.v17.train_stage_2_other_ctc_v17 import directory_sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("frozen")
            stem = root / "stem"
            context = root / "context"
            stem.mkdir(); context.mkdir()
            (stem / "one.stage2_frozen_v17.npz").write_bytes(b"stem")
            (context / "one.stage2_frozen_v17.npz").write_bytes(b"context")
            sidecar = root / "inputs.json"
            sidecar.write_text(__import__("json").dumps({
                "format": "slt_stage2_transition_frozen_inputs_v17", "version": 1,
                "experiment_manifest": str(manifest),
                "experiment_manifest_sha256": sha256(manifest),
                "encoder_sha256": "1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b",
                "inputs": [
                    {"path": str(stem), "source_role": "reviewed STEM train/validation frozen features", "archives": 1, "sha256": directory_sha256(stem)},
                    {"path": str(context), "source_role": "ASLLRP contextual-sign validation frozen features", "archives": 1, "sha256": directory_sha256(context)},
                ],
            }))
            validate_frozen_inputs(sidecar, manifest, stem, context)
            (context / "two.stage2_frozen_v17.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "frozen input"):
                validate_frozen_inputs(sidecar, manifest, stem, context)

    def test_repeated_ctc_targets_require_a_blank_timestep(self):
        self.assertFalse(ctc_lengths_are_feasible(
            torch.tensor([1, 1]), torch.tensor([2]), torch.tensor([2])
        ))
        self.assertTrue(ctc_lengths_are_feasible(
            torch.tensor([1, 1]), torch.tensor([2]), torch.tensor([3])
        ))

    def test_baseline_relative_eligibility_and_order(self):
        baseline = dict(BASELINE_EXPECTATIONS)
        passing = dict(baseline, target_edits=542, citizen_correct=328)
        failing = dict(passing, local_edits=7)
        self.assertTrue(eligibility(passing, baseline))
        self.assertFalse(eligibility(failing, baseline))
        self.assertGreater(candidate_key(passing, baseline, 2), candidate_key(passing, baseline, 3))

    def test_exact_warm_start_and_save_reload_reproduce_predictions(self):
        base = Stage2TemporalHeadV17(Stage2V17Config(dim=16, heads=4, depth=1, dropout=0.0))
        checkpoint = {"format": "slt_stage2_ctc_v17", "model_config": base.config.to_dict(), "model_state_dict": base.state_dict()}
        student = build_transition_student(checkpoint)
        self.assertEqual(student.ctc_head.out_features, 102)
        self.assertTrue(torch.equal(student.ctc_head.weight[:101], base.ctc_head.weight))
        features = torch.randn(1, 1, 32, 612)
        mask = torch.ones(1, 1, dtype=torch.bool)
        before = student.eval()(features, mask)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            torch.save({"model_config": student.config.to_dict(), "model_state_dict": student.state_dict()}, path)
            payload = torch.load(path, weights_only=False)
            reloaded = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
            reloaded.load_state_dict(payload["model_state_dict"])
        self.assertTrue(torch.equal(before, reloaded.eval()(features, mask)[0]))

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
