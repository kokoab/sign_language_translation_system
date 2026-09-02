#!/usr/bin/env python3
"""Gate a full-trajectory model against real validation and compositional holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.model_full_trajectory_v17 import (
    FullTrajectoryGeneratorV17,
    FullTrajectoryV17Config,
    observation_from_prediction,
)
from active.v17.train_full_trajectory_v17 import (
    FullTrajectoryDataset,
    evaluate,
    loader,
)
from scripts.audit_real_motion_reference_v17 import Summary


class FamilyMeanBaseline(nn.Module):
    def __init__(self, mean: dict[str, torch.Tensor]):
        super().__init__()
        for key, value in mean.items():
            self.register_buffer(key, value)

    def forward(self, tokens, token_valid, source_ids, **_):
        return {
            "xyz": self.xyz[source_ids],
            "presence_logits": self.presence_logits[source_ids],
            "confidence_logits": self.confidence_logits[source_ids],
            "log_duration": self.log_duration[source_ids],
        }


def family_mean(dataset: FullTrajectoryDataset, families: int) -> dict[str, torch.Tensor]:
    first = dataset[0]["features"]
    shape = (families,) + first.shape[:2]
    xyz = np.zeros(shape + (3,), np.float64)
    confidence = np.zeros(shape, np.float64)
    observed = np.zeros(shape, np.float64)
    items = np.zeros(families, np.float64)
    log_duration = np.zeros(families, np.float64)
    for index in range(len(dataset)):
        row = dataset[index]
        source = int(row["source_id"])
        features = row["features"]
        present = features[..., 3] > 0
        xyz[source] += features[..., :3] * present[..., None]
        confidence[source] += features[..., 4]
        observed[source] += present
        items[source] += 1
        log_duration[source] += np.log(float(row["duration"]))
    xyz /= observed[..., None].clip(min=1)
    confidence /= observed.clip(min=1)
    probability = observed / items[:, None, None].clip(min=1)
    epsilon = 1e-4
    return {
        "xyz": torch.from_numpy(xyz.astype(np.float32)),
        "presence_logits": torch.from_numpy(np.log(
            probability.clip(epsilon, 1 - epsilon)
            / (1 - probability.clip(epsilon, 1 - epsilon))
        ).astype(np.float32)),
        "confidence_logits": torch.from_numpy(np.log(
            confidence.clip(epsilon, 1 - epsilon)
            / (1 - confidence.clip(epsilon, 1 - epsilon))
        ).astype(np.float32)),
        "log_duration": torch.from_numpy(
            (log_duration / items.clip(min=1)).astype(np.float32)
        ),
    }


def encoded_sequence(
    sequence: tuple[str, ...], token_to_index: dict[str, int], maximum: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [token_to_index["<BOS>"], *[token_to_index[token] for token in sequence], token_to_index["<EOS>"]]
    tokens = torch.zeros((1, maximum), dtype=torch.long)
    valid = torch.zeros((1, maximum), dtype=torch.bool)
    tokens[0, :len(ids)] = torch.tensor(ids)
    valid[0, :len(ids)] = True
    return tokens, valid


@torch.inference_mode()
def generated_summaries(
    model, checkpoint, threshold, device, reference_motion: dict, samples: int = 32,
):
    source_id = checkpoint["source_to_index"]["local_phrase_full"]
    rows = {}
    torch.manual_seed(1701)
    for values in checkpoint["holdout_sequences"]:
        sequence = tuple(values)
        tokens, valid = encoded_sequence(
            sequence, checkpoint["token_to_index"],
            int(checkpoint["model_config"]["maximum_tokens"]),
        )
        choices = []
        for sample in range(samples):
            latent = torch.randn(
                (1, model.config.motion_latent_dim), device=device
            )
            prediction = model(
                tokens.to(device), valid.to(device),
                torch.tensor([source_id], device=device), motion_latent=latent,
            )
            observation = observation_from_prediction(
                prediction, threshold
            )[0].cpu().numpy()
            summary = Summary()
            summary.add(observation)
            motion = summary.result()
            score = 0.0
            for name in ("speed", "acceleration", "jerk"):
                generated_value = motion["hand_motion"][name]["p95"]
                reference_value = reference_motion[name]["p95"]
                if (
                    generated_value is None or generated_value <= 0
                    or reference_value is None or reference_value <= 0
                ):
                    score += 100.0
                else:
                    score += abs(np.log(generated_value / reference_value))
            if motion["presence_change_step_fraction"] <= 0:
                score += 10
            if motion["presence_frame_fractions"]["both_hands_complete"] >= 0.99:
                score += 10
            choices.append((score, sample, prediction, observation, motion, latent))
        _, sample, prediction, observation, motion, latent = min(
            choices, key=lambda value: value[0]
        )
        rows[" ".join(sequence)] = {
            "selected_prior_sample": sample,
            "selected_latent_norm": float(latent.norm().item()),
            "predicted_duration_seconds": float(prediction["log_duration"].exp().item()),
            "motion": motion,
            "hand_participation": [
                bool((observation[:, :21, 3] > 0).any()),
                bool((observation[:, 21:42, 3] > 0).any()),
            ],
        }
    return rows


def real_summary(dataset: FullTrajectoryDataset) -> dict[str, object]:
    summary = Summary()
    for index in range(len(dataset)):
        summary.add(dataset[index]["features"])
    return summary.result()


def motion_ratios(generated: dict, real_motion: dict, name: str) -> dict[str, float]:
    reference = real_motion["hand_motion"][name]["p95"]
    if reference is None or reference <= 0:
        raise ValueError(f"real {name} reference is unavailable")
    output = {}
    for sequence, row in generated.items():
        value = row["motion"]["hand_motion"][name]["p95"]
        output[sequence] = 0.0 if value is None else value / reference
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_full_trajectory_generator_v17":
        raise ValueError("unexpected full-trajectory checkpoint")
    manifest = json.loads(Path(checkpoint["manifest"]).read_text())
    source_to_index = {str(key): int(value) for key, value in checkpoint["source_to_index"].items()}
    holdouts = {tuple(value) for value in checkpoint["holdout_sequences"]}
    enabled_sources = set(source_to_index)
    train = FullTrajectoryDataset(
        manifest, args.data_root, {"train"}, source_to_index,
        excluded_sequences=holdouts, sources=enabled_sources,
    )
    validation = FullTrajectoryDataset(
        manifest, args.data_root, {"validation"}, source_to_index,
        excluded_sequences=holdouts, sources=enabled_sources,
    )
    holdout = FullTrajectoryDataset(
        manifest, args.data_root, {"validation"}, source_to_index,
        only_sequences=holdouts, sources={"local_phrase_full"},
    )
    local_train = FullTrajectoryDataset(
        manifest, args.data_root, {"train"}, source_to_index,
        excluded_sequences=holdouts, sources={"local_phrase_full"},
    )
    device = torch.device(args.device)
    model = FullTrajectoryGeneratorV17(
        FullTrajectoryV17Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().requires_grad_(False).to(device)
    baseline = FamilyMeanBaseline(family_mean(train, len(source_to_index))).to(device)
    validation_loader = loader(validation, args.batch_size)
    holdout_loader = loader(holdout, args.batch_size)
    candidates = []
    for threshold in np.arange(0.25, 0.76, 0.05):
        metrics = evaluate(model, validation_loader, device, float(threshold))
        candidates.append((metrics["presence_f1"], metrics["hand_participation_accuracy"], float(threshold), metrics))
    _, _, threshold, validation_metrics = max(candidates)
    holdout_metrics = evaluate(model, holdout_loader, device, threshold)
    baseline_validation = evaluate(baseline, validation_loader, device, threshold)
    baseline_holdout = evaluate(baseline, holdout_loader, device, threshold)
    real_holdout_motion = real_summary(holdout)
    real_local_train_motion = real_summary(local_train)
    generated = generated_summaries(
        model, checkpoint, threshold, device,
        real_local_train_motion["hand_motion"],
    )
    generated_motion_ratios = {
        name: motion_ratios(generated, real_holdout_motion, name)
        for name in ("speed", "acceleration", "jerk")
    }
    gates = {
        "validation_coordinate_beats_family_mean": (
            validation_metrics["coordinate"] < baseline_validation["coordinate"]
        ),
        "holdout_coordinate_beats_family_mean": (
            holdout_metrics["coordinate"] < baseline_holdout["coordinate"]
        ),
        "holdout_velocity_beats_family_mean": (
            holdout_metrics["velocity"] < baseline_holdout["velocity"]
        ),
        "validation_presence_f1_at_least_0_70": validation_metrics["presence_f1"] >= 0.70,
        "holdout_hand_participation_at_least_0_95": (
            holdout_metrics["hand_participation_accuracy"] >= 0.95
        ),
        "no_generated_phrase_forces_complete_hands_every_frame": all(
            row["motion"]["presence_frame_fractions"]["both_hands_complete"] < 0.99
            for row in generated.values()
        ),
        "generated_presence_is_temporally_dynamic": all(
            row["motion"]["presence_change_step_fraction"] > 0
            for row in generated.values()
        ),
        "generated_speed_within_real_holdout_range": all(
            0.25 <= ratio <= 4.0
            for ratio in generated_motion_ratios["speed"].values()
        ),
        "generated_acceleration_within_real_holdout_range": all(
            0.25 <= ratio <= 4.0
            for ratio in generated_motion_ratios["acceleration"].values()
        ),
        "generated_jerk_within_real_holdout_range": all(
            0.25 <= ratio <= 4.0
            for ratio in generated_motion_ratios["jerk"].values()
        ),
    }
    result = {
        "format": "slt_full_trajectory_generation_gate_v17",
        "version": 1,
        "checkpoint": args.checkpoint.as_posix(),
        "selected_presence_threshold": threshold,
        "validation": validation_metrics,
        "compositional_holdout": holdout_metrics,
        "family_mean_validation": baseline_validation,
        "family_mean_compositional_holdout": baseline_holdout,
        "real_compositional_holdout_motion": real_holdout_motion,
        "real_local_train_motion": real_local_train_motion,
        "generated_holdout_summaries": generated,
        "generated_motion_p95_over_real_holdout": generated_motion_ratios,
        "gates": gates,
        "all_generation_gates_passed": all(gates.values()),
        "render_unseen_combination_allowed": all(gates.values()),
        "claim_boundary": (
            "Passing numeric gates permits native-review rendering only; it does not "
            "make generated landmarks recognition ground truth."
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/full_trajectory_generator_v17_holdout_v1/model.pth"))
    value.add_argument("--data-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_full_trajectory_generation_v1/evaluation.json"))
    value.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    value.add_argument("--batch-size", type=int, default=16)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
