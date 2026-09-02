#!/usr/bin/env python3
"""Gate free-running temporal-code text-to-sign output before any render."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.model_full_trajectory_v17 import observation_from_prediction
from active.v17.model_temporal_code_prior_v17 import (
    TemporalCodePriorV17,
    TemporalCodePriorV17Config,
    apply_temporal_hand_mask,
)
from active.v17.train_full_trajectory_v17 import FullTrajectoryDataset, sequence_key
from active.v17.train_temporal_code_prior_v17 import load_tokenizer
from scripts.audit_real_motion_reference_v17 import Summary
from scripts.evaluate_full_trajectory_generator_v17 import encoded_sequence, real_summary


ONE_HAND_AUDIT = ("SORRY", "I", "LATE")


def longest_run(values):
    best = current = 1
    for before, after in zip(values, values[1:]):
        current = current + 1 if before == after else 1
        best = max(best, current)
    return best


def summary(value):
    result = Summary()
    result.add(value)
    return result.result()


def candidate_score(row, reference_motion, expect_one_hand=False):
    score = 0.0
    for name in ("speed", "acceleration", "jerk"):
        value = row["motion"]["hand_motion"][name]["p95"]
        reference = reference_motion["hand_motion"][name]["p95"]
        score += 100.0 if value is None or value <= 0 else abs(np.log(value / reference))
    if row["motion"]["presence_change_step_fraction"] <= 0:
        score += 100
    if row["motion"]["presence_frame_fractions"]["both_hands_complete"] >= 0.99:
        score += 100
    if row["unique_codes"] < 8:
        score += 8 - row["unique_codes"]
    if row["longest_code_run"] > 8:
        score += row["longest_code_run"] - 8
    if expect_one_hand and sum(row["hand_participation"]) != 1:
        score += 100
    return score


@torch.inference_mode()
def generate_candidates(
    model, tokenizer, checkpoint, sequence, device, reference_motion,
    samples, temperature, top_k,
):
    tokens, valid = encoded_sequence(
        sequence, checkpoint["token_to_index"],
        int(checkpoint["model_config"]["maximum_tokens"]),
    )
    tokens = tokens.repeat(samples, 1).to(device)
    valid = valid.repeat(samples, 1).to(device)
    source_id = checkpoint["source_to_index"]["local_phrase_full"]
    generated = model.generate(
        tokens, valid, torch.full((samples,), source_id, device=device),
        temperature=temperature, top_k=top_k,
    )
    prediction = tokenizer.decode_codes(generated["codes"])
    observation = observation_from_prediction(prediction, 0.25)
    observation = apply_temporal_hand_mask(
        observation, generated["side_logits"], tokenizer.config.downsample_factor,
    ).cpu().numpy()
    codes = generated["codes"].cpu().numpy()
    rows = []
    for index in range(samples):
        values = codes[index].tolist()
        participation = [
            bool((observation[index, :, :21, 3] > 0).any()),
            bool((observation[index, :, 21:42, 3] > 0).any()),
        ]
        row = {
            "sample": index,
            "codes": values,
            "unique_codes": len(set(values)),
            "longest_code_run": longest_run(values),
            "hand_participation": participation,
            "predicted_duration_seconds": float(generated["log_duration"][index].exp()),
            "motion": summary(observation[index]),
            "observation": observation[index],
        }
        row["selection_score"] = candidate_score(
            row, reference_motion, sequence == ONE_HAND_AUDIT
        )
        rows.append(row)
    return min(rows, key=lambda row: (row["selection_score"], row["sample"]))


def dtw_distance(first, second):
    distance = 1 - first @ second.T
    previous = np.full(len(second) + 1, np.inf, np.float32)
    previous[0] = 0
    for row in distance:
        current = np.full(len(second) + 1, np.inf, np.float32)
        for column, value in enumerate(row, start=1):
            current[column] = value + min(
                current[column - 1], previous[column], previous[column - 1]
            )
        previous = current
    return float(previous[-1] / (len(first) + len(second)))


@torch.inference_mode()
def local_validation_codes(tokenizer, dataset, device):
    rows = defaultdict(list)
    for index in range(len(dataset)):
        item = dataset[index]
        features = torch.from_numpy(item["features"])[None].to(device)
        rows[item["sequence"]].append(tokenizer.encode_codes(features)[0].cpu().numpy())
    return rows


def semantic_ranks(generated, genuine, codebook):
    codebook = codebook / np.linalg.norm(codebook, axis=1, keepdims=True).clip(min=1e-8)
    output = {}
    for target, row in generated.items():
        first = codebook[np.asarray(row["codes"])]
        scores = {}
        for phrase, sequences in genuine.items():
            distances = sorted(dtw_distance(first, codebook[codes]) for codes in sequences)
            scores[phrase] = float(np.mean(distances[:min(3, len(distances))]))
        ranking = sorted(scores, key=scores.get)
        output[target] = {
            "target_rank": ranking.index(target) + 1,
            "nearest_phrase": ranking[0],
            "distances": scores,
        }
    return output


def run(args):
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_temporal_code_prior_v17":
        raise ValueError("unexpected temporal prior checkpoint")
    model = TemporalCodePriorV17(TemporalCodePriorV17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().requires_grad_(False).to(device)
    tokenizer, _ = load_tokenizer(Path(checkpoint["tokenizer"]), device)
    manifest = json.loads(Path(checkpoint["manifest"]).read_text())
    source_to_index = {str(key): int(value) for key, value in checkpoint["source_to_index"].items()}
    holdouts = {tuple(value) for value in checkpoint["holdout_sequences"]}
    local_train = FullTrajectoryDataset(
        manifest, args.data_root, {"train"}, source_to_index,
        excluded_sequences=holdouts, sources={"local_phrase_full"},
    )
    local_validation = FullTrajectoryDataset(
        manifest, args.data_root, {"validation"}, source_to_index,
        sources={"local_phrase_full"},
    )
    reference_motion = real_summary(local_train)
    sequences = [*sorted(holdouts), ONE_HAND_AUDIT]
    torch.manual_seed(args.seed)
    selected = {}
    observations = {}
    for sequence in sequences:
        row = generate_candidates(
            model, tokenizer, checkpoint, sequence, device, reference_motion,
            args.samples, args.temperature, args.top_k,
        )
        key = " ".join(sequence)
        observations[key] = row.pop("observation")
        selected[key] = row

    genuine_codes = local_validation_codes(tokenizer, local_validation, device)
    genuine_unique = [len(set(codes.tolist())) for values in genuine_codes.values() for codes in values]
    genuine_runs = [longest_run(codes.tolist()) for values in genuine_codes.values() for codes in values]
    diversity_floor = int(np.floor(np.quantile(genuine_unique, 0.05)))
    repetition_ceiling = int(np.ceil(np.quantile(genuine_runs, 0.95)))
    semantics = semantic_ranks(
        selected, genuine_codes, tokenizer.codebook.weight.detach().cpu().numpy()
    )
    genuine_participation = defaultdict(set)
    for index in range(len(local_validation)):
        item = local_validation[index]
        present = item["features"][..., 3] > 0
        genuine_participation[item["sequence"]].add((
            bool(present[:, :21].any()), bool(present[:, 21:42].any())
        ))
    ratios = {
        sequence: {
            name: row["motion"]["hand_motion"][name]["p95"]
            / reference_motion["hand_motion"][name]["p95"]
            for name in ("speed", "acceleration", "jerk")
        }
        for sequence, row in selected.items()
    }
    unseen = {" ".join(value) for value in holdouts}
    gates = {
        "all_motion_orders_within_0_5_to_2x_genuine_train": all(
            0.5 <= ratio <= 2.0 for row in ratios.values() for ratio in row.values()
        ),
        "all_presence_timelines_dynamic": all(
            row["motion"]["presence_change_step_fraction"] > 0
            for row in selected.values()
        ),
        "no_all_frame_complete_two_hand_collapse": all(
            row["motion"]["presence_frame_fractions"]["both_hands_complete"] < 0.99
            for row in selected.values()
        ),
        "all_code_sequences_diverse": all(
            row["unique_codes"] >= diversity_floor
            and row["longest_code_run"] <= repetition_ceiling
            for row in selected.values()
        ),
        "one_hand_audit_does_not_create_second_hand": (
            sum(selected[" ".join(ONE_HAND_AUDIT)]["hand_participation"]) == 1
        ),
        "generated_participation_matches_a_genuine_phrase_pattern": all(
            tuple(row["hand_participation"]) in genuine_participation[sequence]
            for sequence, row in selected.items()
        ),
        "unseen_combinations_rank_nearest_to_intended_phrase": all(
            semantics[sequence]["target_rank"] == 1 for sequence in unseen
        ),
    }
    serializable_selected = {
        sequence: row for sequence, row in selected.items()
    }
    result = {
        "format": "slt_temporal_text_to_sign_generation_gate_v17",
        "version": 1,
        "checkpoint": args.checkpoint.as_posix(),
        "tokenizer": checkpoint["tokenizer"],
        "samples_per_sequence": args.samples,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "selection_uses_holdout_target": False,
        "selected": serializable_selected,
        "motion_p95_over_genuine_local_train": ratios,
        "semantic_dtw_classification": semantics,
        "genuine_code_diversity_reference": {
            "unique_code_p05_floor": diversity_floor,
            "longest_run_p95_ceiling": repetition_ceiling,
        },
        "gates": gates,
        "all_generation_gates_passed": all(gates.values()),
        "render_unseen_combination_allowed": all(gates.values()),
        "claim_boundary": "Passing numeric gates permits native-review rendering only; generated landmarks remain ineligible as recognition ground truth.",
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/temporal_code_prior_v17_v1/model.pth"))
    value.add_argument("--data-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_temporal_text_to_sign_v1/evaluation.json"))
    value.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    value.add_argument("--samples", type=int, default=48)
    value.add_argument("--temperature", type=float, default=0.85)
    value.add_argument("--top-k", type=int, default=12)
    value.add_argument("--seed", type=int, default=1701)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
