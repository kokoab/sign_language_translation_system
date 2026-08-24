#!/usr/bin/env python3
"""Evaluate a content-controlled latent signing-style profile on held-out voices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_signing_voice_profile_v17 import (
    apply_voice_profile,
    decode_profile,
    encode_profile,
    estimate_voice_profile,
    fit_profile_latent,
)
from active.v17.train_signing_voice_v17 import (
    FOLDS,
    build_prototypes,
    load_content_model,
    rank_auc,
    sha256,
    trajectory_terms,
)


def example_spatial_error(generated: np.ndarray, target: np.ndarray) -> float:
    present = (generated[..., 3] > 0) & (target[..., 3] > 0)
    difference = np.abs(generated[..., :3][present] - target[..., :3][present])
    smooth = np.where(difference < 1.0, 0.5 * difference ** 2, difference - 0.5)
    return float(smooth.mean())


def aggregate_terms(generated: np.ndarray, target: np.ndarray) -> dict[str, float]:
    totals = {key: 0.0 for key in ("spatial", "velocity", "acceleration")}
    seen = 0
    for start in range(0, len(generated), 128):
        prediction = torch.from_numpy(generated[start:start + 128])
        reference = torch.from_numpy(target[start:start + 128])
        terms = trajectory_terms(prediction, reference)
        count = len(prediction)
        seen += count
        for key, value in terms.items():
            totals[key] += float(value) * count
    return {key: value / seen for key, value in totals.items()}


@torch.inference_mode()
def content_predictions(model, values: np.ndarray, device) -> np.ndarray:
    predictions = []
    for start in range(0, len(values), 64):
        features = torch.from_numpy(values[start:start + 64]).to(device)
        predictions.append(model(features).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def content_accuracy(model, values: np.ndarray, targets: np.ndarray, device) -> float:
    return float((content_predictions(model, values, device) == targets).mean())


def run(args):
    started = time.monotonic()
    device_name = (
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    with np.load(args.pool, allow_pickle=False) as payload:
        landmarks = payload["landmarks"].astype(np.float32)
        targets = payload["target_indices"].astype(np.int64)
        signers = payload["signer_ids"].astype(str)
        observed_frames = payload["observed_frames"].astype(np.int16)
    eligible = {
        signer for signer in set(signers.tolist())
        if len(set(targets[signers == signer].tolist())) >= 2
    }
    validation_voices = sorted(FOLDS[args.fold])
    train_voices = sorted(eligible - set(validation_voices))
    all_indices = np.arange(len(landmarks))
    train_indices = all_indices[np.isin(signers, train_voices)]
    validation_indices = all_indices[np.isin(signers, validation_voices)]
    prototypes_tensor, medoids = build_prototypes(landmarks, targets, train_indices)
    prototypes = prototypes_tensor.numpy().astype(np.float32)
    train_profiles = [
        estimate_voice_profile(
            landmarks, targets, train_indices[signers[train_indices] == voice], prototypes
        )
        for voice in train_voices
    ]
    latent_mean, latent_components, train_latents = fit_profile_latent(
        train_profiles, args.latent_dim
    )

    validation_profile_cache = {}
    for voice in validation_voices:
        voice_rows = validation_indices[signers[validation_indices] == voice]
        for target in sorted(set(targets[voice_rows].tolist())):
            reference_rows = voice_rows[targets[voice_rows] != target]
            if len(set(targets[reference_rows].tolist())) < 2:
                reference_rows = voice_rows
            raw = estimate_voice_profile(
                landmarks, targets, reference_rows, prototypes
            )
            latent = encode_profile(raw, latent_mean, latent_components)
            validation_profile_cache[(voice, target)] = decode_profile(
                latent, latent_mean, latent_components
            )

    baseline = np.stack([
        prototypes[int(targets[index])] for index in validation_indices
    ]).astype(np.float32)
    row_profiles = [
        validation_profile_cache[(str(signers[index]), int(targets[index]))]
        for index in validation_indices
    ]
    content_model, label_to_index = load_content_model(args.content_checkpoint, device)
    validation_targets = targets[validation_indices]
    if args.adaptive_content_gate:
        generated = baseline.copy()
        selected_strengths = np.zeros(len(validation_indices), dtype=np.float32)
        assigned = np.zeros(len(validation_indices), dtype=np.bool_)
        for strength in (1.0, 0.75, 0.50, 0.40, 0.25):
            candidates = np.stack([
                apply_voice_profile(
                    prototype, profile, profile_strength=strength,
                    curve_strength=args.curve_strength * strength,
                )
                for prototype, profile in zip(baseline, row_profiles)
            ]).astype(np.float32)
            correct = (
                content_predictions(content_model, candidates, device)
                == validation_targets
            )
            choose = correct & ~assigned
            generated[choose] = candidates[choose]
            selected_strengths[choose] = strength
            assigned |= choose
    else:
        selected_strengths = np.full(
            len(validation_indices), args.profile_strength, dtype=np.float32
        )
        generated = np.stack([
            apply_voice_profile(
                prototype, profile, profile_strength=args.profile_strength,
                curve_strength=args.curve_strength,
            )
            for prototype, profile in zip(baseline, row_profiles)
        ]).astype(np.float32)

    positive_scores = []
    negative_scores = []
    for row_index, index in enumerate(validation_indices):
        target = int(targets[index])
        voice = str(signers[index])
        prototype = prototypes[target]
        strength = float(selected_strengths[row_index])
        positive_scores.append(-example_spatial_error(generated[row_index], landmarks[index]))
        for other_voice in validation_voices:
            if other_voice == voice:
                continue
            other_profile = validation_profile_cache.get((other_voice, target))
            if other_profile is None:
                other_rows = validation_indices[signers[validation_indices] == other_voice]
                raw = estimate_voice_profile(landmarks, targets, other_rows, prototypes)
                other_profile = decode_profile(
                    encode_profile(raw, latent_mean, latent_components),
                    latent_mean, latent_components,
                )
            negative = apply_voice_profile(
                prototype, other_profile, profile_strength=strength,
                curve_strength=args.curve_strength * strength,
            )
            negative_scores.append(-example_spatial_error(negative, landmarks[index]))
    target_values = landmarks[validation_indices]
    generated_terms = aggregate_terms(generated, target_values)
    baseline_terms = aggregate_terms(baseline, target_values)
    generated_content = content_accuracy(
        content_model, generated, validation_targets, device
    )
    baseline_content = content_accuracy(
        content_model, baseline, validation_targets, device
    )
    target_content = content_accuracy(
        content_model, target_values, validation_targets, device
    )
    metrics = {
        "examples": len(validation_indices),
        "generated_content_accuracy": generated_content,
        "prototype_content_accuracy": baseline_content,
        "target_content_accuracy": target_content,
        "adaptive_content_gate": args.adaptive_content_gate,
        "selected_profile_strength_counts": {
            f"{value:.2f}": int((selected_strengths == value).sum())
            for value in sorted(set(selected_strengths.tolist()), reverse=True)
        },
        "style_verification_auc": rank_auc(
            np.asarray(positive_scores), np.asarray(negative_scores)
        ),
        "style_positive_pairs": len(positive_scores),
        "style_negative_pairs": len(negative_scores),
        "style_score_contract": (
            "negative spatial reconstruction error; same target gloss/content; "
            "positive uses target signer's other-gloss profile; negatives use other signers"
        ),
    }
    for term in ("spatial", "velocity", "acceleration"):
        metrics[f"generated_{term}"] = generated_terms[term]
        metrics[f"prototype_{term}"] = baseline_terms[term]
        metrics[f"relative_{term}_improvement"] = (
            baseline_terms[term] - generated_terms[term]
        ) / baseline_terms[term]

    class_medians = np.asarray([
        np.median(observed_frames[train_indices[targets[train_indices] == target]])
        for target in range(100)
    ], dtype=np.float32)
    duration_ratios = []
    for voice in train_voices:
        rows = train_indices[signers[train_indices] == voice]
        duration_ratios.append(float(np.median(
            observed_frames[rows] / class_medians[targets[rows]].clip(min=1)
        )))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_signing_voice_profile_v17",
        "version": 1,
        "latent_dim": args.latent_dim,
        "latent_mean": torch.from_numpy(latent_mean),
        "latent_components": torch.from_numpy(latent_components),
        "train_voice_style_latents": torch.from_numpy(train_latents),
        "train_voice_profiles": torch.from_numpy(np.stack([value.vector() for value in train_profiles])),
        "content_prototypes": prototypes_tensor,
        "content_prototype_pool_indices": medoids,
        "curve_strength": args.curve_strength,
        "profile_strength": args.profile_strength,
        "adaptive_content_gate": args.adaptive_content_gate,
        "label_to_index": label_to_index,
        "train_voices": train_voices,
        "validation_voices": validation_voices,
        "train_voice_duration_ratios": torch.tensor(duration_ratios),
        "class_median_observed_frames": torch.from_numpy(class_medians),
        "fold": args.fold,
        "validation_metrics": metrics,
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "content_checkpoint": args.content_checkpoint.as_posix(),
        "content_checkpoint_sha256": sha256(args.content_checkpoint),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    checkpoint_path = args.output / "best_model.pth"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "fold": args.fold,
        "latent_dim": args.latent_dim,
        "curve_strength": args.curve_strength,
        "profile_strength": args.profile_strength,
        "adaptive_content_gate": args.adaptive_content_gate,
        "train_voices": len(train_voices),
        "validation_voices": validation_voices,
        "validation": metrics,
        "seconds": time.monotonic() - started,
        "claim_boundary": (
            "held-out profile reconstruction/style/content evidence is not a fluent-signer "
            "naturalness or linguistic-correctness rating"
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/local/signing_voice_v17/train_only_landmark_pool.npz"))
    parser.add_argument("--content-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/signing_voice_profile_v17_fold0"))
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--curve-strength", type=float, default=0.25)
    parser.add_argument("--profile-strength", type=float, default=0.50)
    parser.add_argument("--adaptive-content-gate", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
