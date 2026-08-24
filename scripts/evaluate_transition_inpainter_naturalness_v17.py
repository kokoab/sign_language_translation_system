#!/usr/bin/env python3
"""Falsify held-out-signer transition inpainting against genuine motion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
    interpolate_masked_context,
)
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    loss_terms,
)
from scripts.audit_stage2_transition_naturalness_v17 import (
    fixed_random_projection,
    grouped_oof,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_motion_descriptor(
    features: np.ndarray, mask: np.ndarray, frames: int = 24
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    start = max(0, int(indices[0]) - 2)
    stop = min(len(features), int(indices[-1]) + 3)
    stream = features[start:stop].astype(np.float32).reshape(stop - start, -1)
    positions = np.linspace(0.0, len(stream) - 1, frames, dtype=np.float32)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(stream) - 1)
    weight = (positions - lower)[:, None]
    resized = stream[lower] * (1.0 - weight) + stream[upper] * weight
    velocity = np.diff(resized, axis=0)
    acceleration = np.diff(velocity, axis=0)
    descriptor = np.concatenate((
        resized.mean(axis=0),
        resized.std(axis=0),
        np.abs(velocity).mean(axis=0),
        velocity.std(axis=0),
        np.abs(acceleration).mean(axis=0),
        np.abs(velocity).max(axis=0),
        resized[-1] - resized[0],
    ))
    if not np.isfinite(descriptor).all():
        raise ValueError("non-finite local motion descriptor")
    return descriptor.astype(np.float32)


def paired_discrimination(
    genuine: list[np.ndarray],
    generated: list[np.ndarray],
    masks: list[np.ndarray],
    groups: np.ndarray,
    folds: int,
) -> dict[str, Any]:
    descriptors = []
    labels = []
    row_groups = []
    for real, fake, mask, group in zip(genuine, generated, masks, groups):
        descriptors.extend((
            local_motion_descriptor(real, mask),
            local_motion_descriptor(fake, mask),
        ))
        labels.extend((1, 0))
        row_groups.extend((group, group))
    x = fixed_random_projection(np.stack(descriptors))
    y = np.asarray(labels, dtype=np.int64)
    row_groups_array = np.asarray(row_groups)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups.tolist()))))
    dummy = np.zeros(len(y), dtype=np.int64)
    splits = list(splitter.split(dummy, dummy, row_groups_array))
    result = grouped_oof(x, y, row_groups_array, splits)
    result["paired_windows"] = len(genuine)
    result["split_unit"] = "source clip; paired genuine/generated rows never split"
    return result


def per_window_score(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> np.ndarray:
    scores = []
    for index in range(len(target)):
        terms = loss_terms(
            predicted[index:index + 1],
            target[index:index + 1],
            mask[index:index + 1],
        )
        scores.append(float(
            terms["spatial"]
            + 0.25 * terms["velocity"]
            + 0.10 * terms["acceleration"]
        ))
    return np.asarray(scores, dtype=np.float64)


def bootstrap_improvement(
    learned: np.ndarray, linear: np.ndarray, iterations: int
) -> dict[str, float]:
    rng = np.random.default_rng(1701)
    relative = []
    for _ in range(iterations):
        sample = rng.integers(0, len(learned), len(learned))
        learned_mean = float(learned[sample].mean())
        linear_mean = float(linear[sample].mean())
        relative.append((linear_mean - learned_mean) / max(linear_mean, 1e-12))
    return {
        "iterations": iterations,
        "relative_improvement": float((linear.mean() - learned.mean()) / linear.mean()),
        "ci95_low": float(np.quantile(relative, 0.025)),
        "ci95_high": float(np.quantile(relative, 0.975)),
        "windows_improved_fraction": float(np.mean(learned < linear)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_paths = [args.checkpoint, *args.ensemble_checkpoints]
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    if any(row.get("format") != "slt_transition_inpainter_v17" for row in checkpoints):
        raise ValueError("unexpected transition checkpoint")
    if len({json.dumps(row["model_config"], sort_keys=True) for row in checkpoints}) != 1:
        raise ValueError("ensemble checkpoints have different model configurations")
    if len({str(row["held_out_signer"]) for row in checkpoints}) != 1:
        raise ValueError("ensemble checkpoints have different held-out signers")
    config = TransitionInpainterV17Config(**checkpoints[0]["model_config"])
    models = []
    for checkpoint in checkpoints:
        model = TransitionInpainterV17(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
    held_out = str(checkpoints[0]["held_out_signer"])
    dataset = TransitionWindowDataset(
        args.landmark_root, {held_out}, seed=4701, fixed_masks=True
    )

    genuine: list[np.ndarray] = []
    learned: list[np.ndarray] = []
    linear: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    groups = []
    batch_features = []
    batch_masks = []
    for sample in dataset:
        features = sample["features"]
        mask = sample["mask"]
        with torch.inference_mode():
            prediction = torch.stack([
                model(features[None], mask[None])[0] for model in models
            ]).mean(dim=0)
            interpolation = interpolate_masked_context(features[None], mask[None])[0]
        genuine.append(features.numpy())
        learned.append(prediction.numpy())
        linear.append(interpolation.numpy())
        masks.append(mask.numpy())
        groups.append(str(sample["item"]).rsplit(":", 1)[0])
        batch_features.append(features)
        batch_masks.append(mask)

    target_tensor = torch.stack(batch_features)
    mask_tensor = torch.stack(batch_masks)
    learned_tensor = torch.from_numpy(np.stack(learned))
    linear_tensor = torch.from_numpy(np.stack(linear))
    learned_scores = per_window_score(learned_tensor, target_tensor, mask_tensor)
    linear_scores = per_window_score(linear_tensor, target_tensor, mask_tensor)
    group_array = np.asarray(groups)
    report = {
        "format": "transition_inpainter_heldout_naturalness_audit_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoints": [path.as_posix() for path in checkpoint_paths],
        "checkpoint_sha256": [sha256(path) for path in checkpoint_paths],
        "ensemble_size": len(checkpoint_paths),
        "held_out_signer": held_out,
        "paired_windows": len(dataset),
        "held_out_source_clips": len(set(groups)),
        "reconstruction_vs_linear": bootstrap_improvement(
            learned_scores, linear_scores, args.bootstrap_iterations
        ),
        "genuine_vs_learned_discriminator": paired_discrimination(
            genuine, learned, masks, group_array, args.folds
        ),
        "genuine_vs_linear_discriminator": paired_discrimination(
            genuine, linear, masks, group_array, args.folds
        ),
        "interpretation": (
            "Lower learned discriminator performance than linear is evidence of a "
            "closer held-out feature distribution; chance performance still does not "
            "prove rendered-video or human-perceptual naturalness."
        ),
        "descriptor": (
            "masked interval plus two genuine boundary frames; 24-frame resampling; "
            "frame-vector position, velocity, acceleration, endpoint statistics; "
            "fixed label-independent 128D projection; source-clip-grouped CV"
        ),
        "limitations": [
            "This evaluates v17 landmark trajectories, not rendered RGB video.",
            "The held-out data contains one signer and 48 source clips.",
            "Reconstruction of a hidden observed interval is easier than unconstrained text-to-sign generation.",
            "A human Deaf-signer perceptual study is required for a genuine naturalness claim.",
        ],
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/models/transition_inpainter_residual_v17_pilot/best_model.pth"),
    )
    parser.add_argument("--ensemble-checkpoints", type=Path, nargs="*", default=())
    parser.add_argument(
        "--landmark-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/transition_inpainter_naturalness_v17_pilot.json"),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
