#!/usr/bin/env python3
"""Train and gate a temporal discrete tokenizer on genuine v17 trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from active.v17.model_full_trajectory_v17 import observation_from_prediction
from active.v17.model_temporal_motion_tokenizer_v17 import (
    TemporalMotionTokenizerV17,
    TemporalMotionTokenizerV17Config,
)
from active.v17.train_full_trajectory_v17 import (
    FullTrajectoryDataset,
    loader,
    sequence_key,
    trajectory_loss,
)
from active.v17.train_signing_voice_v17 import sha256
from scripts.audit_real_motion_reference_v17 import Summary


def reconstruction_loss(
    prediction: dict[str, torch.Tensor], target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    base, metrics = trajectory_loss(
        prediction, target, torch.ones(len(target), device=target.device),
        coordinate_weight=10.0,
    )
    present = target[..., 3] > 0
    balanced_presence = (
        F.binary_cross_entropy_with_logits(
            prediction["presence_logits"], present.float(), reduction="none"
        ) * torch.where(present, 1.0, 5.0)
    ).mean()
    total = base + 2.0 * balanced_presence + prediction["quantization_loss"]
    metrics["reconstruction_loss"] = float(total.detach())
    metrics["quantization_loss"] = float(prediction["quantization_loss"].detach())
    metrics["balanced_presence"] = float(balanced_presence.detach())
    metrics["codebook_perplexity"] = float(prediction["codebook_perplexity"].detach())
    return total, metrics


@torch.inference_mode()
def evaluate(model, data_loader, device, threshold=0.5):
    model.eval()
    totals: Counter[str] = Counter()
    code_counts = torch.zeros(model.config.codebook_size, dtype=torch.long)
    true_positive = false_positive = false_negative = side_correct = sides = 0
    for batch in data_loader:
        target = batch["features"].to(device)
        prediction = model(target)
        _, metrics = reconstruction_loss(prediction, target)
        for key, value in metrics.items():
            totals[key] += value * len(target)
        code_counts += torch.bincount(
            prediction["codes"].cpu().flatten(), minlength=model.config.codebook_size
        )
        predicted_presence = torch.sigmoid(prediction["presence_logits"]) >= threshold
        target_presence = target[..., 3] > 0
        true_positive += int((predicted_presence & target_presence).sum())
        false_positive += int((predicted_presence & ~target_presence).sum())
        false_negative += int((~predicted_presence & target_presence).sum())
        predicted_side = torch.stack((
            predicted_presence[..., :21].any(dim=(1, 2)),
            predicted_presence[..., 21:42].any(dim=(1, 2)),
        ), dim=1)
        target_side = torch.stack((
            target_presence[..., :21].any(dim=(1, 2)),
            target_presence[..., 21:42].any(dim=(1, 2)),
        ), dim=1)
        side_correct += int((predicted_side == target_side).sum())
        sides += target_side.numel()
        totals["items"] += len(target)
    items = totals.pop("items")
    result = {key: value / items for key, value in totals.items()}
    result.update({
        "presence_f1": 2 * true_positive / max(
            1, 2 * true_positive + false_positive + false_negative
        ),
        "hand_participation_accuracy": side_correct / max(1, sides),
        "active_codes": int((code_counts > 0).sum()),
        "items": int(items),
    })
    probabilities = code_counts.float() / code_counts.sum().clamp(min=1)
    result["aggregate_codebook_perplexity"] = float(torch.exp(
        -(probabilities * probabilities.clamp(min=1e-12).log()).sum()
    ))
    return result


@torch.inference_mode()
def motion_summary(model, dataset, threshold, device):
    real = Summary()
    reconstructed = Summary()
    for index in range(len(dataset)):
        target = torch.from_numpy(dataset[index]["features"])[None].to(device)
        prediction = model(target)
        observation = observation_from_prediction(prediction, threshold)[0].cpu().numpy()
        real.add(target[0].cpu().numpy())
        reconstructed.add(observation)
    return real.result(), reconstructed.result()


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    sources = sorted({str(row["source"]) for row in manifest["rows"]})
    source_to_index = {source: index for index, source in enumerate(sources)}
    holdouts = {sequence_key(value) for value in args.holdout_sequence}
    train = FullTrajectoryDataset(
        manifest, args.data_root, {"train"}, source_to_index,
        excluded_sequences=holdouts,
    )
    validation = FullTrajectoryDataset(
        manifest, args.data_root, {"validation"}, source_to_index,
        excluded_sequences=holdouts,
    )
    holdout = FullTrajectoryDataset(
        manifest, args.data_root, {"validation"}, source_to_index,
        only_sequences=holdouts, sources={"local_phrase_full"},
    )
    train_loader = loader(train, args.batch_size, balanced=True)
    validation_loader = loader(validation, args.batch_size)
    device = torch.device(args.device)
    config = TemporalMotionTokenizerV17Config(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        codebook_size=args.codebook_size,
        downsample_factor=args.downsample_factor,
    )
    model = TemporalMotionTokenizerV17(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = items = 0
        for batch in train_loader:
            target = batch["features"].to(device)
            prediction = model(target)
            loss, _ = reconstruction_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            items += len(target)
        metrics = evaluate(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": total / items, "validation": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["reconstruction_loss"] < best:
            best = metrics["reconstruction_loss"]
            stale = 0
            checkpoint = {
                "format": "slt_temporal_motion_tokenizer_v17",
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "model_config": config.to_dict(),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "manifest": args.manifest.as_posix(),
                "manifest_sha256": sha256(args.manifest),
                "holdout_sequences": [list(value) for value in sorted(holdouts)],
                "validation_metrics": metrics,
                "test_evaluated": False,
            }
            temporary = args.output / "model.pth.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "model.pth")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output / "model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    candidates = []
    for threshold in np.arange(0.25, 0.76, 0.05):
        metrics = evaluate(model, validation_loader, device, float(threshold))
        candidates.append((metrics["presence_f1"], metrics["hand_participation_accuracy"], float(threshold), metrics))
    _, _, threshold, validation_metrics = max(candidates)
    holdout_metrics = evaluate(model, loader(holdout, args.batch_size), device, threshold)
    real_motion, reconstruction_motion = motion_summary(model, holdout, threshold, device)
    ratios = {}
    for name in ("speed", "acceleration", "jerk"):
        reconstructed = reconstruction_motion["hand_motion"][name]["p95"]
        genuine = real_motion["hand_motion"][name]["p95"]
        ratios[name] = 0.0 if reconstructed is None else reconstructed / genuine
    gates = {
        "validation_presence_f1_at_least_0_90": validation_metrics["presence_f1"] >= 0.90,
        "holdout_presence_f1_at_least_0_90": holdout_metrics["presence_f1"] >= 0.90,
        "holdout_hand_participation_at_least_0_95": holdout_metrics["hand_participation_accuracy"] >= 0.95,
        "holdout_coordinate_below_0_04": holdout_metrics["coordinate"] < 0.04,
        "at_least_16_active_codes": validation_metrics["active_codes"] >= 16,
        **{
            f"holdout_{name}_within_0_5_to_2x": 0.5 <= ratio <= 2.0
            for name, ratio in ratios.items()
        },
    }
    result = {
        "format": "slt_temporal_motion_tokenizer_training_result_v17",
        "version": 1,
        "checkpoint": (args.output / "model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "model.pth"),
        "selected_epoch": checkpoint["epoch"],
        "parameters": sum(value.numel() for value in model.parameters()),
        "train_items": len(train),
        "validation_items": len(validation),
        "holdout_items": len(holdout),
        "selected_presence_threshold": threshold,
        "validation": validation_metrics,
        "holdout": holdout_metrics,
        "real_holdout_motion": real_motion,
        "reconstructed_holdout_motion": reconstruction_motion,
        "reconstructed_motion_p95_over_real": ratios,
        "gates": gates,
        "all_tokenizer_gates_passed": all(gates.values()),
        "temporal_prior_training_allowed": all(gates.values()),
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=Path("active/v17/full_trajectory_generation_manifest_v17.json"))
    value.add_argument("--data-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/temporal_motion_tokenizer_v17_v1"))
    value.add_argument("--holdout-sequence", action="append", default=["GOOD,MORNING", "TOMORROW,SCHOOL,GO"])
    value.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    value.add_argument("--epochs", type=int, default=40)
    value.add_argument("--patience", type=int, default=6)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--learning-rate", type=float, default=3e-4)
    value.add_argument("--hidden-dim", type=int, default=192)
    value.add_argument("--latent-dim", type=int, default=64)
    value.add_argument("--codebook-size", type=int, default=128)
    value.add_argument("--downsample-factor", type=int, choices=(2, 4), default=2)
    value.add_argument("--seed", type=int, default=1701)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
