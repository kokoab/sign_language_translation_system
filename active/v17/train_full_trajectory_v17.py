#!/usr/bin/env python3
"""Train the first gloss-conditioned whole-utterance landmark model."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from active.v17.model_full_trajectory_v17 import (
    FullTrajectoryGeneratorV17,
    FullTrajectoryV17Config,
)
from active.v17.train_signing_voice_v17 import sha256
from scripts.extract_full_trajectory_landmarks_v17 import destination_for


def sequence_key(value: str) -> tuple[str, ...]:
    return tuple(part.strip().upper() for part in value.replace("_", ",").split(",") if part.strip())


class FullTrajectoryDataset(Dataset):
    def __init__(
        self, manifest: dict[str, object], root: Path, roles: set[str],
        source_to_index: dict[str, int], *,
        excluded_sequences: set[tuple[str, ...]] | None = None,
        only_sequences: set[tuple[str, ...]] | None = None,
        sources: set[str] | None = None,
    ):
        excluded_sequences = excluded_sequences or set()
        rows = []
        for row in manifest["rows"]:
            key = tuple(row["target_sequence"])
            if (
                row["role"] not in roles
                or key in excluded_sequences
                or (only_sequences is not None and key not in only_sequences)
                or (sources is not None and row["source"] not in sources)
            ):
                continue
            archive = destination_for(root, row)
            if not archive.is_file():
                raise FileNotFoundError(archive)
            rows.append((row, archive))
        if not rows:
            raise ValueError("full-trajectory dataset selection is empty")
        self.rows = rows
        self.source_to_index = source_to_index
        self.maximum_tokens = int(manifest["maximum_tokens"]) + 2
        self.bos = int(manifest["token_to_index"]["<BOS>"])
        self.eos = int(manifest["token_to_index"]["<EOS>"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row, archive = self.rows[index]
        with np.load(archive, allow_pickle=False) as payload:
            features = payload["observation_features"].astype(np.float32)
            metadata = json.loads(str(payload["metadata_json"]))
        token_ids = [self.bos, *map(int, row["target_token_ids"]), self.eos]
        tokens = np.zeros(self.maximum_tokens, dtype=np.int64)
        valid = np.zeros(self.maximum_tokens, dtype=np.bool_)
        tokens[:len(token_ids)] = token_ids
        valid[:len(token_ids)] = True
        duration = float(metadata["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"invalid duration in {archive}")
        return {
            "features": features,
            "tokens": tokens,
            "token_valid": valid,
            "source_id": self.source_to_index[str(row["source"])],
            "duration": duration,
            "source": str(row["source"]),
            "sequence": " ".join(row["target_sequence"]),
        }


def trajectory_loss(
    prediction: dict[str, torch.Tensor], target: torch.Tensor,
    duration: torch.Tensor, kl_weight: float = 0.0,
    coordinate_weight: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    present = target[..., 3] > 0
    node_weight = torch.ones(target.shape[-2], device=target.device)
    node_weight[:42] = 2.0
    weight = present * node_weight[None, None]
    coordinate = F.smooth_l1_loss(
        prediction["xyz"], target[..., :3], reduction="none"
    ).mean(dim=-1)
    coordinate = (coordinate * weight).sum() / weight.sum().clamp(min=1)
    presence_error = F.binary_cross_entropy_with_logits(
        prediction["presence_logits"], present.float(), reduction="none"
    )
    presence = (
        presence_error * torch.where(present, 1.0, 1.5)
    ).mean()
    predicted_presence_probability = torch.sigmoid(prediction["presence_logits"])
    presence_motion = F.smooth_l1_loss(
        predicted_presence_probability[:, 1:] - predicted_presence_probability[:, :-1],
        present[:, 1:].float() - present[:, :-1].float(),
    )
    confidence = F.smooth_l1_loss(
        torch.sigmoid(prediction["confidence_logits"]), target[..., 4], reduction="none"
    )
    confidence = (confidence * weight).sum() / weight.sum().clamp(min=1)

    common = present[:, 1:] & present[:, :-1]
    velocity_error = F.smooth_l1_loss(
        prediction["xyz"][:, 1:] - prediction["xyz"][:, :-1],
        target[:, 1:, :, :3] - target[:, :-1, :, :3], reduction="none",
    ).mean(dim=-1)
    velocity_weight = common * node_weight[None, None]
    velocity = (
        (velocity_error * velocity_weight).sum()
        / velocity_weight.sum().clamp(min=1)
    )
    common3 = present[:, 2:] & present[:, 1:-1] & present[:, :-2]
    prediction_acceleration = (
        prediction["xyz"][:, 2:] - 2 * prediction["xyz"][:, 1:-1]
        + prediction["xyz"][:, :-2]
    )
    target_acceleration = (
        target[:, 2:, :, :3] - 2 * target[:, 1:-1, :, :3]
        + target[:, :-2, :, :3]
    )
    acceleration_error = F.smooth_l1_loss(
        prediction_acceleration, target_acceleration, reduction="none"
    ).mean(dim=-1)
    acceleration_weight = common3 * node_weight[None, None]
    acceleration = (
        (acceleration_error * acceleration_weight).sum()
        / acceleration_weight.sum().clamp(min=1)
    )

    def motion_scale_error(order: int) -> torch.Tensor:
        predicted_delta = torch.diff(prediction["xyz"][..., :42, :], n=order, dim=1)
        target_delta = torch.diff(target[..., :42, :3], n=order, dim=1)
        valid = torch.ones(
            predicted_delta.shape[:-1], dtype=torch.bool, device=target.device
        )
        hand_present = present[..., :42]
        for offset in range(order + 1):
            valid &= hand_present[:, offset:offset + predicted_delta.shape[1]]
        predicted_speed = predicted_delta.norm(dim=-1)
        target_speed = target_delta.norm(dim=-1)
        count = valid.sum(dim=(1, 2)).clamp(min=1)
        predicted_mean = (predicted_speed * valid).sum(dim=(1, 2)) / count
        target_mean = (target_speed * valid).sum(dim=(1, 2)) / count
        usable = valid.any(dim=(1, 2))
        return torch.abs(predicted_mean[usable] - target_mean[usable]).mean()

    speed_scale = motion_scale_error(1)
    acceleration_scale = motion_scale_error(2)
    jerk_scale = motion_scale_error(3)
    predicted_sides = torch.stack((
        prediction["presence_logits"][..., :21].amax(dim=(1, 2)),
        prediction["presence_logits"][..., 21:42].amax(dim=(1, 2)),
    ), dim=1)
    target_sides = torch.stack((
        present[..., :21].any(dim=(1, 2)), present[..., 21:42].any(dim=(1, 2)),
    ), dim=1).float()
    participation = F.binary_cross_entropy_with_logits(predicted_sides, target_sides)
    duration_loss = F.smooth_l1_loss(prediction["log_duration"], duration.log())
    kl = target.new_zeros(())
    if "motion_mean" in prediction:
        kl = -0.5 * torch.mean(
            1 + prediction["motion_log_variance"]
            - prediction["motion_mean"].square()
            - prediction["motion_log_variance"].exp()
        )
    total = (
        coordinate_weight * coordinate + 2.0 * presence + 0.5 * presence_motion
        + 0.2 * confidence + velocity + 0.3 * acceleration
        + 2.0 * speed_scale + acceleration_scale + 0.5 * jerk_scale
        + participation + 0.2 * duration_loss + kl_weight * kl
    )
    return total, {
        "loss": float(total.detach()),
        "coordinate": float(coordinate.detach()),
        "presence": float(presence.detach()),
        "presence_motion": float(presence_motion.detach()),
        "confidence": float(confidence.detach()),
        "velocity": float(velocity.detach()),
        "acceleration": float(acceleration.detach()),
        "speed_scale": float(speed_scale.detach()),
        "acceleration_scale": float(acceleration_scale.detach()),
        "jerk_scale": float(jerk_scale.detach()),
        "participation": float(participation.detach()),
        "duration": float(duration_loss.detach()),
        "kl": float(kl.detach()),
    }


@torch.inference_mode()
def evaluate(
    model, loader, device, presence_threshold: float = 0.5,
    use_posterior: bool = False,
) -> dict[str, float]:
    model.eval()
    totals: Counter[str] = Counter()
    true_positive = false_positive = false_negative = side_correct = sides = 0
    for batch in loader:
        target = batch["features"].to(device)
        prediction = model(
            batch["tokens"].to(device), batch["token_valid"].to(device),
            batch["source_id"].to(device),
            target_features=target if use_posterior else None,
        )
        _, metrics = trajectory_loss(
            prediction, target, batch["duration"].float().to(device)
        )
        for key, value in metrics.items():
            totals[key] += value * len(target)
        predicted_presence = (
            torch.sigmoid(prediction["presence_logits"]) >= presence_threshold
        )
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
    denominator = totals["items"]
    output = {key: value / denominator for key, value in totals.items() if key != "items"}
    output["presence_f1"] = 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)
    output["hand_participation_accuracy"] = side_correct / max(1, sides)
    output["items"] = int(denominator)
    return output


def loader(dataset, batch_size, shuffle=False, balanced=False):
    sampler = None
    if balanced:
        counts = Counter(row[0]["source"] for row in dataset.rows)
        weights = [1.0 / counts[row[0]["source"]] for row in dataset.rows]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle and sampler is None,
        sampler=sampler, num_workers=0,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest = json.loads(args.manifest.read_text())
    holdouts = {sequence_key(value) for value in args.holdout_sequence}
    enabled_sources = set(args.sources) if args.sources else None
    sources = sorted({
        str(row["source"]) for row in manifest["rows"]
        if enabled_sources is None or row["source"] in enabled_sources
    })
    source_to_index = {source: index for index, source in enumerate(sources)}
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
    ) if holdouts and "local_phrase_full" in source_to_index else None
    train_loader = loader(train, args.batch_size, balanced=True)
    validation_loader = loader(validation, args.batch_size)
    holdout_loader = loader(holdout, args.batch_size) if holdout else None
    config = FullTrajectoryV17Config(
        vocabulary_size=len(manifest["token_to_index"]),
        source_families=len(sources),
        maximum_tokens=int(manifest["maximum_tokens"]) + 2,
        model_dim=args.model_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        motion_latent_dim=args.motion_latent_dim,
    )
    device = torch.device(args.device)
    model = FullTrajectoryGeneratorV17(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        items = 0
        for batch in train_loader:
            target = batch["features"].to(device)
            prediction = model(
                batch["tokens"].to(device), batch["token_valid"].to(device),
                batch["source_id"].to(device),
                target_features=target, sample_latent=True,
            )
            loss, _ = trajectory_loss(
                prediction, target, batch["duration"].float().to(device),
                args.kl_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(target)
            items += len(target)
        validation_metrics = evaluate(
            model, validation_loader, device, use_posterior=True
        )
        validation_prior = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_total / items,
            "validation": validation_metrics,
            "validation_prior": validation_prior,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_prior["loss"] < best:
            best = validation_prior["loss"]
            stale = 0
            checkpoint = {
                "format": "slt_full_trajectory_generator_v17",
                "version": 2,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "model_config": config.to_dict(),
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "manifest": args.manifest.as_posix(),
                "manifest_sha256": sha256(args.manifest),
                "token_to_index": manifest["token_to_index"],
                "source_to_index": source_to_index,
                "holdout_sequences": [list(value) for value in sorted(holdouts)],
                "validation_metrics": validation_metrics,
                "validation_prior_metrics": validation_prior,
                "selection_metric": "validation_prior.loss",
                "test_evaluated": False,
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "local_test_accessed": False,
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
    model.to(device)
    holdout_metrics = evaluate(model, holdout_loader, device) if holdout_loader else None
    holdout_posterior = (
        evaluate(model, holdout_loader, device, use_posterior=True)
        if holdout_loader else None
    )
    report = {
        "format": "slt_full_trajectory_training_result_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": (args.output / "model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "model.pth"),
        "parameters": sum(value.numel() for value in model.parameters()),
        "train_items": len(train),
        "validation_items": len(validation),
        "holdout_items": len(holdout) if holdout else 0,
        "holdout_sequences": [list(value) for value in sorted(holdouts)],
        "selected_epoch": checkpoint["epoch"],
        "selection_metric": checkpoint["selection_metric"],
        "validation": checkpoint["validation_metrics"],
        "validation_prior": checkpoint["validation_prior_metrics"],
        "compositional_holdout": holdout_metrics,
        "compositional_holdout_posterior": holdout_posterior,
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": (
            "A held-out combination reconstruction metric is not native naturalness; "
            "generated rendering remains gated separately."
        ),
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=Path("active/v17/full_trajectory_generation_manifest_v17.json"))
    value.add_argument("--data-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/full_trajectory_generator_v17_holdout_v1"))
    value.add_argument(
        "--holdout-sequence", action="append",
        default=["GOOD,MORNING", "TOMORROW,SCHOOL,GO"],
    )
    value.add_argument("--sources", nargs="*", default=[])
    value.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    value.add_argument("--epochs", type=int, default=40)
    value.add_argument("--patience", type=int, default=7)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--learning-rate", type=float, default=3e-4)
    value.add_argument("--model-dim", type=int, default=128)
    value.add_argument("--encoder-layers", type=int, default=2)
    value.add_argument("--decoder-layers", type=int, default=3)
    value.add_argument("--motion-latent-dim", type=int, default=32)
    value.add_argument("--kl-weight", type=float, default=1e-4)
    value.add_argument("--seed", type=int, default=1701)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
