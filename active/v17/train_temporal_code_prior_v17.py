#!/usr/bin/env python3
"""Train a gloss-conditioned autoregressive prior over genuine motion codes."""

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

from active.v17.model_temporal_code_prior_v17 import (
    TemporalCodePriorV17,
    TemporalCodePriorV17Config,
)
from active.v17.model_temporal_motion_tokenizer_v17 import (
    TemporalMotionTokenizerV17,
    TemporalMotionTokenizerV17Config,
)
from active.v17.train_full_trajectory_v17 import (
    FullTrajectoryDataset,
    loader,
    sequence_key,
)
from active.v17.train_signing_voice_v17 import sha256


def load_tokenizer(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_temporal_motion_tokenizer_v17":
        raise ValueError("unexpected tokenizer checkpoint")
    model = TemporalMotionTokenizerV17(
        TemporalMotionTokenizerV17Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval().requires_grad_(False).to(device), checkpoint


def target_sides(features, factor):
    present = features[..., 3] > 0
    batch, frames, nodes = present.shape
    blocked = present.reshape(batch, frames // factor, factor, nodes)
    return torch.stack((
        blocked[..., :21].any(dim=(2, 3)),
        blocked[..., 21:42].any(dim=(2, 3)),
    ), dim=-1).float()


def prior_loss(prediction, codes, sides, duration):
    code = F.cross_entropy(
        prediction["code_logits"].flatten(0, 1), codes.flatten(),
        label_smoothing=0.02,
    )
    side_error = F.binary_cross_entropy_with_logits(
        prediction["side_logits"], sides, reduction="none"
    )
    side = (side_error * torch.where(sides > 0, 1.0, 3.0)).mean()
    duration_loss = F.smooth_l1_loss(prediction["log_duration"], duration.log())
    total = code + side + 0.2 * duration_loss
    metrics = {
        "loss": float(total.detach()),
        "code_cross_entropy": float(code.detach()),
        "side_loss": float(side.detach()),
        "duration": float(duration_loss.detach()),
    }
    if "text_code_logits" in prediction:
        text_code = F.cross_entropy(
            prediction["text_code_logits"].flatten(0, 1), codes.flatten(),
            label_smoothing=0.02,
        )
        text_side_error = F.binary_cross_entropy_with_logits(
            prediction["text_side_logits"], sides, reduction="none"
        )
        text_side = (
            text_side_error * torch.where(sides > 0, 1.0, 3.0)
        ).mean()
        weight = prediction.get("text_anchor_loss_weight", 0.5)
        total = total + weight * (text_code + text_side)
        metrics.update({
            "loss": float(total.detach()),
            "text_code_cross_entropy": float(text_code.detach()),
            "text_side_loss": float(text_side.detach()),
        })
    return total, metrics


@torch.inference_mode()
def evaluate(model, tokenizer, data_loader, device, side_threshold=0.5):
    model.eval()
    totals: Counter[str] = Counter()
    code_correct = code_total = side_tp = side_fp = side_fn = 0
    exact_codes = exact_sides = 0
    factor = tokenizer.config.downsample_factor
    for batch in data_loader:
        features = batch["features"].to(device)
        codes = tokenizer.encode_codes(features)
        sides = target_sides(features, factor)
        prediction = model(
            batch["tokens"].to(device), batch["token_valid"].to(device),
            batch["source_id"].to(device), codes,
        )
        _, metrics = prior_loss(
            prediction, codes, sides, batch["duration"].float().to(device)
        )
        for key, value in metrics.items():
            totals[key] += value * len(features)
        predicted_codes = prediction["code_logits"].argmax(dim=-1)
        predicted_sides = torch.sigmoid(prediction["side_logits"]) >= side_threshold
        true_sides = sides > 0
        code_correct += int((predicted_codes == codes).sum())
        code_total += codes.numel()
        side_tp += int((predicted_sides & true_sides).sum())
        side_fp += int((predicted_sides & ~true_sides).sum())
        side_fn += int((~predicted_sides & true_sides).sum())
        exact_codes += int((predicted_codes == codes).all(dim=1).sum())
        exact_sides += int((predicted_sides == true_sides).all(dim=(1, 2)).sum())
        totals["items"] += len(features)
    items = totals.pop("items")
    result = {key: value / items for key, value in totals.items()}
    result.update({
        "code_top1": code_correct / max(1, code_total),
        "exact_code_sequence": exact_codes / max(1, items),
        "side_f1": 2 * side_tp / max(1, 2 * side_tp + side_fp + side_fn),
        "exact_side_timeline": exact_sides / max(1, items),
        "items": int(items),
    })
    return result


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    tokenizer, tokenizer_checkpoint = load_tokenizer(args.tokenizer, device)
    tokenizer_result = json.loads((args.tokenizer.parent / "result.json").read_text())
    if not tokenizer_result.get("all_tokenizer_gates_passed"):
        raise ValueError("tokenizer reconstruction gates did not pass")
    manifest = json.loads(args.manifest.read_text())
    holdouts = {sequence_key(value) for value in args.holdout_sequence}
    sources = sorted({str(row["source"]) for row in manifest["rows"]})
    source_to_index = {source: index for index, source in enumerate(sources)}
    train = FullTrajectoryDataset(
        manifest, args.data_root, {"train"}, source_to_index,
        excluded_sequences=holdouts,
    )
    validation = FullTrajectoryDataset(
        manifest, args.data_root, {"validation"}, source_to_index,
        excluded_sequences=holdouts,
    )
    train_loader = loader(train, args.batch_size, balanced=True)
    validation_loader = loader(validation, args.batch_size)
    config = TemporalCodePriorV17Config(
        vocabulary_size=len(manifest["token_to_index"]),
        source_families=len(sources),
        codebook_size=tokenizer.config.codebook_size,
        code_steps=tokenizer.code_steps,
        maximum_tokens=int(manifest["maximum_tokens"]) + 2,
        model_dim=args.model_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        text_anchor_loss_weight=args.text_anchor_loss_weight,
        text_guidance_scale=args.text_guidance_scale,
    )
    model = TemporalCodePriorV17(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history = []
    started = time.monotonic()
    factor = tokenizer.config.downsample_factor
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = items = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            with torch.inference_mode():
                codes = tokenizer.encode_codes(features)
                sides = target_sides(features, factor)
            codes = codes.clone()
            sides = sides.clone()
            prediction = model(
                batch["tokens"].to(device), batch["token_valid"].to(device),
                batch["source_id"].to(device), codes,
            )
            loss, _ = prior_loss(
                prediction, codes, sides, batch["duration"].float().to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
            items += len(features)
        metrics = evaluate(model, tokenizer, validation_loader, device)
        row = {"epoch": epoch, "train_loss": total / items, "validation": metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["loss"] < best:
            best = metrics["loss"]
            stale = 0
            checkpoint = {
                "format": "slt_temporal_code_prior_v17",
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "model_config": config.to_dict(),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "manifest": args.manifest.as_posix(),
                "manifest_sha256": sha256(args.manifest),
                "tokenizer": args.tokenizer.as_posix(),
                "tokenizer_sha256": sha256(args.tokenizer),
                "token_to_index": manifest["token_to_index"],
                "source_to_index": source_to_index,
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
    result = {
        "format": "slt_temporal_code_prior_training_result_v17",
        "version": 1,
        "checkpoint": (args.output / "model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "model.pth"),
        "tokenizer_sha256": sha256(args.tokenizer),
        "parameters": sum(value.numel() for value in model.parameters()),
        "train_items": len(train),
        "validation_items": len(validation),
        "selected_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation_metrics"],
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "claim_boundary": "Teacher-forced code accuracy does not authorize rendering; autoregressive motion gates run separately.",
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=Path("active/v17/full_trajectory_generation_manifest_v17.json"))
    value.add_argument("--data-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--tokenizer", type=Path, default=Path("artifacts/models/temporal_motion_tokenizer_v17_v2/model.pth"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/temporal_code_prior_v17_v1"))
    value.add_argument("--holdout-sequence", action="append", default=["GOOD,MORNING", "TOMORROW,SCHOOL,GO"])
    value.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    value.add_argument("--epochs", type=int, default=40)
    value.add_argument("--patience", type=int, default=7)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--learning-rate", type=float, default=3e-4)
    value.add_argument("--model-dim", type=int, default=128)
    value.add_argument("--encoder-layers", type=int, default=2)
    value.add_argument("--decoder-layers", type=int, default=3)
    value.add_argument("--text-anchor-loss-weight", type=float, default=0.0)
    value.add_argument("--text-guidance-scale", type=float, default=1.0)
    value.add_argument("--seed", type=int, default=1701)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
