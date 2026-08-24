#!/usr/bin/env python3
"""Pretrain the part-wise v17 encoder by reconstructing masked anatomical spans.

Only approved Citizen-train and SemLex-train landmarks are loaded. Left hand, right
hand, face, and body spans are masked independently; validation and test are absent.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
    from active.v17.schema_v17 import NUM_CHANNELS, NUM_NODES, V17Config, schema_fingerprint
    from active.v17.train_stage_1_v17 import (
        Citizen100V17Dataset,
        SemLexSupplementV17Dataset,
        sha256_file,
    )
else:
    from .model_v17 import SLTStage1V17, Stage1V17Config
    from .schema_v17 import NUM_CHANNELS, NUM_NODES, V17Config, schema_fingerprint
    from .train_stage_1_v17 import (
        Citizen100V17Dataset,
        SemLexSupplementV17Dataset,
        sha256_file,
    )


PARTS = ((0, 21), (21, 42), (42, 57), (57, 61))
ENCODER_PREFIXES = ("part_temporal_encoder.", "position", "blocks.")


def anatomical_span_mask(
    features: torch.Tensor,
    *,
    mask_ratio: float,
    span_length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a [B,T,61] mask with independently sampled part-level spans."""
    if features.ndim != 4 or features.shape[-2:] != (NUM_NODES, NUM_CHANNELS):
        raise ValueError(f"expected [B,T,{NUM_NODES},{NUM_CHANNELS}]")
    if not 0.0 < mask_ratio < 1.0 or span_length < 1:
        raise ValueError("mask_ratio must be in (0,1) and span_length positive")
    batch, frames = features.shape[:2]
    probability = min(1.0, mask_ratio / span_length)
    starts = torch.rand(
        batch, frames, len(PARTS), device=features.device, generator=generator
    ) < probability
    # Every sample/part must contribute a reconstruction target.
    missing = ~starts.any(dim=1)
    if missing.any():
        fallback = torch.randint(
            frames, (batch, len(PARTS)), device=features.device, generator=generator
        )
        row, part = torch.where(missing)
        starts[row, fallback[row, part], part] = True
    part_mask = torch.zeros_like(starts)
    for offset in range(span_length):
        if offset == 0:
            part_mask |= starts
        else:
            part_mask[:, offset:] |= starts[:, :-offset]
    node_mask = torch.zeros(
        batch, frames, NUM_NODES, dtype=torch.bool, device=features.device
    )
    for part, (start, end) in enumerate(PARTS):
        node_mask[:, :, start:end] = part_mask[:, :, part].unsqueeze(-1)
    return node_mask


def reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, masked_nodes: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    present = target[..., 3] > 0.5
    masked_present = masked_nodes & present
    if not masked_nodes.any() or not masked_present.any():
        raise RuntimeError("masked pose batch has no reconstruction targets")
    xyz = F.smooth_l1_loss(
        prediction[..., :3][masked_present], target[..., :3][masked_present]
    )
    confidence = F.mse_loss(
        prediction[..., 4][masked_present], target[..., 4][masked_present]
    )
    presence = F.binary_cross_entropy_with_logits(
        prediction[..., 3][masked_nodes], target[..., 3][masked_nodes]
    )
    loss = xyz + 0.25 * confidence + 0.25 * presence
    return loss, {
        "xyz": float(xyz.detach()),
        "confidence": float(confidence.detach()),
        "presence": float(presence.detach()),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("epochs must be positive and batch size at least two")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = bool(args.amp and device.type == "cuda")
    expected_schema = schema_fingerprint(V17Config())
    citizen = Citizen100V17Dataset(
        args.data_root, "train", args.manifest, args.rejections,
        cache=False, expected_schema=expected_schema,
    )
    semlex = SemLexSupplementV17Dataset(
        args.supplement_root,
        args.supplement_manifest,
        citizen.label_to_index,
        cache=False,
        expected_schema=expected_schema,
    )
    dataset = ConcatDataset((citizen, semlex))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    config = Stage1V17Config(
        num_classes=citizen.num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        drop_path=args.drop_path,
        temporal_encoder="partwise_global",
        part_depth=args.part_depth,
    )
    model = SLTStage1V17(config).to(device)
    decoder = nn.Linear(config.dim, NUM_NODES * NUM_CHANNELS).to(device)
    encoder_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(ENCODER_PREFIXES)
    ]
    optimizer = torch.optim.AdamW(
        [*encoder_parameters, *decoder.parameters()],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(loader)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.02
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mask_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        decoder.train()
        sums = {"loss": 0.0, "xyz": 0.0, "confidence": 0.0, "presence": 0.0}
        samples = 0
        for features, _ in loader:
            if step >= total_steps:
                break
            features = features.to(device, non_blocking=device.type == "cuda").float()
            masked_nodes = anatomical_span_mask(
                features,
                mask_ratio=args.mask_ratio,
                span_length=args.span_length,
                generator=mask_generator,
            )
            masked = features.clone()
            masked[masked_nodes] = 0.0
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with context:
                encoded, _ = model.encode(masked)
                prediction = decoder(encoded).reshape_as(features)
                loss, pieces = reconstruction_loss(prediction, features, masked_nodes)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([*encoder_parameters, *decoder.parameters()], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch = len(features)
            samples += batch
            sums["loss"] += float(loss.detach()) * batch
            for name, value in pieces.items():
                sums[name] += value * batch
            step += 1
        row = {
            "epoch": float(epoch),
            **{name: value / max(samples, 1) for name, value in sums.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
            "steps": float(step),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if step >= total_steps:
            break

    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(ENCODER_PREFIXES)
    }
    checkpoint = {
        "format": "slt_v17_masked_pose_pretrain",
        "encoder_state_dict": state,
        "model_config": config.to_dict(),
        "manifest_sha256": sha256_file(Path(args.manifest)),
        "supplement_manifest_sha256": sha256_file(Path(args.supplement_manifest)),
        "schema_fingerprint": expected_schema,
        "epochs": len(history),
        "steps": step,
        "objective": (
            "independent_part_span_masked_xyz_presence_confidence_reconstruction"
        ),
        "mask_ratio": args.mask_ratio,
        "span_length": args.span_length,
        "citizen_train_clips": len(citizen),
        "semlex_train_clips": len(semlex),
        "validation_accessed": False,
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "masked_pose_pretrained.pth")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = {key: value for key, value in checkpoint.items() if key != "encoder_state_dict"}
    result["encoder_parameters"] = sum(parameter.numel() for parameter in encoder_parameters)
    result["decoder_parameters"] = sum(parameter.numel() for parameter in decoder.parameters())
    result["final_loss"] = history[-1]["loss"]
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--supplement-root", type=Path, default=Path("data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"))
    parser.add_argument("--supplement-manifest", type=Path, default=Path("data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mask-ratio", type=float, default=0.35)
    parser.add_argument("--span-length", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--part-depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--drop-path", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
