#!/usr/bin/env python3
"""Train a bounded stochastic residual transition model around a frozen mean."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import copy
import gc
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_diffusion_v17 import (
    TransitionResidualDiffusionV17,
    TransitionResidualDiffusionV17Config,
)
from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
)
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    discover_signers,
    landmark_tree_fingerprint,
    sha256,
)


LOG = logging.getLogger("train_transition_diffusion_v17")


def fixed_mask(index: int, seed: int = 5701) -> torch.Tensor:
    rng = random.Random(seed + index * 104729)
    length = rng.randint(4, 12)
    start = rng.randint(3, 32 - length - 3)
    mask = torch.zeros(32, dtype=torch.bool)
    mask[start:start + length] = True
    return mask


def load_mean_model(path: Path, device: torch.device) -> TransitionInpainterV17:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_transition_inpainter_v17":
        raise ValueError("unexpected deterministic mean checkpoint")
    model = TransitionInpainterV17(
        TransitionInpainterV17Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().requires_grad_(False)
    return model.to(device)


@torch.inference_mode()
def estimate_residual_scale(
    dataset: TransitionWindowDataset,
    mean_model: TransitionInpainterV17,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    sum_square = torch.zeros(61, 3, dtype=torch.float64)
    count = torch.zeros(61, 1, dtype=torch.float64)
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        features = torch.stack([
            torch.from_numpy(dataset.preloaded_features[index].astype(np.float32))
            for index in range(start, stop)
        ]).to(device)
        mask = torch.stack([fixed_mask(index) for index in range(start, stop)]).to(device)
        mean = mean_model(features, mask)
        residual = features[..., :3] - mean[..., :3]
        valid = (mask[:, :, None] & (features[..., 3] > 0)).to(residual.dtype)
        sum_square += (residual.square() * valid[..., None]).sum(
            dim=(0, 1)
        ).cpu().double()
        count += valid.sum(dim=(0, 1)).cpu().double()[:, None]
    scale = torch.sqrt(sum_square / count.clamp_min(1.0)).float().clamp_min(0.01)
    return scale


def denoising_batch(
    diffusion: TransitionResidualDiffusionV17,
    mean_model: TransitionInpainterV17,
    features: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        mean = mean_model(features, mask)
    clean = (features[..., :3] - mean[..., :3]) / scale[None, None]
    clean = clean * mask[:, :, None, None]
    if noise is None:
        noise = torch.randn_like(clean)
    noise = noise * mask[:, :, None, None]
    if timesteps is None:
        timesteps = torch.randint(
            0, diffusion.config.timesteps, (len(features),), device=features.device
        )
    noisy = diffusion.q_sample(clean, noise, timesteps)
    predicted_noise = diffusion(mean, mask, noisy, timesteps)
    valid = (
        mask[:, :, None] & (features[..., 3] > 0)
    )[..., None].expand_as(clean)
    noise_loss = F.mse_loss(
        predicted_noise[valid], noise[valid]
    )
    alpha_bar = diffusion.alpha_bars[timesteps].to(clean.dtype)[:, None, None, None]
    predicted_clean = (
        noisy - (1.0 - alpha_bar).sqrt() * predicted_noise
    ) / alpha_bar.sqrt()
    clean_loss = F.smooth_l1_loss(predicted_clean[valid], clean[valid])
    loss = noise_loss + 0.10 * clean_loss
    return loss, {
        "noise_loss": float(noise_loss.detach()),
        "clean_loss": float(clean_loss.detach()),
    }


@torch.inference_mode()
def evaluate(
    diffusion: TransitionResidualDiffusionV17,
    mean_model: TransitionInpainterV17,
    loader: DataLoader,
    scale: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    diffusion.eval()
    generator = torch.Generator().manual_seed(8701)
    totals = {"loss": 0.0, "noise_loss": 0.0, "clean_loss": 0.0}
    rows = 0
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        shape = features.shape[:-1] + (3,)
        noise = torch.randn(shape, generator=generator).to(device)
        timesteps = torch.randint(
            0, diffusion.config.timesteps, (len(features),), generator=generator
        ).to(device)
        loss, terms = denoising_batch(
            diffusion, mean_model, features, mask, scale,
            noise=noise, timesteps=timesteps,
        )
        count = len(features)
        rows += count
        totals["loss"] += float(loss) * count
        totals["noise_loss"] += terms["noise_loss"] * count
        totals["clean_loss"] += terms["clean_loss"] * count
    return {"windows": rows, **{key: value / rows for key, value in totals.items()}}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    all_signers = discover_signers(args.landmark_root)
    if args.held_out_signer not in all_signers:
        raise ValueError("held-out signer is absent")
    train_signers = {
        signer for signer in all_signers - {args.held_out_signer}
        if not any(signer.startswith(prefix) for prefix in args.exclude_train_signer_prefix)
    }
    train_dataset = TransitionWindowDataset(
        args.landmark_root, train_signers, seed=1701, fixed_masks=False
    )
    validation_dataset = TransitionWindowDataset(
        args.landmark_root, {args.held_out_signer}, seed=4701, fixed_masks=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    mean_model = load_mean_model(args.mean_checkpoint, device)
    scale = estimate_residual_scale(
        train_dataset, mean_model, device, args.batch_size
    ).to(device)
    archive_count, archive_fingerprint = landmark_tree_fingerprint(args.landmark_root)
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    started = time.monotonic()
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        diffusion = TransitionResidualDiffusionV17(
            TransitionResidualDiffusionV17Config(
                dim=args.dim, depth=args.depth, heads=args.heads,
                dropout=args.dropout, timesteps=args.timesteps,
            )
        ).to(device)
        optimizer = torch.optim.AdamW(
            diffusion.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
            generator=torch.Generator().manual_seed(seed),
        )
        best_metrics = evaluate(
            diffusion, mean_model, validation_loader, scale, device
        )
        best_epoch = 0
        best_state = copy.deepcopy(diffusion.state_dict())
        patience = 0
        history = [{"epoch": 0, "validation": best_metrics}]
        for epoch in range(1, args.epochs + 1):
            diffusion.train()
            total = 0.0
            seen = 0
            for batch in loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = denoising_batch(
                    diffusion, mean_model, features, mask, scale
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite diffusion loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    diffusion.parameters(), args.gradient_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                total += float(loss.detach()) * len(features)
                seen += len(features)
            metrics = evaluate(
                diffusion, mean_model, validation_loader, scale, device
            )
            history.append({
                "epoch": epoch, "train_loss": total / seen,
                "validation": metrics,
            })
            if metrics["loss"] < best_metrics["loss"]:
                best_metrics = metrics
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in diffusion.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
            LOG.info(
                "seed=%d epoch=%d train=%.6f heldout=%.6f best=%d patience=%d",
                seed, epoch, total / seen, metrics["loss"], best_epoch, patience,
            )
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            if patience >= args.patience:
                break
        seed_root = args.output / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format": "slt_transition_residual_diffusion_v17",
            "version": 1,
            "model_config": diffusion.config.to_dict(),
            "model_state_dict": best_state,
            "residual_scale": scale.cpu(),
            "mean_checkpoint": args.mean_checkpoint.as_posix(),
            "mean_checkpoint_sha256": sha256(args.mean_checkpoint),
            "held_out_signer": args.held_out_signer,
            "train_signers": sorted(train_signers),
            "seed": seed,
            "epoch": best_epoch,
            "validation_metrics": best_metrics,
            "landmark_archive_count": archive_count,
            "landmark_tree_sha256": archive_fingerprint,
            "test_evaluated": False,
            "how2sign_validation_accessed": False,
            "how2sign_test_accessed": False,
        }
        torch.save(checkpoint, seed_root / "best_model.pth")
        (seed_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        candidates.append({
            "seed": seed, "best_epoch": best_epoch, "metrics": best_metrics,
            "checkpoint": (seed_root / "best_model.pth").as_posix(),
        })
    winner = min(candidates, key=lambda row: row["metrics"]["loss"])
    selected = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    torch.save(selected, args.output / "best_model.pth")
    report = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "metrics": winner["metrics"],
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "mean_checkpoint": args.mean_checkpoint.as_posix(),
        "mean_checkpoint_sha256": sha256(args.mean_checkpoint),
        "held_out_signer": args.held_out_signer,
        "train_signers": sorted(train_signers),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "candidates": candidates,
        "seconds": time.monotonic() - started,
        "claim_boundary": "stochastic landmark inpainting is not human-perceptual naturalness",
        "test_evaluated": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--landmark-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--mean-checkpoint", type=Path,
        default=Path("artifacts/models/transition_inpainter_residual_v17_full_h8_how2sign_only/best_model.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/transition_residual_diffusion_v17_h8_pilot"),
    )
    parser.add_argument("--held-out-signer", default="how2sign:8")
    parser.add_argument("--exclude-train-signer-prefix", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=(11701,))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
