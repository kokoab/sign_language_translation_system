#!/usr/bin/env python3
"""Train and evaluate a genuine multi-signer masked-transition inpainter."""

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
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
    interpolate_masked_context,
)


LOG = logging.getLogger("train_transition_inpainter_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_signers(root: Path) -> set[str]:
    signers = set()
    for path in root.glob("*/*.transition_landmarks_v17.npz"):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        signers.add(str(metadata["signer_id"]))
    return signers


def landmark_tree_fingerprint(root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    paths = sorted(root.glob("*/*.transition_landmarks_v17.npz"))
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return len(paths), digest.hexdigest()


class TransitionWindowDataset(Dataset):
    def __init__(
        self, root: Path, signers: set[str], *, seed: int, fixed_masks: bool,
        preload: bool = True,
    ):
        self.rows: list[tuple[Path, int, str]] = []
        self.preloaded_features: list[np.ndarray] | None = [] if preload else None
        self.seed = seed
        self.fixed_masks = fixed_masks
        for path in sorted(root.glob("*/*.transition_landmarks_v17.npz")):
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                valid = payload["window_valid"].astype(np.bool_)
                signer = str(metadata["signer_id"])
                if signer not in signers:
                    continue
                landmarks = payload["landmarks"] if self.preloaded_features is not None else None
                for window in np.flatnonzero(valid):
                    self.rows.append((path, int(window), signer))
                    if self.preloaded_features is not None:
                        self.preloaded_features.append(
                            landmarks[int(window)].copy()
                        )
        if not self.rows:
            raise ValueError(f"no transition windows for {sorted(signers)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, window, signer = self.rows[index]
        if self.preloaded_features is None:
            with np.load(path, allow_pickle=False) as payload:
                features = payload["landmarks"][window].astype(np.float32)
        else:
            features = self.preloaded_features[index].astype(np.float32)
        if self.fixed_masks:
            rng = random.Random(self.seed + index * 104729)
        else:
            rng = random.Random(random.getrandbits(64))
        length = rng.randint(4, 12)
        start = rng.randint(3, 32 - length - 3)
        mask = np.zeros(32, dtype=np.bool_)
        mask[start:start + length] = True
        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "signer": signer,
            "item": f"{path.name}:{window}",
        }


def linear_interpolation(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return interpolate_masked_context(features, mask)


def loss_terms(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    present = target[..., 3] > 0
    spatial_mask = mask[:, :, None] & present
    spatial_error = F.smooth_l1_loss(
        predicted[..., :3], target[..., :3], reduction="none"
    ).mean(dim=-1)
    spatial = (spatial_error * spatial_mask).sum() / spatial_mask.sum().clamp_min(1)

    auxiliary_error = F.mse_loss(
        predicted[..., 3:], target[..., 3:], reduction="none"
    ).mean(dim=-1)
    auxiliary_mask = mask[:, :, None].expand_as(auxiliary_error)
    auxiliary = (
        (auxiliary_error * auxiliary_mask).sum()
        / auxiliary_mask.sum().clamp_min(1)
    )

    target_velocity = target[:, 1:, ..., :3] - target[:, :-1, ..., :3]
    predicted_velocity = predicted[:, 1:, ..., :3] - predicted[:, :-1, ..., :3]
    velocity_focus = mask[:, 1:] | mask[:, :-1]
    velocity_present = present[:, 1:] & present[:, :-1]
    velocity_mask = velocity_focus[:, :, None] & velocity_present
    velocity_error = (predicted_velocity - target_velocity).square().mean(dim=-1)
    velocity = (
        (velocity_error * velocity_mask).sum() / velocity_mask.sum().clamp_min(1)
    )

    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    predicted_acceleration = predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
    acceleration_focus = velocity_focus[:, 1:] | velocity_focus[:, :-1]
    acceleration_present = (
        present[:, 2:] & present[:, 1:-1] & present[:, :-2]
    )
    acceleration_mask = acceleration_focus[:, :, None] & acceleration_present
    acceleration_error = (
        predicted_acceleration - target_acceleration
    ).square().mean(dim=-1)
    acceleration = (
        (acceleration_error * acceleration_mask).sum()
        / acceleration_mask.sum().clamp_min(1)
    )
    return {
        "spatial": spatial,
        "auxiliary": auxiliary,
        "velocity": velocity,
        "acceleration": acceleration,
    }


def motion_distribution_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Match per-articulator velocity/acceleration moments over a training batch."""
    present = target[..., 3] > 0
    losses = []
    target_velocity = target[:, 1:, ..., :3] - target[:, :-1, ..., :3]
    predicted_velocity = predicted[:, 1:, ..., :3] - predicted[:, :-1, ..., :3]
    velocity_mask = (
        (mask[:, 1:] | mask[:, :-1])[:, :, None]
        & present[:, 1:] & present[:, :-1]
    )
    target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
    predicted_acceleration = predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
    acceleration_mask = (
        (mask[:, 2:] | mask[:, 1:-1] | mask[:, :-2])[:, :, None]
        & present[:, 2:] & present[:, 1:-1] & present[:, :-2]
    )
    for predicted_motion, target_motion, motion_mask in (
        (predicted_velocity, target_velocity, velocity_mask),
        (predicted_acceleration, target_acceleration, acceleration_mask),
    ):
        weight = motion_mask.to(target.dtype)[..., None]
        count = weight.sum(dim=(0, 1)).clamp_min(1.0)
        target_mean_abs = (target_motion.abs() * weight).sum(dim=(0, 1)) / count
        predicted_mean_abs = (predicted_motion.abs() * weight).sum(dim=(0, 1)) / count
        target_mean = (target_motion * weight).sum(dim=(0, 1)) / count
        predicted_mean = (predicted_motion * weight).sum(dim=(0, 1)) / count
        target_std = torch.sqrt(
            ((target_motion - target_mean).square() * weight).sum(dim=(0, 1))
            / count + 1e-8
        )
        predicted_std = torch.sqrt(
            ((predicted_motion - predicted_mean).square() * weight).sum(dim=(0, 1))
            / count + 1e-8
        )
        for predicted_stat, target_stat in (
            (predicted_mean_abs, target_mean_abs),
            (predicted_std, target_std),
        ):
            scale = target_stat.detach().clamp_min(1e-3)
            losses.append(F.smooth_l1_loss(
                predicted_stat / scale, target_stat / scale
            ))
    return torch.stack(losses).mean()


def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {key: 0.0 for key in ("spatial", "auxiliary", "velocity", "acceleration")}
    baseline = totals.copy()
    rows = 0
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            predicted = model(features, mask)
            interpolated = linear_interpolation(features, mask)
            learned_terms = loss_terms(predicted, features, mask)
            baseline_terms = loss_terms(interpolated, features, mask)
            count = len(features)
            rows += count
            for key in totals:
                totals[key] += float(learned_terms[key]) * count
                baseline[key] += float(baseline_terms[key]) * count
    learned = {key: value / rows for key, value in totals.items()}
    linear = {key: value / rows for key, value in baseline.items()}
    score = learned["spatial"] + 0.25 * learned["velocity"] + 0.10 * learned["acceleration"]
    baseline_score = linear["spatial"] + 0.25 * linear["velocity"] + 0.10 * linear["acceleration"]
    return {
        "windows": rows,
        **{f"learned_{key}": value for key, value in learned.items()},
        **{f"linear_{key}": value for key, value in linear.items()},
        "learned_score": score,
        "linear_score": baseline_score,
        "relative_score_improvement": (baseline_score - score) / max(baseline_score, 1e-12),
    }


def train_seed(
    seed: int,
    train_dataset: TransitionWindowDataset,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TransitionInpainterV17(
        TransitionInpainterV17Config(
            dim=args.dim, depth=args.depth, heads=args.heads, dropout=args.dropout
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )
    best_metrics = evaluate(model, validation_loader, device)
    best_score = best_metrics["learned_score"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "validation": best_metrics}]
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(features, mask)
            terms = loss_terms(predicted, features, mask)
            loss = (
                terms["spatial"]
                + args.auxiliary_weight * terms["auxiliary"]
                + args.velocity_weight * terms["velocity"]
                + args.acceleration_weight * terms["acceleration"]
                + args.motion_distribution_weight
                * motion_distribution_loss(predicted, features, mask)
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite transition loss seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        metrics = evaluate(model, validation_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": total / seen,
            "validation": metrics,
        })
        if metrics["learned_score"] < best_score:
            best_score = metrics["learned_score"]
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d train=%.6f heldout=%.6f linear=%.6f best=%d patience=%d",
            seed, epoch, total / seen, metrics["learned_score"],
            metrics["linear_score"], best_epoch, patience,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break
    output = args.output / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_transition_inpainter_v17",
        "version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": best_state,
        "seed": seed,
        "epoch": best_epoch,
        "held_out_signer": args.held_out_signer,
        "validation_metrics": best_metrics,
        "test_evaluated": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    torch.save(checkpoint, output / "best_model.pth")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "metrics": best_metrics,
        "checkpoint": (output / "best_model.pth").as_posix(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    all_signers = discover_signers(args.landmark_root)
    if args.held_out_signer not in all_signers:
        raise ValueError(
            f"held-out signer {args.held_out_signer} absent from {sorted(all_signers)}"
        )
    train_signers = {
        signer for signer in all_signers - {args.held_out_signer}
        if not any(signer.startswith(prefix) for prefix in args.exclude_train_signer_prefix)
    }
    if not train_signers:
        raise ValueError("no training signers remain after source exclusions")
    train_dataset = TransitionWindowDataset(
        args.landmark_root, train_signers, seed=1701, fixed_masks=False
    )
    validation_dataset = TransitionWindowDataset(
        args.landmark_root, {args.held_out_signer}, seed=4701, fixed_masks=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    archive_count, archive_fingerprint = landmark_tree_fingerprint(args.landmark_root)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    candidates = [
        train_seed(seed, train_dataset, validation_loader, args, device)
        for seed in args.seeds
    ]
    winner = min(candidates, key=lambda row: row["metrics"]["learned_score"])
    selected = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    selected.update({
        "train_signers": sorted(train_signers),
        "excluded_train_signer_prefixes": list(args.exclude_train_signer_prefix),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "landmark_root": args.landmark_root.as_posix(),
        "landmark_archive_count": archive_count,
        "landmark_tree_sha256": archive_fingerprint,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    })
    torch.save(selected, args.output / "best_model.pth")
    report = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "metrics": winner["metrics"],
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "held_out_signer": args.held_out_signer,
        "train_signers": sorted(train_signers),
        "excluded_train_signer_prefixes": list(args.exclude_train_signer_prefix),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "landmark_archive_count": archive_count,
        "landmark_tree_sha256": archive_fingerprint,
        "candidates": candidates,
        "seconds": time.monotonic() - started,
        "claim_boundary": (
            "feature reconstruction is not rendered-video or human-perceptual naturalness"
        ),
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
        "--output", type=Path,
        default=Path("artifacts/models/transition_inpainter_v17_pilot"),
    )
    parser.add_argument("--held-out-signer", default="how2sign:8")
    parser.add_argument("--exclude-train-signer-prefix", action="append", default=[])
    parser.add_argument("--seeds", type=int, nargs="+", default=(10701, 10702, 10703))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--auxiliary-weight", type=float, default=0.10)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-weight", type=float, default=0.10)
    parser.add_argument("--motion-distribution-weight", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
