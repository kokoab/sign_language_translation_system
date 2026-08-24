#!/usr/bin/env python3
"""Train source-balanced transition inpainting with channel/signer-held-out gates."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import copy
import gc
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
)
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    discover_signers,
    evaluate,
    landmark_tree_fingerprint,
    loss_terms,
    motion_distribution_loss,
    sha256,
)


LOG = logging.getLogger("train_transition_inpainter_multicorpus_v17")


def manifest_signers(path: Path) -> tuple[set[str], set[str]]:
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "continuous_unlabeled_transition_manifest_v17":
        raise ValueError("unexpected web transition manifest")
    train = {
        str(row["signer_id"]) for row in manifest["rows"] if row["role"] == "train"
    }
    validation = {
        str(row["signer_id"])
        for row in manifest["rows"] if row["role"] == "validation"
    }
    if not train or not validation or train & validation:
        raise ValueError("web train/validation voice sets must be non-empty and disjoint")
    return train, validation


def weighted_sampler(
    primary_windows: int,
    web_windows: int,
    web_probability: float,
    samples: int,
    seed: int,
) -> WeightedRandomSampler:
    if not 0.0 < web_probability < 1.0:
        raise ValueError("web probability must be strictly inside (0, 1)")
    weights = torch.cat((
        torch.full((primary_windows,), (1.0 - web_probability) / primary_windows),
        torch.full((web_windows,), web_probability / web_windows),
    )).double()
    return WeightedRandomSampler(
        weights,
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def combined_selection_score(
    how2sign: dict[str, float], youtube_asl: dict[str, float]
) -> float:
    return 0.5 * (
        how2sign["relative_score_improvement"]
        + youtube_asl["relative_score_improvement"]
    )


def passes_primary_floor(
    how2sign: dict[str, float], floor: float
) -> bool:
    return how2sign["relative_score_improvement"] >= floor


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.how2sign_regression_tolerance < 0.0:
        raise ValueError("How2Sign regression tolerance must be non-negative")
    device_name = (
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)

    how2sign_all = {
        signer for signer in discover_signers(args.how2sign_root)
        if signer.startswith("how2sign:")
    }
    if args.held_out_how2sign not in how2sign_all:
        raise ValueError("held-out How2Sign signer is absent")
    how2sign_train = how2sign_all - {args.held_out_how2sign}
    manifest_web_train, manifest_web_validation = manifest_signers(args.web_manifest)
    available_web = discover_signers(args.web_root)
    web_train = manifest_web_train & available_web
    web_validation = manifest_web_validation & available_web
    missing_web_train = manifest_web_train - available_web
    missing_web_validation = manifest_web_validation - available_web
    if not web_train or not web_validation:
        raise ValueError("usable web train/validation voice sets must both be non-empty")
    if len(web_train) < args.minimum_web_train_voices:
        raise ValueError(
            f"only {len(web_train)} usable web train voices; "
            f"minimum is {args.minimum_web_train_voices}"
        )
    if len(web_validation) < args.minimum_web_validation_voices:
        raise ValueError(
            f"only {len(web_validation)} usable web validation voices; "
            f"minimum is {args.minimum_web_validation_voices}"
        )
    primary_dataset = TransitionWindowDataset(
        args.how2sign_root, how2sign_train, seed=1701, fixed_masks=False
    )
    web_dataset = TransitionWindowDataset(
        args.web_root, web_train, seed=2701, fixed_masks=False
    )
    primary_validation = TransitionWindowDataset(
        args.how2sign_root, {args.held_out_how2sign}, seed=4701, fixed_masks=True
    )
    web_validation_dataset = TransitionWindowDataset(
        args.web_root, web_validation, seed=5701, fixed_masks=True
    )
    primary_loader = DataLoader(
        primary_validation, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    web_loader = DataLoader(
        web_validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    train_dataset = ConcatDataset((primary_dataset, web_dataset))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    requested_config = TransitionInpainterV17Config(
        dim=args.dim, depth=args.depth, heads=args.heads, dropout=args.dropout
    )
    model = TransitionInpainterV17(requested_config).to(device)
    initial_checkpoint_sha256 = None
    if args.initial_checkpoint:
        initial_checkpoint = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=False
        )
        if initial_checkpoint.get("format") != "slt_transition_inpainter_v17":
            raise ValueError("unexpected initial transition checkpoint")
        if initial_checkpoint.get("held_out_signer") != args.held_out_how2sign:
            raise ValueError("initial checkpoint held-out signer does not match fold")
        if initial_checkpoint.get("model_config") != requested_config.to_dict():
            raise ValueError("initial checkpoint model configuration does not match")
        model.load_state_dict(initial_checkpoint["model_state_dict"])
        initial_checkpoint_sha256 = sha256(args.initial_checkpoint)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sampler = weighted_sampler(
        len(primary_dataset),
        len(web_dataset),
        args.web_probability,
        args.samples_per_epoch or len(train_dataset),
        args.seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    primary_metrics = evaluate(model, primary_loader, device)
    web_metrics = evaluate(model, web_loader, device)
    primary_floor = (
        primary_metrics["relative_score_improvement"]
        - args.how2sign_regression_tolerance
    )
    initial_metrics = {"how2sign": primary_metrics, "youtube_asl": web_metrics}
    best_score = combined_selection_score(primary_metrics, web_metrics)
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = initial_metrics
    history = [{
        "epoch": 0,
        "validation": best_metrics,
        "selection_score": best_score,
        "passes_how2sign_floor": True,
    }]
    patience = 0
    started = time.monotonic()
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
                raise RuntimeError("non-finite multi-corpus loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        primary_metrics = evaluate(model, primary_loader, device)
        web_metrics = evaluate(model, web_loader, device)
        metrics = {"how2sign": primary_metrics, "youtube_asl": web_metrics}
        score = combined_selection_score(primary_metrics, web_metrics)
        eligible = passes_primary_floor(primary_metrics, primary_floor)
        history.append({
            "epoch": epoch,
            "train_loss": total / seen,
            "validation": metrics,
            "selection_score": score,
            "passes_how2sign_floor": eligible,
        })
        if eligible and score > best_score:
            best_score = score
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
            "epoch=%d train=%.6f h2s=%.4f web=%.4f score=%.4f best=%d",
            epoch,
            total / seen,
            primary_metrics["relative_score_improvement"],
            web_metrics["relative_score_improvement"],
            score,
            best_epoch,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break

    how2sign_count, how2sign_hash = landmark_tree_fingerprint(args.how2sign_root)
    web_count, web_hash = landmark_tree_fingerprint(args.web_root)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_transition_inpainter_v17",
        "version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": best_state,
        "seed": args.seed,
        "epoch": best_epoch,
        "initial_checkpoint": (
            args.initial_checkpoint.as_posix() if args.initial_checkpoint else None
        ),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "held_out_signer": args.held_out_how2sign,
        "train_signers": sorted(how2sign_train | web_train),
        "how2sign_train_signers": sorted(how2sign_train),
        "youtube_asl_train_voice_proxies": sorted(web_train),
        "youtube_asl_validation_voice_proxies": sorted(web_validation),
        "youtube_asl_missing_train_voice_proxies": sorted(missing_web_train),
        "youtube_asl_missing_validation_voice_proxies": sorted(missing_web_validation),
        "how2sign_train_windows": len(primary_dataset),
        "youtube_asl_train_windows": len(web_dataset),
        "web_probability": args.web_probability,
        "how2sign_relative_improvement_floor": primary_floor,
        "how2sign_regression_tolerance": args.how2sign_regression_tolerance,
        "motion_distribution_weight": args.motion_distribution_weight,
        "initial_validation_metrics": initial_metrics,
        "validation_metrics": best_metrics,
        "selection_score": best_score,
        "how2sign_landmark_archive_count": how2sign_count,
        "how2sign_landmark_tree_sha256": how2sign_hash,
        "youtube_asl_landmark_archive_count": web_count,
        "youtube_asl_landmark_tree_sha256": web_hash,
        "youtube_asl_manifest": args.web_manifest.as_posix(),
        "youtube_asl_manifest_sha256": sha256(args.web_manifest),
        "test_evaluated": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    checkpoint_path = args.output / "best_model.pth"
    torch.save(checkpoint, checkpoint_path)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selected_epoch": best_epoch,
        "selection_score": best_score,
        "initial_checkpoint": (
            args.initial_checkpoint.as_posix() if args.initial_checkpoint else None
        ),
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "initial_validation": initial_metrics,
        "validation": best_metrics,
        "web_probability": args.web_probability,
        "how2sign_relative_improvement_floor": primary_floor,
        "how2sign_regression_tolerance": args.how2sign_regression_tolerance,
        "motion_distribution_weight": args.motion_distribution_weight,
        "how2sign_train_signers": sorted(how2sign_train),
        "youtube_asl_train_voices": len(web_train),
        "youtube_asl_validation_voices": len(web_validation),
        "youtube_asl_missing_train_voices": len(missing_web_train),
        "youtube_asl_missing_validation_voices": len(missing_web_validation),
        "how2sign_train_windows": len(primary_dataset),
        "youtube_asl_train_windows": len(web_dataset),
        "seconds": time.monotonic() - started,
        "claim_boundary": (
            "channel-disjoint web validation plus signer-held-out How2Sign are "
            "machine landmark gates, not human-perceptual naturalness"
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
        "--how2sign-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--web-root", type=Path,
        default=Path("data/local/youtube_asl_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--web-manifest", type=Path,
        default=Path("active/v17/youtube_asl_transition_manifest_v17.json"),
    )
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        help="fold-matched How2Sign checkpoint used as the no-regression baseline",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/transition_inpainter_multicorpus_v17_h8_w025"),
    )
    parser.add_argument("--held-out-how2sign", default="how2sign:8")
    parser.add_argument("--web-probability", type=float, default=0.25)
    parser.add_argument("--minimum-web-train-voices", type=int, default=80)
    parser.add_argument("--minimum-web-validation-voices", type=int, default=16)
    parser.add_argument("--how2sign-regression-tolerance", type=float, default=0.0)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10703)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--auxiliary-weight", type=float, default=0.10)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--motion-distribution-weight", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
