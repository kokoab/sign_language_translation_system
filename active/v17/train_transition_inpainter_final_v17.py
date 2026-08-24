#!/usr/bin/env python3
"""Train the final all-voice transition mean using LOSO-selected settings."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import gc
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

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
    landmark_tree_fingerprint,
    loss_terms,
    sha256,
)


LOG = logging.getLogger("train_transition_inpainter_final_v17")


def run(args: argparse.Namespace) -> dict[str, object]:
    device_name = (
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    signers = {
        signer for signer in discover_signers(args.landmark_root)
        if not any(signer.startswith(prefix) for prefix in args.exclude_signer_prefix)
    }
    dataset = TransitionWindowDataset(
        args.landmark_root, signers, seed=1701, fixed_masks=False
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = TransitionInpainterV17(
        TransitionInpainterV17Config(
            dim=args.dim, depth=args.depth, heads=args.heads, dropout=args.dropout
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    history = []
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
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite final transition loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        history.append({"epoch": epoch, "train_loss": total / seen})
        LOG.info("epoch=%d/%d train=%.6f", epoch, args.epochs, total / seen)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    archive_count, tree_hash = landmark_tree_fingerprint(args.landmark_root)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_transition_inpainter_v17",
        "version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "seed": args.seed,
        "epoch": args.epochs,
        "held_out_signer": None,
        "train_signers": sorted(signers),
        "train_windows": len(dataset),
        "landmark_archive_count": archive_count,
        "landmark_tree_sha256": tree_hash,
        "selection_basis": (
            "architecture/hyperparameters from three-fold train-only LOSO; "
            "fixed epoch is median LOSO selected epoch"
        ),
        "test_evaluated": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    torch.save(checkpoint, args.output / "model.pth")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {
        "checkpoint": (args.output / "model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "model.pth"),
        "seed": args.seed,
        "epochs": args.epochs,
        "train_signers": sorted(signers),
        "train_windows": len(dataset),
        "final_train_loss": history[-1]["train_loss"],
        "landmark_tree_sha256": tree_hash,
        "seconds": time.monotonic() - started,
        "claim_boundary": "train-all artifact inherits LOSO evidence; it has no independent accuracy score",
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
        default=Path("artifacts/models/transition_inpainter_v17_all6_final"),
    )
    parser.add_argument("--exclude-signer-prefix", action="append", default=["ncslgr:"])
    parser.add_argument("--seed", type=int, default=10703)
    parser.add_argument("--epochs", type=int, default=58)
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
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
