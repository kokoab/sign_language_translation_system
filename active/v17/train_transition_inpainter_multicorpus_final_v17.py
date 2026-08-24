#!/usr/bin/env python3
"""Train the final all-voice transition mean with frozen multi-corpus settings."""

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
from torch.utils.data import ConcatDataset, DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
)
from active.v17.train_transition_inpainter_multicorpus_v17 import (
    manifest_signers,
    weighted_sampler,
)
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    discover_signers,
    landmark_tree_fingerprint,
    loss_terms,
    sha256,
)


LOG = logging.getLogger("train_transition_inpainter_multicorpus_final_v17")


def run(args: argparse.Namespace) -> dict[str, object]:
    device_name = (
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)

    how2sign = {
        signer for signer in discover_signers(args.how2sign_root)
        if signer.startswith("how2sign:")
    }
    manifest_train, manifest_validation = manifest_signers(args.web_manifest)
    available_web = discover_signers(args.web_root)
    web = (manifest_train | manifest_validation) & available_web
    if len(how2sign) != 6 or len(web) < 96:
        raise ValueError("final transition voice breadth floor failed")
    how2sign_dataset = TransitionWindowDataset(
        args.how2sign_root, how2sign, seed=1701, fixed_masks=False
    )
    web_dataset = TransitionWindowDataset(
        args.web_root, web, seed=2701, fixed_masks=False
    )
    combined = ConcatDataset((how2sign_dataset, web_dataset))
    sampler = weighted_sampler(
        len(how2sign_dataset), len(web_dataset), args.web_probability,
        args.samples_per_epoch or len(combined), args.seed,
    )
    loader = DataLoader(
        combined, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    requested_config = TransitionInpainterV17Config(
        dim=args.dim, depth=args.depth, heads=args.heads, dropout=args.dropout
    )
    initial = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    if (
        initial.get("format") != "slt_transition_inpainter_v17"
        or initial.get("held_out_signer") is not None
        or initial.get("model_config") != requested_config.to_dict()
    ):
        raise ValueError("unexpected all-voice initial transition checkpoint")
    model = TransitionInpainterV17(requested_config).to(device)
    model.load_state_dict(initial["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
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
                raise RuntimeError("non-finite final multi-corpus transition loss")
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

    how2sign_count, how2sign_hash = landmark_tree_fingerprint(args.how2sign_root)
    web_count, web_hash = landmark_tree_fingerprint(args.web_root)
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
        "initial_checkpoint": args.initial_checkpoint.as_posix(),
        "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
        "how2sign_train_signers": sorted(how2sign),
        "youtube_asl_train_voice_proxies": sorted(web),
        "web_probability": args.web_probability,
        "how2sign_train_windows": len(how2sign_dataset),
        "youtube_asl_train_windows": len(web_dataset),
        "how2sign_landmark_archive_count": how2sign_count,
        "how2sign_landmark_tree_sha256": how2sign_hash,
        "youtube_asl_landmark_archive_count": web_count,
        "youtube_asl_landmark_tree_sha256": web_hash,
        "youtube_asl_manifest_sha256": sha256(args.web_manifest),
        "selection_basis": (
            "10% web replay and LR 5e-5 frozen on signer 8, confirmed on signers "
            "3/5; fixed 63 epochs is the median selected fold epoch"
        ),
        "test_evaluated": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    checkpoint_path = args.output / "model.pth"
    torch.save(checkpoint, checkpoint_path)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "epochs": args.epochs,
        "how2sign_voices": len(how2sign),
        "youtube_asl_voice_proxies": len(web),
        "how2sign_train_windows": len(how2sign_dataset),
        "youtube_asl_train_windows": len(web_dataset),
        "final_train_loss": history[-1]["train_loss"],
        "seconds": time.monotonic() - started,
        "claim_boundary": (
            "train-all artifact inherits LOSO/channel-held-out evidence and has "
            "no independent accuracy or human-naturalness score"
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
        default=Path("artifacts/models/transition_inpainter_v17_all6_final/model.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final"),
    )
    parser.add_argument("--web-probability", type=float, default=0.10)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10703)
    parser.add_argument("--epochs", type=int, default=63)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=5e-5)
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
