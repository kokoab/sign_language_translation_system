#!/usr/bin/env python3
"""Conservatively adapt v2 to signer-voice synthesis with teacher preservation."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import Stage2TemporalHeadV17, Stage2V17Config, make_stage2_checkpoint
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collate,
    evaluate,
    sampler_weights,
)


LOG = logging.getLogger("train_stage_2_signer_voice_adaptation_v17")
SEEDS = (4701, 4702, 4703)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_key(metrics: dict[str, object]) -> tuple[float, float, float]:
    return (
        -float(metrics["worst_domain_wer"]),
        -float(metrics["equal_domain_mean_wer"]),
        float(metrics["equal_domain_mean_sequence_accuracy"]),
    )


def train_seed(seed, teacher, train_dataset, weights, val_loader, args, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = copy.deepcopy(teacher).to(device).train()
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch, replacement=True, generator=generator
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, collate_fn=collate,
    )
    baseline = evaluate(model, val_loader, device)
    best_metrics = baseline
    best_key = selection_key(baseline)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_epoch = 0
    history = [{"epoch": 0, "train_loss": None, "ctc_loss": None, "distill_loss": None, **baseline}]
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = total_ctc = total_distill = seen = 0.0
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, input_lengths = model(features, mask)
            with torch.inference_mode():
                teacher_logits, _ = teacher(features, mask)
            ctc = criterion(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets, input_lengths, target_lengths,
            )
            temperature = args.temperature
            token_mask = mask.repeat_interleave(model.config.tokens_per_window, dim=1)
            per_token = F.kl_div(
                F.log_softmax(logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            distill = (per_token * token_mask).sum() / token_mask.sum().clamp_min(1)
            distill = distill * temperature * temperature
            loss = ctc + args.distill_weight * distill
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite adaptation loss at seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            count = len(mask)
            total += float(loss.detach()) * count
            total_ctc += float(ctc.detach()) * count
            total_distill += float(distill.detach()) * count
            seen += count
        metrics = evaluate(model, val_loader, device)
        key = selection_key(metrics)
        history.append({
            "epoch": epoch,
            "train_loss": total / seen,
            "ctc_loss": total_ctc / seen,
            "distill_loss": total_distill / seen,
            **metrics,
        })
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d loss=%.4f ASLLRP=%.4f local=%.4f best=%d patience=%d",
            seed, epoch, total / seen,
            metrics["domains"]["asllrp_contiguous"]["wer"],
            metrics["domains"]["local_phrases"]["wer"], best_epoch, patience,
        )
        if patience >= args.patience:
            break
    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = make_stage2_checkpoint(
        model, best_state, seed=seed, epoch=best_epoch,
        validation_metrics=best_metrics, selection_key=list(best_key),
        warm_started_from=args.base_checkpoint.as_posix(),
        warm_started_from_sha256=sha256(args.base_checkpoint),
        distill_weight=args.distill_weight,
        temperature=args.temperature,
    )
    torch.save(checkpoint, seed_dir / "best_head.pth")
    (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "validation_metrics": best_metrics,
        "checkpoint": (seed_dir / "best_head.pth").as_posix(),
    }


def run(args):
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    teacher = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    real_train = RealPhraseDataset(args.cache_root, "train")
    real_val = RealPhraseDataset(args.cache_root, "validation")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    train_dataset = CombinedDataset([real_train, synthetic])
    weights, sampling_mass = sampler_weights(real_train, synthetic)
    val_loader = DataLoader(
        real_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = [
        train_seed(seed, teacher, train_dataset, weights, val_loader, args, device)
        for seed in SEEDS
    ]
    winner = max(results, key=lambda row: tuple(row["selection_key"]))
    selected = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    selected.update({
        "synthetic_pool": args.synthetic_pool.as_posix(),
        "synthetic_pool_sha256": sha256(args.synthetic_pool),
        "synthetic_plan": args.synthetic_plan.as_posix(),
        "synthetic_plan_sha256": sha256(args.synthetic_plan),
        "training_source_sampling_mass": sampling_mass,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    })
    torch.save(selected, args.output / "best_model.pth")
    report = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "selection_key": winner["selection_key"],
        "validation_metrics": winner["validation_metrics"],
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "candidate_results": results,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("artifacts/models/stage2_v17_unified_ctc_v2/best_model.pth"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument(
        "--synthetic-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"),
    )
    parser.add_argument(
        "--synthetic-plan", type=Path,
        default=Path("active/v17/stage2_signer_voice_plan_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_signer_voice_adaptation_pilot_v1"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distill-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
