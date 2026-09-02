#!/usr/bin/env python3
"""Train the Stage-1-anchored v17 CTC boundary head on non-test caches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.model_stage2_v17 import Stage2IdentityAnchoredHeadV17, Stage2V17Config
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collate,
    evaluate,
)


LOG = logging.getLogger("train_stage_2_identity_anchored_v17")


def loader(dataset, batch_size: int, *, sampler=None) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=sampler is None,
        sampler=sampler, num_workers=0, collate_fn=collate,
    )


def source_weights(dataset: CombinedDataset) -> torch.Tensor:
    sources = []
    for child in dataset.datasets:
        if isinstance(child, SyntheticCompositionDataset):
            sources.extend(str(row["source"]) for row in child.rows)
        else:
            sources.extend(sample.source for sample in child.samples)
    counts = Counter(sources)
    desired = {
        "local_phrases": 0.20,
        "asllrp_contiguous": 0.15,
        "asllrp_segmented_train": 0.20,
        "synthetic_citizen_train": 0.10,
        "synthetic_multivoice_train": 0.15,
        "synthetic_balanced_multivoice_train": 0.20,
    }
    if set(counts) != set(desired):
        raise ValueError(f"unexpected sources: {dict(counts)}")
    return torch.tensor([desired[source] / counts[source] for source in sources], dtype=torch.double)


def selection_key(phrase: dict, context: dict) -> tuple[float, ...]:
    wers = [value["wer"] for value in phrase["domains"].values()]
    wers.extend(value["wer"] for value in context["domains"].values())
    accuracies = [value["sequence_accuracy"] for value in phrase["domains"].values()]
    accuracies.extend(value["sequence_accuracy"] for value in context["domains"].values())
    return (-max(wers), -float(np.mean(wers)), float(np.mean(accuracies)))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(args: argparse.Namespace) -> dict[str, object]:
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.phrase_root, args.context_train_root, args.context_validation_root,
        args.synthetic_pool, args.synthetic_plan,
    )):
        raise ValueError("identity-anchored training is restricted to train/validation data")
    seed_all(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    phrase_train = RealPhraseDataset(args.phrase_root, "train")
    context_train = RealPhraseDataset(args.context_train_root, "train")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    training = CombinedDataset([phrase_train, context_train, synthetic])
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        source_weights(training), args.samples_per_epoch, replacement=True, generator=generator
    )
    train_loader = loader(training, args.batch_size, sampler=sampler)
    phrase_validation = RealPhraseDataset(args.phrase_root, "validation")
    phrase_loader = loader(
        phrase_validation, args.validation_batch_size,
        sampler=torch.utils.data.SequentialSampler(phrase_validation),
    )
    context_validation = RealPhraseDataset(args.context_validation_root, "validation")
    context_loader = loader(
        context_validation, args.validation_batch_size,
        sampler=torch.utils.data.SequentialSampler(context_validation),
    )
    model = Stage2IdentityAnchoredHeadV17(Stage2V17Config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best_key = None
    best_state = None
    best = None
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = samples = 0
        epoch_started = time.monotonic()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = model(
                batch["features"].to(device), batch["window_mask"].to(device)
            )
            loss = criterion(
                logits.log_softmax(-1).transpose(0, 1), batch["targets"].to(device),
                lengths, batch["target_lengths"].to(device),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch["window_mask"])
            samples += len(batch["window_mask"])
        scheduler.step()
        phrase = evaluate(model, phrase_loader, device)
        context = evaluate(model, context_loader, device)
        key = selection_key(phrase, context)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, samples),
            "epoch_seconds": time.monotonic() - epoch_started,
            "phrase_validation": phrase,
            "context_validation": context,
            "selection_key": list(key),
        }
        history.append(row)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best = row
        LOG.info(
            "epoch=%d loss=%.4f phrase=%.4f context=%.4f seconds=%.1f",
            epoch, row["train_loss"], phrase["equal_domain_mean_wer"],
            context["equal_domain_mean_wer"], row["epoch_seconds"],
        )
    if best_state is None or best is None:
        raise RuntimeError("training did not produce a checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_stage2_identity_anchored_ctc_v17",
        "format_version": 1,
        "model_config": Stage2V17Config().to_dict(),
        "model_state_dict": best_state,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "validation": best,
        "test_evaluated": False,
    }
    torch.save(checkpoint, args.output / "best_model.pth")
    report = {
        "format": checkpoint["format"],
        "version": 1,
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "parameter_count": model.parameter_count,
        "device": str(device),
        "samples_per_epoch": args.samples_per_epoch,
        "best": best,
        "history": history,
        "wall_seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phrase-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    value.add_argument("--context-train-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_train_frozen_features"))
    value.add_argument("--context-validation-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"))
    value.add_argument("--synthetic-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"))
    value.add_argument("--synthetic-plan", type=Path, default=Path("active/v17/stage2_balanced_multivoice_plan_v17.json"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_identity_anchored_v1"))
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_identity_anchored_v1/validation.json"))
    value.add_argument("--device", default="auto")
    value.add_argument("--mps-memory-fraction", type=float, default=0.12)
    value.add_argument("--seed", type=int, default=17091)
    value.add_argument("--epochs", type=int, default=25)
    value.add_argument("--samples-per-epoch", type=int, default=6000)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--validation-batch-size", type=int, default=64)
    value.add_argument("--lr", type=float, default=3e-4)
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(parser().parse_args()), indent=2))
