#!/usr/bin/env python3
"""Adapt a separate Stage 1 to local and ASLLRP phrase segments with replay gates."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_reel_emission_v17 import (
    BoundarySample,
    asllrp_annotations,
    collect_samples,
    isolated_samples,
    sha256,
)
from active.v17.train_stage_1_v17 import augment_v17


class CompleteSamples(Dataset):
    def __init__(self, samples: list[BoundarySample]):
        self.samples = [sample for sample in samples if sample.kind == "complete"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return torch.from_numpy(sample.features.copy()), sample.target, sample.domain


@torch.inference_mode()
def metrics(
    model: SLTStage1V17, dataset: CompleteSamples, device: torch.device, batch: int
) -> dict[str, dict[str, float]]:
    counts: dict[str, list[int]] = {}
    model.eval()
    for features, targets, domains in DataLoader(dataset, batch_size=batch):
        predictions = model(features.to(device)).argmax(1).cpu()
        for prediction, target, domain in zip(predictions, targets, domains):
            row = counts.setdefault(str(domain), [0, 0])
            row[0] += int(prediction == target)
            row[1] += 1
    return {
        domain: {"correct": row[0], "samples": row[1], "accuracy": row[0] / row[1]}
        for domain, row in counts.items()
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if any("test" in {part.casefold() for part in path.parts} for path in (
        args.citizen_root, args.semlex_train_root, args.semlex_validation_root,
        args.phrase_root,
    )):
        raise ValueError("contextual adaptation refuses protected test paths")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    annotations = asllrp_annotations(args.span_manifest, args.segmented_manifest)
    train_rows = collect_samples(
        split="train", labels=labels, no_emit_index=100,
        citizen_root=args.citizen_root, semlex_root=args.semlex_train_root,
        phrase_root=args.phrase_root, annotations=annotations,
        per_class=args.replay_per_class,
    )
    if not any(sample.domain == "semlex" for sample in train_rows):
        train_rows.extend(isolated_samples(
            args.semlex_train_root / "landmarks_v17", labels, 100,
            domain="semlex", per_class=args.replay_per_class,
            prefix_fractions=(),
        ))
    train = CompleteSamples(train_rows)
    validation = CompleteSamples(collect_samples(
        split="validation", labels=labels, no_emit_index=100,
        citizen_root=args.citizen_root, semlex_root=args.semlex_validation_root,
        phrase_root=args.phrase_root, annotations=annotations, per_class=10000,
    ))
    source_share = {
        "citizen": 0.35, "semlex": 0.20,
        "local_phrase": 0.25, "asllrp_phrase": 0.20,
    }
    counts = Counter(sample.domain for sample in train.samples)
    weights = torch.tensor([
        source_share[sample.domain] / counts[sample.domain]
        for sample in train.samples
    ], dtype=torch.double)
    loader = DataLoader(
        train, batch_size=args.batch_size,
        sampler=WeightedRandomSampler(
            weights, args.samples_per_epoch, replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        ), num_workers=0,
    )
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    baseline = metrics(model, validation, device, args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    selected = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for features, targets, domains in loader:
            value = augment_v17(features.to(device))
            targets = targets.to(device)
            logits = model(value)
            with torch.no_grad():
                teacher_logits = teacher(value)
            loss = F.cross_entropy(logits, targets) + args.distill_weight * F.kl_div(
                F.log_softmax(logits / 2.0, dim=1),
                F.softmax(teacher_logits / 2.0, dim=1),
                reduction="batchmean",
            ) * 4.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        measured = metrics(model, validation, device, args.batch_size)
        passes = all(
            measured[source]["accuracy"] >= baseline[source]["accuracy"] - 0.01
            for source in ("citizen", "semlex")
        )
        score = (
            measured["asllrp_phrase"]["accuracy"] * 0.6
            + measured["local_phrase"]["accuracy"] * 0.4
        )
        row = {
            "epoch": epoch, "loss": total / batches, "validation": measured,
            "passes_replay_gate": passes, "selection_score": score,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if passes and (selected is None or score > selected["selection_score"]):
            selected = {**row, "state": copy.deepcopy(model.state_dict())}
    if selected is None:
        raise RuntimeError("no contextual epoch passed isolated replay gates")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = selected.pop("state")
    output_checkpoint = copy.deepcopy(checkpoint)
    output_checkpoint["model_state_dict"] = state
    output_checkpoint["contextual_phrase_adaptation"] = {
        "base": str(args.base), "base_sha256": sha256(args.base),
        "training_counts": dict(counts), "source_share": source_share,
        "selected": selected, "test_accessed": False,
        "external_evaluation_reserved_accessed": False,
    }
    torch.save(output_checkpoint, args.output_dir / "best_model.pth")
    result = {
        "format": "slt_stage1_contextual_phrase_adaptation_v17",
        "base": str(args.base), "baseline": baseline, "selected": selected,
        "history": history, "training_counts": dict(counts),
        "elapsed_seconds": time.perf_counter() - started, "device": str(device),
        "test_accessed": False, "external_evaluation_reserved_accessed": False,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "selected": selected}))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base", type=Path, default=Path("artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"))
    value.add_argument("--citizen-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    value.add_argument("--semlex-train-root", type=Path, default=Path("data/local/semlex_citizen100_train_audit"))
    value.add_argument("--semlex-validation-root", type=Path, default=Path("data/local/semlex_citizen100_val_audit"))
    value.add_argument("--phrase-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    value.add_argument("--span-manifest", type=Path, default=Path("data/local/asllrp_contiguous_phrases_v17/manifest.json"))
    value.add_argument("--segmented-manifest", type=Path, default=Path("data/local/asllrp_segmented_citizen100_v17/manifest.json"))
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/models/stage1_v17_contextual_adapt_experiment_v1"))
    value.add_argument("--epochs", type=int, default=6)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--samples-per-epoch", type=int, default=4000)
    value.add_argument("--replay-per-class", type=int, default=10)
    value.add_argument("--learning-rate", type=float, default=5e-6)
    value.add_argument("--weight-decay", type=float, default=1e-3)
    value.add_argument("--distill-weight", type=float, default=0.3)
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--seed", type=int, default=17043)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
