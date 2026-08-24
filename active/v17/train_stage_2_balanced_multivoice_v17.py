#!/usr/bin/env python3
"""Adapt Stage 2 with exhaustive pair replay and genuine multi-signer context."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from collections import Counter
import copy
import gc
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

from active.v17.model_stage2_v17 import (
    load_stage2_context_adapted,
    Stage2ContextAdapterV17,
    Stage2TemporalHeadV17,
    Stage2V17Config,
    make_stage2_checkpoint,
)
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collate,
    evaluate,
)


LOG = logging.getLogger("train_stage_2_balanced_multivoice_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def domain_edits(metrics: dict[str, object], domain: str) -> int:
    row = metrics["domains"][domain]
    return int(round(float(row["wer"]) * int(row["tokens"])))


def candidate_key(phrase: dict[str, object], context: dict[str, object]) -> tuple[int, ...]:
    asllrp = domain_edits(phrase, "asllrp_contiguous")
    local = domain_edits(phrase, "local_phrases")
    contextual = domain_edits(context, "asllrp_segmented_validation")
    local_gate = int(local <= 7)
    context_gate = int(contextual <= 42)
    return (
        local_gate + context_gate,
        local_gate,
        context_gate,
        -asllrp,
        -contextual,
        -local,
    )


def make_context_wrapper(
    base: Stage2TemporalHeadV17, adapter_payload: dict[str, object]
) -> Stage2ContextAdapterV17:
    state = adapter_payload["model_state_dict"]
    config = adapter_payload["context_adapter_config"]
    return Stage2ContextAdapterV17(
        base,
        feature_mode=str(config["feature_mode"]),
        scaler_mean=state["scaler_mean"],
        scaler_scale=state["scaler_scale"],
        coefficients=state["coefficients"],
        intercept=state["intercept"],
        class_indices=state["class_indices"],
        target_class_indices=tuple(int(value) for value in config["target_class_indices"]),
        weight=float(config["weight"]),
    )


def source_weights(
    phrase: RealPhraseDataset,
    context: RealPhraseDataset,
    synthetic: SyntheticCompositionDataset,
    desired: dict[str, float],
) -> tuple[torch.Tensor, dict[str, int]]:
    counts = Counter(sample.source for sample in phrase.samples)
    counts.update(sample.source for sample in context.samples)
    counts.update(str(row["source"]) for row in synthetic.rows)
    if set(counts) != set(desired):
        raise ValueError(f"unexpected sources: {dict(counts)}")
    if abs(sum(desired.values()) - 1.0) > 1e-8:
        raise ValueError("sampling masses must sum to one")
    values = [desired[sample.source] / counts[sample.source] for sample in phrase.samples]
    values.extend(desired[sample.source] / counts[sample.source] for sample in context.samples)
    values.extend(
        desired[str(row["source"])] / counts[str(row["source"])]
        for row in synthetic.rows
    )
    return torch.as_tensor(values, dtype=torch.double), dict(sorted(counts.items()))


def train_seed(
    seed: int,
    teacher: Stage2TemporalHeadV17,
    adapter_payload: dict[str, object],
    train_dataset,
    weights,
    phrase_loader,
    context_loader,
    args,
    device,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = copy.deepcopy(teacher).to(device)
    for parameter in model.parameters():
        parameter.requires_grad = True
    adapted = make_context_wrapper(model, adapter_payload).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
    )
    baseline_phrase = evaluate(adapted, phrase_loader, device)
    baseline_context = evaluate(adapted, context_loader, device)
    best_key = candidate_key(baseline_phrase, baseline_context)
    best_phrase = baseline_phrase
    best_context = baseline_context
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    history = [{
        "epoch": 0,
        "train_loss": None,
        "ctc_loss": None,
        "distill_loss": None,
        "phrase_validation": baseline_phrase,
        "context_validation": baseline_context,
        "selection_key": list(best_key),
    }]
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
                targets,
                input_lengths,
                target_lengths,
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
                raise RuntimeError(
                    f"non-finite loss at seed={seed} epoch={epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            count = len(mask)
            total += float(loss.detach()) * count
            total_ctc += float(ctc.detach()) * count
            total_distill += float(distill.detach()) * count
            seen += count
        scheduler.step()
        phrase = evaluate(adapted, phrase_loader, device)
        context = evaluate(adapted, context_loader, device)
        key = candidate_key(phrase, context)
        history.append({
            "epoch": epoch,
            "train_loss": total / seen,
            "ctc_loss": total_ctc / seen,
            "distill_loss": total_distill / seen,
            "phrase_validation": phrase,
            "context_validation": context,
            "selection_key": list(key),
        })
        if key > best_key:
            best_key = key
            best_phrase = phrase
            best_context = context
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d loss=%.4f edits(asllrp/local/context)=%d/%d/%d "
            "best=%d patience=%d",
            seed,
            epoch,
            total / seen,
            domain_edits(phrase, "asllrp_contiguous"),
            domain_edits(phrase, "local_phrases"),
            domain_edits(context, "asllrp_segmented_validation"),
            best_epoch,
            patience,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break

    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = make_stage2_checkpoint(
        model,
        best_state,
        seed=seed,
        epoch=best_epoch,
        validation_metrics=best_phrase,
        contextual_validation_metrics=best_context,
        selection_key=list(best_key),
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
        "phrase_validation": best_phrase,
        "context_validation": best_context,
        "checkpoint": (seed_dir / "best_head.pth").as_posix(),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    teacher = Stage2TemporalHeadV17(Stage2V17Config(**base_payload["model_config"]))
    teacher.load_state_dict(base_payload["model_state_dict"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    _, adapter_payload = load_stage2_context_adapted(args.context_template)

    phrase_train = RealPhraseDataset(args.phrase_cache_root, "train")
    context_train = RealPhraseDataset(args.context_train_root, "train")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    train_dataset = CombinedDataset([phrase_train, context_train, synthetic])
    desired = {
        "local_phrases": args.local_mass,
        "asllrp_contiguous": args.phrase_mass,
        "asllrp_segmented_train": args.context_mass,
        "synthetic_citizen_train": args.citizen_mass,
        "synthetic_multivoice_train": args.transfer_mass,
        "synthetic_balanced_multivoice_train": args.balanced_mass,
    }
    weights, counts = source_weights(phrase_train, context_train, synthetic, desired)
    phrase_validation = RealPhraseDataset(args.phrase_cache_root, "validation")
    context_validation = RealPhraseDataset(args.context_validation_root, "validation")
    phrase_loader = DataLoader(
        phrase_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    context_loader = DataLoader(
        context_validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = [
        train_seed(
            seed,
            teacher,
            adapter_payload,
            train_dataset,
            weights,
            phrase_loader,
            context_loader,
            args,
            device,
        )
        for seed in args.seeds
    ]
    winner = max(results, key=lambda row: tuple(row["selection_key"]))
    selected = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    selected.update({
        "synthetic_pool": args.synthetic_pool.as_posix(),
        "synthetic_pool_sha256": sha256(args.synthetic_pool),
        "synthetic_plan": args.synthetic_plan.as_posix(),
        "synthetic_plan_sha256": sha256(args.synthetic_plan),
        "context_train_root": args.context_train_root.as_posix(),
        "training_source_counts": counts,
        "training_source_sampling_mass": desired,
        "context_template": args.context_template.as_posix(),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    })
    torch.save(selected, args.output / "best_model.pth")
    report = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "selection_key": winner["selection_key"],
        "phrase_validation": winner["phrase_validation"],
        "context_validation": winner["context_validation"],
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "base_checkpoint": args.base_checkpoint.as_posix(),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "synthetic_plan_sha256": sha256(args.synthetic_plan),
        "training_source_counts": counts,
        "training_source_sampling_mass": desired,
        "candidate_results": results,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path(
            "artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/"
            "seed_4703/best_head.pth"
        ),
    )
    parser.add_argument(
        "--context-template", type=Path,
        default=Path(
            "artifacts/models/stage2_v17_multivoice_transfer_context_adapted_v3/model.pth"
        ),
    )
    parser.add_argument(
        "--phrase-cache-root", type=Path,
        default=Path("data/local/stage2_v17_frozen_features"),
    )
    parser.add_argument(
        "--context-train-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_train_frozen_features"),
    )
    parser.add_argument(
        "--context-validation-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"),
    )
    parser.add_argument(
        "--synthetic-pool", type=Path,
        default=Path(
            "data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"
        ),
    )
    parser.add_argument(
        "--synthetic-plan", type=Path,
        default=Path("active/v17/stage2_balanced_multivoice_plan_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_balanced_multivoice_v1"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(8701, 8702, 8703))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=7.5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distill-weight", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--local-mass", type=float, default=0.22)
    parser.add_argument("--phrase-mass", type=float, default=0.18)
    parser.add_argument("--context-mass", type=float, default=0.20)
    parser.add_argument("--citizen-mass", type=float, default=0.10)
    parser.add_argument("--transfer-mass", type=float, default=0.15)
    parser.add_argument("--balanced-mass", type=float, default=0.15)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
