#!/usr/bin/env python3
"""Distill the general two-head Stage-2 selector into one temporal head."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
    Stage2V17Config,
    Stage2ContextAdapterV17,
    Stage2TemporalHeadV17,
    load_stage2_context_adapted,
    load_stage2_general_ctc_selector,
)
from active.v17.train_stage_2_balanced_multivoice_v17 import (
    domain_edits,
    source_weights,
)
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collate,
    collapse_ctc,
    evaluate,
    make_stage2_checkpoint,
)


LOG = logging.getLogger("train_stage_2_general_selector_distill_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def context_wrapper(
    base: Stage2TemporalHeadV17, teacher
) -> Stage2ContextAdapterV17:
    source = teacher.primary
    target_class_indices = tuple(
        int(value) - 1
        for value in torch.nonzero(
            source.target_projection.sum(dim=0), as_tuple=False
        ).flatten().tolist()
    )
    return Stage2ContextAdapterV17(
        base,
        feature_mode=source.feature_mode,
        scaler_mean=source.scaler_mean.detach().cpu(),
        scaler_scale=source.scaler_scale.detach().cpu(),
        coefficients=source.coefficients.detach().cpu(),
        intercept=source.intercept.detach().cpu(),
        class_indices=source.class_indices.detach().cpu(),
        target_class_indices=target_class_indices,
        weight=source.weight,
    )


def context_adapter_config(model: Stage2ContextAdapterV17) -> dict[str, object]:
    target_ctc_indices = torch.nonzero(
        model.target_projection.sum(dim=0), as_tuple=False
    ).flatten().tolist()
    return {
        "feature_mode": model.feature_mode,
        "weight": model.weight,
        "target_class_indices": [int(value) - 1 for value in target_ctc_indices],
        "normalization": "per-window population z-score over fitted adapter classes",
    }


def package_context_adapted_student(
    student_checkpoint: Path,
    teacher,
    *,
    output: Path,
    phrase_loader: DataLoader,
    context_loader: DataLoader,
    device: torch.device,
    extra: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    payload = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("compact student checkpoint is not a v17 CTC head")
    student = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    student.load_state_dict(payload["model_state_dict"], strict=True)
    adapted = context_wrapper(student, teacher).to(device).eval()
    phrase = evaluate(adapted, phrase_loader, device)
    context = evaluate(adapted, context_loader, device)
    combined = dict(payload)
    combined.update(extra or {})
    combined.update({
        "format": "slt_stage2_context_adapted_ctc_v17",
        "format_version": 1,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in adapted.state_dict().items()
        },
        "context_adapter_config": context_adapter_config(adapted),
        "base_checkpoint": student_checkpoint.as_posix(),
        "base_checkpoint_sha256": sha256(student_checkpoint),
        "validation_metrics": {"phrases": phrase, "contextual": context},
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, output)
    reloaded, _ = load_stage2_context_adapted(output)
    reloaded.to(device).eval()
    if evaluate(reloaded, phrase_loader, device) != phrase:
        raise RuntimeError("packaged student phrase metrics do not cold-reload")
    if evaluate(reloaded, context_loader, device) != context:
        raise RuntimeError("packaged student context metrics do not cold-reload")
    return combined, phrase, context


def validation_edits(
    phrase: dict[str, object], context: dict[str, object]
) -> tuple[int, int, int]:
    return (
        domain_edits(phrase, "asllrp_contiguous"),
        domain_edits(phrase, "local_phrases"),
        domain_edits(context, "asllrp_segmented_validation"),
    )


def candidate_key(
    phrase: dict[str, object], context: dict[str, object], baseline: tuple[int, int, int]
) -> tuple[int, ...]:
    edits = validation_edits(phrase, context)
    not_worse = tuple(int(value <= limit) for value, limit in zip(edits, baseline))
    return (
        int(all(not_worse)),
        sum(not_worse),
        -sum(edits),
        -edits[0],
        -edits[1],
        -edits[2],
    )


def source_masses_for_synthetic_ratio(synthetic_mass: float) -> dict[str, float]:
    """Keep real replay ratios fixed while sweeping total synthetic exposure."""
    if not 0.0 <= synthetic_mass <= 0.5:
        raise ValueError("synthetic mass must be in [0, 0.5]")
    real = 1.0 - synthetic_mass
    return {
        "local_phrases": real * 0.30,
        "asllrp_contiguous": real * 0.20,
        "asllrp_segmented_train": real * 0.50,
        "synthetic_citizen_train": synthetic_mass * 0.25,
        "synthetic_multivoice_train": synthetic_mass * 0.375,
        "synthetic_balanced_multivoice_train": synthetic_mass * 0.375,
    }


def gradients_are_finite(parameters: list[torch.nn.Parameter]) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


def apply_index_weight_boost(
    weights: torch.Tensor, indices: list[int], boost: float
) -> torch.Tensor:
    if boost < 1.0 or any(index < 0 or index >= len(weights) for index in indices):
        raise ValueError("invalid selector sampling boost")
    output = weights.clone()
    if indices:
        output[torch.as_tensor(indices, dtype=torch.long)] *= boost
    return output


def correct_selector_training_indices(
    teacher,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[int], dict[str, int]]:
    """Find train rows where the selector fires and its final sequence is correct."""
    correct: list[int] = []
    selected_count = 0
    teacher.eval()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            indices = list(range(start, min(len(dataset), start + batch_size)))
            batch = collate([dataset[index] for index in indices])
            logits, lengths, selected = teacher.forward_with_selection(
                batch["features"].to(device), batch["window_mask"].to(device)
            )
            predictions = logits.argmax(dim=-1).cpu().numpy()
            offset = 0
            targets = batch["targets"].tolist()
            for row, (index, length, target_length) in enumerate(zip(
                indices, lengths.cpu().tolist(), batch["target_lengths"].tolist()
            )):
                reference = [
                    int(value) - 1
                    for value in targets[offset:offset + target_length]
                ]
                offset += target_length
                if not bool(selected[row]):
                    continue
                selected_count += 1
                if collapse_ctc(predictions[row, :length]) == reference:
                    correct.append(index)
    return correct, {
        "selector_owned_rows": selected_count,
        "selector_owned_exact_rows": len(correct),
        "scanned_rows": len(dataset),
    }


def load_student_initialization(
    path: Path | None, fallback: Stage2TemporalHeadV17
) -> tuple[Stage2TemporalHeadV17, dict[str, object] | None]:
    if path is None:
        return copy.deepcopy(fallback), None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_temporal_pretrain_v17":
        raise ValueError("student initialization is not v17 temporal pretraining")
    model = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    if model.config.to_dict() != fallback.config.to_dict():
        raise ValueError("student initialization/teacher model configurations differ")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if payload.get("ctc_head_trained") is not False:
        raise ValueError("temporal pretraining must leave the CTC head frozen")
    return model, payload


def train_seed(
    seed: int,
    teacher,
    student_template: Stage2TemporalHeadV17,
    train_dataset,
    weights: torch.Tensor,
    phrase_loader: DataLoader,
    context_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    student = copy.deepcopy(student_template).to(device)
    adapted = context_wrapper(student, teacher).to(device)
    for name, parameter in student.named_parameters():
        parameter.requires_grad = (
            args.train_scope == "all" or name.startswith("ctc_head.")
        )
    trainable_parameters = [
        parameter for parameter in student.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("distillation has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate,
    )
    baseline_phrase = evaluate(adapted, phrase_loader, device)
    baseline_context = evaluate(adapted, context_loader, device)
    baseline_edits = validation_edits(baseline_phrase, baseline_context)
    best_key = candidate_key(baseline_phrase, baseline_context, baseline_edits)
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
    }
    best_epoch = 0
    best_phrase = baseline_phrase
    best_context = baseline_context
    history = [{
        "epoch": 0,
        "phrase_validation": baseline_phrase,
        "context_validation": baseline_context,
        "edits": list(baseline_edits),
        "selection_key": list(best_key),
    }]
    patience = 0
    for epoch in range(1, args.epochs + 1):
        student.train()
        teacher.eval()
        totals = {
            "loss": 0.0, "ctc": 0.0, "distill": 0.0,
            "samples": 0, "teacher_selected": 0, "nonfinite_batches": 0,
        }
        for batch in train_loader:
            features = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, input_lengths = adapted(features, mask)
            with torch.inference_mode():
                teacher_logits, teacher_lengths, teacher_selected = (
                    teacher.forward_with_selection(features, mask)
                )
            if not torch.equal(input_lengths, teacher_lengths):
                raise RuntimeError("teacher/student CTC lengths differ")
            ctc = ctc_criterion(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )
            temperature = args.temperature
            token_mask = mask.repeat_interleave(student.config.tokens_per_window, dim=1)
            per_token = F.kl_div(
                F.log_softmax(logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1)
            row_weight = 1.0 + teacher_selected.to(per_token.dtype) * (
                args.selector_distill_boost - 1.0
            )
            distill_mask = token_mask * row_weight.unsqueeze(1)
            distill = (
                (per_token * distill_mask).sum() / distill_mask.sum().clamp_min(1)
            ) * temperature * temperature
            loss = args.ctc_weight * ctc + args.distill_weight * distill
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss seed={seed} epoch={epoch}")
            loss.backward()
            if not gradients_are_finite(trainable_parameters):
                optimizer.zero_grad(set_to_none=True)
                totals["nonfinite_batches"] += 1
                LOG.warning(
                    "discarding non-finite MPS/CTC gradient seed=%d epoch=%d count=%d",
                    seed, epoch, totals["nonfinite_batches"],
                )
                if totals["nonfinite_batches"] > args.max_nonfinite_batches:
                    raise RuntimeError(
                        f"repeated non-finite gradients seed={seed} epoch={epoch}"
                    )
                continue
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            count = len(mask)
            totals["loss"] += float(loss.detach()) * count
            totals["ctc"] += float(ctc.detach()) * count
            totals["distill"] += float(distill.detach()) * count
            totals["samples"] += count
            totals["teacher_selected"] += int(teacher_selected.sum().item())
        scheduler.step()
        phrase = evaluate(adapted, phrase_loader, device)
        context = evaluate(adapted, context_loader, device)
        edits = validation_edits(phrase, context)
        key = candidate_key(phrase, context, baseline_edits)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / totals["samples"],
            "ctc_loss": totals["ctc"] / totals["samples"],
            "distill_loss": totals["distill"] / totals["samples"],
            "teacher_selected_rows": totals["teacher_selected"],
            "discarded_nonfinite_batches": totals["nonfinite_batches"],
            "phrase_validation": phrase,
            "context_validation": context,
            "edits": list(edits),
            "selection_key": list(key),
        }
        history.append(row)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_phrase = phrase
            best_context = context
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d loss=%.5f edits=%s best=%d patience=%d",
            seed, epoch, row["loss"], edits, best_epoch, patience,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break

    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = make_stage2_checkpoint(
        student,
        best_state,
        seed=seed,
        epoch=best_epoch,
        validation_metrics=best_phrase,
        contextual_validation_metrics=best_context,
        selection_key=list(best_key),
        distilled_from=args.teacher.as_posix(),
        distilled_from_sha256=sha256(args.teacher),
        distill_weight=args.distill_weight,
        ctc_weight=args.ctc_weight,
        temperature=args.temperature,
        selector_distill_boost=args.selector_distill_boost,
        train_scope=args.train_scope,
        baseline_edits=list(baseline_edits),
        maximum_discarded_nonfinite_batches_per_epoch=args.max_nonfinite_batches,
    )
    torch.save(checkpoint, seed_dir / "best_head.pth")
    (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "edits": list(validation_edits(best_phrase, best_context)),
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
    teacher, _ = load_stage2_general_ctc_selector(args.teacher)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    student_template, initialization_payload = load_student_initialization(
        args.student_initialization, teacher.primary.base
    )

    phrase_train = RealPhraseDataset(args.phrase_root, "train")
    context_train = RealPhraseDataset(args.context_train_root, "train")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    train_dataset = CombinedDataset([phrase_train, context_train, synthetic])
    desired = (
        source_masses_for_synthetic_ratio(args.synthetic_mass)
        if args.synthetic_mass is not None else {
            "local_phrases": args.local_mass,
            "asllrp_contiguous": args.phrase_mass,
            "asllrp_segmented_train": args.context_mass,
            "synthetic_citizen_train": args.citizen_mass,
            "synthetic_multivoice_train": args.transfer_mass,
            "synthetic_balanced_multivoice_train": args.balanced_mass,
        }
    )
    weights, counts = source_weights(
        phrase_train, context_train, synthetic, desired
    )
    selector_sampling_audit = {
        "selector_owned_rows": 0,
        "selector_owned_exact_rows": 0,
        "scanned_rows": 0,
    }
    if args.selector_sampling_boost > 1.0:
        correct_indices, selector_sampling_audit = correct_selector_training_indices(
            teacher, train_dataset, device=device,
            batch_size=args.selector_scan_batch_size,
        )
        weights = apply_index_weight_boost(
            weights, correct_indices, args.selector_sampling_boost
        )
        LOG.info(
            "selector sampling scan rows=%d owned=%d owned_exact=%d boost=%.1f",
            selector_sampling_audit["scanned_rows"],
            selector_sampling_audit["selector_owned_rows"],
            selector_sampling_audit["selector_owned_exact_rows"],
            args.selector_sampling_boost,
        )
    phrase_loader = DataLoader(
        RealPhraseDataset(args.phrase_root, "validation"),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    )
    context_loader = DataLoader(
        RealPhraseDataset(args.context_validation_root, "validation"),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = [
        train_seed(
            seed, teacher, student_template, train_dataset, weights, phrase_loader, context_loader,
            args, device,
        )
        for seed in args.seeds
    ]
    winner = max(results, key=lambda row: tuple(row["selection_key"]))
    package_extra = {
        "synthetic_pool": args.synthetic_pool.as_posix(),
        "synthetic_pool_sha256": sha256(args.synthetic_pool),
        "synthetic_plan": args.synthetic_plan.as_posix(),
        "synthetic_plan_sha256": sha256(args.synthetic_plan),
        "training_source_counts": counts,
        "training_source_sampling_mass": desired,
        "student_initialization": (
            args.student_initialization.as_posix()
            if args.student_initialization is not None else None
        ),
        "student_initialization_sha256": (
            sha256(args.student_initialization)
            if args.student_initialization is not None else None
        ),
        "temporal_pretraining_source_split": (
            initialization_payload.get("source_split")
            if initialization_payload is not None else None
        ),
        "selector_sampling_boost": args.selector_sampling_boost,
        "selector_sampling_audit": selector_sampling_audit,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    selected, packaged_phrase, packaged_context = package_context_adapted_student(
        Path(winner["checkpoint"]), teacher,
        output=args.output / "best_model.pth",
        phrase_loader=phrase_loader,
        context_loader=context_loader,
        device=device,
        extra=package_extra,
    )
    if packaged_phrase != winner["phrase_validation"]:
        raise RuntimeError("packaged winner phrase metrics differ from selection")
    if packaged_context != winner["context_validation"]:
        raise RuntimeError("packaged winner context metrics differ from selection")
    report = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "selection_key": winner["selection_key"],
        "edits": winner["edits"],
        "phrase_validation": winner["phrase_validation"],
        "context_validation": winner["context_validation"],
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "teacher": args.teacher.as_posix(),
        "teacher_sha256": sha256(args.teacher),
        "synthetic_plan_sha256": sha256(args.synthetic_plan),
        "training_source_counts": counts,
        "training_source_sampling_mass": desired,
        "student_initialization": (
            args.student_initialization.as_posix()
            if args.student_initialization is not None else None
        ),
        "student_initialization_sha256": (
            sha256(args.student_initialization)
            if args.student_initialization is not None else None
        ),
        "selector_sampling_boost": args.selector_sampling_boost,
        "selector_sampling_audit": selector_sampling_audit,
        "packaged_artifact_format": selected["format"],
        "cold_reload_verified": True,
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
        "--teacher", type=Path,
        default=Path("artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"),
    )
    parser.add_argument(
        "--student-initialization", type=Path,
        help="Strict label-free v17 temporal-pretraining checkpoint",
    )
    parser.add_argument(
        "--phrase-root", type=Path,
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
        default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"),
    )
    parser.add_argument(
        "--synthetic-plan", type=Path,
        default=Path("active/v17/stage2_balanced_multivoice_plan_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_general_selector_distill_v1"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(9701, 9702, 9703))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--samples-per-epoch", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ctc-weight", type=float, default=0.25)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--selector-distill-boost", type=float, default=32.0)
    parser.add_argument(
        "--selector-sampling-boost", type=float, default=1.0,
        help="Oversample only selector-owned training rows whose final sequence is exact",
    )
    parser.add_argument("--selector-scan-batch-size", type=int, default=32)
    parser.add_argument("--train-scope", choices=("all", "ctc_head"), default="all")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument(
        "--max-nonfinite-batches", type=int, default=2,
        help="Fail after this many explicitly discarded non-finite MPS/CTC batches per epoch",
    )
    parser.add_argument("--local-mass", type=float, default=0.22)
    parser.add_argument("--phrase-mass", type=float, default=0.18)
    parser.add_argument("--context-mass", type=float, default=0.20)
    parser.add_argument("--citizen-mass", type=float, default=0.10)
    parser.add_argument("--transfer-mass", type=float, default=0.15)
    parser.add_argument("--balanced-mass", type=float, default=0.15)
    parser.add_argument(
        "--synthetic-mass", type=float,
        help="Sweep total synthetic exposure while preserving fixed real/synthetic sub-ratios",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
