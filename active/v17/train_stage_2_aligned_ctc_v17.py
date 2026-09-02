#!/usr/bin/env python3
"""Fine-tune the v17 CTC head with genuine ASLLRP sign-boundary supervision.

The ordinary CTC objective knows the ordered glosses but not their positions.  ASLLRP
also publishes manual start/end frames for every sign, so this experiment adds a
frame-aligned cross-entropy term for ASLLRP training spans while retaining CTC for all
real phrases.  Citizen, SemLex, local, ASLLRP, and RIT test data are never loaded.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import Stage2TemporalHeadV17, Stage2V17Config
from active.v17.train_stage_2_v17 import (
    RealPhraseDataset,
    collate,
    collapse_ctc,
    evaluate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_bins(frame_count: int, tokens_per_window: int = 8) -> list[tuple[float, float]]:
    """Map each compressed CTC step to its source-frame interval."""
    if frame_count < 1 or tokens_per_window < 1:
        raise ValueError("frame_count and tokens_per_window must be positive")
    bins: list[tuple[float, float]] = []
    for window_start in range(0, frame_count, 32):
        window_length = min(32, frame_count - window_start)
        if window_length < 4 and window_start:
            break
        for step in range(tokens_per_window):
            bins.append((
                window_start + window_length * step / tokens_per_window,
                window_start + window_length * (step + 1) / tokens_per_window,
            ))
    return bins


def aligned_ctc_targets(
    frame_count: int,
    token_intervals: list[tuple[float, float]],
    token_indices: list[int],
) -> np.ndarray:
    """Create blank/one-based-class targets using maximum temporal overlap."""
    if len(token_intervals) != len(token_indices) or not token_indices:
        raise ValueError("token intervals and indices must be non-empty and aligned")
    bins = output_bins(frame_count)
    targets = np.zeros(len(bins), dtype=np.int64)
    for index, (start, stop) in enumerate(bins):
        overlaps = [max(0.0, min(stop, right) - max(start, left)) for left, right in token_intervals]
        best = int(np.argmax(overlaps))
        if overlaps[best] > 0:
            targets[index] = int(token_indices[best]) + 1

    # A very short annotated sign can fall between two compressed-step centers.  Give
    # every token its nearest unused step, in monotonic order, without inventing a label.
    centers = np.asarray([(left + right) * 0.5 for left, right in bins])
    previous = -1
    for interval, token_index in zip(token_intervals, token_indices):
        candidates = np.flatnonzero(centers > previous)
        if not len(candidates):
            raise ValueError("not enough CTC steps for aligned target sequence")
        center = (interval[0] + interval[1]) * 0.5
        nearest = int(candidates[np.argmin(np.abs(centers[candidates] - center))])
        targets[nearest] = int(token_index) + 1
        previous = nearest
    return targets


def build_alignment_map(
    phrase_manifest_path: Path,
    span_manifest_path: Path,
    segmented_manifest_path: Path,
) -> dict[str, np.ndarray]:
    phrase = json.loads(phrase_manifest_path.read_text())
    spans = json.loads(span_manifest_path.read_text())["spans"]
    signs = json.loads(segmented_manifest_path.read_text())["videos"]
    phrase_rows = {
        str(row["source_item_id"]): row for row in phrase["rows"]
        if row["source"] == "asllrp_contiguous" and row["role"] in {"train", "validation"}
    }
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for sign in signs:
        by_parent.setdefault(str(sign["utterance_video_filename"]), []).append(sign)
    output: dict[str, np.ndarray] = {}
    for span in spans:
        item_id = f"asllrp:{span['utterance_video_filename']}:span{int(span['span_index_in_utterance']):02d}"
        row = phrase_rows.get(item_id)
        if row is None:
            continue
        first = int(span["span_start_frame_global"])
        last = int(span["span_end_frame_global"])
        selected = sorted((
            sign for sign in by_parent.get(str(span["utterance_video_filename"]), [])
            if first <= int(sign["sign_start_frame"]) <= int(sign["sign_end_frame"]) <= last
        ), key=lambda sign: int(sign["sign_start_frame"]))
        labels = [str(sign["canonical_label"]) for sign in selected]
        if labels != list(row["target_sequence"]):
            raise ValueError(f"{item_id}: annotation sequence mismatch: {labels}")
        crop_global_start = (
            int(span["utterance_start_frame_global"]) + int(span["crop_start_frame_local"])
        )
        intervals = [
            (
                float(int(sign["sign_start_frame"]) - crop_global_start),
                float(int(sign["sign_end_frame"]) - crop_global_start + 1),
            )
            for sign in selected
        ]
        aligned = aligned_ctc_targets(
            int(row["frame_count"]), intervals, [int(value) for value in row["target_indices"]]
        )
        if collapse_ctc(aligned) != [int(value) for value in row["target_indices"]]:
            raise ValueError(f"{item_id}: aligned targets do not CTC-collapse to the glosses")
        output[item_id] = aligned
    missing = sorted(set(phrase_rows) - set(output))
    if missing:
        raise ValueError(f"missing ASLLRP alignments: {missing[:5]}")
    return output


def collate_aligned(samples, alignment_map: dict[str, np.ndarray]) -> dict[str, Any]:
    batch = collate(samples)
    steps = batch["features"].shape[1] * 8
    aligned = np.full((len(samples), steps), -100, dtype=np.int64)
    for index, sample in enumerate(samples):
        target = alignment_map.get(sample.item_id)
        if target is not None:
            aligned[index, :len(target)] = target
    batch["aligned_targets"] = torch.from_numpy(aligned)
    return batch


def load_warm(path: Path) -> tuple[Stage2TemporalHeadV17, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("warm checkpoint must be a bare v17 CTC head")
    model = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def domain_edits(metrics: dict[str, Any], source: str) -> int:
    value = metrics["domains"][source]
    return int(round(float(value["wer"]) * int(value["tokens"])))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.cache_root, args.phrase_manifest, args.span_manifest, args.segmented_manifest):
        if "test" in {part.lower() for part in path.parts}:
            raise ValueError("aligned CTC training is restricted to train/validation data")
    seed_all(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    alignments = build_alignment_map(
        args.phrase_manifest, args.span_manifest, args.segmented_manifest
    )
    train = RealPhraseDataset(args.cache_root, "train")
    validation = RealPhraseDataset(args.cache_root, "validation")
    warm, warm_payload = load_warm(args.warm_checkpoint)
    teacher = deepcopy(warm).to(device).eval()
    model = warm.to(device)
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )
    warm_metrics = evaluate(model, validation_loader, device)

    counts = Counter(sample.source for sample in train.samples)
    source_mass = {"asllrp_contiguous": 0.5, "local_phrases": 0.5}
    weights = torch.tensor(
        [source_mass[sample.source] / counts[sample.source] for sample in train.samples],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        train, batch_size=args.batch_size, sampler=sampler, num_workers=0,
        collate_fn=lambda samples: collate_aligned(samples, alignments),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best_state = deepcopy(model.state_dict())
    best_metrics = warm_metrics
    best_epoch = 0
    best_key = (domain_edits(warm_metrics, "asllrp_contiguous"), 0)
    local_floor = domain_edits(warm_metrics, "local_phrases")
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            logits, lengths = model(features, mask)
            with torch.no_grad():
                teacher_logits, _ = teacher(features, mask)
            loss_ctc = ctc(
                logits.log_softmax(-1).transpose(0, 1), batch["targets"].to(device),
                lengths, batch["target_lengths"].to(device),
            )
            aligned = batch["aligned_targets"].to(device)
            loss_aligned = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), aligned.reshape(-1), ignore_index=-100
            )
            valid = mask.unsqueeze(-1).expand(-1, -1, 8).reshape(mask.shape[0], -1)
            temperature = args.distill_temperature
            loss_distill = F.kl_div(
                F.log_softmax(logits[valid] / temperature, dim=-1),
                F.softmax(teacher_logits[valid] / temperature, dim=-1),
                reduction="batchmean",
            ) * temperature * temperature
            loss = (
                loss_ctc + args.alignment_weight * loss_aligned
                + args.distill_weight * loss_distill
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["batches"] += 1
            totals["loss"] += float(loss.detach())
            totals["ctc"] += float(loss_ctc.detach())
            totals["aligned"] += float(loss_aligned.detach())
            totals["distill"] += float(loss_distill.detach())
        metrics = evaluate(model, validation_loader, device)
        asllrp_edits = domain_edits(metrics, "asllrp_contiguous")
        local_edits = domain_edits(metrics, "local_phrases")
        eligible = local_edits <= local_floor
        key = (asllrp_edits, local_edits)
        if eligible and key < best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            best_state = deepcopy(model.state_dict())
        history.append({
            "epoch": epoch,
            "loss": totals["loss"] / totals["batches"],
            "ctc_loss": totals["ctc"] / totals["batches"],
            "aligned_loss": totals["aligned"] / totals["batches"],
            "distill_loss": totals["distill"] / totals["batches"],
            "asllrp_edits": asllrp_edits,
            "local_edits": local_edits,
            "eligible": eligible,
        })
        print(json.dumps(history[-1]))

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(warm_payload)
    checkpoint.update({
        "model_state_dict": best_state,
        "aligned_ctc_finetune": True,
        "aligned_ctc_epoch": best_epoch,
        "aligned_ctc_validation_metrics": best_metrics,
        "aligned_ctc_training_config": vars(args) | {"device": str(device)},
        "warm_checkpoint": args.warm_checkpoint.as_posix(),
        "warm_checkpoint_sha256": sha256(args.warm_checkpoint),
        "phrase_manifest_sha256": sha256(args.phrase_manifest),
        "span_manifest_sha256": sha256(args.span_manifest),
        "segmented_manifest_sha256": sha256(args.segmented_manifest),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    })
    checkpoint["aligned_ctc_training_config"] = {
        key: value.as_posix() if isinstance(value, Path) else value
        for key, value in checkpoint["aligned_ctc_training_config"].items()
    }
    torch.save(checkpoint, args.output / "best_model.pth")
    result = {
        "warm_validation_metrics": warm_metrics,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "promotable": best_epoch > 0 and best_key < (
            domain_edits(warm_metrics, "asllrp_contiguous"), local_floor
        ),
        "alignment_rows": len(alignments),
        "history": history,
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--cache-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    value.add_argument("--phrase-manifest", type=Path, default=Path("active/v17/stage2_training_manifest_v17.json"))
    value.add_argument("--span-manifest", type=Path, default=Path("data/local/asllrp_contiguous_phrases_v17/manifest.json"))
    value.add_argument("--segmented-manifest", type=Path, default=Path("data/local/asllrp_segmented_citizen100_v17/manifest.json"))
    value.add_argument("--warm-checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_aligned_ctc_v1"))
    value.add_argument("--device", default="auto")
    value.add_argument("--mps-memory-fraction", type=float, default=0.12)
    value.add_argument("--seed", type=int, default=17101)
    value.add_argument("--epochs", type=int, default=16)
    value.add_argument("--samples-per-epoch", type=int, default=512)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--lr", type=float, default=2e-5)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--alignment-weight", type=float, default=0.4)
    value.add_argument("--distill-weight", type=float, default=0.5)
    value.add_argument("--distill-temperature", type=float, default=2.0)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
