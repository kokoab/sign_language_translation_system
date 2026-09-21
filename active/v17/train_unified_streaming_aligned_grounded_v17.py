#!/usr/bin/env python3
"""Causal CTC experiment with frame-aligned NCSLGR transition supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.approved_phrase_data_v17 import DEFAULT_MANIFEST, APPROVED_ROOT, require_training_manifest
from active.v17.model_unified_streaming_ctc_v17 import (
    UnifiedStreamingCTCConfig, UnifiedStreamingCTCHeadV17,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
import active.v17.train_unified_streaming_ctc_v17 as base_train
from active.v17.train_unified_streaming_grounded_ctc_v17 import grounded_evaluate
from active.v17.train_stage_1_reel_emission_v17 import asllrp_annotations


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def endpoints(frames: int, stride: int, window: int) -> list[int]:
    values = list(range(min(12, window), frames + 1, stride))
    if not values or values[-1] != frames:
        values.append(frames)
    return values


def ncslgr_alignments(
    manifest: Path, labels: dict[str, int], stride: int, window: int,
    source_frames: dict[str, int],
) -> dict[str, np.ndarray]:
    payload = json.loads(manifest.read_text())
    other = len(labels) + 1
    output = {}
    for row in payload["rows"]:
        if not row["has_strict_target"] or row["source_item_id"] not in source_frames:
            continue
        frames = source_frames[row["source_item_id"]]
        if not 0 < frames <= int(row["source_frame_count"]):
            raise ValueError(f"invalid evidence frame count: {row['source_item_id']}")
        ends = endpoints(frames, stride, window)
        aligned = np.zeros(len(ends), dtype=np.int64)
        intervals = []
        for event in row["events"]:
            target = labels.get(event["canonical_label"], other - 1) + 1
            intervals.append((
                int(event["source_start_frame"]),
                int(event["source_end_frame_exclusive"]), target,
            ))
        if any(left < 0 or right <= left or right > frames for left, right, _ in intervals):
            raise ValueError(f"annotation outside cached evidence: {row['source_item_id']}")
        for index, end in enumerate(ends):
            start = max(0, end - window)
            overlaps = [max(0, min(end, right) - max(start, left)) for left, right, _ in intervals]
            if overlaps and max(overlaps) > 0:
                aligned[index] = intervals[int(np.argmax(overlaps))][2]
        # Guarantee at least one monotonic step for each annotated event. This uses
        # the published interval center and never guesses a gloss.
        previous = -1
        centers = np.asarray(ends) - window / 2
        for left, right, target in intervals:
            choices = np.flatnonzero(np.arange(len(ends)) > previous)
            if not len(choices):
                break
            nearest = int(choices[np.argmin(np.abs(centers[choices] - (left + right) / 2))])
            aligned[nearest] = target
            previous = nearest
        output[row["source_item_id"]] = aligned
    return output


def collate_aligned(samples, alignments: dict[str, np.ndarray]):
    batch = base_train.collate(samples)
    width = batch["evidence"].shape[1]
    aligned = np.full((len(samples), width), -100, dtype=np.int64)
    for index, sample in enumerate(samples):
        values = alignments.get(sample.identity)
        if values is not None:
            length = len(sample.evidence)
            if len(values) != length:
                raise ValueError(f"alignment/evidence length mismatch: {sample.identity}")
            aligned[index, :length] = values
    batch["aligned"] = torch.from_numpy(aligned)
    return batch


def supplement_weight(original_count: int, supplemental_count: int, mass: float) -> float:
    if original_count <= 0 or supplemental_count <= 0 or not 0 < mass <= 1:
        raise ValueError("invalid supplemental training weight")
    return original_count * mass / supplemental_count


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_provenance = require_training_manifest(args)
    for path in (args.phrase_root, args.other_root, args.citizen_root,
                 args.semlex_train_root, args.semlex_val_root):
        if "test" in {part.casefold() for part in path.parts}:
            raise ValueError("aligned experiment refuses test paths")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    stage1 = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    stage1.load_state_dict(checkpoint["model_state_dict"], strict=True)
    annotations = asllrp_annotations(args.span_manifest, args.segmented_manifest)

    train_raw = base_train.phrase_sequences(
        args.phrase_root, "train", args.rolling_stride, args.window_frames, labels=labels)
    validation_raw = base_train.phrase_sequences(
        args.phrase_root, "validation", args.rolling_stride, args.window_frames, labels=labels)
    train_raw += base_train.phrase_sequences(
        args.other_root, "train", args.rolling_stride, args.window_frames, labels=labels)
    validation_raw += base_train.phrase_sequences(
        args.other_root, "validation", args.rolling_stride, args.window_frames, labels=labels)
    train_raw += base_train.isolated_sequences([
        args.citizen_root / "train",
        args.semlex_train_root / "full_clean_landmarks_v17",
    ], labels)
    validation_raw += base_train.isolated_sequences([
        args.citizen_root / "val", args.semlex_val_root / "landmarks_v17",
    ], labels)
    train_raw += base_train.blank_sequences("train", labels, args, annotations)
    validation_raw += base_train.blank_sequences("validation", labels, args, annotations)
    started = time.perf_counter()
    train = base_train.encode(stage1, train_raw, device, args.embedding_batch_size, "window")
    validation = base_train.encode(
        stage1, validation_raw, device, args.embedding_batch_size, "window")
    original_train_count = len(train)
    supplemental_raw = []
    if getattr(args, "supplement_root", None) is not None:
        base_train.refuse_protected(args.supplement_root)
        supplemental_raw = base_train.phrase_sequences(
            args.supplement_root, "train", args.rolling_stride, args.window_frames, labels=labels)
        if not supplemental_raw or {sample.source for sample in supplemental_raw} != {"flores_other"}:
            raise ValueError("supplement must contain only audited flores_other training archives")
        identities = {sample.identity for sample in train_raw + validation_raw}
        if any(sample.identity in identities for sample in supplemental_raw):
            raise ValueError("supplement identity overlaps original data")
        train += base_train.encode(stage1, supplemental_raw, device, args.embedding_batch_size, "window")
    alignments = ncslgr_alignments(
        args.ncslgr_manifest, labels, args.rolling_stride, args.window_frames,
        {sample.identity: sample.source_frames for sample in train_raw + validation_raw
         if sample.source == "ncslgr_strict"})
    config = UnifiedStreamingCTCConfig(
        stage1_dim=stage1.config.dim, num_glosses=100,
        hidden_dim=args.hidden_dim, blocks=args.blocks, dropout=args.dropout,
    )
    model = UnifiedStreamingCTCHeadV17(config).to(device)
    if getattr(args, "initial_head", None) is not None:
        initial = torch.load(args.initial_head, map_location="cpu", weights_only=False)
        if initial.get("head_config") != config.to_dict():
            raise ValueError("initial temporal head configuration mismatch")
        model.load_state_dict(initial["head_state_dict"], strict=True)
    loader = DataLoader(
        base_train.Sequences(train), batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda samples: collate_aligned(samples, alignments), num_workers=0,
    )
    counts = Counter(base_train.group(value, config.other_index, True) for value in train[:original_train_count])
    weights = {key: original_train_count / (len(counts) * count) for key, count in counts.items()}
    added_weight = supplement_weight(original_train_count, len(supplemental_raw), args.supplement_mass) if supplemental_raw else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = aligned_total = 0.0; seen = 0
        for batch in loader:
            logits = model(batch["evidence"].to(device))
            rows = ctc(
                logits.log_softmax(-1).transpose(0, 1), batch["targets"].to(device),
                batch["lengths"].to(device), batch["target_lengths"].to(device),
            )
            sample_weights = torch.tensor([
                added_weight if value.source == "flores_other" else weights[base_train.group(value, config.other_index, True)]
                for value in batch["samples"]
            ], dtype=rows.dtype, device=device)
            loss_ctc = (rows * sample_weights).mean()
            aligned = batch["aligned"].to(device)
            loss_aligned = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), aligned.reshape(-1), ignore_index=-100,
            ) if (aligned != -100).any() else torch.zeros((), device=device)
            loss = loss_ctc + args.alignment_weight * loss_aligned
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            ready_fd = os.environ.pop("SLT_TRAIN_READY_FD", None)
            if ready_fd is not None:
                os.write(int(ready_fd), b"TRAINING_STARTED\n")
                os.close(int(ready_fd))
            total += float(loss.detach().cpu()) * len(batch["samples"])
            aligned_total += float(loss_aligned.detach().cpu()) * len(batch["samples"])
            seen += len(batch["samples"])
        metrics = grounded_evaluate(model, validation, device, args.batch_size, config.other_index)
        source = metrics["by_source"]
        score = (
            0.45 * source["local_phrases"]["known_wer"]
            + 0.15 * source["ncslgr_strict"]["known_wer"]
            + 0.10 * source["asllrp_contiguous"]["known_wer"]
            + 0.20 * metrics["isolated"]["known_wer"]
            + 0.10 * metrics["blank_boundaries"]["false_emission_rate"]
        )
        row = {"epoch": epoch, "loss": total / seen, "aligned_loss": aligned_total / seen,
               "score": score, "local_wer": source["local_phrases"]["known_wer"],
               "ncslgr_wer": source["ncslgr_strict"]["known_wer"],
               "isolated_wer": metrics["isolated"]["known_wer"],
               "blank_false": metrics["blank_boundaries"]["false_emission_rate"]}
        history.append(row); print(json.dumps(row), flush=True)
        if best is None or score < best["score"]:
            best = {"score": score, "epoch": epoch, "metrics": metrics,
                    "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "best_model.pth"
    torch.save({
        "dataset_provenance": dataset_provenance,
        "format": "slt_unified_streaming_ctc_v17", "version": 1,
        "head_config": config.to_dict(), "head_state_dict": model.cpu().state_dict(),
        "base_checkpoint": str(args.base), "base_checkpoint_sha256": sha256(args.base),
        "label_to_index": labels, "ctc_blank_index": 0,
        "other_label": "__OTHER__", "other_index": config.other_index,
        "decode_policy": "greedy causal CTC; no phrase grammar or language prior",
        "ncslgr_alignment_weight": args.alignment_weight,
        "initial_head": str(args.initial_head) if getattr(args, "initial_head", None) else None,
        "initial_head_sha256": sha256(args.initial_head) if getattr(args, "initial_head", None) else None,
        "phrase_root": str(args.phrase_root),
        "selected_epoch": best["epoch"], "validation": best["metrics"],
        "test_accessed": False,
    }, output)
    result = {
        "dataset_provenance": dataset_provenance,
        "format": "slt_unified_streaming_aligned_grounded_v17", "output": str(output),
        "selected_epoch": best["epoch"], "validation": best["metrics"],
        "history": history, "elapsed_seconds": time.perf_counter() - started,
        "train_samples": len(train), "validation_samples": len(validation),
        "alignment_weight": args.alignment_weight, "test_accessed": False,
        "initial_head": str(args.initial_head) if getattr(args, "initial_head", None) else None,
        "phrase_root": str(args.phrase_root),
        "supplement_mass": args.supplement_mass if supplemental_raw else 0,
        "supplement_sample_weight": added_weight,
        "original_group_weights": weights,
        "training_sources": dict(Counter(sample.source for sample in train_raw + supplemental_raw)),
        "validation_sources": dict(Counter(sample.source for sample in validation_raw)),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--base", type=Path, default=Path(
        "artifacts/models/stage1_v17_asllrp_core_adapt_v1/best_model.pth"))
    value.add_argument("--phrase-root", type=Path, default=APPROVED_ROOT / "phrases")
    value.add_argument("--other-root", type=Path, default=APPROVED_ROOT / "other")
    value.add_argument("--ncslgr-manifest", type=Path, default=Path(
        "active/v17/ncslgr_supervised_manifest_v17.json"))
    value.add_argument("--citizen-root", type=Path, default=Path(
        "data/local/citizen100_v17/landmarks"))
    value.add_argument("--semlex-train-root", type=Path, default=Path(
        "data/local/semlex_citizen100_train_audit"))
    value.add_argument("--semlex-val-root", type=Path, default=Path(
        "data/local/semlex_citizen100_val_audit"))
    value.add_argument("--span-manifest", type=Path, default=Path(
        "data/local/asllrp_contiguous_phrases_v17/manifest.json"))
    value.add_argument("--segmented-manifest", type=Path, default=Path(
        "data/local/asllrp_segmented_citizen100_v17/manifest.json"))
    value.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/models/unified_streaming_aligned_grounded_v17_v1"))
    value.add_argument("--epochs", type=int, default=18)
    value.add_argument("--supplement-root", type=Path)
    value.add_argument("--supplement-mass", type=float, default=.1)
    value.add_argument("--initial-head", type=Path,
                       help="Matched experiment initialization; strict full head state")
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--embedding-batch-size", type=int, default=128)
    value.add_argument("--boundary-per-class", type=int, default=10)
    value.add_argument("--blank-policy", choices=("all_prefixes", "transitions_only"),
                       default="transitions_only")
    value.add_argument("--rolling-stride", type=int, default=4)
    value.add_argument("--window-frames", type=int, default=8)
    value.add_argument("--hidden-dim", type=int, default=128)
    value.add_argument("--blocks", type=int, default=3)
    value.add_argument("--dropout", type=float, default=0.10)
    value.add_argument("--learning-rate", type=float, default=2e-3)
    value.add_argument("--alignment-weight", type=float, default=0.12)
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--seed", type=int, default=17081)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
