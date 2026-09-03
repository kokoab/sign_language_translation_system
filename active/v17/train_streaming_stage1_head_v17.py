#!/usr/bin/env python3
"""Train a tiny causal CTC head over rolling predictions from the accepted Stage 1."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.geometry_v17 import resample_features
from active.v17.model_streaming_stage1_head_v17 import (
    StreamingStage1CTCHeadV17,
    StreamingStage1HeadConfig,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_streaming_tcn_ctc_v17 import (
    SequenceSample,
    collapse_ctc,
    edit_distance,
    load_isolated_samples,
    load_phrase_samples,
    refuse_protected,
    sha256,
)


@dataclass(frozen=True)
class EvidenceSample:
    logits: np.ndarray
    targets: tuple[int, ...]
    source: str
    item_id: str


def rolling_clips(frames: np.ndarray, stride: int, minimum_frames: int) -> list[np.ndarray]:
    if len(frames) <= 32:
        return [resample_features(frames, 32).astype(np.float32)]
    endpoints = list(range(minimum_frames, len(frames) + 1, stride))
    if endpoints[-1] != len(frames):
        endpoints.append(len(frames))
    return [
        resample_features(frames[max(0, stop - 32) : stop], 32).astype(np.float32)
        for stop in endpoints
    ]


@torch.inference_mode()
def extract_evidence(
    model: SLTStage1V17,
    samples: list[SequenceSample],
    device: torch.device,
    batch_size: int,
    stride: int,
    minimum_frames: int,
) -> list[EvidenceSample]:
    collected: list[list[np.ndarray]] = [[] for _ in samples]
    pending: list[np.ndarray] = []
    owners: list[int] = []

    def flush() -> None:
        if not pending:
            return
        value = torch.from_numpy(np.stack(pending)).to(device)
        logits = model(value).float().cpu().numpy()
        for owner, row in zip(owners, logits):
            collected[owner].append(row)
        pending.clear()
        owners.clear()

    for owner, sample in enumerate(samples):
        for clip in rolling_clips(sample.frames, stride, minimum_frames):
            pending.append(clip)
            owners.append(owner)
            if len(pending) == batch_size:
                flush()
    flush()
    return [
        EvidenceSample(
            np.stack(rows), sample.targets, sample.source, sample.item_id
        )
        for rows, sample in zip(collected, samples)
    ]


class EvidenceTrainingDataset(Dataset):
    def __init__(
        self,
        phrases: list[EvidenceSample],
        isolated: list[EvidenceSample],
        synthetic_count: int,
        phrase_repeats: int,
        seed: int,
    ):
        self.phrases = phrases
        self.isolated = isolated
        self.synthetic_count = synthetic_count
        self.phrase_repeats = phrase_repeats
        self.real_count = len(phrases) * phrase_repeats + len(isolated)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.real_count + self.synthetic_count

    def __getitem__(self, index: int) -> EvidenceSample:
        if index < len(self.phrases) * self.phrase_repeats:
            return self.phrases[index % len(self.phrases)]
        if index < self.real_count:
            return self.isolated[index - len(self.phrases) * self.phrase_repeats]
        rng = random.Random(self.seed + self.epoch * self.synthetic_count + index)
        chosen = [
            self.isolated[rng.randrange(len(self.isolated))]
            for _ in range(rng.randint(2, 5))
        ]
        return EvidenceSample(
            np.concatenate([sample.logits for sample in chosen]),
            tuple(value for sample in chosen for value in sample.targets),
            "synthetic_stage1_composition",
            f"synthetic:{self.epoch}:{index}",
        )


def collate(samples: list[EvidenceSample]) -> dict[str, object]:
    lengths = torch.tensor([len(sample.logits) for sample in samples], dtype=torch.long)
    target_lengths = torch.tensor([len(sample.targets) for sample in samples], dtype=torch.long)
    features = np.zeros((len(samples), int(lengths.max()), 100), np.float32)
    for index, sample in enumerate(samples):
        features[index, : len(sample.logits)] = sample.logits
    return {
        "features": torch.from_numpy(features),
        "lengths": lengths,
        "targets": torch.tensor(
            [value for sample in samples for value in sample.targets], dtype=torch.long
        ),
        "target_lengths": target_lengths,
        "samples": samples,
    }


@torch.inference_mode()
def evaluate(
    model: StreamingStage1CTCHeadV17,
    samples: list[EvidenceSample],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    buckets: dict[str, dict[str, float]] = {}
    examples = []
    for offset in range(0, len(samples), batch_size):
        batch = collate(samples[offset : offset + batch_size])
        paths = model(batch["features"].to(device)).argmax(-1).cpu().numpy()
        for sample, length, path in zip(batch["samples"], batch["lengths"], paths):
            predicted = collapse_ctc(path[: int(length)])
            bucket = buckets.setdefault(sample.source, {
                "samples": 0, "exact": 0, "edits": 0, "target_tokens": 0,
                "predicted_tokens": 0,
            })
            bucket["samples"] += 1
            bucket["exact"] += predicted == sample.targets
            bucket["edits"] += edit_distance(sample.targets, predicted)
            bucket["target_tokens"] += len(sample.targets)
            bucket["predicted_tokens"] += len(predicted)
            if sample.source in {"local_phrases", "asllrp_contiguous"} and len(examples) < 30:
                examples.append({
                    "source": sample.source, "item_id": sample.item_id,
                    "target": sample.targets, "predicted": predicted,
                })
    for bucket in buckets.values():
        bucket["exact_accuracy"] = bucket["exact"] / bucket["samples"]
        bucket["wer"] = bucket["edits"] / bucket["target_tokens"]
    phrase = [buckets[key] for key in ("local_phrases", "asllrp_contiguous") if key in buckets]
    aggregate = {
        key: sum(bucket[key] for bucket in phrase)
        for key in ("samples", "exact", "edits", "target_tokens", "predicted_tokens")
    }
    aggregate["exact_accuracy"] = aggregate["exact"] / aggregate["samples"]
    aggregate["wer"] = aggregate["edits"] / aggregate["target_tokens"]
    return {"phrase_aggregate": aggregate, "by_source": buckets, "examples": examples}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    refuse_protected((
        args.phrase_root, args.citizen_train, args.citizen_validation,
        args.semlex_train, args.semlex_validation,
    ))
    checkpoint = torch.load(args.stage1, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("expected a landmark-only v17 Stage-1 checkpoint")
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    stage1 = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    stage1.load_state_dict(checkpoint["model_state_dict"], strict=True)
    evidence_device = torch.device(
        "mps" if args.evidence_device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.evidence_device == "auto" else args.evidence_device
    )
    head_device = torch.device(args.head_device)
    stage1.to(evidence_device).eval()
    raw_train_phrase = load_phrase_samples(args.phrase_root, "train")
    raw_val_phrase = load_phrase_samples(args.phrase_root, "validation")
    raw_train_isolated = load_isolated_samples(
        (args.citizen_train, args.semlex_train), labels, "isolated_train"
    )
    raw_val_isolated = load_isolated_samples(
        (args.citizen_validation, args.semlex_validation), labels, "isolated_validation"
    )
    evidence_started = time.perf_counter()
    train_phrase = extract_evidence(
        stage1, raw_train_phrase, evidence_device, args.evidence_batch_size,
        args.evidence_stride, args.minimum_prefix_frames,
    )
    val_phrase = extract_evidence(
        stage1, raw_val_phrase, evidence_device, args.evidence_batch_size,
        args.evidence_stride, args.minimum_prefix_frames,
    )
    train_isolated = extract_evidence(
        stage1, raw_train_isolated, evidence_device, args.evidence_batch_size,
        args.evidence_stride, args.minimum_prefix_frames,
    )
    val_isolated = extract_evidence(
        stage1, raw_val_isolated, evidence_device, args.evidence_batch_size,
        args.evidence_stride, args.minimum_prefix_frames,
    )
    evidence_seconds = time.perf_counter() - evidence_started
    del stage1
    if evidence_device.type == "mps":
        torch.mps.empty_cache()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = StreamingStage1CTCHeadV17(StreamingStage1HeadConfig(
        num_glosses=len(labels), hidden_dim=args.hidden_dim, blocks=args.blocks,
        dropout=args.dropout,
    )).to(head_device)
    dataset = EvidenceTrainingDataset(
        train_phrase, train_isolated, args.synthetic_count, args.phrase_repeats,
        args.seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = val_phrase + val_isolated
    history = []
    best_score = float("inf")
    best_epoch = 0
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        total = 0.0
        seen = 0
        for batch in loader:
            features = batch["features"].to(head_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = ctc(
                logits.log_softmax(-1).transpose(0, 1),
                batch["targets"].to(head_device), batch["lengths"].to(head_device),
                batch["target_lengths"].to(head_device),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        metrics = evaluate(model, validation, head_device, args.batch_size)
        sources = metrics["by_source"]
        local = sources["local_phrases"]["wer"]
        heldout = sources["asllrp_contiguous"]["wer"]
        score = 0.6 * heldout + 0.4 * local
        row = {
            "epoch": epoch, "train_loss": total / seen,
            "selection_score": score, "validation": metrics,
        }
        history.append(row)
        print(json.dumps({
            "epoch": epoch, "loss": row["train_loss"], "score": score,
            "local_wer": local, "heldout_asllrp_wer": heldout,
        }), flush=True)
        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "format": "slt_streaming_stage1_ctc_head_v17",
                "model_config": model.config.to_dict(),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "label_to_index": labels, "ctc_blank_index": 0,
                "stage1_checkpoint": str(args.stage1),
                "stage1_checkpoint_sha256": sha256(args.stage1),
                "evidence_stride_frames": args.evidence_stride,
                "minimum_prefix_frames": args.minimum_prefix_frames,
                "epoch": epoch, "validation": metrics,
                "test_accessed": False,
                "external_evaluation_reserved_accessed": False,
            }, args.output_dir / "best_model.pth")
    best = torch.load(args.output_dir / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final = evaluate(model.to(head_device), validation, head_device, args.batch_size)
    result = {
        "format": "slt_streaming_stage1_head_experiment_v17",
        "architecture": "frozen_stage1_rolling_evidence_plus_causal_depthwise_tcn_ctc",
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "checkpoint_bytes": (args.output_dir / "best_model.pth").stat().st_size,
        "receptive_field_steps": model.config.receptive_field_steps,
        "best_epoch": best_epoch, "best_selection_score": best_score,
        "validation": final, "history": history,
        "data_counts": {
            "train_phrase": len(train_phrase), "validation_phrase": len(val_phrase),
            "train_isolated": len(train_isolated),
            "validation_isolated": len(val_isolated),
        },
        "evidence_extraction_seconds": evidence_seconds,
        "head_training_seconds": time.perf_counter() - training_started,
        "evidence_device": str(evidence_device), "head_device": str(head_device),
        "test_accessed": False,
        "external_evaluation_reserved_accessed": False,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "best_epoch": best_epoch}))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage1", type=Path, default=Path("artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"))
    value.add_argument("--phrase-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    value.add_argument("--citizen-train", type=Path, default=Path("data/local/citizen100_v17/landmarks/train"))
    value.add_argument("--citizen-validation", type=Path, default=Path("data/local/citizen100_v17/landmarks/val"))
    value.add_argument("--semlex-train", type=Path, default=Path("data/local/semlex_citizen100_train_audit/landmarks_v17"))
    value.add_argument("--semlex-validation", type=Path, default=Path("data/local/semlex_citizen100_val_audit/landmarks_v17"))
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/models/streaming_stage1_head_v17_experiment_v1"))
    value.add_argument("--epochs", type=int, default=20)
    value.add_argument("--batch-size", type=int, default=128)
    value.add_argument("--evidence-batch-size", type=int, default=128)
    value.add_argument("--synthetic-count", type=int, default=3000)
    value.add_argument("--phrase-repeats", type=int, default=5)
    value.add_argument("--evidence-stride", type=int, default=4)
    value.add_argument("--minimum-prefix-frames", type=int, default=8)
    value.add_argument("--hidden-dim", type=int, default=64)
    value.add_argument("--blocks", type=int, default=3)
    value.add_argument("--dropout", type=float, default=0.10)
    value.add_argument("--learning-rate", type=float, default=2e-3)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--evidence-device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--head-device", choices=("cpu", "mps"), default="cpu")
    value.add_argument("--seed", type=int, default=17042)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
