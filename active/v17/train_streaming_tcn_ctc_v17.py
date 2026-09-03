#!/usr/bin/env python3
"""Train and compare tiny causal v17 landmark CTC models without test access."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.geometry_v17 import resample_features
from active.v17.model_streaming_tcn_ctc_v17 import (
    StreamingLandmarkTCNCTCV17,
    StreamingTCNConfig,
)


@dataclass(frozen=True)
class SequenceSample:
    frames: np.ndarray
    targets: tuple[int, ...]
    source: str
    item_id: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_protected(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered = {part.casefold() for part in path.parts}
        if "test" in lowered or "external_evaluation_reserved" in lowered:
            raise ValueError(f"streaming training refuses protected path: {path}")


def restore_source_frames(windows: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    restored = []
    for window, (start, stop) in zip(windows, ranges):
        count = int(stop) - int(start)
        if count > 0:
            restored.append(resample_features(window.astype(np.float32), count))
    if not restored:
        raise ValueError("phrase cache contains no recoverable source frames")
    return np.concatenate(restored).astype(np.float32)


def load_phrase_samples(root: Path, role: str) -> list[SequenceSample]:
    samples = []
    for path in sorted((root / role).glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata["role"] != role:
                raise ValueError(f"{path}: split mismatch")
            frames = restore_source_frames(
                payload["landmarks"], payload["window_source_ranges"]
            )
            targets = tuple(int(value) + 1 for value in payload["target_indices"])
        samples.append(SequenceSample(
            frames, targets, str(metadata["source"]), str(metadata["source_item_id"])
        ))
    if not samples:
        raise ValueError(f"no phrase samples under {root / role}")
    return samples


def load_isolated_samples(
    roots: Iterable[Path], labels: dict[str, int], source_prefix: str
) -> list[SequenceSample]:
    samples = []
    for root in roots:
        refuse_protected((root,))
        for label, index in labels.items():
            for path in sorted((root / label).glob("*.v17.npz")):
                with np.load(path, allow_pickle=False) as payload:
                    frames = payload["features"].astype(np.float32)
                if frames.shape != (32, 61, 5) or not np.isfinite(frames).all():
                    raise ValueError(f"invalid v17 archive: {path}")
                samples.append(SequenceSample(
                    frames, (index + 1,), f"{source_prefix}:{root.name}", str(path)
                ))
    return samples


class TrainingSequences(Dataset):
    def __init__(
        self,
        phrases: list[SequenceSample],
        isolated: list[SequenceSample],
        synthetic_count: int,
        seed: int,
    ):
        self.phrases = phrases
        self.isolated = isolated
        self.synthetic_count = synthetic_count
        self.seed = seed
        self.epoch = 0
        self.real_count = len(phrases) * 3 + len(isolated)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.real_count + self.synthetic_count

    def __getitem__(self, index: int) -> SequenceSample:
        if index < len(self.phrases) * 3:
            return self.phrases[index % len(self.phrases)]
        if index < self.real_count:
            return self.isolated[index - len(self.phrases) * 3]
        rng = random.Random(self.seed + self.epoch * self.synthetic_count + index)
        count = rng.randint(2, 5)
        chosen = [self.isolated[rng.randrange(len(self.isolated))] for _ in range(count)]
        pieces = []
        for sample in chosen:
            left, right = rng.randint(0, 3), rng.randint(0, 3)
            piece = sample.frames[left : len(sample.frames) - right or None]
            speed_frames = max(12, int(round(len(piece) * rng.uniform(0.65, 1.05))))
            pieces.append(resample_features(piece, speed_frames).astype(np.float32))
        return SequenceSample(
            np.concatenate(pieces),
            tuple(value for sample in chosen for value in sample.targets),
            "synthetic_isolated_composition",
            f"synthetic:{self.epoch}:{index}",
        )


def collate(samples: list[SequenceSample]) -> dict[str, object]:
    lengths = torch.tensor([len(sample.frames) for sample in samples], dtype=torch.long)
    target_lengths = torch.tensor([len(sample.targets) for sample in samples], dtype=torch.long)
    maximum = int(lengths.max())
    features = np.zeros((len(samples), maximum, 61, 5), dtype=np.float32)
    for index, sample in enumerate(samples):
        features[index, : len(sample.frames)] = sample.frames
    targets = torch.tensor(
        [value for sample in samples for value in sample.targets], dtype=torch.long
    )
    return {
        "features": torch.from_numpy(features),
        "lengths": lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "samples": samples,
    }


def collapse_ctc(path: np.ndarray) -> tuple[int, ...]:
    output = []
    previous = -1
    for value in path.tolist():
        if value != 0 and value != previous:
            output.append(int(value))
        previous = int(value)
    return tuple(output)


def aligned_auxiliary_loss(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    samples: list[SequenceSample],
) -> torch.Tensor:
    """A weak monotonic guide that prevents early all-blank CTC collapse."""
    pooled_segments = []
    targets = []
    for index, (length, sample) in enumerate(zip(lengths.tolist(), samples)):
        usable = logits[index, :length, 1:]
        for position, target in enumerate(sample.targets):
            start = round(position * len(usable) / len(sample.targets))
            stop = round((position + 1) * len(usable) / len(sample.targets))
            stop = max(start + 1, stop)
            pooled_segments.append(usable[start:stop].mean(dim=0))
            targets.append(target - 1)
    return nn.functional.cross_entropy(
        torch.stack(pooled_segments), torch.tensor(targets, device=logits.device)
    )


def edit_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(row[j] + 1, next_row[-1] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


@torch.inference_mode()
def evaluate(
    model: nn.Module, samples: list[SequenceSample], device: torch.device, batch_size: int
) -> dict[str, object]:
    model.eval()
    by_source: dict[str, dict[str, float]] = {}
    for offset in range(0, len(samples), batch_size):
        batch = collate(samples[offset : offset + batch_size])
        logits = model(batch["features"].to(device))[:, :: model.config.output_stride]
        paths = logits.argmax(dim=-1).cpu().numpy()
        output_lengths = (batch["lengths"] + model.config.output_stride - 1) // model.config.output_stride
        for sample, length, path in zip(batch["samples"], output_lengths, paths):
            predicted = collapse_ctc(path[: int(length)])
            bucket = by_source.setdefault(sample.source, {
                "samples": 0, "exact": 0, "edits": 0, "target_tokens": 0,
                "predicted_tokens": 0,
            })
            bucket["samples"] += 1
            bucket["exact"] += predicted == sample.targets
            bucket["edits"] += edit_distance(sample.targets, predicted)
            bucket["target_tokens"] += len(sample.targets)
            bucket["predicted_tokens"] += len(predicted)
    for bucket in by_source.values():
        bucket["exact_accuracy"] = bucket["exact"] / bucket["samples"]
        bucket["wer"] = bucket["edits"] / bucket["target_tokens"]
    phrase_buckets = [value for key, value in by_source.items() if key in {"local_phrases", "asllrp_contiguous"}]
    aggregate = {
        key: sum(bucket[key] for bucket in phrase_buckets)
        for key in ("samples", "exact", "edits", "target_tokens", "predicted_tokens")
    }
    aggregate["exact_accuracy"] = aggregate["exact"] / aggregate["samples"]
    aggregate["wer"] = aggregate["edits"] / aggregate["target_tokens"]
    return {"phrase_aggregate": aggregate, "by_source": by_source}


def checkpoint_payload(
    model: StreamingLandmarkTCNCTCV17,
    labels: dict[str, int],
    epoch: int,
    metrics: dict[str, object],
    provenance: dict[str, object],
) -> dict[str, object]:
    return {
        "format": "slt_streaming_tcn_ctc_v17",
        "model_config": model.config.to_dict(),
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "label_to_index": labels,
        "ctc_blank_index": 0,
        "epoch": epoch,
        "validation": metrics,
        "provenance": provenance,
        "test_accessed": False,
        "external_evaluation_reserved_accessed": False,
    }


def train_variant(
    name: str,
    config: StreamingTCNConfig,
    train_phrases: list[SequenceSample],
    train_isolated: list[SequenceSample],
    validation: list[SequenceSample],
    labels: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    provenance: dict[str, object],
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = StreamingLandmarkTCNCTCV17(config).to(device)
    dataset = TrainingSequences(
        train_phrases, train_isolated, args.synthetic_count, args.seed
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            frame_logits = model(features)
            logits = frame_logits[:, :: model.config.output_stride]
            output_lengths = (
                batch["lengths"] + model.config.output_stride - 1
            ) // model.config.output_stride
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            ctc_loss = loss_function(
                log_probs, targets, output_lengths.to(device),
                batch["target_lengths"].to(device),
            )
            auxiliary = aligned_auxiliary_loss(
                logits, output_lengths, batch["samples"]
            )
            loss = ctc_loss + args.auxiliary_weight * auxiliary
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
            seen += len(features)
        metrics = evaluate(model, validation, device, args.batch_size)
        by_source = metrics["by_source"]
        local_wer = by_source.get("local_phrases", {"wer": 1.0})["wer"]
        heldout_wer = by_source.get("asllrp_contiguous", {"wer": 1.0})["wer"]
        score = 0.6 * heldout_wer + 0.4 * local_wer
        row = {
            "epoch": epoch, "train_loss": total_loss / seen,
            "selection_score": score, "validation": metrics,
        }
        history.append(row)
        print(json.dumps({"variant": name, **row}), flush=True)
        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save(
                checkpoint_payload(model, labels, epoch, metrics, provenance),
                output / "best_model.pth",
            )
    checkpoint = torch.load(output / "best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate(model.to(device), validation, device, args.batch_size)
    result = {
        "variant": name,
        "model_config": config.to_dict(),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "validation": final_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(output / "best_model.pth"),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    input_paths = (
        args.phrase_root, args.citizen_train, args.citizen_validation,
        args.semlex_train, args.semlex_validation, args.training_manifest,
    )
    refuse_protected(input_paths)
    manifest = json.loads(args.training_manifest.read_text())
    if any(manifest.get(key) for key in (
        "citizen_test_accessed", "local_test_accessed", "semlex_test_accessed"
    )):
        raise ValueError("training manifest reports protected test access")
    labels = {str(key): int(value) for key, value in manifest["label_to_index"].items()}
    train_phrases = load_phrase_samples(args.phrase_root, "train")
    val_phrases = load_phrase_samples(args.phrase_root, "validation")
    train_isolated = load_isolated_samples(
        (args.citizen_train, args.semlex_train), labels, "isolated_train"
    )
    val_isolated = load_isolated_samples(
        (args.citizen_validation, args.semlex_validation), labels, "isolated_validation"
    )
    validation = val_phrases + val_isolated
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "training_manifest": str(args.training_manifest),
        "training_manifest_sha256": sha256(args.training_manifest),
        "counts": {
            "train_phrases": len(train_phrases),
            "train_isolated": len(train_isolated),
            "validation_phrases": len(val_phrases),
            "validation_isolated": len(val_isolated),
        },
        "device": str(device),
        "test_accessed": False,
        "external_evaluation_reserved_accessed": False,
    }
    variants = {}
    for name, use_face_body in (
        ("hands_only", False), ("all_v17_landmarks", True)
    ):
        if name not in args.variants:
            continue
        variants[name] = train_variant(
            name,
            StreamingTCNConfig(
                num_glosses=len(labels), group_dim=args.group_dim,
                hidden_dim=args.hidden_dim, blocks=args.blocks,
                dropout=args.dropout, use_face_body=use_face_body,
            ),
            train_phrases, train_isolated, validation, labels, args, device, provenance,
        )
    selected = min(variants, key=lambda key: variants[key]["best_selection_score"])
    result = {
        "format": "slt_streaming_tcn_ctc_experiment_v17",
        "selected_variant": selected,
        "variants": variants,
        "provenance": provenance,
        "test_accessed": False,
        "external_evaluation_reserved_accessed": False,
    }
    (args.output_dir / "experiment_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"selected_variant": selected, "output": str(args.output_dir)}))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phrase-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    value.add_argument("--citizen-train", type=Path, default=Path("data/local/citizen100_v17/landmarks/train"))
    value.add_argument("--citizen-validation", type=Path, default=Path("data/local/citizen100_v17/landmarks/val"))
    value.add_argument("--semlex-train", type=Path, default=Path("data/local/semlex_citizen100_train_audit/landmarks_v17"))
    value.add_argument("--semlex-validation", type=Path, default=Path("data/local/semlex_citizen100_val_audit/landmarks_v17"))
    value.add_argument("--training-manifest", type=Path, default=Path("active/v17/stage2_training_manifest_v17.json"))
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/models/streaming_tcn_ctc_v17_experiment_v1"))
    value.add_argument("--epochs", type=int, default=12)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--synthetic-count", type=int, default=2500)
    value.add_argument("--learning-rate", type=float, default=2e-3)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--group-dim", type=int, default=24)
    value.add_argument("--hidden-dim", type=int, default=96)
    value.add_argument("--blocks", type=int, default=4)
    value.add_argument("--dropout", type=float, default=0.10)
    value.add_argument("--auxiliary-weight", type=float, default=1.0)
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--seed", type=int, default=17041)
    value.add_argument(
        "--variants", nargs="+", choices=("hands_only", "all_v17_landmarks"),
        default=("hands_only", "all_v17_landmarks"),
    )
    return value


if __name__ == "__main__":
    run(parser().parse_args())
