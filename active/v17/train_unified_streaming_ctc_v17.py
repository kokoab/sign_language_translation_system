#!/usr/bin/env python3
"""Train a separate CTC head on frozen, pooled v17 Stage-1 window evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.approved_phrase_data_v17 import DEFAULT_MANIFEST, APPROVED_ROOT, require_training_manifest
from active.v17.model_unified_streaming_ctc_v17 import (
    UnifiedStreamingCTCConfig,
    UnifiedStreamingCTCHeadV17,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.geometry_v17 import resample_features
from active.v17.schema_stage2_features_v17 import landmark_config
from active.v17.schema_v17 import schema_fingerprint
from active.v17.train_stage_1_reel_emission_v17 import (
    asllrp_annotations,
    collect_samples,
)
from active.v17.train_streaming_tcn_ctc_v17 import (
    collapse_ctc,
    edit_distance,
    load_isolated_samples,
    restore_source_frames,
)


@dataclass(frozen=True)
class RawSequence:
    windows: np.ndarray
    targets: tuple[int, ...]
    source: str
    identity: str
    source_frames: int | None = None


@dataclass(frozen=True)
class EvidenceSequence:
    evidence: np.ndarray
    targets: tuple[int, ...]
    source: str
    identity: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_protected(*paths: Path) -> None:
    for path in paths:
        if "test" in {part.casefold() for part in path.parts}:
            raise ValueError(f"refusing protected path: {path}")


def rolling_windows(
    frames: np.ndarray, stride: int, window_frames: int = 32,
) -> np.ndarray:
    minimum_frames = min(12, window_frames)
    endpoints = list(range(minimum_frames, len(frames) + 1, stride))
    if not endpoints or endpoints[-1] != len(frames):
        endpoints.append(len(frames))
    return np.stack([
        resample_features(
            frames[max(0, end - window_frames) : end], 32
        ).astype(np.float32)
        for end in endpoints
    ])


def validate_phrase_archive(cached, ranges, targets, metadata, labels):
    """Reject incompatible or incomplete supervision before temporal resampling."""
    fingerprint = metadata.get("schema", {}).get(
        "landmark_schema_fingerprint", metadata.get("schema_fingerprint"))
    if fingerprint != schema_fingerprint(landmark_config()):
        raise ValueError("incompatible landmark schema")
    if cached.ndim != 4 or cached.shape[1:] != (32, 61, 5) or not np.isfinite(cached).all():
        raise ValueError("invalid landmark tensor")
    if (not len(cached) or ranges.shape != (len(cached), 2)
            or not np.issubdtype(ranges.dtype, np.integer)
            or ranges[0, 0] != 0 or np.any(ranges[:, 1] <= ranges[:, 0])
            or np.any(ranges[1:, 0] != ranges[:-1, 1])):
        raise ValueError("source ranges must be contiguous, non-overlapping and start at zero")
    frames = metadata.get("sampled_source_frames")
    if (not isinstance(frames, int) or isinstance(frames, bool)
            or frames != int(ranges[-1, 1]) or metadata.get("dropped_tail_frames", 0) != 0):
        raise ValueError("incomplete source-frame coverage; recover/review cache before admission")
    if (targets.ndim != 1 or not len(targets) or not np.issubdtype(targets.dtype, np.integer)
            or np.any(targets < 0) or np.any(targets > len(labels))):
        raise ValueError("invalid target indices")
    names = metadata.get("target_sequence")
    # Some existing NCSLGR metadata preserves raw unsupported gloss names.
    # This checks the stored mapping, not the semantic validity of an OOV label.
    if (not isinstance(names, list) or len(names) != len(targets)
            or any(not isinstance(name, str) for name in names)
            or [labels.get(name, len(labels)) for name in names] != targets.tolist()):
        raise ValueError("target names disagree with frozen vocabulary indices")


def phrase_sequences(
    root: Path, role: str, rolling_stride: int, rolling_window_frames: int = 32,
    *, labels: dict[str, int] | None = None,
    evidence_level: str = "window",
) -> list[RawSequence]:
    refuse_protected(root / role)
    if (rolling_stride < 0 or not 1 <= rolling_window_frames <= 32
            or evidence_level not in {"window", "frame"}):
        raise ValueError("invalid rolling window configuration")
    manifest = json.loads(Path(__file__).with_name("citizen100_manifest.json").read_text())
    frozen = {row["canonical_label"]: row["class_index"] for row in manifest["classes"]}
    if labels is not None and labels != frozen:
        raise ValueError("checkpoint vocabulary differs from frozen Citizen100 mapping")
    labels = frozen
    output = []
    for path in sorted((root / role).glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if metadata["role"] != role:
                raise ValueError(f"split mismatch: {path}")
            cached = payload["landmarks"].astype(np.float32)
            ranges, indices = payload["window_source_ranges"], payload["target_indices"]
            try:
                validate_phrase_archive(cached, ranges, indices, metadata, labels)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
            if rolling_stride:
                frames = restore_source_frames(
                    cached, ranges
                )
                windows = rolling_windows(
                    frames, rolling_stride, rolling_window_frames
                )
            else:
                windows = cached
            targets = tuple(int(value) + 1 for value in indices)
            minimum_steps = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
            if evidence_level == "window" and minimum_steps > len(windows):
                raise ValueError(f"{path}: CTC target cannot fit available observation steps")
        output.append(RawSequence(
            windows, targets, str(metadata["source"]),
            str(metadata["source_item_id"]),
            int(ranges[-1, 1]),
        ))
    return output


def isolated_sequences(
    roots: list[Path], labels: dict[str, int]
) -> list[RawSequence]:
    loaded = load_isolated_samples(roots, labels, "isolated")
    return [
        RawSequence(value.frames[None], value.targets, value.source, value.item_id)
        for value in loaded
    ]


def blank_sequences(
    split: str, labels: dict[str, int], args: argparse.Namespace,
    annotations: dict[str, tuple[dict[str, object], list[dict[str, object]]]],
) -> list[RawSequence]:
    semlex_root = args.semlex_train_root if split == "train" else args.semlex_val_root
    samples = collect_samples(
        split=split, labels=labels, no_emit_index=100,
        citizen_root=args.citizen_root, semlex_root=semlex_root,
        phrase_root=args.phrase_root, annotations=annotations,
        per_class=args.boundary_per_class,
    )
    allowed = {"transition"} if args.blank_policy == "transitions_only" else {
        "prefix", "transition"
    }
    return [
        RawSequence(
            value.features[None], (), f"blank:{value.domain}:{value.kind}",
            value.identity,
        )
        for value in samples if value.kind in allowed
    ]


def full_local_other_sequences(
    root: Path, role: str, labels: dict[str, int], other_index: int,
    rolling_stride: int, rolling_window_frames: int = 32,
) -> list[RawSequence]:
    output = []
    aliases = {"ME": "I"}
    for path in sorted((root / role / "local_phrase_full").glob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            sequence = [
                aliases.get(str(value), str(value))
                for value in metadata["target_sequence"]
            ]
            if all(value in labels for value in sequence):
                continue
            frames = payload["observation_features"].astype(np.float32)
        targets = []
        for label in sequence:
            target = labels[label] + 1 if label in labels else other_index
            if not targets or target != other_index or targets[-1] != other_index:
                targets.append(target)
        windows = (
            rolling_windows(frames, rolling_stride, rolling_window_frames)
            if rolling_stride else np.stack([
                resample_features(frames[start : start + 32], 32).astype(np.float32)
                for start in range(0, len(frames), 32)
            ])
        )
        output.append(RawSequence(
            windows, tuple(targets), "local_phrase_other",
            str(metadata["source_item_id"]),
        ))
    return output


@torch.inference_mode()
def encode(
    model: SLTStage1V17, samples: list[RawSequence], device: torch.device,
    batch_size: int, evidence_level: str,
) -> list[EvidenceSequence]:
    counts = [len(sample.windows) for sample in samples]
    windows = np.concatenate([sample.windows for sample in samples])
    rows = []
    model.to(device).eval()
    for start in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[start : start + batch_size]).to(device)
        if evidence_level == "window":
            logits, pooled = model(batch, return_embeddings=True)
            evidence = torch.cat((pooled, logits), dim=-1).unsqueeze(1)
        else:
            encoded, _ = model.encode(batch)
            logits = model.classifier(encoded)
            evidence = torch.cat((encoded, logits), dim=-1)
        rows.append(evidence.half().cpu().numpy())
    model.cpu()
    evidence = np.concatenate(rows)
    output, cursor = [], 0
    for sample, count in zip(samples, counts):
        output.append(EvidenceSequence(
            evidence[cursor : cursor + count].reshape(-1, evidence.shape[-1]),
            sample.targets,
            sample.source, sample.identity,
        ))
        minimum_steps = len(sample.targets) + sum(
            a == b for a, b in zip(sample.targets, sample.targets[1:]))
        if minimum_steps > len(output[-1].evidence):
            raise ValueError(f"{sample.identity}: CTC target cannot fit encoded evidence")
        cursor += count
    return output


class Sequences(Dataset):
    def __init__(self, samples: list[EvidenceSequence]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> EvidenceSequence:
        return self.samples[index]


def collate(samples: list[EvidenceSequence]) -> dict[str, object]:
    lengths = torch.tensor([len(value.evidence) for value in samples], dtype=torch.long)
    target_lengths = torch.tensor([len(value.targets) for value in samples], dtype=torch.long)
    evidence = np.zeros(
        (len(samples), int(lengths.max()), samples[0].evidence.shape[-1]), np.float32
    )
    for index, sample in enumerate(samples):
        evidence[index, : len(sample.evidence)] = sample.evidence
    targets = torch.tensor(
        [target for sample in samples for target in sample.targets], dtype=torch.long
    )
    return {
        "evidence": torch.from_numpy(evidence), "lengths": lengths,
        "targets": targets, "target_lengths": target_lengths, "samples": samples,
    }


def group(
    sample: EvidenceSequence, other_index: int, source_balanced: bool = False,
) -> str:
    if not sample.targets:
        return "blank"
    if other_index in sample.targets:
        return "other"
    if sample.source.startswith("isolated:"):
        if source_balanced:
            return "isolated:citizen" if sample.source == "isolated:train" else "isolated:semlex"
        return "isolated"
    if source_balanced:
        return (
            "phrase:asllrp" if sample.source == "asllrp_contiguous"
            else "phrase:local"
        )
    return "phrase"


def known(sequence: tuple[int, ...], other_index: int) -> tuple[int, ...]:
    return tuple(value for value in sequence if value != other_index)


@torch.inference_mode()
def evaluate(
    model: nn.Module, samples: list[EvidenceSequence], device: torch.device,
    batch_size: int, other_index: int,
) -> dict[str, object]:
    model.eval()
    by_source: dict[str, dict[str, float]] = {}
    for start in range(0, len(samples), batch_size):
        batch = collate(samples[start : start + batch_size])
        paths = model(batch["evidence"].to(device)).argmax(-1).cpu().numpy()
        for sample, length, path in zip(batch["samples"], batch["lengths"], paths):
            predicted = collapse_ctc(path[: int(length)])
            expected_known, predicted_known = (
                known(sample.targets, other_index), known(predicted, other_index)
            )
            bucket = by_source.setdefault(sample.source, {
                "samples": 0, "exact": 0, "known_edits": 0,
                "known_target_tokens": 0, "known_predicted_tokens": 0,
                "false_emission_samples": 0,
            })
            bucket["samples"] += 1
            bucket["exact"] += predicted == sample.targets
            bucket["known_edits"] += edit_distance(expected_known, predicted_known)
            bucket["known_target_tokens"] += len(expected_known)
            bucket["known_predicted_tokens"] += len(predicted_known)
            bucket["false_emission_samples"] += not sample.targets and bool(predicted_known)

    def finish(bucket: dict[str, float]) -> dict[str, float]:
        bucket["exact_accuracy"] = bucket["exact"] / max(bucket["samples"], 1)
        bucket["known_wer"] = bucket["known_edits"] / max(bucket["known_target_tokens"], 1)
        bucket["false_emission_rate"] = (
            bucket["false_emission_samples"] / max(bucket["samples"], 1)
        )
        return bucket

    for bucket in by_source.values():
        finish(bucket)

    def aggregate(prefixes: tuple[str, ...]) -> dict[str, float]:
        selected = [value for key, value in by_source.items() if key.startswith(prefixes)]
        keys = (
            "samples", "exact", "known_edits", "known_target_tokens",
            "known_predicted_tokens", "false_emission_samples",
        )
        return finish({key: sum(value[key] for value in selected) for key in keys})

    return {
        "exact_phrases": aggregate(("local_phrases", "asllrp_contiguous")),
        "isolated": aggregate(("isolated:",)),
        "blank_boundaries": aggregate(("blank:",)),
        "other_spans": aggregate(("asllrp_other_ctc",)),
        "by_source": by_source,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    dataset_provenance = require_training_manifest(args)
    refuse_protected(
        args.phrase_root, args.other_root, args.citizen_root,
        args.semlex_train_root, args.semlex_val_root,
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("base must be a v17 Stage-1 checkpoint")
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    if sorted(labels.values()) != list(range(100)):
        raise ValueError("expected exactly 100 locked glosses")
    base = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    annotations = asllrp_annotations(args.span_manifest, args.segmented_manifest)

    train_raw = phrase_sequences(
        args.phrase_root, "train", args.rolling_stride,
        args.rolling_window_frames, labels=labels, evidence_level=args.evidence_level,
    )
    validation_raw = phrase_sequences(
        args.phrase_root, "validation", args.rolling_stride,
        args.rolling_window_frames, labels=labels, evidence_level=args.evidence_level,
    )
    train_raw += phrase_sequences(
        args.other_root, "train", args.rolling_stride,
        args.rolling_window_frames, labels=labels, evidence_level=args.evidence_level,
    )
    validation_raw += phrase_sequences(
        args.other_root, "validation", args.rolling_stride,
        args.rolling_window_frames, labels=labels, evidence_level=args.evidence_level,
    )
    if args.full_local_root is not None:
        other_index = len(labels) + 1
        train_raw += full_local_other_sequences(
            args.full_local_root, "train", labels, other_index,
            args.rolling_stride, args.rolling_window_frames,
        )
        validation_raw += full_local_other_sequences(
            args.full_local_root, "validation", labels, other_index,
            args.rolling_stride, args.rolling_window_frames,
        )
    train_raw += isolated_sequences([
        args.citizen_root / "train",
        args.semlex_train_root / "full_clean_landmarks_v17",
    ], labels)
    validation_raw += isolated_sequences([
        args.citizen_root / "val", args.semlex_val_root / "landmarks_v17",
    ], labels)
    train_raw += blank_sequences("train", labels, args, annotations)
    validation_raw += blank_sequences("validation", labels, args, annotations)

    started = time.perf_counter()
    train = encode(
        base, train_raw, device, args.embedding_batch_size, args.evidence_level
    )
    validation = encode(
        base, validation_raw, device, args.embedding_batch_size,
        args.evidence_level,
    )
    config = UnifiedStreamingCTCConfig(
        stage1_dim=base.config.dim, num_glosses=base.config.num_classes,
        hidden_dim=args.hidden_dim, blocks=args.blocks, dropout=args.dropout,
    )
    model = UnifiedStreamingCTCHeadV17(config).to(device)
    if args.freeze_gloss_delta:
        for parameter in model.gloss_delta.parameters():
            parameter.requires_grad = False
    loader = DataLoader(
        Sequences(train), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    counts = Counter(
        group(value, config.other_index, args.source_balanced_groups)
        for value in train
    )
    weights = {
        key: len(train) / (len(counts) * count) for key, count in counts.items()
    }
    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    ctc = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for batch in loader:
            logits = model(batch["evidence"].to(device))
            rows = ctc(
                logits.log_softmax(-1).transpose(0, 1),
                batch["targets"].to(device), batch["lengths"].to(device),
                batch["target_lengths"].to(device),
            )
            weight = torch.tensor(
                [
                    weights[group(
                        value, config.other_index, args.source_balanced_groups
                    )]
                    for value in batch["samples"]
                ],
                dtype=rows.dtype, device=device,
            )
            loss = (rows * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch["samples"])
            seen += len(batch["samples"])
        metrics = evaluate(model, validation, device, args.batch_size, config.other_index)
        phrase, isolated, blank = (
            metrics["exact_phrases"], metrics["isolated"], metrics["blank_boundaries"]
        )
        if args.selection_policy == "continuous_balanced":
            sources = metrics["by_source"]
            score = (
                0.50 * sources["asllrp_contiguous"]["known_wer"]
                + 0.20 * sources["local_phrases"]["known_wer"]
                + 0.20 * sources["isolated:val"]["known_wer"]
                + 0.10 * blank["false_emission_rate"]
            )
        else:
            score = (
                phrase["known_wer"] + 0.5 * blank["false_emission_rate"]
                + 0.25 * isolated["known_wer"]
            )
        row = {
            "epoch": epoch, "loss": total / seen, "selection_score": score,
            "phrase_exact_accuracy": phrase["exact_accuracy"],
            "phrase_known_wer": phrase["known_wer"],
            "asllrp_exact_accuracy": metrics["by_source"][
                "asllrp_contiguous"
            ]["exact_accuracy"],
            "asllrp_known_wer": metrics["by_source"][
                "asllrp_contiguous"
            ]["known_wer"],
            "isolated_exact_accuracy": isolated["exact_accuracy"],
            "blank_false_emission_rate": blank["false_emission_rate"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or score < best["score"]:
            best = {
                "score": score, "epoch": epoch,
                "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            }
    if best is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best["state"])
    metrics = evaluate(model, validation, device, args.batch_size, config.other_index)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "best_model.pth"
    torch.save({
        "dataset_provenance": dataset_provenance,
        "format": "slt_unified_streaming_ctc_v17", "version": 1,
        "head_config": config.to_dict(), "head_state_dict": model.cpu().state_dict(),
        "base_checkpoint": str(args.base), "base_checkpoint_sha256": sha256(args.base),
        "label_to_index": labels, "ctc_blank_index": 0,
        "other_label": "__OTHER__", "other_index": config.other_index,
        "decode_policy": "greedy CTC; drop blank and OTHER; no phrase grammar or language prior",
        "training_policy": {
            "rolling_stride_frames": args.rolling_stride,
            "rolling_window_frames": args.rolling_window_frames,
            "gloss_delta_frozen": args.freeze_gloss_delta,
            "evidence_level": args.evidence_level,
            "blank_policy": args.blank_policy,
            "full_local_root": None if args.full_local_root is None else str(
                args.full_local_root
            ),
            "source_balanced_groups": args.source_balanced_groups,
            "selection_policy": args.selection_policy,
        },
        "selected_epoch": best["epoch"], "validation": metrics,
        "test_accessed": False, "external_evaluation_reserved_accessed": False,
    }, output)
    result = {
        "dataset_provenance": dataset_provenance,
        "format": "slt_unified_streaming_ctc_experiment_v17",
        "output": str(output), "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "selected_epoch": best["epoch"],
        "head_parameters": sum(value.numel() for value in model.parameters()),
        "head_receptive_field_steps": config.receptive_field_steps,
        "rolling_stride_frames": args.rolling_stride,
        "rolling_window_frames": args.rolling_window_frames,
        "gloss_delta_frozen": args.freeze_gloss_delta,
        "evidence_level": args.evidence_level,
        "blank_policy": args.blank_policy,
        "source_balanced_groups": args.source_balanced_groups,
        "selection_policy": args.selection_policy,
        "train_samples": len(train), "validation_samples": len(validation),
        "train_sources": dict(Counter(value.source for value in train)),
        "validation_sources": dict(Counter(value.source for value in validation)),
        "group_counts": dict(counts), "validation": metrics, "history": history,
        "test_accessed": False, "external_evaluation_reserved_accessed": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--base", type=Path, default=Path(
        "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"
    ))
    value.add_argument("--phrase-root", type=Path, default=APPROVED_ROOT / "phrases")
    value.add_argument("--other-root", type=Path, default=APPROVED_ROOT / "other")
    value.add_argument("--citizen-root", type=Path, default=Path(
        "data/local/citizen100_v17/landmarks"
    ))
    value.add_argument("--semlex-train-root", type=Path, default=Path(
        "data/local/semlex_citizen100_train_audit"
    ))
    value.add_argument("--semlex-val-root", type=Path, default=Path(
        "data/local/semlex_citizen100_val_audit"
    ))
    value.add_argument("--span-manifest", type=Path, default=Path(
        "data/local/asllrp_contiguous_phrases_v17/manifest.json"
    ))
    value.add_argument("--segmented-manifest", type=Path, default=Path(
        "data/local/asllrp_segmented_citizen100_v17/manifest.json"
    ))
    value.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/models/unified_streaming_window_ctc_v17_experiment_v1"
    ))
    value.add_argument("--epochs", type=int, default=12)
    value.add_argument("--batch-size", type=int, default=32)
    value.add_argument("--embedding-batch-size", type=int, default=128)
    value.add_argument("--boundary-per-class", type=int, default=10)
    value.add_argument(
        "--blank-policy", choices=("all_prefixes", "transitions_only"),
        default="all_prefixes",
    )
    value.add_argument("--full-local-root", type=Path)
    value.add_argument(
        "--source-balanced-groups", action="store_true",
        help="give local/ASLLRP phrase and Citizen/SemLex replay equal task mass",
    )
    value.add_argument(
        "--selection-policy",
        choices=("aggregate", "continuous_balanced"), default="aggregate",
    )
    value.add_argument(
        "--evidence-level", choices=("window", "frame"), default="window",
    )
    value.add_argument(
        "--rolling-stride", type=int, default=4,
        help="source-frame stride; zero keeps the cached non-overlapping windows",
    )
    value.add_argument(
        "--rolling-window-frames", type=int, default=32,
        help="source frames in each trailing window before resampling to 32",
    )
    value.add_argument("--hidden-dim", type=int, default=128)
    value.add_argument("--blocks", type=int, default=3)
    value.add_argument("--dropout", type=float, default=0.10)
    value.add_argument("--learning-rate", type=float, default=2e-3)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument(
        "--freeze-gloss-delta", action="store_true",
        help="preserve Stage-1's 100-way gloss scores and learn timing/OOV only",
    )
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--seed", type=int, default=17051)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
