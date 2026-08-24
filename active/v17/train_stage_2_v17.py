#!/usr/bin/env python3
"""Train the cached frozen-encoder v17 Stage-2 CTC head."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    Stage2TemporalHeadV17,
    Stage2V17Config,
    make_stage2_checkpoint,
)


LOG = logging.getLogger("train_stage_2_v17")
SEEDS = (1701, 1702, 1703)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Sample:
    features: np.ndarray
    targets: np.ndarray
    source: str
    item_id: str
    target_sequence: tuple[str, ...]


class RealPhraseDataset(Dataset):
    def __init__(self, root: Path, role: str):
        self.samples = []
        for path in sorted((root / role).glob("*/*.stage2_frozen_v17.npz")):
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                features = payload["frozen_features"].astype(np.float16)
                targets = payload["target_indices"].astype(np.int64) + 1
            if metadata.get("role") != role:
                raise ValueError(f"{path}: role mismatch")
            self.samples.append(Sample(
                features=features,
                targets=targets,
                source=str(metadata["source"]),
                item_id=str(metadata["source_item_id"]),
                target_sequence=tuple(metadata["target_sequence"]),
            ))
        if not self.samples:
            raise ValueError(f"no {role} frozen phrase features under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SyntheticCompositionDataset(Dataset):
    def __init__(self, pool_path: Path, plan_path: Path):
        with np.load(pool_path, allow_pickle=False) as payload:
            self.pool = payload["frozen_features"].astype(np.float16)
            self.pool_targets = payload["target_indices"].astype(np.int64)
            self.pool_sources = (
                payload["source_codes"].astype(np.uint8) if "source_codes" in payload.files else None
            )
            metadata = json.loads(str(payload["metadata_json"]))
        self.source_code_map = {
            int(key): str(value) for key, value in metadata.get("source_code_map", {}).items()
        }
        plan = json.loads(plan_path.read_text())
        if plan["pool_sha256"] != sha256(pool_path):
            raise ValueError("synthetic plan/pool hash mismatch")
        if metadata.get("source_split") not in {
            "citizen_official_train_only", "citizen_asllrp_train_only_replay",
            "citizen_semlex_asllrp_train_only_replay",
        }:
            raise ValueError("synthetic pool is not an approved training-only pool")
        self.rows = plan["rows"]
        self.signer_neutral: dict[str, np.ndarray] = {}
        for signer, indices in plan.get("signer_pool_indices", {}).items():
            index_array = np.asarray(indices, dtype=np.int64)
            if len(index_array) == 0 or self.pool_sources is None:
                raise ValueError(f"{signer}: invalid signer voice pool")
            declared_codes = np.unique(self.pool_sources[index_array])
            if len(declared_codes) != 1:
                raise ValueError(f"{signer}: signer voice spans multiple source datasets")
            endpoints = np.concatenate(
                (self.pool[index_array, :2], self.pool[index_array, -2:]), axis=1
            ).astype(np.float32)
            self.signer_neutral[str(signer)] = endpoints.mean(axis=(0, 1))
        transition_source_code = int(metadata.get("transition_scale_source_code", 1))
        asllrp_mask = (
            self.pool_sources == transition_source_code if self.pool_sources is not None else None
        )
        self.transition_scale = (
            self.pool[asllrp_mask].astype(np.float32).std(axis=(0, 1)).clip(0.05)
            if asllrp_mask is not None and asllrp_mask.any()
            else np.ones(FROZEN_TEMPORAL_FEATURE_DIM, dtype=np.float32)
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        pool_indices = np.asarray(row["pool_indices"], dtype=np.int64)
        targets = np.asarray(row["target_indices"], dtype=np.int64)
        if not np.array_equal(self.pool_targets[pool_indices], targets):
            raise ValueError(f"{row['sequence_id']}: pool label mismatch")
        source = str(row.get("source", "synthetic_citizen_train"))
        expected_source_code = None
        if self.pool_sources is not None:
            expected_source_code = row.get("pool_source_code")
            if expected_source_code is None:
                expected_source_code = {
                    "synthetic_citizen_train": 0,
                    "synthetic_asllrp_contextual_train": 1,
                }.get(source)
            expected_source_code = (
                int(expected_source_code) if expected_source_code is not None else None
            )
            if expected_source_code is None or not np.all(
                self.pool_sources[pool_indices] == expected_source_code
            ):
                raise ValueError(f"{row['sequence_id']}: pool source mismatch")
        features = self.pool[pool_indices]
        voice = row.get("signer_voice_synthesis")
        if voice is not None:
            signer = str(voice["signer_id"])
            if signer not in self.signer_neutral:
                raise ValueError(f"{row['sequence_id']}: signer voice is not declared")
            voice_source_code = int(voice.get(
                "content_pool_source_code", voice.get("pool_source_code", expected_source_code)
            ))
            if self.pool_sources is None or not np.all(
                self.pool_sources[pool_indices] == voice_source_code
            ):
                raise ValueError(f"{row['sequence_id']}: signer voice source mismatch")
            durations = [int(value) for value in voice["token_duration_frames"]]
            if len(durations) != len(features) or any(not 4 <= value <= 32 for value in durations):
                raise ValueError(f"{row['sequence_id']}: invalid signer timing")
            features = compose_signer_voice(
                features.astype(np.float32),
                durations,
                self.signer_neutral[signer],
                self.transition_scale,
                context_frames=int(voice.get("context_frames", 5)),
                bridge_frames=int(voice.get("bridge_frames", 2)),
                max_trim_frames=int(voice.get("max_trim_frames", 3)),
                minimum_keep_frames=int(voice.get("minimum_keep_frames", 4)),
            ).astype(np.float16)
        leading_padding = row.get("leading_padding_frames")
        if leading_padding is not None:
            leading_padding = int(leading_padding)
            if not 0 <= leading_padding < 32:
                raise ValueError(f"{row['sequence_id']}: invalid leading padding")
            stream = features.reshape(-1, FROZEN_TEMPORAL_FEATURE_DIM)
            if leading_padding:
                stream = np.concatenate([
                    np.zeros((leading_padding, FROZEN_TEMPORAL_FEATURE_DIM), np.float16),
                    stream,
                ])
            trailing_padding = (-len(stream)) % 32
            if trailing_padding:
                stream = np.concatenate([
                    stream,
                    np.zeros((trailing_padding, FROZEN_TEMPORAL_FEATURE_DIM), np.float16),
                ])
            features = stream.reshape(-1, 32, FROZEN_TEMPORAL_FEATURE_DIM)
        return Sample(
            features=features,
            targets=targets + 1,
            source=source,
            item_id=row["sequence_id"],
            target_sequence=tuple(str(value) for value in targets),
        )


def resample_temporal(value: np.ndarray, output_frames: int) -> np.ndarray:
    """Linearly resample a temporal feature trajectory without changing its endpoints."""
    if value.ndim != 2 or value.shape[0] < 1 or output_frames < 1:
        raise ValueError("invalid temporal resampling request")
    if value.shape[0] == output_frames:
        return value.astype(np.float32, copy=True)
    positions = np.linspace(0.0, value.shape[0] - 1, output_frames, dtype=np.float32)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, value.shape[0] - 1)
    weight = (positions - lower).reshape(-1, 1)
    return (
        value[lower].astype(np.float32) * (1.0 - weight)
        + value[upper].astype(np.float32) * weight
    )


def compose_signer_voice(
    tokens: np.ndarray,
    durations: list[int],
    neutral: np.ndarray,
    transition_scale: np.ndarray,
    *,
    context_frames: int,
    bridge_frames: int,
    max_trim_frames: int,
    minimum_keep_frames: int,
) -> np.ndarray:
    """Make a style-coherent, timing-aware phrase from one signer's real trajectories.

    Each normalized isolated trajectory is first restored to its observed duration.
    At a boundary, a monotonic frame-selection search trims only a small peripheral
    region, then inserts a short smooth bridge.  The resulting raw-time stream is
    repacked into the same resampled 32-frame windows used by the real extractor.
    """
    if tokens.ndim != 3 or tokens.shape[1:] != (32, FROZEN_TEMPORAL_FEATURE_DIM):
        raise ValueError("unexpected signer token tensor")
    if len(tokens) != len(durations) or not len(tokens):
        raise ValueError("signer tokens and durations differ")
    if neutral.shape != (FROZEN_TEMPORAL_FEATURE_DIM,) or transition_scale.shape != neutral.shape:
        raise ValueError("invalid signer style statistics")
    if min(context_frames, bridge_frames, max_trim_frames) < 0 or minimum_keep_frames < 2:
        raise ValueError("invalid coarticulation configuration")

    trajectories = [resample_temporal(token, duration) for token, duration in zip(tokens, durations)]
    stream = trajectories[0]
    for following in trajectories[1:]:
        tail_trim_limit = min(max_trim_frames, max(0, len(stream) - minimum_keep_frames))
        head_trim_limit = min(max_trim_frames, max(0, len(following) - minimum_keep_frames))
        best = (float("inf"), 0, 0)
        for tail_trim in range(tail_trim_limit + 1):
            tail = stream[-1 - tail_trim]
            for head_trim in range(head_trim_limit + 1):
                head = following[head_trim]
                distance = float(np.mean(np.square((tail - head) / transition_scale)))
                best = min(best, (distance, tail_trim, head_trim))
        _, tail_trim, head_trim = best
        if tail_trim:
            stream = stream[:-tail_trim]
        following = following[head_trim:]
        if bridge_frames:
            alpha = np.linspace(0.0, 1.0, bridge_frames + 2, dtype=np.float32)[1:-1, None]
            bridge = stream[-1:] * (1.0 - alpha) + following[:1] * alpha
            stream = np.concatenate((stream, bridge, following), axis=0)
        else:
            stream = np.concatenate((stream, following), axis=0)

    if context_frames:
        alpha = np.linspace(0.0, 1.0, context_frames + 2, dtype=np.float32)[1:-1, None]
        lead = neutral[None] * (1.0 - alpha) + stream[:1] * alpha
        trail = stream[-1:] * (1.0 - alpha) + neutral[None] * alpha
        stream = np.concatenate((lead, stream, trail), axis=0)
    windows = []
    for start in range(0, len(stream), 32):
        chunk = stream[start:start + 32]
        if len(chunk) < 4 and windows:
            windows[-1] = resample_temporal(
                np.concatenate((windows[-1], chunk), axis=0), 32
            )
        else:
            windows.append(resample_temporal(chunk, 32))
    if not windows or len(windows) > 8:
        raise ValueError("signer voice synthesis exceeds the Stage-2 window contract")
    return np.stack(windows).astype(np.float32)


class CombinedDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for dataset in datasets:
            self.offsets.append(total)
            total += len(dataset)
        self.length = total

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for dataset, offset in reversed(list(zip(self.datasets, self.offsets))):
            if index >= offset:
                return dataset[index - offset]
        raise IndexError(index)


def collate(samples: list[Sample]) -> dict[str, Any]:
    max_windows = max(sample.features.shape[0] for sample in samples)
    if max_windows > 8:
        raise ValueError("Stage-2 sample exceeds eight windows")
    features = np.zeros(
        (len(samples), max_windows, 32, FROZEN_TEMPORAL_FEATURE_DIM), dtype=np.float32
    )
    window_mask = np.zeros((len(samples), max_windows), dtype=np.bool_)
    target_lengths = np.asarray([len(sample.targets) for sample in samples], dtype=np.int64)
    for index, sample in enumerate(samples):
        windows = sample.features.shape[0]
        features[index, :windows] = sample.features
        window_mask[index, :windows] = True
    return {
        "features": torch.from_numpy(features),
        "window_mask": torch.from_numpy(window_mask),
        "targets": torch.from_numpy(np.concatenate([sample.targets for sample in samples])),
        "target_lengths": torch.from_numpy(target_lengths),
        "sources": [sample.source for sample in samples],
        "item_ids": [sample.item_id for sample in samples],
        "target_sequences": [sample.target_sequence for sample in samples],
    }


def sampler_weights(real: RealPhraseDataset, synthetic: SyntheticCompositionDataset):
    counts = Counter(sample.source for sample in real.samples)
    counts.update(str(row.get("source", "synthetic_citizen_train")) for row in synthetic.rows)
    if counts.get("synthetic_balanced_multivoice_train"):
        desired = {
            "local_phrases": 0.25,
            "asllrp_contiguous": 0.20,
            "synthetic_citizen_train": 0.15,
            "synthetic_multivoice_train": 0.15,
            "synthetic_balanced_multivoice_train": 0.25,
        }
    elif counts.get("synthetic_multivoice_train"):
        desired = {
            "local_phrases": 0.30,
            "asllrp_contiguous": 0.20,
            "synthetic_citizen_train": 0.20,
            "synthetic_multivoice_train": 0.30,
        }
    elif counts.get("synthetic_asllrp_contextual_train"):
        desired = {
            "local_phrases": 0.30,
            "asllrp_contiguous": 0.20,
            "synthetic_citizen_train": 0.20,
            "synthetic_asllrp_contextual_train": 0.30,
        }
    else:
        desired = {
            "local_phrases": 0.35,
            "asllrp_contiguous": 0.25,
            "synthetic_citizen_train": 0.40,
        }
    if set(counts) != set(desired):
        raise ValueError(f"unexpected training sources: {dict(counts)}")
    weights = [desired[sample.source] / counts[sample.source] for sample in real.samples]
    weights.extend(
        desired[str(row.get("source", "synthetic_citizen_train"))]
        / counts[str(row.get("source", "synthetic_citizen_train"))]
        for row in synthetic.rows
    )
    return torch.as_tensor(weights, dtype=torch.double), desired


def collapse_ctc(sequence: np.ndarray) -> list[int]:
    output = []
    previous = None
    for token in sequence.tolist():
        token = int(token)
        if token != previous and token != 0:
            output.append(token - 1)
        previous = token
    return output


def edit_distance(reference: list[int], hypothesis: list[int]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, predicted in enumerate(hypothesis, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + int(expected != predicted),
            ))
        previous = current
    return previous[-1]


def evaluate(model, loader, device) -> dict[str, Any]:
    model.eval()
    domains: dict[str, dict[str, float]] = defaultdict(
        lambda: {"edits": 0.0, "tokens": 0.0, "exact": 0.0, "samples": 0.0}
    )
    phrase_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"edits": 0.0, "tokens": 0.0, "exact": 0.0, "samples": 0.0}
    )
    with torch.inference_mode():
        for batch in loader:
            logits, lengths = model(
                batch["features"].to(device), batch["window_mask"].to(device)
            )
            predictions = logits.argmax(dim=-1).cpu().numpy()
            lengths = lengths.cpu().tolist()
            offset = 0
            target_lengths = batch["target_lengths"].tolist()
            flat_targets = batch["targets"].tolist()
            for index, (source, length, target_length) in enumerate(
                zip(batch["sources"], lengths, target_lengths)
            ):
                reference = [int(value) - 1 for value in flat_targets[offset:offset + target_length]]
                offset += target_length
                hypothesis = collapse_ctc(predictions[index, :length])
                edits = edit_distance(reference, hypothesis)
                key = " ".join(batch["target_sequences"][index])
                for stats in (domains[source], phrase_stats[f"{source}:{key}"]):
                    stats["edits"] += edits
                    stats["tokens"] += len(reference)
                    stats["exact"] += float(reference == hypothesis)
                    stats["samples"] += 1
    domain_metrics = {
        source: {
            "wer": values["edits"] / max(1.0, values["tokens"]),
            "sequence_accuracy": values["exact"] / max(1.0, values["samples"]),
            "samples": int(values["samples"]),
            "tokens": int(values["tokens"]),
        }
        for source, values in sorted(domains.items())
    }
    phrase_metrics = {
        key: {
            "wer": values["edits"] / max(1.0, values["tokens"]),
            "sequence_accuracy": values["exact"] / max(1.0, values["samples"]),
            "samples": int(values["samples"]),
        }
        for key, values in sorted(phrase_stats.items())
    }
    wers = [value["wer"] for value in domain_metrics.values()]
    accuracies = [value["sequence_accuracy"] for value in domain_metrics.values()]
    return {
        "domains": domain_metrics,
        "phrases": phrase_metrics,
        "equal_domain_mean_wer": float(np.mean(wers)),
        "worst_domain_wer": float(max(wers)),
        "equal_domain_mean_sequence_accuracy": float(np.mean(accuracies)),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_seed(seed, train_dataset, weights, val_loader, args, device):
    seed_everything(seed)
    model = Stage2TemporalHeadV17(Stage2V17Config()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch, replacement=True, generator=generator
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, collate_fn=collate,
    )
    best_key = None
    best_metrics = None
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = seen = 0
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, input_lengths = model(features, mask)
            loss = criterion(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets, input_lengths, target_lengths,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(mask)
            seen += len(mask)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        key = (
            -metrics["worst_domain_wer"],
            -metrics["equal_domain_mean_wer"],
            metrics["equal_domain_mean_sequence_accuracy"],
        )
        history.append({"epoch": epoch, "train_loss": running / max(1, seen), **metrics})
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 5 == 0 or patience == 0:
            LOG.info(
                "seed=%d epoch=%d loss=%.4f meanWER=%.4f worstWER=%.4f meanSeq=%.4f patience=%d",
                seed, epoch, running / max(1, seen), metrics["equal_domain_mean_wer"],
                metrics["worst_domain_wer"], metrics["equal_domain_mean_sequence_accuracy"], patience,
            )
        if patience >= args.patience:
            break
    if best_state is None or best_metrics is None or best_key is None:
        raise RuntimeError("training produced no checkpoint")
    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        make_stage2_checkpoint(
            model, best_state, seed=seed, epoch=best_epoch,
            validation_metrics=best_metrics, selection_key=list(best_key),
        ),
        seed_dir / "best_head.pth",
    )
    (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "validation_metrics": best_metrics,
        "checkpoint": (seed_dir / "best_head.pth").as_posix(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
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
        train_seed(seed, train_dataset, weights, val_loader, args, device) for seed in SEEDS
    ]
    winner = max(results, key=lambda result: tuple(result["selection_key"]))
    selected = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    selected.update({
        "selected_stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "selected_stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "real_training_manifest": args.training_manifest.as_posix(),
        "real_training_manifest_sha256": sha256(args.training_manifest),
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
    temporary = args.output / "best_model.pth.tmp"
    torch.save(selected, temporary)
    temporary.replace(args.output / "best_model.pth")
    report = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "selection_key": winner["selection_key"],
        "validation_metrics": winner["validation_metrics"],
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "train_real_samples": len(real_train),
        "train_synthetic_sequences": len(synthetic),
        "validation_samples": len(real_val),
        "trainable_parameters": Stage2TemporalHeadV17().parameter_count,
        "candidate_results": results,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument(
        "--synthetic-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
    )
    parser.add_argument(
        "--synthetic-plan", type=Path,
        default=Path("active/v17/stage2_synthetic_plan_v17.json"),
    )
    parser.add_argument(
        "--training-manifest", type=Path,
        default=Path("active/v17/stage2_training_manifest_v17.json"),
    )
    parser.add_argument(
        "--stage1-checkpoint", type=Path,
        default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_unified_ctc_v1"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
