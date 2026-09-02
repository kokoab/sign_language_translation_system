#!/usr/bin/env python3
"""Repair Stage 2 with class-balanced isolated replay and boundary augmentation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.18")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.08")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    Stage2TemporalHeadV17,
    Stage2V17Config,
    load_stage2_model_v17,
    make_stage2_checkpoint,
)
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    Sample,
    SyntheticCompositionDataset,
    collate,
    collapse_ctc,
    evaluate,
    resample_temporal,
)


LOG = logging.getLogger("train_stage_2_accuracy_repair_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IsolatedPoolDataset(Dataset):
    def __init__(self, path: Path, source: str, *, augment_boundaries: bool):
        with np.load(path, allow_pickle=False) as payload:
            self.features = payload["frozen_features"].astype(np.float16)
            self.targets = payload["target_indices"].astype(np.int64)
            metadata = json.loads(str(payload["metadata_json"]))
            raw_ids = (
                payload["item_ids"].astype(str).tolist()
                if "item_ids" in payload.files else metadata.get("item_ids", [])
            )
        if self.features.shape != (len(self.targets), 32, FROZEN_TEMPORAL_FEATURE_DIM):
            raise ValueError(f"{path}: invalid isolated feature shape")
        expected_hash = metadata.get("stage1_checkpoint_sha256")
        if expected_hash is None:
            raise ValueError(f"{path}: missing Stage-1 checkpoint pin")
        self.stage1_checkpoint_sha256 = str(expected_hash)
        self.source = source
        self.augment_boundaries = augment_boundaries
        self.item_ids = raw_ids or [f"{source}:{index}" for index in range(len(self.targets))]
        if len(self.item_ids) != len(self.targets):
            raise ValueError(f"{path}: isolated item IDs differ from features")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Sample:
        features = self.features[index].astype(np.float32)[None]
        if self.augment_boundaries:
            features = boundary_shift(features)
        target = int(self.targets[index])
        return Sample(
            features=features,
            targets=np.asarray([target + 1], dtype=np.int64),
            source=self.source,
            item_id=self.item_ids[index],
            target_sequence=(str(target),),
        )


class BoundaryAugmentedDataset(Dataset):
    def __init__(self, base: Dataset, probability: float):
        self.base = base
        self.probability = probability

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Sample:
        sample = self.base[index]
        if random.random() >= self.probability:
            return sample
        return replace(sample, features=boundary_shift(sample.features))


def boundary_shift(features: np.ndarray) -> np.ndarray:
    """Shift a frozen stream across arbitrary 32-frame recording boundaries."""
    if features.ndim not in (2, 3) or features.shape[-1] != FROZEN_TEMPORAL_FEATURE_DIM:
        raise ValueError("invalid frozen stream for boundary shift")
    stream = features.reshape(-1, FROZEN_TEMPORAL_FEATURE_DIM).astype(np.float32)
    padding_budget = max(0, 256 - len(stream))
    leading = random.randint(0, min(31, padding_budget))
    trailing = random.randint(0, min(31, padding_budget - leading))
    if leading:
        stream = np.concatenate((np.zeros((leading, stream.shape[1]), np.float32), stream))
    if trailing:
        stream = np.concatenate((stream, np.zeros((trailing, stream.shape[1]), np.float32)))
    windows: list[np.ndarray] = []
    for start in range(0, len(stream), 32):
        chunk = stream[start:start + 32]
        if len(chunk) < 4 and windows:
            windows[-1] = resample_temporal(np.concatenate((windows[-1], chunk)), 32)
        else:
            windows.append(resample_temporal(chunk, 32))
    if not 1 <= len(windows) <= 8:
        raise ValueError("boundary augmentation exceeds Stage-2 window contract")
    return np.stack(windows).astype(np.float32)


def balanced_weights(
    real: RealPhraseDataset,
    synthetic: SyntheticCompositionDataset,
    isolated: list[IsolatedPoolDataset],
) -> tuple[torch.Tensor, dict[str, float], dict[str, int]]:
    metadata: list[tuple[str, tuple[int, ...]]] = []
    metadata.extend(
        (sample.source, tuple(int(value) for value in sample.targets))
        for sample in real.samples
    )
    metadata.extend(
        (
            str(row.get("source", "synthetic_citizen_train")),
            tuple(int(value) + 1 for value in row["target_indices"]),
        )
        for row in synthetic.rows
    )
    for dataset in isolated:
        metadata.extend(
            (dataset.source, (int(target) + 1,)) for target in dataset.targets
        )
    source_mass = {
        "local_phrases": 0.12,
        "asllrp_contiguous": 0.08,
        "synthetic_citizen_train": 0.06,
        "synthetic_multivoice_train": 0.06,
        "synthetic_balanced_multivoice_train": 0.08,
        "isolated_citizen_train": 0.20,
        "isolated_semlex_train": 0.20,
        "isolated_local_train": 0.20,
    }
    present = {source for source, _ in metadata}
    if present != set(source_mass):
        raise ValueError(f"unexpected accuracy-repair sources: {sorted(present)}")
    sequence_counts = Counter(metadata)
    source_keys: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for source, sequence in metadata:
        source_keys[source].add(sequence)
    weights = [
        source_mass[source]
        / len(source_keys[source])
        / sequence_counts[(source, sequence)]
        for source, sequence in metadata
    ]
    return (
        torch.as_tensor(weights, dtype=torch.double),
        source_mass,
        Counter(source for source, _ in metadata),
    )


def isolated_metrics(
    model: Stage2TemporalHeadV17,
    datasets: dict[str, IsolatedPoolDataset],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    domains: dict[str, dict[str, float | int]] = {}
    with torch.inference_mode():
        for name, dataset in datasets.items():
            correct = duplicates = 0
            confusion = np.zeros((100, 100), dtype=np.int64)
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                collate_fn=collate,
            )
            for batch in loader:
                logits, lengths = model(
                    batch["features"].to(device), batch["window_mask"].to(device)
                )
                predictions = logits.argmax(-1).cpu().numpy()
                for row, length, target in zip(
                    predictions, lengths.cpu().tolist(), batch["targets"].tolist()
                ):
                    hypothesis = collapse_ctc(row[:length])
                    expected = int(target) - 1
                    correct += int(hypothesis == [expected])
                    duplicates += int(len(hypothesis) > 1)
                    if hypothesis:
                        confusion[expected, hypothesis[0]] += 1
            true_positive = np.diag(confusion).astype(np.float64)
            recall = true_positive / np.maximum(confusion.sum(1), 1)
            domains[name] = {
                "accuracy": correct / len(dataset),
                "correct": correct,
                "samples": len(dataset),
                "macro_recall": float(recall.mean()),
                "multi_token_rate": duplicates / len(dataset),
            }
    accuracies = [float(value["accuracy"]) for value in domains.values()]
    return {
        "domains": domains,
        "equal_domain_mean_accuracy": float(np.mean(accuracies)),
        "worst_domain_accuracy": float(min(accuracies)),
        "total_correct": sum(int(value["correct"]) for value in domains.values()),
        "total_samples": sum(int(value["samples"]) for value in domains.values()),
    }


def validation_bundle(
    model: Stage2TemporalHeadV17,
    phrase_loader: DataLoader,
    isolated_validation: dict[str, IsolatedPoolDataset],
    device: torch.device,
    batch_size: int,
    stage1_baseline: float,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    phrase = evaluate(model, phrase_loader, device)
    isolated = isolated_metrics(model, isolated_validation, device, batch_size)
    local_phrase = phrase["domains"]["local_phrases"]
    asllrp_phrase = phrase["domains"]["asllrp_contiguous"]
    eligible = (
        isolated["equal_domain_mean_accuracy"] > stage1_baseline
        and local_phrase["wer"] <= 0.04
        and asllrp_phrase["wer"] <= 0.4583333333333333
    )
    key = (
        float(eligible),
        isolated["equal_domain_mean_accuracy"],
        isolated["worst_domain_accuracy"],
        -phrase["equal_domain_mean_wer"],
        phrase["equal_domain_mean_sequence_accuracy"],
    )
    return {"eligible": eligible, "isolated": isolated, "phrases": phrase}, key


def train_seed(
    seed: int,
    train_dataset: Dataset,
    weights: torch.Tensor,
    phrase_loader: DataLoader,
    isolated_validation: dict[str, IsolatedPoolDataset],
    args: argparse.Namespace,
    device: torch.device,
    stage1_baseline: float,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    warm, warm_checkpoint = load_stage2_model_v17(args.warm_start)
    if not isinstance(warm, Stage2TemporalHeadV17):
        raise ValueError("accuracy repair must warm-start from a bare Stage-2 CTC head")
    model = warm.to(device)
    teacher = Stage2TemporalHeadV17(Stage2V17Config(**warm_checkpoint["model_config"]))
    teacher.load_state_dict(warm_checkpoint["model_state_dict"], strict=True)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sampler = WeightedRandomSampler(
        weights, args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, collate_fn=collate,
    )
    best_key: tuple[float, ...] | None = None
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = seen = 0
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            target_starts = torch.cat((
                torch.zeros(1, dtype=torch.long, device=device),
                target_lengths.cumsum(0)[:-1],
            ))
            first_targets = targets[target_starts] - 1
            optimizer.zero_grad(set_to_none=True)
            logits, input_lengths = model(features, mask)
            # MPS does not implement CTC. Keep the transformer and all large
            # tensors on Metal, but copy this small [time,batch,class] loss
            # input to CPU; autograd carries its gradient back to MPS.
            ctc_logits = logits.log_softmax(-1).transpose(0, 1)
            if device.type == "mps":
                ctc_logits = ctc_logits.cpu()
                ctc_targets = targets.cpu()
                ctc_input_lengths = input_lengths.cpu()
                ctc_target_lengths = target_lengths.cpu()
            else:
                ctc_targets = targets
                ctc_input_lengths = input_lengths
                ctc_target_lengths = target_lengths
            ctc = criterion(
                ctc_logits, ctc_targets, ctc_input_lengths, ctc_target_lengths
            )
            single = target_lengths == 1
            if single.any():
                token_mask = (
                    torch.arange(logits.shape[1], device=device)[None]
                    < input_lengths[:, None]
                )
                nonblank = logits[..., 1:].masked_fill(~token_mask[..., None], -1e4)
                classification = F.cross_entropy(
                    torch.logsumexp(nonblank[single], dim=1), first_targets[single]
                )
            else:
                classification = logits.sum() * 0.0
            multi = target_lengths > 1
            if multi.any():
                with torch.inference_mode():
                    teacher_logits, _ = teacher(features, mask)
                temperature = args.temperature
                distill = F.kl_div(
                    F.log_softmax(logits[multi] / temperature, dim=-1),
                    F.softmax(teacher_logits[multi] / temperature, dim=-1),
                    reduction="batchmean",
                ) * temperature**2
            else:
                distill = logits.sum() * 0.0
            loss = (
                ctc + args.isolated_classification_weight * classification
                + args.phrase_distill_weight * distill
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(mask)
            seen += len(mask)
        scheduler.step()
        should_validate = epoch == 1 or epoch % args.validation_interval == 0
        if not should_validate:
            LOG.info(
                "seed=%d epoch=%d loss=%.4f validation=deferred",
                seed, epoch, running / max(1, seen),
            )
            continue
        metrics, key = validation_bundle(
            model, phrase_loader, isolated_validation, device, args.batch_size,
            stage1_baseline,
        )
        history.append({
            "epoch": epoch, "train_loss": running / max(1, seen),
            "selection_key": list(key), **metrics,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = metrics
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        LOG.info(
            "seed=%d epoch=%d loss=%.4f isolated=%.4f phraseWER=%.4f eligible=%s stale=%d",
            seed, epoch, running / max(1, seen),
            metrics["isolated"]["equal_domain_mean_accuracy"],
            metrics["phrases"]["equal_domain_mean_wer"],
            metrics["eligible"], stale,
        )
        if stale >= args.patience:
            break
    if best_state is None or best_metrics is None or best_key is None:
        raise RuntimeError("Stage-2 repair produced no checkpoint")
    output = args.output / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = make_stage2_checkpoint(
        model, best_state, seed=seed, epoch=best_epoch,
        validation_metrics=best_metrics, selection_key=list(best_key),
        warm_started_from=args.warm_start.as_posix(),
        warm_started_from_sha256=sha256(args.warm_start),
        training_design="class_sequence_balanced_isolated_phrase_boundary_replay",
    )
    torch.save(checkpoint, output / "best_model.pth")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "seed": seed,
        "selected_epoch": best_epoch,
        "selection_key": list(best_key),
        "validation_metrics": best_metrics,
        "checkpoint": (output / "best_model.pth").as_posix(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 < args.mps_memory_fraction <= 0.25:
        raise ValueError("MPS memory fraction must be in (0, 0.25]")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    stage1_result = json.loads(args.stage1_result.read_text())
    stage1_domains = stage1_result["validation_metrics"]
    stage1_baseline = float(np.mean([
        float(stage1_domains[name]["top1"]) / 100.0
        for name in ("citizen", "semlex", "local")
    ]))
    real_train = RealPhraseDataset(args.cache_root, "train")
    real_validation = RealPhraseDataset(args.cache_root, "validation")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    isolated_train = [
        IsolatedPoolDataset(args.citizen_train_pool, "isolated_citizen_train", augment_boundaries=True),
        IsolatedPoolDataset(args.semlex_train_pool, "isolated_semlex_train", augment_boundaries=True),
        IsolatedPoolDataset(args.local_train_pool, "isolated_local_train", augment_boundaries=True),
    ]
    isolated_validation = {
        "citizen": IsolatedPoolDataset(args.citizen_validation_pool, "isolated_citizen_validation", augment_boundaries=False),
        "semlex": IsolatedPoolDataset(args.semlex_validation_pool, "isolated_semlex_validation", augment_boundaries=False),
        "local": IsolatedPoolDataset(args.local_validation_pool, "isolated_local_validation", augment_boundaries=False),
    }
    expected_stage1 = sha256(args.stage1_checkpoint)
    observed_stage1 = {
        dataset.stage1_checkpoint_sha256
        for dataset in isolated_train + list(isolated_validation.values())
    }
    if observed_stage1 != {expected_stage1}:
        raise ValueError("isolated pools do not match the selected Stage-1 checkpoint")
    train_dataset = CombinedDataset([
        BoundaryAugmentedDataset(real_train, args.phrase_boundary_probability),
        BoundaryAugmentedDataset(synthetic, args.phrase_boundary_probability),
        *isolated_train,
    ])
    weights, source_mass, source_counts = balanced_weights(
        real_train, synthetic, isolated_train
    )
    phrase_loader = DataLoader(
        real_validation, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    results = [
        train_seed(
            seed, train_dataset, weights, phrase_loader, isolated_validation,
            args, device, stage1_baseline,
        )
        for seed in args.seeds
    ]
    winner = max(results, key=lambda row: tuple(row["selection_key"]))
    selected = torch.load(winner["checkpoint"], map_location="cpu", weights_only=False)
    selected.update({
        "selected_stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "selected_stage1_checkpoint_sha256": expected_stage1,
        "stage1_equal_domain_validation_accuracy": stage1_baseline,
        "training_source_sampling_mass": source_mass,
        "training_source_counts": dict(source_counts),
        "synthetic_pool": args.synthetic_pool.as_posix(),
        "synthetic_pool_sha256": sha256(args.synthetic_pool),
        "synthetic_plan": args.synthetic_plan.as_posix(),
        "synthetic_plan_sha256": sha256(args.synthetic_plan),
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
        "selected_epoch": winner["selected_epoch"],
        "selection_key": winner["selection_key"],
        "validation_metrics": winner["validation_metrics"],
        "stage1_equal_domain_validation_accuracy": stage1_baseline,
        "stage2_exceeds_stage1": bool(winner["validation_metrics"]["eligible"]),
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
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
    parser.add_argument("--synthetic-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"))
    parser.add_argument("--synthetic-plan", type=Path, default=Path("active/v17/stage2_balanced_multivoice_plan_v17.json"))
    parser.add_argument("--citizen-train-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"))
    parser.add_argument("--semlex-train-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/semlex_train_isolated_pool.npz"))
    parser.add_argument("--local-train-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/local_train.npz"))
    parser.add_argument("--citizen-validation-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/citizen_validation.npz"))
    parser.add_argument("--semlex-validation-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/semlex_validation.npz"))
    parser.add_argument("--local-validation-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/local_validation.npz"))
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--stage1-result", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/result.json"))
    parser.add_argument("--warm-start", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_accuracy_repair_v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1701])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.18)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--isolated-classification-weight", type=float, default=0.5)
    parser.add_argument("--phrase-distill-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--phrase-boundary-probability", type=float, default=0.5)
    parser.add_argument("--validation-interval", type=int, default=2)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
