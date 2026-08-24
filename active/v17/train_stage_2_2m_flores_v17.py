#!/usr/bin/env python3
"""Fine-tune Stage 2 with locked-100 replay and full-gloss 2M-Flores auxiliary CTC."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from collections import defaultdict
from functools import partial
import gc
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    Stage2DualHeadV17,
    Stage2V17Config,
    warm_start_dual_stage2,
)
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collapse_ctc,
    edit_distance,
    sampler_weights,
    sha256,
)


LOG = logging.getLogger("train_stage_2_2m_flores_v17")


def collate_long(samples, maximum_windows: int) -> dict[str, Any]:
    max_windows = max(sample.features.shape[0] for sample in samples)
    if max_windows > maximum_windows:
        raise ValueError(f"Stage-2 sample exceeds {maximum_windows} windows")
    features = np.zeros(
        (len(samples), max_windows, 32, FROZEN_TEMPORAL_FEATURE_DIM), dtype=np.float32
    )
    window_mask = np.zeros((len(samples), max_windows), dtype=np.bool_)
    target_lengths = np.asarray([len(sample.targets) for sample in samples], dtype=np.int64)
    locked_partial = [sample.targets[(sample.targets >= 1) & (sample.targets <= 100)] for sample in samples]
    locked_partial_lengths = np.asarray([len(value) for value in locked_partial], dtype=np.int64)
    for index, sample in enumerate(samples):
        windows = sample.features.shape[0]
        features[index, :windows] = sample.features
        window_mask[index, :windows] = True
    return {
        "features": torch.from_numpy(features),
        "window_mask": torch.from_numpy(window_mask),
        "targets": torch.from_numpy(np.concatenate([sample.targets for sample in samples])),
        "target_lengths": torch.from_numpy(target_lengths),
        "locked_partial_targets": torch.from_numpy(np.concatenate(locked_partial)),
        "locked_partial_target_lengths": torch.from_numpy(locked_partial_lengths),
        "sources": [sample.source for sample in samples],
        "item_ids": [sample.item_id for sample in samples],
        "target_sequences": [sample.target_sequence for sample in samples],
    }


def evaluate_locked(model, loader, device) -> dict[str, Any]:
    model.eval()
    domains: dict[str, dict[str, float]] = defaultdict(
        lambda: {"edits": 0.0, "tokens": 0.0, "exact": 0.0, "samples": 0.0}
    )
    with torch.inference_mode():
        for batch in loader:
            logits, lengths = model.forward_locked(
                batch["features"].to(device), batch["window_mask"].to(device)
            )
            predictions = logits.argmax(dim=-1).cpu().numpy()
            lengths = lengths.cpu().tolist()
            offset = 0
            flat_targets = batch["targets"].tolist()
            for index, (source, length, target_length) in enumerate(
                zip(batch["sources"], lengths, batch["target_lengths"].tolist())
            ):
                reference = [
                    int(value) - 1 for value in flat_targets[offset:offset + target_length]
                ]
                offset += target_length
                hypothesis = collapse_ctc(predictions[index, :length])
                edits = edit_distance(reference, hypothesis)
                values = domains[source]
                values["edits"] += edits
                values["tokens"] += len(reference)
                values["exact"] += float(reference == hypothesis)
                values["samples"] += 1
    return {
        source: {
            "wer": values["edits"] / max(1.0, values["tokens"]),
            "sequence_accuracy": values["exact"] / max(1.0, values["samples"]),
            "samples": int(values["samples"]),
            "tokens": int(values["tokens"]),
        }
        for source, values in sorted(domains.items())
    }


def metrics_key(phrase: dict[str, Any], contextual: dict[str, Any]) -> tuple[float, ...]:
    local = phrase["local_phrases"]["wer"]
    asllrp_phrase = phrase["asllrp_contiguous"]["wer"]
    contextual_wer = contextual["asllrp_segmented_validation"]["wer"]
    # The 254-row signer-held-out contextual set is the high-power primary gate.
    # Preserve the existing local and 12-phrase gates as explicit secondary checks.
    return (-contextual_wer, -asllrp_phrase, -local)


def make_loader(dataset, batch_size, collate, sampler=None, shuffle=False):
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle if sampler is None else False,
        num_workers=0, collate_fn=collate,
    )


def train_seed(
    seed: int,
    locked_train,
    locked_weights,
    auxiliary_train,
    phrase_loader,
    contextual_loader,
    auxiliary_num_classes: int,
    args,
    device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = Stage2V17Config(max_windows=args.maximum_windows)
    model = Stage2DualHeadV17(config, auxiliary_num_classes).to(device)
    warm_start = torch.load(args.warm_start, map_location="cpu", weights_only=False)
    warm_start_dual_stage2(model, warm_start)
    initial_phrase = evaluate_locked(model, phrase_loader, device)
    initial_contextual = evaluate_locked(model, contextual_loader, device)

    optimizer = torch.optim.AdamW([
        {
            "params": model.locked.parameters(),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": model.auxiliary_ctc_head.parameters(),
            "lr": args.auxiliary_head_lr,
            "weight_decay": args.weight_decay,
        },
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    generator = torch.Generator().manual_seed(seed)
    locked_sampler = WeightedRandomSampler(
        locked_weights, num_samples=args.locked_samples_per_epoch,
        replacement=True, generator=generator,
    )
    collate = partial(collate_long, maximum_windows=args.maximum_windows)
    locked_loader = make_loader(locked_train, args.locked_batch_size, collate, locked_sampler)
    best_key = metrics_key(initial_phrase, initial_contextual)
    best_epoch = 0
    best_phrase = initial_phrase
    best_contextual = initial_contextual
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    patience = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        auxiliary_generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        auxiliary_loader = DataLoader(
            auxiliary_train, batch_size=args.auxiliary_batch_size, shuffle=True,
            generator=auxiliary_generator, num_workers=0, collate_fn=collate,
        )
        auxiliary_iterator = iter(auxiliary_loader)
        model.train()
        locked_loss_sum = auxiliary_loss_sum = partial_loss_sum = 0.0
        locked_seen = auxiliary_seen = 0
        for step, batch in enumerate(locked_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            logits, input_lengths = model.forward_locked(
                batch["features"].to(device), batch["window_mask"].to(device)
            )
            loss_locked = criterion(
                logits.log_softmax(dim=-1).transpose(0, 1),
                batch["targets"].to(device), input_lengths,
                batch["target_lengths"].to(device),
            )
            if not torch.isfinite(loss_locked):
                raise FloatingPointError(
                    f"non-finite locked CTC loss at seed={seed} epoch={epoch} step={step}"
                )
            loss = loss_locked
            loss_auxiliary = loss_partial = None
            if step % args.auxiliary_every_locked_steps == 0:
                try:
                    auxiliary_batch = next(auxiliary_iterator)
                except StopIteration:
                    auxiliary_iterator = iter(auxiliary_loader)
                    auxiliary_batch = next(auxiliary_iterator)
                auxiliary_value, auxiliary_lengths = model.locked.encode(
                    auxiliary_batch["features"].to(device),
                    auxiliary_batch["window_mask"].to(device),
                )
                # First learn the expanded decoder over the frozen shared representation.
                # Only the ordered locked-label subsequence is allowed to adapt the
                # deployment/shared graph during this conservative transfer phase.
                auxiliary_logits = model.auxiliary_ctc_head(auxiliary_value.detach())
                loss_auxiliary = criterion(
                    auxiliary_logits.log_softmax(dim=-1).transpose(0, 1),
                    auxiliary_batch["targets"].to(device), auxiliary_lengths,
                    auxiliary_batch["target_lengths"].to(device),
                )
                if not torch.isfinite(loss_auxiliary):
                    raise FloatingPointError(
                        f"non-finite auxiliary CTC loss at seed={seed} epoch={epoch} step={step}"
                    )
                partial_logits = model.locked.ctc_head(auxiliary_value)
                loss_partial = criterion(
                    partial_logits.log_softmax(dim=-1).transpose(0, 1),
                    auxiliary_batch["locked_partial_targets"].to(device), auxiliary_lengths,
                    auxiliary_batch["locked_partial_target_lengths"].to(device),
                )
                if not torch.isfinite(loss_partial):
                    raise FloatingPointError(
                        f"non-finite partial locked CTC loss at seed={seed} epoch={epoch} step={step}"
                    )
                loss = (
                    loss + args.auxiliary_loss_weight * loss_auxiliary
                    + args.partial_locked_loss_weight * loss_partial
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.locked.parameters(), 1.0, error_if_nonfinite=True
            )
            torch.nn.utils.clip_grad_norm_(
                model.auxiliary_ctc_head.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            locked_loss_sum += float(loss_locked.detach()) * len(batch["window_mask"])
            locked_seen += len(batch["window_mask"])
            if loss_auxiliary is not None:
                auxiliary_loss_sum += float(loss_auxiliary.detach()) * len(auxiliary_batch["window_mask"])
                partial_loss_sum += float(loss_partial.detach()) * len(auxiliary_batch["window_mask"])
                auxiliary_seen += len(auxiliary_batch["window_mask"])
        scheduler.step()
        phrase = evaluate_locked(model, phrase_loader, device)
        contextual = evaluate_locked(model, contextual_loader, device)
        key = metrics_key(phrase, contextual)
        record = {
            "epoch": epoch,
            "locked_train_loss": locked_loss_sum / max(1, locked_seen),
            "auxiliary_train_loss": auxiliary_loss_sum / max(1, auxiliary_seen),
            "partial_locked_train_loss": partial_loss_sum / max(1, auxiliary_seen),
            "phrase_validation": phrase,
            "contextual_validation": contextual,
            "selection_key": list(key),
        }
        history.append(record)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_phrase = phrase
            best_contextual = contextual
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d locked=%.4f aux=%.4f partial=%.4f contextual=%.4f asllrp_phrase=%.4f local=%.4f patience=%d",
            seed, epoch, record["locked_train_loss"], record["auxiliary_train_loss"],
            record["partial_locked_train_loss"],
            contextual["asllrp_segmented_validation"]["wer"],
            phrase["asllrp_contiguous"]["wer"], phrase["local_phrases"]["wer"], patience,
        )
        if patience >= args.patience:
            break
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    teacher_checkpoint = {
        "format": "slt_stage2_dual_ctc_v17",
        "format_version": 1,
        "model_config": config.to_dict(),
        "auxiliary_num_classes": auxiliary_num_classes,
        "model_state_dict": best_state,
        "seed": seed,
        "epoch": best_epoch,
        "phrase_validation": best_phrase,
        "contextual_validation": best_contextual,
        "selection_key": list(best_key),
        "test_evaluated": False,
    }
    torch.save(teacher_checkpoint, seed_dir / "best_teacher.pth")
    (seed_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "phrase_validation": best_phrase,
        "contextual_validation": best_contextual,
        "initial_phrase_validation": initial_phrase,
        "initial_contextual_validation": initial_contextual,
        "teacher_checkpoint": (seed_dir / "best_teacher.pth").as_posix(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    vocabulary = json.loads(args.auxiliary_vocabulary.read_text())
    if int(vocabulary["locked_prefix_count"]) != 100:
        raise ValueError("auxiliary vocabulary does not preserve the locked 100-prefix")
    auxiliary_num_classes = int(vocabulary["expanded_vocabulary_count"])
    locked_real = RealPhraseDataset(args.locked_cache_root, "train")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    locked_train = CombinedDataset([locked_real, synthetic])
    locked_weights, sampling_mass = sampler_weights(locked_real, synthetic)
    auxiliary_train = RealPhraseDataset(args.auxiliary_cache_root, "train")
    phrase_validation = RealPhraseDataset(args.locked_cache_root, "validation")
    contextual_validation = RealPhraseDataset(args.contextual_cache_root, "validation")
    collate = partial(collate_long, maximum_windows=args.maximum_windows)
    phrase_loader = make_loader(phrase_validation, args.validation_batch_size, collate)
    contextual_loader = make_loader(contextual_validation, args.validation_batch_size, collate)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    candidates = [
        train_seed(
            seed, locked_train, locked_weights, auxiliary_train, phrase_loader,
            contextual_loader, auxiliary_num_classes, args, device,
        )
        for seed in seeds
    ]
    winner = max(candidates, key=lambda value: tuple(value["selection_key"]))
    teacher = torch.load(winner["teacher_checkpoint"], map_location="cpu", weights_only=False)
    locked_state = {
        name.removeprefix("locked."): value
        for name, value in teacher["model_state_dict"].items()
        if name.startswith("locked.")
    }
    deployment = {
        "format": "slt_stage2_ctc_v17",
        "format_version": 1,
        "model_config": teacher["model_config"],
        "model_state_dict": locked_state,
        "blank_index": 0,
        "seed": winner["seed"],
        "epoch": winner["best_epoch"],
        "validation_metrics": {
            "phrase": winner["phrase_validation"],
            "contextual": winner["contextual_validation"],
        },
        "selection_key": winner["selection_key"],
        "warm_start_checkpoint": args.warm_start.as_posix(),
        "warm_start_checkpoint_sha256": sha256(args.warm_start),
        "auxiliary_vocabulary": args.auxiliary_vocabulary.as_posix(),
        "auxiliary_vocabulary_sha256": sha256(args.auxiliary_vocabulary),
        "auxiliary_training_manifest": args.auxiliary_manifest.as_posix(),
        "auxiliary_training_manifest_sha256": sha256(args.auxiliary_manifest),
        "training_source_sampling_mass": sampling_mass,
        "auxiliary_loss_weight": args.auxiliary_loss_weight,
        "partial_locked_loss_weight": args.partial_locked_loss_weight,
        "shared_learning_rate": args.lr,
        "auxiliary_head_learning_rate": args.auxiliary_head_lr,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    temporary = args.output / "best_model.pth.tmp"
    torch.save(deployment, temporary)
    temporary.replace(args.output / "best_model.pth")
    result = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "selection_key": winner["selection_key"],
        "phrase_validation": winner["phrase_validation"],
        "contextual_validation": winner["contextual_validation"],
        "checkpoint": (args.output / "best_model.pth").as_posix(),
        "checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "locked_real_samples": len(locked_real),
        "locked_synthetic_sequences": len(synthetic),
        "auxiliary_real_sentences": len(auxiliary_train),
        "auxiliary_num_classes": auxiliary_num_classes,
        "candidates": candidates,
        "seconds": time.monotonic() - started,
        "device": str(device),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-cache-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument("--contextual-cache-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"))
    parser.add_argument("--auxiliary-cache-root", type=Path, default=Path("data/local/stage2_v17_2m_flores_frozen_features"))
    parser.add_argument("--synthetic-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"))
    parser.add_argument("--synthetic-plan", type=Path, default=Path("active/v17/stage2_mixed_synthetic_plan_v17.json"))
    parser.add_argument("--warm-start", type=Path, default=Path("artifacts/models/stage2_v17_unified_ctc_v2/best_model.pth"))
    parser.add_argument("--auxiliary-vocabulary", type=Path, default=Path("active/v17/stage2_2m_flores_vocabulary_v17.json"))
    parser.add_argument("--auxiliary-manifest", type=Path, default=Path("active/v17/stage2_2m_flores_training_manifest_v17.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_2m_flores_aux_ctc_v1"))
    parser.add_argument(
        "--device", default="cpu",
        help="CPU is the safe default because PyTorch CTC falls back from MPS and can yield non-finite gradients",
    )
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    parser.add_argument("--maximum-windows", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--locked-batch-size", type=int, default=16)
    parser.add_argument("--auxiliary-batch-size", type=int, default=2)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument("--locked-samples-per-epoch", type=int, default=2000)
    parser.add_argument("--auxiliary-every-locked-steps", type=int, default=4)
    parser.add_argument("--auxiliary-loss-weight", type=float, default=0.10)
    parser.add_argument("--partial-locked-loss-weight", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--auxiliary-head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--seeds", default="2701,2702,2703")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
