#!/usr/bin/env python3
"""Adapt the selected v17 CTC head to full ASLLRP spans with OTHER."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from collections import Counter, defaultdict
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from active.v17.model_stage2_v17 import (
    Stage2TemporalHeadV17,
    Stage2V17Config,
    make_stage2_checkpoint,
    warm_start_stage2_with_other,
)
from active.v17.train_stage_2_v17 import (
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collate,
    edit_distance,
)


LOG = logging.getLogger("train_stage_2_other_ctc_v17")
OTHER_CLASS_INDEX = 100
OTHER_CTC_INDEX = 101


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def collapse_ctc(sequence: np.ndarray) -> list[int]:
    output = []
    previous = None
    for value in sequence.tolist():
        token = int(value)
        if token != previous and token != 0:
            output.append(token - 1)
        previous = token
    return output


def metric_accumulator():
    return {"edits": 0, "tokens": 0, "exact": 0, "samples": 0}


def add_metric(stats, reference, hypothesis):
    stats["edits"] += edit_distance(reference, hypothesis)
    stats["tokens"] += len(reference)
    stats["exact"] += int(reference == hypothesis)
    stats["samples"] += 1


def finish_metric(stats):
    return {
        **stats,
        "wer": stats["edits"] / max(1, stats["tokens"]),
        "sequence_accuracy": stats["exact"] / max(1, stats["samples"]),
    }


def evaluate(model, loader, device):
    model.eval()
    full = defaultdict(metric_accumulator)
    target_only = defaultdict(metric_accumulator)
    with torch.inference_mode():
        for batch in loader:
            logits, lengths = model(
                batch["features"].to(device), batch["window_mask"].to(device)
            )
            predictions = logits.argmax(dim=-1).cpu().numpy()
            flat_targets = batch["targets"].tolist()
            offset = 0
            for index, (source, length, target_length) in enumerate(zip(
                batch["sources"], lengths.cpu().tolist(), batch["target_lengths"].tolist()
            )):
                reference = [
                    int(value) - 1
                    for value in flat_targets[offset:offset + target_length]
                ]
                offset += target_length
                hypothesis = collapse_ctc(predictions[index, :length])
                add_metric(full[source], reference, hypothesis)
                clean_reference = [value for value in reference if value != OTHER_CLASS_INDEX]
                clean_hypothesis = [value for value in hypothesis if value != OTHER_CLASS_INDEX]
                add_metric(target_only[source], clean_reference, clean_hypothesis)
    return {
        "full": {key: finish_metric(value) for key, value in sorted(full.items())},
        "target_only": {
            key: finish_metric(value) for key, value in sorted(target_only.items())
        },
    }


def selection_key(metrics):
    target = metrics["target_only"]
    local = target["local_phrases"]
    sparse = target["asllrp_contiguous"]
    natural = target["asllrp_other_ctc"]
    natural_full = metrics["full"]["asllrp_other_ctc"]
    old_guard = int(local["edits"] <= 7 and sparse["edits"] <= 11)
    mean_wer = np.mean([
        local["wer"], sparse["wer"], natural["wer"], natural_full["wer"],
    ])
    return (
        old_guard,
        -float(mean_wer),
        -float(natural_full["wer"]),
        -float(sparse["wer"]),
        -float(natural["wer"]),
        -float(local["wer"]),
    )


def sampling_weights(old, natural, synthetic, masses):
    samples = old.samples + natural.samples
    counts = Counter(sample.source for sample in samples)
    counts.update(str(row["source"]) for row in synthetic.rows)
    if set(counts) != set(masses):
        raise ValueError(f"unexpected training sources: {dict(counts)}")
    weights = [masses[sample.source] / counts[sample.source] for sample in samples]
    weights.extend(
        masses[str(row["source"])] / counts[str(row["source"])]
        for row in synthetic.rows
    )
    return torch.as_tensor(weights, dtype=torch.double), counts


def distillation_loss(student_logits, teacher_logits, lengths, replay_mask, temperature):
    if not replay_mask.any():
        return student_logits.sum() * 0.0
    token_mask = torch.arange(student_logits.shape[1], device=lengths.device)[None] < lengths[:, None]
    mask = token_mask & replay_mask[:, None]
    student = student_logits[..., :101] / temperature
    teacher = teacher_logits / temperature
    values = F.kl_div(
        student.log_softmax(dim=-1), teacher.softmax(dim=-1), reduction="none"
    ).sum(dim=-1)
    return values[mask].mean() * temperature * temperature


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_initialization(path: Path, temporal_mix: float = 1.0):
    """Return student initialization and an unchanged 100-class distillation teacher."""
    if not 0.0 <= temporal_mix <= 1.0:
        raise ValueError("temporal mix must be in [0,1]")
    initialization = torch.load(path, map_location="cpu", weights_only=False)
    if initialization.get("format") == "slt_stage2_ctc_v17":
        return initialization, initialization, None
    if initialization.get("format") != "slt_stage2_temporal_pretrain_v17":
        raise ValueError("warm-start must be v17 Stage 2 or temporal pretraining")
    if initialization.get("ctc_head_trained") is not False:
        raise ValueError("temporal pretraining must leave the CTC head frozen")
    teacher_path = Path(str(initialization["base_checkpoint"]))
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if teacher.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("temporal pretraining base is not v17 Stage 2")
    if sha256(teacher_path) != initialization.get("base_checkpoint_sha256"):
        raise ValueError("temporal pretraining base checkpoint hash changed")
    if initialization["model_config"] != teacher["model_config"]:
        raise ValueError("temporal initialization and teacher configurations differ")
    mixed_state = {}
    for name, value in initialization["model_state_dict"].items():
        base = teacher["model_state_dict"][name]
        if name.startswith("ctc_head."):
            if not torch.equal(value, base):
                raise ValueError("temporal pretraining changed the frozen CTC head")
            mixed_state[name] = base
        else:
            mixed_state[name] = base + temporal_mix * (value - base)
    student = dict(initialization)
    student["format"] = "slt_stage2_ctc_v17"
    student["model_state_dict"] = mixed_state
    return student, teacher, {
        "checkpoint": path.as_posix(),
        "checkpoint_sha256": sha256(path),
        "source_split": initialization.get("source_split"),
        "temporal_mix": temporal_mix,
        "teacher": teacher_path.as_posix(),
        "teacher_sha256": sha256(teacher_path),
    }


def train_one(
    seed, train_dataset, weights, validation_loader,
    initialization, teacher_checkpoint, args, device,
):
    seed_everything(seed)
    source_config = Stage2V17Config(**initialization["model_config"])
    model = Stage2TemporalHeadV17(Stage2V17Config(
        **{**source_config.to_dict(), "num_classes": 101}
    )).to(device)
    warm_start_stage2_with_other(model, initialization)
    teacher = Stage2TemporalHeadV17(source_config).to(device).eval()
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"], strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    head_parameters = list(model.ctc_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    for parameter in backbone_parameters:
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": args.lr},
        {"params": backbone_parameters, "lr": args.backbone_lr},
    ], weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, collate_fn=collate,
    )
    baseline = evaluate(model, validation_loader, device)
    best_key = selection_key(baseline)
    best_metrics = baseline
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_epoch = 0
    history = [{"epoch": 0, "validation": baseline, "selection_key": list(best_key)}]
    patience = 0
    peak_mps_current = peak_mps_driver = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_epochs + 1:
            for parameter in backbone_parameters:
                parameter.requires_grad = True
        model.train()
        total_ctc = total_distill = seen = 0.0
        for batch in loader:
            features = batch["features"].to(device)
            window_mask = batch["window_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            replay_mask = torch.as_tensor(
                [source != "asllrp_other_ctc" for source in batch["sources"]],
                dtype=torch.bool, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = model(features, window_mask)
            ctc = criterion(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets, lengths, target_lengths,
            )
            with torch.inference_mode():
                teacher_logits, teacher_lengths = teacher(features, window_mask)
            if not torch.equal(lengths, teacher_lengths):
                raise RuntimeError("teacher/student temporal lengths differ")
            distill = distillation_loss(
                logits, teacher_logits, lengths, replay_mask, args.temperature
            )
            loss = ctc + args.distill_weight * distill
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(window_mask)
            total_ctc += float(ctc.detach()) * count
            total_distill += float(distill.detach()) * count
            seen += count
        metrics = evaluate(model, validation_loader, device)
        if device.type == "mps":
            torch.mps.synchronize()
            peak_mps_current = max(
                peak_mps_current, int(torch.mps.current_allocated_memory())
            )
            peak_mps_driver = max(
                peak_mps_driver, int(torch.mps.driver_allocated_memory())
            )
            torch.mps.empty_cache()
        key = selection_key(metrics)
        history.append({
            "epoch": epoch,
            "ctc_loss": total_ctc / seen,
            "distill_loss": total_distill / seen,
            "validation": metrics,
            "selection_key": list(key),
        })
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d ctc=%.4f distill=%.4f key=%s patience=%d",
            seed, epoch, total_ctc / seen, total_distill / seen,
            tuple(round(value, 5) for value in key), patience,
        )
        if epoch > args.head_epochs and patience >= args.patience:
            break
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "validation_metrics": best_metrics,
        "state_dict": best_state,
        "history": history,
        "model_config": model.config.to_dict(),
        "peak_mps_current_bytes": peak_mps_current,
        "peak_mps_driver_bytes": peak_mps_driver,
    }


def run(args):
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    device = torch.device(device_name)
    if device.type == "mps":
        if not 0 < args.mps_memory_fraction <= 0.12:
            raise ValueError("MPS memory fraction must be in (0, 0.12]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    initialization, teacher_checkpoint, temporal_provenance = resolve_initialization(
        args.warm_start, args.temporal_mix
    )
    old_train = RealPhraseDataset(args.replay_root, "train")
    natural_train = RealPhraseDataset(args.other_root, "train")
    synthetic_train = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    old_validation = RealPhraseDataset(args.replay_root, "validation")
    natural_validation = RealPhraseDataset(args.other_root, "validation")
    train_dataset = CombinedDataset([old_train, natural_train, synthetic_train])
    validation_dataset = CombinedDataset([old_validation, natural_validation])
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    masses = {
        "local_phrases": args.local_mass,
        "asllrp_contiguous": args.sparse_asllrp_mass,
        "asllrp_other_ctc": args.natural_asllrp_mass,
        "synthetic_citizen_train": args.synthetic_citizen_mass,
        "synthetic_multivoice_train": args.synthetic_multivoice_mass,
    }
    if not np.isclose(sum(masses.values()), 1.0):
        raise ValueError("training source masses must sum to one")
    weights, counts = sampling_weights(old_train, natural_train, synthetic_train, masses)
    started = time.monotonic()
    results = [
        train_one(
            seed, train_dataset, weights, validation_loader,
            initialization, teacher_checkpoint, args, device,
        )
        for seed in args.seeds
    ]
    winner = max(results, key=lambda value: tuple(value["selection_key"]))
    args.output.mkdir(parents=True, exist_ok=True)
    model = Stage2TemporalHeadV17(Stage2V17Config(**winner["model_config"]))
    selected = make_stage2_checkpoint(
        model, winner.pop("state_dict"), seed=winner["seed"], epoch=winner["best_epoch"],
        validation_metrics=winner["validation_metrics"],
        selection_key=winner["selection_key"],
        warm_started_from=args.warm_start.as_posix(),
        warm_started_from_sha256=sha256(args.warm_start),
        temporal_initialization=temporal_provenance,
        replay_cache_root=args.replay_root.as_posix(),
        other_cache_root=args.other_root.as_posix(),
        other_manifest=args.other_manifest.as_posix(),
        other_manifest_sha256=sha256(args.other_manifest),
        synthetic_pool=args.synthetic_pool.as_posix(),
        synthetic_pool_sha256=sha256(args.synthetic_pool),
        synthetic_plan=args.synthetic_plan.as_posix(),
        synthetic_plan_sha256=sha256(args.synthetic_plan),
        other_class_index=OTHER_CLASS_INDEX,
        other_ctc_index=OTHER_CTC_INDEX,
        decoder_policy="collapse CTC then remove class index 100 (OTHER)",
        training_source_sampling_mass=masses,
        distill_weight=args.distill_weight,
        temperature=args.temperature,
        citizen_test_accessed=False, semlex_test_accessed=False,
        local_test_accessed=False, rit_external_evaluation_accessed=False,
    )
    checkpoint_path = args.output / "best_model.pth"
    torch.save(selected, checkpoint_path)
    for result in results:
        result.pop("state_dict", None)
    report = {
        "format": "slt_stage2_other_ctc_training_result_v17",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "selection_key": winner["selection_key"],
        "validation_metrics": winner["validation_metrics"],
        "training_counts": dict(counts),
        "training_source_sampling_mass": masses,
        "temporal_initialization": temporal_provenance,
        "validation_counts": {
            "legacy": len(old_validation), "natural_asllrp": len(natural_validation),
        },
        "seconds": time.monotonic() - started,
        "device": str(device),
        "candidate_results": results,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "rit_external_evaluation_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--replay-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    value.add_argument("--other-root", type=Path, default=Path("data/local/stage2_v17_asllrp_other_frozen_features"))
    value.add_argument("--other-manifest", type=Path, default=Path("active/v17/stage2_asllrp_other_ctc_manifest_v17.json"))
    value.add_argument("--warm-start", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"))
    value.add_argument("--temporal-mix", type=float, default=1.0)
    value.add_argument("--synthetic-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"))
    value.add_argument("--synthetic-plan", type=Path, default=Path("active/v17/stage2_multivoice_transfer_plan_v17.json"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_asllrp_other_ctc_v1"))
    value.add_argument("--device", default="auto")
    value.add_argument("--mps-memory-fraction", type=float, default=0.12)
    value.add_argument("--seeds", type=lambda text: tuple(int(value) for value in text.split(",")), default=(1701,))
    value.add_argument("--epochs", type=int, default=40)
    value.add_argument("--head-epochs", type=int, default=5)
    value.add_argument("--patience", type=int, default=10)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--samples-per-epoch", type=int, default=1800)
    value.add_argument("--lr", type=float, default=3e-4)
    value.add_argument("--backbone-lr", type=float, default=4e-5)
    value.add_argument("--weight-decay", type=float, default=0.02)
    value.add_argument("--distill-weight", type=float, default=0.35)
    value.add_argument("--temperature", type=float, default=2.0)
    value.add_argument("--local-mass", type=float, default=0.18)
    value.add_argument("--sparse-asllrp-mass", type=float, default=0.12)
    value.add_argument("--natural-asllrp-mass", type=float, default=0.40)
    value.add_argument("--synthetic-citizen-mass", type=float, default=0.12)
    value.add_argument("--synthetic-multivoice-mass", type=float, default=0.18)
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(parser().parse_args()), indent=2))
