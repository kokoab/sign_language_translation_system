#!/usr/bin/env python3
"""Label-free Stage-2 temporal pretraining on genuine 2M-Flores sentences.

The deployment vocabulary and CTC head are never trained here.  A frozen copy of
the selected Stage-2 head encodes clean, contiguous sentence crops.  The student
sees bounded masked spans and learns to recover the clean token trajectory with a
small training-only predictor.  Only the student's temporal encoder is exported.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.10")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.05")
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
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import Stage2TemporalHeadV17, Stage2V17Config
from active.v17.train_stage_2_v17 import sha256


LOG = logging.getLogger("pretrain_stage2_temporal_v17")


def stable_fold(value: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("at least two deterministic folds are required")
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


class GenuineSentenceCropDataset(Dataset):
    """Lazily load bounded contiguous crops from training-only sentence features."""

    def __init__(
        self,
        root: Path,
        role: str,
        *,
        max_windows: int,
        folds: int = 5,
        validation_fold: int = 0,
        repeats: int = 1,
        seed: int = 117,
    ):
        if role not in {"train", "validation"}:
            raise ValueError("temporal pretraining role must be train or validation")
        if max_windows < 1 or repeats < 1 or not 0 <= validation_fold < folds:
            raise ValueError("invalid crop/split configuration")
        paths = sorted((root / "train").glob("*/*.stage2_frozen_v17.npz"))
        selected = [
            path for path in paths
            if (stable_fold(path.name, folds) == validation_fold) == (role == "validation")
        ]
        if not selected:
            raise ValueError(f"no {role} genuine sentence features under {root}")
        self.paths = selected
        self.role = role
        self.max_windows = max_windows
        self.repeats = repeats
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths) * self.repeats

    def __getitem__(self, index: int) -> dict[str, object]:
        path_index = index % len(self.paths)
        repeat = index // len(self.paths)
        path = self.paths[path_index]
        with np.load(path, allow_pickle=False) as payload:
            features = payload["frozen_features"].astype(np.float32)
            metadata = json.loads(str(payload["metadata_json"]))
        if metadata.get("source") != "two_m_flores_asl" or metadata.get("role") != "train":
            raise ValueError(f"{path}: temporal pretraining requires 2M-Flores dev/train only")
        windows = min(len(features), self.max_windows)
        limit = len(features) - windows
        if limit:
            salt = 0 if self.role == "validation" else self.epoch * 1_000_003
            rng = random.Random(self.seed + salt + path_index * 10_007 + repeat * 101)
            start = rng.randrange(limit + 1)
        else:
            start = 0
        return {
            "features": features[start:start + windows],
            "item_id": str(metadata["source_item_id"]),
            "crop_start_window": start,
        }


def collate_temporal(samples: list[dict[str, object]]) -> dict[str, object]:
    maximum = max(np.asarray(sample["features"]).shape[0] for sample in samples)
    dimension = np.asarray(samples[0]["features"]).shape[-1]
    features = np.zeros((len(samples), maximum, 32, dimension), dtype=np.float32)
    window_mask = np.zeros((len(samples), maximum), dtype=np.bool_)
    for row, sample in enumerate(samples):
        value = np.asarray(sample["features"], dtype=np.float32)
        features[row, :len(value)] = value
        window_mask[row, :len(value)] = True
    return {
        "features": torch.from_numpy(features),
        "window_mask": torch.from_numpy(window_mask),
        "item_ids": [str(sample["item_id"]) for sample in samples],
    }


def contiguous_token_mask(
    window_mask: torch.Tensor,
    *,
    tokens_per_window: int,
    ratio: float,
    span_tokens: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if window_mask.ndim != 2 or not 0.0 < ratio < 1.0 or span_tokens < 1:
        raise ValueError("invalid temporal mask request")
    valid = window_mask.cpu().repeat_interleave(tokens_per_window, dim=1)
    masked = torch.zeros_like(valid)
    for row in range(len(valid)):
        length = int(valid[row].sum())
        target = max(1, min(length - 1, int(round(length * ratio))))
        attempts = 0
        while int(masked[row].sum()) < target:
            start = int(torch.randint(length, (1,), generator=generator).item())
            stop = min(length, start + span_tokens)
            masked[row, start:stop] = True
            attempts += 1
            if attempts > length * 4:
                available = torch.nonzero(valid[row] & ~masked[row], as_tuple=False).flatten()
                masked[row, available[:target - int(masked[row].sum())]] = True
                break
        if int(masked[row].sum()) >= length:
            masked[row, -1] = False
    return masked & valid


def apply_token_mask(
    features: torch.Tensor, token_mask: torch.Tensor, *, tokens_per_window: int
) -> torch.Tensor:
    if features.ndim != 4 or features.shape[2] % tokens_per_window:
        raise ValueError("unexpected frozen feature shape")
    frames_per_token = features.shape[2] // tokens_per_window
    frame_mask = token_mask.repeat_interleave(frames_per_token, dim=1)
    flattened = features.flatten(1, 2).clone()
    if tuple(frame_mask.shape) != tuple(flattened.shape[:2]):
        raise ValueError("temporal token mask does not match features")
    return flattened.masked_fill(frame_mask.to(features.device).unsqueeze(-1), 0.0).reshape_as(features)


def temporal_pretraining_loss(
    student_tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    masked_tokens: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    temperature: float,
    contrastive_weight: float,
    visible_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0.0 or contrastive_weight < 0.0 or visible_weight < 0.0:
        raise ValueError("invalid temporal objective weights")
    masked_tokens = masked_tokens.to(student_tokens.device)
    valid_tokens = valid_tokens.to(student_tokens.device)
    selected_prediction = predicted_tokens[masked_tokens]
    selected_target = teacher_tokens[masked_tokens].detach()
    if not len(selected_prediction):
        raise RuntimeError("temporal batch has no masked targets")
    reconstruction = F.smooth_l1_loss(selected_prediction, selected_target)
    cosine = 1.0 - F.cosine_similarity(selected_prediction, selected_target, dim=-1).mean()
    if len(selected_prediction) > 1:
        logits = F.normalize(selected_prediction, dim=-1) @ F.normalize(
            selected_target, dim=-1
        ).transpose(0, 1)
        contrastive = F.cross_entropy(
            logits / temperature,
            torch.arange(len(logits), device=logits.device),
        )
    else:
        contrastive = reconstruction.new_zeros(())
    visible = valid_tokens & ~masked_tokens
    preservation = F.smooth_l1_loss(
        student_tokens[visible], teacher_tokens[visible].detach()
    )
    loss = (
        reconstruction + cosine + contrastive_weight * contrastive
        + visible_weight * preservation
    )
    return loss, {
        "reconstruction": float(reconstruction.detach()),
        "cosine": float(cosine.detach()),
        "contrastive": float(contrastive.detach()),
        "visible_preservation": float(preservation.detach()),
    }


def load_base(path: Path) -> tuple[Stage2TemporalHeadV17, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("temporal pretraining base must be a compact v17 Stage-2 head")
    model = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def evaluate_ssl(model, teacher, predictor, loader, args, device, seed: int) -> dict[str, float]:
    model.eval()
    predictor.eval()
    totals: dict[str, float] = {name: 0.0 for name in (
        "loss", "reconstruction", "cosine", "contrastive", "visible_preservation"
    )}
    batches = 0
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            features = batch["features"].to(device)
            window_mask = batch["window_mask"].to(device)
            token_mask = contiguous_token_mask(
                batch["window_mask"], tokens_per_window=model.config.tokens_per_window,
                ratio=args.mask_ratio, span_tokens=args.mask_span_tokens,
                generator=torch.Generator().manual_seed(seed + step),
            )
            clean, _ = teacher.encode(features, window_mask)
            masked = apply_token_mask(
                features, token_mask, tokens_per_window=model.config.tokens_per_window
            )
            encoded, _ = model.encode(masked, window_mask)
            loss, pieces = temporal_pretraining_loss(
                encoded, predictor(encoded), clean, token_mask,
                batch["window_mask"].repeat_interleave(model.config.tokens_per_window, dim=1),
                temperature=args.temperature,
                contrastive_weight=args.contrastive_weight,
                visible_weight=args.visible_weight,
            )
            totals["loss"] += float(loss)
            for name, value in pieces.items():
                totals[name] += value
            batches += 1
    return {name: value / max(1, batches) for name, value in totals.items()}


def run(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    student, base_payload = load_base(args.base_checkpoint)
    teacher = copy.deepcopy(student).to(device).eval()
    student.to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    for parameter in student.ctc_head.parameters():
        parameter.requires_grad = False
    predictor = nn.Sequential(
        nn.LayerNorm(student.config.dim),
        nn.Linear(student.config.dim, student.config.dim),
        nn.GELU(),
        nn.Linear(student.config.dim, student.config.dim),
    ).to(device)
    train = GenuineSentenceCropDataset(
        args.cache_root, "train", max_windows=student.config.max_windows,
        folds=args.folds, validation_fold=args.validation_fold,
        repeats=args.train_repeats, seed=args.seed,
    )
    validation = GenuineSentenceCropDataset(
        args.cache_root, "validation", max_windows=student.config.max_windows,
        folds=args.folds, validation_fold=args.validation_fold,
        repeats=args.validation_repeats, seed=args.seed,
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_temporal,
    )
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable.extend(predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(student.state_dict())
    best_predictor = copy.deepcopy(predictor.state_dict())
    best_validation: dict[str, float] | None = None
    history = []
    patience = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        train.set_epoch(epoch)
        generator = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        loader = DataLoader(
            train, batch_size=args.batch_size, shuffle=True, generator=generator,
            num_workers=0, collate_fn=collate_temporal,
        )
        student.train()
        predictor.train()
        total = seen = 0.0
        for step, batch in enumerate(loader):
            features = batch["features"].to(device)
            window_mask = batch["window_mask"].to(device)
            token_mask = contiguous_token_mask(
                batch["window_mask"], tokens_per_window=student.config.tokens_per_window,
                ratio=args.mask_ratio, span_tokens=args.mask_span_tokens,
                generator=torch.Generator().manual_seed(args.seed + epoch * 10_000 + step),
            )
            with torch.inference_mode():
                clean, _ = teacher.encode(features, window_mask)
            masked = apply_token_mask(
                features, token_mask, tokens_per_window=student.config.tokens_per_window
            )
            optimizer.zero_grad(set_to_none=True)
            encoded, _ = student.encode(masked, window_mask)
            loss, _ = temporal_pretraining_loss(
                encoded, predictor(encoded), clean, token_mask,
                batch["window_mask"].repeat_interleave(student.config.tokens_per_window, dim=1),
                temperature=args.temperature,
                contrastive_weight=args.contrastive_weight,
                visible_weight=args.visible_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite temporal pretraining loss at epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        scheduler.step()
        validation_metrics = evaluate_ssl(
            student, teacher, predictor, validation_loader, args, device,
            args.seed + epoch * 100_000,
        )
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, seen),
            "validation": validation_metrics,
        }
        history.append(row)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_validation = validation_metrics
            best_state = copy.deepcopy(student.state_dict())
            best_predictor = copy.deepcopy(predictor.state_dict())
            patience = 0
        else:
            patience += 1
        LOG.info(
            "epoch=%d train=%.5f validation=%.5f best=%d patience=%d",
            epoch, row["train_loss"], validation_metrics["loss"], best_epoch, patience,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break
    if best_validation is None:
        raise RuntimeError("temporal pretraining produced no selected epoch")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_stage2_temporal_pretrain_v17",
        "format_version": 1,
        "model_config": student.config.to_dict(),
        "model_state_dict": best_state,
        "predictor_state_dict": best_predictor,
        "base_checkpoint": args.base_checkpoint.as_posix(),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "source_cache_root": args.cache_root.as_posix(),
        "source_manifest": args.source_manifest.as_posix(),
        "source_manifest_sha256": sha256(args.source_manifest),
        "source_split": "2m_flores_dev_train_only",
        "split_contract": {
            "folds": args.folds,
            "validation_fold": args.validation_fold,
            "train_items": len(train.paths),
            "validation_items": len(validation.paths),
        },
        "objective": "masked_clean_teacher_reconstruction_plus_token_contrastive_alignment",
        "selected_epoch": best_epoch,
        "validation": best_validation,
        "ctc_head_trained": False,
        "two_m_flores_devtest_accessed": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    checkpoint_path = args.output / "temporal_pretrained.pth"
    torch.save(checkpoint, checkpoint_path)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    probe = Stage2TemporalHeadV17(Stage2V17Config(**reloaded["model_config"]))
    probe.load_state_dict(reloaded["model_state_dict"], strict=True)
    result = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selected_epoch": best_epoch,
        "validation": best_validation,
        "train_items": len(train.paths),
        "validation_items": len(validation.paths),
        "train_crops_per_epoch": len(train),
        "validation_crops": len(validation),
        "seconds": time.monotonic() - started,
        "device": str(device),
        "ctc_head_trained": False,
        "two_m_flores_devtest_accessed": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"),
    )
    parser.add_argument(
        "--cache-root", type=Path,
        default=Path("data/local/stage2_v17_2m_flores_frozen_features"),
    )
    parser.add_argument(
        "--source-manifest", type=Path,
        default=Path("active/v17/stage2_2m_flores_training_manifest_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_2m_flores_temporal_pretrain_v1"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=11701)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--train-repeats", type=int, default=4)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--mask-span-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--contrastive-weight", type=float, default=0.10)
    parser.add_argument("--visible-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
