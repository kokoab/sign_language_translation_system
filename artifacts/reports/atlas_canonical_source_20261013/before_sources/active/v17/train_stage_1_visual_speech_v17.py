#!/usr/bin/env python3
"""Train an Auto-AVSR-initialized visual-only Citizen100 face/mouth teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_visual_speech_v17 import (
    VisualSpeechTeacherV17,
    VisualSpeechTeacherV17Config,
    load_auto_avsr_frontend,
)
from active.v17.schema_visual_speech_v17 import (
    CROP_SIZE,
    VIEW_NAMES,
    VisualSpeechV17Config,
    schema_fingerprint,
)
from active.v17.train_stage_1_v17 import ExponentialMovingAverage, select_device


LOG = logging.getLogger("visual_speech_stage1_v17")
AUTO_AVSR_SOURCE = "https://github.com/mpc001/auto_avsr"
AUTO_AVSR_MODEL_NAME = "vsr_trlrs2lrs3vox2avsp_base.pth"
AUTO_AVSR_TRAINING_HOURS = 3291
AUTO_AVSR_REPORTED_LRS3_WER = 20.3


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rejections(path: Path | None) -> set[tuple[str, str, str]]:
    if path is None or not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["split"], row["canonical_label"], row["video"])
            for row in csv.DictReader(handle)
        }


def decode_view(blob: np.ndarray, offsets: np.ndarray, view_index: int) -> np.ndarray:
    output = np.zeros((len(offsets), CROP_SIZE, CROP_SIZE, 3), np.uint8)
    for frame, (start, length) in enumerate(offsets[:, view_index].tolist()):
        if start < 0:
            continue
        decoded = cv2.imdecode(blob[start:start + length], cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != output[frame].shape:
            raise ValueError("invalid packed visual-speech JPEG")
        output[frame] = decoded
    return output


class CitizenVisualSpeechV17Dataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        manifest_path: Path,
        rejection_path: Path | None,
        view: str,
    ):
        if split not in ("train", "val"):
            raise ValueError("visual-speech development accepts only train/val")
        if view not in VIEW_NAMES:
            raise ValueError(f"unknown visual-speech view: {view}")
        self.root = root
        self.split = split
        self.view = view
        self.view_index = VIEW_NAMES.index(view)
        self.expected_schema = schema_fingerprint(VisualSpeechV17Config())
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = sorted(manifest["classes"], key=lambda row: int(row["class_index"]))
        self.label_to_index = {
            str(row["canonical_label"]): int(row["class_index"]) for row in classes
        }
        rejected = load_rejections(rejection_path)
        files: list[Path] = []
        targets: list[int] = []
        for label, target in self.label_to_index.items():
            selected = [
                path for path in sorted((root / split / label).glob("*.visual_speech_v17.npz"))
                if (
                    split, label,
                    path.name.removesuffix(".visual_speech_v17.npz") + ".mp4",
                ) not in rejected
            ]
            if not selected:
                raise ValueError(f"no visual-speech {split} samples for {label}")
            files.extend(selected)
            targets.extend([target] * len(selected))
        self.files = files
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            if metadata.get("schema_fingerprint") != self.expected_schema:
                raise ValueError(f"visual-speech schema mismatch: {path}")
            if metadata.get("split") != self.split or metadata.get("canonical_label") != path.parent.name:
                raise ValueError(f"visual-speech provenance mismatch: {path}")
            if metadata.get("audio_accessed") is not False:
                raise ValueError(f"visual-speech archive does not prove visual-only input: {path}")
            pixels = decode_view(payload["jpeg_blob"], payload["jpeg_offsets"], self.view_index)
            valid = payload["valid"][:, self.view_index].astype(np.bool_, copy=True)
        # Retain uint8 until the batch reaches its execution device.
        return (
            torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(valid),
            self.targets[index],
        )


def prepare_pixels(
    pixels: torch.Tensor, valid: torch.Tensor, training: bool,
    augmentation: str = "strong",
) -> tuple[torch.Tensor, torch.Tensor]:
    if augmentation not in ("mild", "strong"):
        raise ValueError("pixel augmentation must be mild or strong")
    pixels = pixels.float().div_(255.0)
    batch, frames, _, height, width = pixels.shape
    crop = 88
    output = torch.empty((batch, frames, 3, crop, crop), device=pixels.device)
    for sample in range(batch):
        if training:
            if augmentation == "mild":
                center_top, center_left = (height - crop) // 2, (width - crop) // 2
                top = max(0, min(height - crop, center_top + int(torch.randint(-4, 5, (), device=pixels.device))))
                left = max(0, min(width - crop, center_left + int(torch.randint(-4, 5, (), device=pixels.device))))
            else:
                top = int(torch.randint(0, height - crop + 1, (), device=pixels.device))
                left = int(torch.randint(0, width - crop + 1, (), device=pixels.device))
        else:
            top, left = (height - crop) // 2, (width - crop) // 2
        output[sample] = pixels[sample, :, :, top:top + crop, left:left + crop]
    if training:
        flipped = torch.rand(batch, device=pixels.device) < 0.5
        output[flipped] = output[flipped].flip(-1)
        spread = 0.04 if augmentation == "mild" else 0.20
        contrast = 1.0 - spread / 2 + spread * torch.rand(batch, 1, 1, 1, 1, device=pixels.device)
        brightness_spread = 0.04 if augmentation == "mild" else 0.12
        brightness = (torch.rand(batch, 1, 1, 1, 1, device=pixels.device) - 0.5) * brightness_spread
        output = (output * contrast + brightness).clamp(0.0, 1.0)
    # Input archives are BGR. Auto-AVSR uses torchvision Grayscale then 0.421/0.165.
    gray = (
        0.1140 * output[:, :, 0:1]
        + 0.5870 * output[:, :, 1:2]
        + 0.2989 * output[:, :, 2:3]
    )
    gray = (gray - 0.421) / 0.165
    if training and augmentation == "strong":
        for sample in range(batch):
            if torch.rand((), device=pixels.device) < 0.35:
                length = int(torch.randint(1, 5, (), device=pixels.device))
                start = int(torch.randint(0, max(1, frames - length + 1), (), device=pixels.device))
                gray[sample, start:start + length] = 0
                valid[sample, start:start + length] = False
    gray *= valid[:, :, None, None, None]
    return gray.contiguous(), valid.contiguous()


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predicted = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    confusion = np.zeros((logits.shape[1], logits.shape[1]), np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "top1": float(100 * np.mean(predicted == targets)),
        "top5": float(100 * np.mean((top5 == targets[:, None]).any(axis=1))),
        "macro_f1": float(100 * f1.mean()),
        "samples": float(len(targets)),
    }


@torch.no_grad()
def evaluate(
    model: VisualSpeechTeacherV17, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    loss_sum = 0.0
    for pixels, valid, targets in loader:
        targets_all.append(targets.numpy())
        pixels, valid = prepare_pixels(pixels.to(device), valid.to(device), False)
        targets = targets.to(device)
        logits = model(pixels, valid)
        loss_sum += float(F.cross_entropy(logits, targets).cpu()) * len(targets)
        logits_all.append(logits.cpu().numpy())
    logits = np.concatenate(logits_all)
    targets = np.concatenate(targets_all)
    return {
        "loss": loss_sum / len(targets),
        **classification_metrics(logits, targets),
    }, logits, targets


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.freeze_frontend_epochs < 0:
        raise ValueError("freeze-frontend-epochs must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_set = CitizenVisualSpeechV17Dataset(
        args.data_root, "train", args.manifest, args.rejections, args.view
    )
    validation_set = CitizenVisualSpeechV17Dataset(
        args.data_root, "val", args.manifest, args.rejections, args.view
    )
    counts = torch.bincount(train_set.targets, minlength=len(train_set.label_to_index)).float()
    weights = 1.0 / counts[train_set.targets]
    sampler = WeightedRandomSampler(
        weights, num_samples=len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, drop_last=False,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    config = VisualSpeechTeacherV17Config(
        num_classes=len(train_set.label_to_index), dim=args.dim,
        depth=args.depth, heads=args.heads,
        dropout=args.dropout, head_dropout=args.head_dropout,
    )
    model = VisualSpeechTeacherV17(config)
    frontend_load = load_auto_avsr_frontend(model.frontend, str(args.pretrained_checkpoint))
    initialize_from = None
    if args.initialize_from:
        initialize_from = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        if initialize_from.get("format") != "slt_stage1_visual_speech_v17":
            raise ValueError("visual-speech warm start format mismatch")
        if initialize_from.get("model_config") != config.to_dict():
            raise ValueError("visual-speech warm start architecture mismatch")
        model.load_state_dict(initialize_from["model_state_dict"], strict=True)
    model = model.to(device)
    def set_frontend_trainable(enabled: bool) -> None:
        for parameter in model.frontend.parameters():
            parameter.requires_grad_(False)
        if not enabled:
            return
        modules = (
            (model.frontend.trunk.layer4,)
            if args.frontend_trainable_scope == "layer4"
            else (model.frontend,)
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    set_frontend_trainable(args.freeze_frontend_epochs == 0)
    frontend_parameters = list(model.frontend.parameters())
    frontend_ids = {id(parameter) for parameter in frontend_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in frontend_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": frontend_parameters, "lr": args.frontend_lr},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=min(args.lr, args.frontend_lr) * 0.02
    )
    ema = ExponentialMovingAverage(model, args.ema_decay)
    if initialize_from is not None:
        # A warm start is already a trained solution. Skip the from-scratch EMA
        # ramp that deliberately forgets random initialization in early updates.
        ema.updates = 10_000
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "visual_only": True,
        "audio_accessed": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "view": args.view,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "crop_schema_fingerprint": train_set.expected_schema,
        "class_balanced_sampling": True,
        "frontend_trainable_scope": args.frontend_trainable_scope,
        "pixel_augmentation": args.pixel_augmentation,
        "initialize_from": str(args.initialize_from) if args.initialize_from else None,
        "initialize_from_sha256": (
            sha256_file(args.initialize_from) if args.initialize_from else None
        ),
        "pretraining": {
            "source": AUTO_AVSR_SOURCE,
            "model": AUTO_AVSR_MODEL_NAME,
            "checkpoint": str(args.pretrained_checkpoint),
            "sha256": sha256_file(args.pretrained_checkpoint),
            "reported_training_hours": AUTO_AVSR_TRAINING_HOURS,
            "reported_lrs3_visual_wer": AUTO_AVSR_REPORTED_LRS3_WER,
            "transferred_component": "frontend3D plus ResNet18 trunk only",
            "load_result": frontend_load,
            "license_boundary": (
                "Auto-AVSR code is Apache-2.0; checkpoint terms may derive from its "
                "training datasets and require separate deployment review"
            ),
        },
    }
    (args.output / "training_data_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info(
        "device=%s train=%d val=%d params=%d view=%s frontend_keys=%d",
        device, len(train_set), len(validation_set), model.parameter_count,
        args.view, frontend_load["loaded_keys"],
    )
    history: list[dict[str, float]] = []
    best = -1.0
    stale = 0
    if initialize_from is not None:
        metrics, logits, targets = evaluate(model, validation_loader, device)
        best = metrics["top1"]
        row = {
            "epoch": 0.0, "train_loss": None, **metrics, "seconds": 0.0,
            "frontend_frozen": args.freeze_frontend_epochs > 0,
        }
        history.append(row)
        checkpoint = {
            "format": "slt_stage1_visual_speech_v17", "epoch": 0,
            "model_config": config.to_dict(),
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "validation_metrics": metrics,
            "label_to_index": train_set.label_to_index,
            "manifest_sha256": sha256_file(args.manifest),
            "training_data_provenance": provenance, "test_evaluated": False,
        }
        temporary = args.output / "best_model.pth.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "best_model.pth")
        np.savez_compressed(
            args.output / "best_validation_logits.npz",
            logits=logits.astype(np.float32), targets=targets,
            item_ids=np.asarray([
                str(path.relative_to(args.data_root)) for path in validation_set.files
            ]),
        )
        LOG.info("epoch=0 warm-start top1=%.2f top5=%.2f", metrics["top1"], metrics["top5"])
    for epoch in range(1, args.epochs + 1):
        if args.freeze_frontend_epochs and epoch == args.freeze_frontend_epochs + 1:
            set_frontend_trainable(True)
            LOG.info("unfroze Auto-AVSR frontend at epoch=%d", epoch)
        model.train()
        total_loss = 0.0
        seen = 0
        started = time.monotonic()
        for batch_index, (pixels, valid, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            pixels, valid = prepare_pixels(
                pixels.to(device), valid.to(device), True, args.pixel_augmentation
            )
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixels, valid)
            loss = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            total_loss += float(loss.detach().cpu()) * len(targets)
            seen += len(targets)
        live = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema.shadow)
        metrics, logits, targets = evaluate(model, validation_loader, device)
        model.load_state_dict(live)
        scheduler.step()
        row = {
            "epoch": float(epoch), "train_loss": total_loss / max(seen, 1),
            **metrics, "seconds": time.monotonic() - started,
            "frontend_frozen": epoch <= args.freeze_frontend_epochs,
        }
        history.append(row)
        LOG.info(
            "epoch=%d train=%.4f val=%.4f top1=%.2f top5=%.2f seconds=%.1f",
            epoch, row["train_loss"], row["loss"], row["top1"], row["top5"], row["seconds"],
        )
        if metrics["top1"] > best:
            best = metrics["top1"]
            stale = 0
            checkpoint = {
                "format": "slt_stage1_visual_speech_v17",
                "epoch": epoch,
                "model_config": config.to_dict(),
                "model_state_dict": {
                    key: value.detach().cpu().clone() for key, value in ema.shadow.items()
                },
                "validation_metrics": metrics,
                "label_to_index": train_set.label_to_index,
                "manifest_sha256": sha256_file(args.manifest),
                "training_data_provenance": provenance,
                "test_evaluated": False,
            }
            temporary = args.output / "best_model.pth.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best_model.pth")
            np.savez_compressed(
                args.output / "best_validation_logits.npz",
                logits=logits.astype(np.float32),
                targets=targets,
                item_ids=np.asarray([
                    str(path.relative_to(args.data_root)) for path in validation_set.files
                ]),
            )
        else:
            stale += 1
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= args.patience:
            LOG.info("early stopping after %d stale epochs", stale)
            break
    result = {
        "best_validation_top1": best,
        "epochs_completed": len(history),
        "parameters": model.parameter_count,
        "view": args.view,
        "device": str(device),
        "visual_only": True,
        "audio_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/visual_speech_rgb"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--view", choices=VIEW_NAMES, default="mouth")
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17_visual_speech_teacher"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--frontend-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--freeze-frontend-epochs", type=int, default=5)
    parser.add_argument(
        "--frontend-trainable-scope", choices=("layer4", "full"), default="full"
    )
    parser.add_argument(
        "--pixel-augmentation", choices=("mild", "strong"), default="strong"
    )
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(train(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
