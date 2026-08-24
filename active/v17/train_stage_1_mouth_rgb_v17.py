#!/usr/bin/env python3
"""Train the separate visual-only v17 mouth-crop classifier on Citizen train/val."""

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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_mouth_rgb_v17 import decode_packed_crops
    from active.v17.model_mouth_rgb_v17 import MouthRGBStage1, MouthRGBStage1Config
    from active.v17.schema_mouth_rgb_v17 import MouthRGBV17Config, schema_fingerprint
    from active.v17.train_stage_1_v17 import ExponentialMovingAverage, select_device
else:
    from .extract_mouth_rgb_v17 import decode_packed_crops
    from .model_mouth_rgb_v17 import MouthRGBStage1, MouthRGBStage1Config
    from .schema_mouth_rgb_v17 import MouthRGBV17Config, schema_fingerprint
    from .train_stage_1_v17 import ExponentialMovingAverage, select_device


LOG = logging.getLogger("mouth_rgb_stage1_v17")


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


class CitizenMouthRGBV17Dataset(Dataset):
    def __init__(
        self, root: Path, split: str, manifest_path: Path, rejection_path: Path | None,
    ):
        if split not in ("train", "val"):
            raise ValueError("mouth RGB development accepts only train or val")
        self.root = root
        self.split = split
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = sorted(manifest["classes"], key=lambda row: int(row["class_index"]))
        self.label_to_index = {
            str(row["canonical_label"]): int(row["class_index"]) for row in classes
        }
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.expected_schema = schema_fingerprint(MouthRGBV17Config())
        rejected = load_rejections(rejection_path)
        files: list[Path] = []
        targets: list[int] = []
        for label, target in self.label_to_index.items():
            class_root = root / split / label
            selected = [
                path for path in sorted(class_root.glob("*.mouth_rgb_v17.npz"))
                if (
                    split, label,
                    path.name.removesuffix(".mouth_rgb_v17.npz") + ".mp4",
                ) not in rejected
            ]
            if not selected:
                raise ValueError(f"no mouth RGB {split} samples for {label}")
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
                raise ValueError(f"mouth RGB schema mismatch: {path}")
            if metadata.get("split") != self.split or metadata.get("canonical_label") != path.parent.name:
                raise ValueError(f"mouth RGB provenance mismatch: {path}")
            pixels = decode_packed_crops(payload["jpeg_blob"], payload["jpeg_offsets"])
            valid = payload["valid"].astype(np.bool_, copy=False)
        pixels = pixels[..., ::-1].copy()  # BGR to RGB
        tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        tensor *= torch.from_numpy(valid.copy())[:, None, None, None]
        return tensor, torch.from_numpy(valid.copy()), self.targets[index]


def augment_mouth(pixels: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    output = pixels.clone()
    batch, frames = output.shape[:2]
    device = output.device
    flip = torch.rand(batch, device=device) < 0.5
    output[flip] = output[flip].flip(-1)
    contrast = 0.85 + 0.30 * torch.rand(batch, 1, 1, 1, 1, device=device)
    brightness = (torch.rand(batch, 1, 1, 1, 1, device=device) - 0.5) * 0.20
    output = (output * contrast + brightness).clamp(-1.0, 1.0)
    if torch.rand((), device=device) < 0.5:
        base = torch.linspace(0.0, 1.0, frames, device=device)
        warped = output.clone()
        warped_valid = valid.clone()
        for sample in range(batch):
            rate = 0.85 + 0.30 * torch.rand((), device=device)
            indices = (((base - 0.5) * rate + 0.5).clamp(0, 1) * (frames - 1)).round().long()
            warped[sample] = output[sample].index_select(0, indices)
            warped_valid[sample] = valid[sample].index_select(0, indices)
        output, valid = warped, warped_valid
    output *= valid[:, :, None, None, None]
    return output.contiguous(), valid.contiguous()


def metrics_from_logits(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predicted = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    confusion = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
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
def evaluate(model: MouthRGBStage1, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    loss_sum = 0.0
    for pixels, valid, targets in loader:
        targets_all.append(targets.numpy())
        pixels, valid, targets = pixels.to(device), valid.to(device), targets.to(device)
        logits = model(pixels, valid)
        loss_sum += float(F.cross_entropy(logits, targets).cpu()) * len(targets)
        logits_all.append(logits.cpu().numpy())
    logits = np.concatenate(logits_all)
    targets = np.concatenate(targets_all)
    return {"loss": loss_sum / len(targets), **metrics_from_logits(logits, targets)}, logits, targets


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = select_device(args.device)
    train_set = CitizenMouthRGBV17Dataset(args.data_root, "train", args.manifest, args.rejections)
    val_set = CitizenMouthRGBV17Dataset(args.data_root, "val", args.manifest, args.rejections)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.architecture == "mobilenet_v3_small":
        from active.v17.model_mouth_rgb_mobilenet_v17 import (
            MouthMobileNetV17, MouthMobileNetV17Config,
        )
        model = MouthMobileNetV17(
            MouthMobileNetV17Config(len(train_set.label_to_index), 128, args.dropout)
        ).to(device)
        backbone_parameters = list(model.backbone.parameters())
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        head_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in backbone_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": args.backbone_lr},
                {"params": head_parameters, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        model = MouthRGBStage1(
            MouthRGBStage1Config(len(train_set.label_to_index), args.dropout)
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=min(args.lr, args.backbone_lr) * 0.02
    )
    ema = ExponentialMovingAverage(model, args.ema_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []; best = -1.0; stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = seen = 0; started = time.monotonic()
        for pixels, valid, targets in train_loader:
            pixels, valid = augment_mouth(pixels.to(device), valid.to(device))
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixels, valid)
            loss = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ema.update(model)
            total_loss += float(loss.detach().cpu()) * len(targets); seen += len(targets)
        live = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema.shadow)
        metrics, logits, targets = evaluate(model, val_loader, device)
        model.load_state_dict(live); scheduler.step()
        row = {"epoch": epoch, "train_loss": total_loss / seen, **metrics, "seconds": time.monotonic() - started}
        history.append(row); LOG.info("epoch=%d train=%.4f val=%.4f top1=%.2f top5=%.2f", epoch, row["train_loss"], row["loss"], row["top1"], row["top5"])
        if metrics["top1"] > best:
            best = metrics["top1"]; stale = 0
            checkpoint = {
                "format": "slt_stage1_mouth_rgb_v17", "epoch": epoch,
                "architecture": args.architecture,
                "model_config": model.config.to_dict(),
                "model_state_dict": {key: value.detach().cpu().clone() for key, value in ema.shadow.items()},
                "validation_metrics": metrics,
                "label_to_index": train_set.label_to_index,
                "manifest_sha256": sha256_file(args.manifest),
                "crop_schema_fingerprint": train_set.expected_schema,
                "training_samples": len(train_set), "validation_samples": len(val_set),
                "visual_only": True, "audio_accessed": False, "test_evaluated": False,
            }
            temporary = args.output / "best_model.pth.tmp"; torch.save(checkpoint, temporary); temporary.replace(args.output / "best_model.pth")
            np.savez_compressed(args.output / "best_validation_logits.npz", logits=logits.astype(np.float32), targets=targets, item_ids=np.asarray([str(path.relative_to(args.data_root)) for path in val_set.files]))
        else:
            stale += 1
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= args.patience:
            break
    result = {"best_validation_top1": best, "epochs_completed": len(history), "parameters": model.parameter_count, "architecture": args.architecture, "device": str(device), "visual_only": True, "audio_accessed": False, "test_evaluated": False}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/mouth_rgb"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17_mouth_rgb"))
    parser.add_argument("--architecture", choices=("tiny", "mobilenet_v3_small"), default="tiny")
    parser.add_argument("--epochs", type=int, default=80); parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4); parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.10); parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--ema-decay", type=float, default=0.999); parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(); print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    main()
