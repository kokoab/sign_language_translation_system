#!/usr/bin/env python3
"""Train learned mouth/lower-face fusion on frozen Auto-AVSR features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_visual_speech_v17 import (
    MultiViewVisualSpeechHeadV17,
    MultiViewVisualSpeechV17Config,
)
from active.v17.train_stage_1_v17 import ExponentialMovingAverage, select_device
from active.v17.train_stage_1_visual_speech_features_v17 import (
    FrozenVisualFeatureDataset,
)
from active.v17.train_stage_1_visual_speech_v17 import (
    classification_metrics,
    sha256_file,
)


LOG = logging.getLogger("visual_speech_multiview_stage1_v17")
VIEW_ORDER = ("mouth", "lower_face")


class MultiViewFeatureDataset(Dataset):
    def __init__(self, paths: tuple[Path, Path], expected_split: str):
        datasets = [
            FrozenVisualFeatureDataset(path, expected_split) for path in paths
        ]
        if tuple(dataset.metadata.get("view") for dataset in datasets) != VIEW_ORDER:
            raise ValueError(f"multi-view caches must be ordered as {VIEW_ORDER}")
        reference = datasets[0]
        for dataset in datasets[1:]:
            for key in (
                "crop_schema_fingerprint", "manifest_sha256",
                "pretrained_checkpoint_sha256", "split",
            ):
                if dataset.metadata.get(key) != reference.metadata.get(key):
                    raise ValueError(f"multi-view cache mismatch: {key}")
            if not np.array_equal(dataset.item_ids, reference.item_ids):
                raise ValueError("multi-view cache item IDs are not aligned")
            if not torch.equal(dataset.targets, reference.targets):
                raise ValueError("multi-view cache targets are not aligned")
        self.features = torch.stack(
            [dataset.features for dataset in datasets], dim=2
        )
        self.valid = torch.stack([dataset.valid for dataset in datasets], dim=2)
        self.targets = reference.targets
        self.item_ids = reference.item_ids
        self.metadata = reference.metadata

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.features[index], self.valid[index], self.targets[index]


def augment_multiview(
    features: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    output = features.clone()
    valid = valid.clone()
    batch, frames = output.shape[:2]
    base = torch.linspace(0.0, 1.0, frames, device=output.device)
    for sample in range(batch):
        if torch.rand((), device=output.device) < 0.5:
            rate = 0.90 + 0.20 * torch.rand((), device=output.device)
            indices = (
                ((base - 0.5) * rate + 0.5).clamp(0, 1) * (frames - 1)
            ).round().long()
            output[sample] = output[sample].index_select(0, indices)
            valid[sample] = valid[sample].index_select(0, indices)
        if torch.rand((), device=output.device) < 0.35:
            length = int(torch.randint(1, 5, (), device=output.device))
            start = int(torch.randint(0, frames - length + 1, (), device=output.device))
            output[sample, start:start + length] = 0
            valid[sample, start:start + length] = False
        # Occasional single-view dropout prevents either crop from becoming mandatory.
        if torch.rand((), device=output.device) < 0.20:
            view = int(torch.randint(0, 2, (), device=output.device))
            output[sample, :, view] = 0
            valid[sample, :, view] = False
    scale = output.detach().std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
    output = output + 0.01 * scale * torch.randn_like(output)
    output *= valid.unsqueeze(-1)
    return output.contiguous(), valid.contiguous()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, targets_all = [], []
    loss_sum = 0.0
    for features, valid, targets in loader:
        targets_all.append(targets.numpy())
        features, valid, targets = (
            features.to(device), valid.to(device), targets.to(device)
        )
        logits = model(features, valid)
        loss_sum += float(F.cross_entropy(logits, targets).cpu()) * len(targets)
        logits_all.append(logits.cpu().numpy())
    logits = np.concatenate(logits_all)
    targets = np.concatenate(targets_all)
    return {
        "loss": loss_sum / len(targets),
        **classification_metrics(logits, targets),
    }, logits, targets


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_paths = (args.mouth_train_cache, args.lower_face_train_cache)
    validation_paths = (
        args.mouth_validation_cache, args.lower_face_validation_cache
    )
    train_set = MultiViewFeatureDataset(train_paths, "train")
    validation_set = MultiViewFeatureDataset(validation_paths, "val")
    for key in (
        "crop_schema_fingerprint", "manifest_sha256",
        "pretrained_checkpoint_sha256",
    ):
        if train_set.metadata.get(key) != validation_set.metadata.get(key):
            raise ValueError(f"train/validation cache mismatch: {key}")
    counts = torch.bincount(train_set.targets).float()
    sampler = WeightedRandomSampler(
        1.0 / counts[train_set.targets], len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
    )
    device = select_device(args.device)
    config = MultiViewVisualSpeechV17Config(
        num_classes=int(counts.numel()), dim=args.dim, view_dim=args.view_dim,
        depth=args.depth, heads=args.heads, dropout=args.dropout,
        head_dropout=args.head_dropout,
    )
    model = MultiViewVisualSpeechHeadV17(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.02
    )
    ema = ExponentialMovingAverage(model, args.ema_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "visual_only": True,
        "audio_accessed": False,
        "test_evaluated": False,
        "views": list(VIEW_ORDER),
        "train_caches": [str(path) for path in train_paths],
        "train_cache_sha256": [sha256_file(path) for path in train_paths],
        "validation_caches": [str(path) for path in validation_paths],
        "validation_cache_sha256": [
            sha256_file(path) for path in validation_paths
        ],
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "pretrained_checkpoint_sha256": train_set.metadata[
            "pretrained_checkpoint_sha256"
        ],
        "frontend_frozen": True,
        "class_balanced_sampling": True,
        "view_dropout_probability": 0.20,
        "cache_metadata": train_set.metadata,
    }
    (args.output / "training_data_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info(
        "device=%s train=%d val=%d params=%d views=%s",
        device, len(train_set), len(validation_set), model.parameter_count,
        ",".join(VIEW_ORDER),
    )
    history, best, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        started = time.monotonic()
        for features, valid, targets in train_loader:
            features, valid = augment_multiview(
                features.to(device), valid.to(device)
            )
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, valid)
            loss = F.cross_entropy(
                logits, targets, label_smoothing=args.label_smoothing
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            total_loss += float(loss.detach().cpu()) * len(targets)
            seen += len(targets)
        live = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        model.load_state_dict(ema.shadow)
        metrics, logits, targets = evaluate(model, validation_loader, device)
        model.load_state_dict(live)
        scheduler.step()
        row = {
            "epoch": float(epoch), "train_loss": total_loss / seen,
            **metrics, "seconds": time.monotonic() - started,
        }
        history.append(row)
        LOG.info(
            "epoch=%d train=%.4f val=%.4f top1=%.2f top5=%.2f seconds=%.1f",
            epoch, row["train_loss"], row["loss"], row["top1"], row["top5"],
            row["seconds"],
        )
        if metrics["top1"] > best:
            best, stale = metrics["top1"], 0
            checkpoint = {
                "format": "slt_stage1_visual_speech_multiview_v17",
                "epoch": epoch,
                "model_config": config.to_dict(),
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in ema.shadow.items()
                },
                "validation_metrics": metrics,
                "manifest_sha256": train_set.metadata["manifest_sha256"],
                "training_data_provenance": provenance,
                "test_evaluated": False,
            }
            temporary = args.output / "best_model.pth.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best_model.pth")
            np.savez_compressed(
                args.output / "best_validation_logits.npz",
                logits=logits.astype(np.float32), targets=targets,
                item_ids=validation_set.item_ids,
            )
        else:
            stale += 1
        (args.output / "history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        if stale >= args.patience:
            LOG.info("early stopping after %d stale epochs", stale)
            break
    result = {
        "best_validation_top1": best,
        "epochs_completed": len(history),
        "parameters": model.parameter_count,
        "views": list(VIEW_ORDER),
        "device": str(device),
        "visual_only": True,
        "audio_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouth-train-cache", type=Path, required=True)
    parser.add_argument("--lower-face-train-cache", type=Path, required=True)
    parser.add_argument("--mouth-validation-cache", type=Path, required=True)
    parser.add_argument("--lower-face-validation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--view-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(train(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
