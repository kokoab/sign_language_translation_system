#!/usr/bin/env python3
"""Train the v17 frozen-MobileCLIP2 RGB Stage 1 challenger."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
from torch.utils.data import DataLoader, Dataset, Subset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.model_mobileclip2_v17 import (
        MobileCLIP2Stage1Config,
        MobileCLIP2Stage1V17,
        make_checkpoint,
    )
    from active.v17.schema_mobileclip2_v17 import (
        EMBEDDING_DIM,
        SEQUENCE_LENGTH,
        MobileCLIP2V17Config,
        schema_fingerprint,
    )
    from active.v17.train_stage_1_v17 import ExponentialMovingAverage, load_rejections
else:
    from .model_mobileclip2_v17 import (
        MobileCLIP2Stage1Config,
        MobileCLIP2Stage1V17,
        make_checkpoint,
    )
    from .schema_mobileclip2_v17 import (
        EMBEDDING_DIM,
        SEQUENCE_LENGTH,
        MobileCLIP2V17Config,
        schema_fingerprint,
    )
    from .train_stage_1_v17 import ExponentialMovingAverage, load_rejections


LOG = logging.getLogger("stage1_mobileclip2_v17")
EXPECTED_SHAPE = (SEQUENCE_LENGTH, EMBEDDING_DIM)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MobileCLIP2CitizenDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        manifest_path: Path,
        rejection_path: Path | None = None,
        *,
        cache: bool = True,
        expected_schema: str | None = None,
    ):
        if split not in ("train", "val"):
            raise ValueError("the Citizen test split is sealed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        classes = sorted(manifest["classes"], key=lambda item: item["class_index"])
        self.label_to_index = {
            str(item["canonical_label"]): int(item["class_index"])
            for item in classes
        }
        if sorted(self.label_to_index.values()) != list(range(len(classes))):
            raise ValueError("manifest class indices are not contiguous")
        self.num_classes = len(classes)
        self.expected_schema = expected_schema or schema_fingerprint(MobileCLIP2V17Config())
        rejected = load_rejections(rejection_path)
        self.files: list[Path] = []
        targets = []
        for label, target in self.label_to_index.items():
            class_root = root / split / label
            if not class_root.is_dir():
                raise FileNotFoundError(class_root)
            selected = [
                path for path in sorted(class_root.glob("*.mobileclip2_v17.npz"))
                if (
                    split,
                    label,
                    path.name.removesuffix(".mobileclip2_v17.npz") + ".mp4",
                ) not in rejected
            ]
            if not selected:
                raise ValueError(f"no usable {split} samples for {label}")
            self.files.extend(selected)
            targets.extend([target] * len(selected))
        self.targets = torch.tensor(targets, dtype=torch.long)
        self._cache = [self._load(path) for path in self.files] if cache else None

    def _load(self, path: Path) -> torch.Tensor:
        with np.load(path, allow_pickle=False) as payload:
            value = payload["embeddings"]
            metadata = json.loads(str(payload["metadata_json"]))
        if tuple(value.shape) != EXPECTED_SHAPE:
            raise ValueError(f"{path}: expected {EXPECTED_SHAPE}, got {value.shape}")
        if metadata.get("schema_fingerprint") != self.expected_schema:
            raise ValueError(f"{path}: MobileCLIP2 schema mismatch")
        value = value.astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: non-finite embeddings")
        return torch.from_numpy(value.copy())

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = self._cache[index] if self._cache is not None else self._load(self.files[index])
        return value, self.targets[index]

    def balanced_subset(self, samples_per_class: int) -> Subset:
        remaining = {index: samples_per_class for index in range(self.num_classes)}
        selected = []
        for index, target in enumerate(self.targets.tolist()):
            if remaining[target] > 0:
                selected.append(index)
                remaining[target] -= 1
        if any(remaining.values()):
            raise ValueError("requested subset exceeds at least one class")
        return Subset(self, selected)


def augment_embeddings(value: torch.Tensor) -> torch.Tensor:
    """Temporal-only augmentation; never fabricate alternate RGB embeddings."""
    result = value.clone()
    batch, frames, _ = result.shape
    device = result.device
    if torch.rand((), device=device) < 0.7:
        base = torch.linspace(0.0, 1.0, frames, device=device)
        warped = result.clone()
        for sample in range(batch):
            rate = 0.82 + 0.36 * torch.rand((), device=device)
            offset = (torch.rand((), device=device) - 0.5) * 0.08
            positions = ((base - 0.5) * rate + 0.5 + offset).clamp(0.0, 1.0)
            indices = (positions * (frames - 1)).round().long()
            warped[sample] = result[sample].index_select(0, indices)
        result = warped
    if torch.rand((), device=device) < 0.4:
        mask = torch.rand(batch, frames, 1, device=device) < 0.10
        # Replacing with an adjacent real embedding is temporal dropout, not
        # interpolation or a synthetic visual observation.
        previous = torch.cat((result[:, :1], result[:, :-1]), dim=1)
        result = torch.where(mask, previous, result)
    if torch.rand((), device=device) < 0.5:
        result = result + torch.randn_like(result) * 0.006
        result = F.normalize(result, dim=-1)
    return result


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_sum = correct = top5_correct = total = 0
    confusion = np.zeros((model.config.num_classes, model.config.num_classes), dtype=np.int64)
    for embeddings, targets in loader:
        targets_numpy = targets.numpy().copy()
        logits = model(embeddings.to(device))
        targets_device = targets.to(device)
        loss = F.cross_entropy(logits, targets_device)
        predictions = logits.argmax(1)
        top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
        if device.type == "mps":
            torch.mps.synchronize()
        predictions = predictions.cpu().numpy()
        top5 = top5.cpu().numpy()
        count = len(targets_numpy)
        loss_sum += float(loss.cpu()) * count
        correct += int((predictions == targets_numpy).sum())
        top5_correct += int((top5 == targets_numpy[:, None]).any(1).sum())
        total += count
        np.add.at(confusion, (targets_numpy, predictions), 1)
    tp = np.diag(confusion).astype(float)
    recall = tp / np.maximum(confusion.sum(1), 1)
    precision = tp / np.maximum(confusion.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "loss": loss_sum / max(total, 1),
        "top1": 100.0 * correct / max(total, 1),
        "top5": 100.0 * top5_correct / max(total, 1),
        "macro_f1": 100.0 * float(f1.mean()),
        "samples": float(total),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    expected_schema = schema_fingerprint(MobileCLIP2V17Config())
    train_dataset = MobileCLIP2CitizenDataset(
        args.data_root, "train", args.manifest, args.rejections, cache=not args.no_cache,
        expected_schema=expected_schema,
    )
    val_dataset = MobileCLIP2CitizenDataset(
        args.data_root, "val", args.manifest, args.rejections, cache=not args.no_cache,
        expected_schema=expected_schema,
    )
    if train_dataset.label_to_index != val_dataset.label_to_index:
        raise ValueError("train and validation class maps differ")
    dim, depth, epochs = args.dim, args.depth, args.epochs
    train_data, val_data = train_dataset, val_dataset
    if args.smoke:
        train_data = train_dataset.balanced_subset(2)
        val_data = val_dataset.balanced_subset(1)
        dim, depth, epochs = 64, 1, 1
        args.max_train_batches = 2
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    config = MobileCLIP2Stage1Config(
        num_classes=train_dataset.num_classes,
        dim=dim,
        depth=depth,
        heads=args.heads if dim % args.heads == 0 else 4,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        drop_path=args.drop_path,
    )
    model = MobileCLIP2Stage1V17(config).to(device)
    LOG.info("device=%s train=%d val=%d parameters=%d", device, len(train_data), len(val_data), model.parameter_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(args.warmup_epochs, epochs))

    def lr_scale(epoch_index: int) -> float:
        if epoch_index < warmup:
            return (epoch_index + 1) / warmup
        progress = (epoch_index - warmup) / max(epochs - warmup, 1)
        return args.minimum_lr_ratio + (1 - args.minimum_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    ema = ExponentialMovingAverage(model, args.ema_decay)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_top1 = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = seen = 0
        started = time.monotonic()
        for batch_index, (embeddings, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            embeddings = augment_embeddings(embeddings.to(device))
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                model(embeddings), targets, label_smoothing=args.label_smoothing
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            ema.update(model)
            total_loss += float(loss.detach()) * targets.numel()
            seen += targets.numel()
        live = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(ema.shadow)
        metrics = evaluate(model, val_loader, device)
        model.load_state_dict(live)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        LOG.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f top1=%.2f top5=%.2f macro_f1=%.2f seconds=%.1f",
            epoch, row["train_loss"], row["loss"], row["top1"], row["top5"], row["macro_f1"], row["seconds"],
        )
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            stale = 0
            state = {key: value.detach().cpu().clone() for key, value in ema.shadow.items()}
            checkpoint = make_checkpoint(
                model, state, epoch=epoch, validation_metrics=metrics,
                label_to_index=train_dataset.label_to_index,
                manifest_sha256=sha256_file(args.manifest),
                schema_fingerprint=expected_schema,
            )
            temporary = output / "best_model.pth.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best_model.pth")
        else:
            stale += 1
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= args.patience:
            LOG.info("early stopping after %d stale epochs", stale)
            break
    result = {
        "best_validation_top1": best_top1,
        "epochs_completed": len(history),
        "parameters": model.parameter_count,
        "image_encoder_parameters": 11_406_976,
        "device": str(device),
        "schema_fingerprint": expected_schema,
        "test_evaluated": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/mobileclip2_s0"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17_mobileclip2_s0"))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--drop-path", type=float, default=0.08)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(train(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
