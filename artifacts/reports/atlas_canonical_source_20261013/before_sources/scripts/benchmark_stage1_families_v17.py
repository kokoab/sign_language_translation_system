#!/usr/bin/env python3
"""Matched 100-gloss Stage-1 architecture-family benchmark.

This is an experiment-only runner. It does not alter the production v17 model.
Every family uses the same train/validation archives, balanced sampler, augmentation,
optimizer, schedule, label smoothing, EMA selection, seed, and validation protocol.
The protected test split is never loaded.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active.v17.model_v17 import (
    SLTStage1V17,
    Stage1V17Config,
    anatomical_adjacency_v17,
    masked_temporal_features,
)
from active.v17.train_stage_1_v17 import (
    Citizen100V17Dataset,
    ExponentialMovingAverage,
    SemLexSupplementV17Dataset,
    augment_v17,
    class_source_balanced_weights,
    dataset_targets_and_sources,
    extractor_schema_fingerprint,
)


OUTPUT = ROOT / "artifacts/reports/capstone1_v17_revision_checklist_v1/stage1_family_benchmark"
FAMILIES = (
    "bilstm",
    "bigru",
    "tcn",
    "transformer",
    "compact_transformer",
    "conv_transformer",
    "anatomical_token_transformer",
    "partwise_transformer",
    "stgcn",
    "squeezeformer",
)
DISPLAY_NAMES = {
    "bilstm": "BiLSTM",
    "bigru": "BiGRU",
    "tcn": "Temporal CNN",
    "transformer": "Flat Transformer",
    "compact_transformer": "Compact Transformer",
    "conv_transformer": "Conv-augmented Transformer",
    "anatomical_token_transformer": "Anatomical-token Transformer",
    "partwise_transformer": "Part-wise + global Transformer",
    "stgcn": "ST-GCN",
    "squeezeformer": "Part-wise + global Squeezeformer",
}


class FrameProjector(nn.Module):
    """Use the selected model's exact derived per-frame landmark representation."""

    def __init__(self, dim: int, dropout: float = 0.12):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(61 * 11 + 66, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def pairwise(features: torch.Tensor) -> torch.Tensor:
        xyz = features[..., :3]
        presence = features[..., 3]
        output = []
        for offset in (0, 21):
            for first, second in SLTStage1V17.HAND_PAIRS:
                valid = presence[:, :, offset + first] * presence[:, :, offset + second]
                distance = torch.linalg.vector_norm(
                    xyz[:, :, offset + first] - xyz[:, :, offset + second], dim=-1
                )
                output.append(distance * valid)
        return torch.stack(output, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        derived = masked_temporal_features(features).flatten(2)
        return self.projection(torch.cat((derived, self.pairwise(features)), dim=-1))


class AttentionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, max(32, input_dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, input_dim // 4), 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, num_classes),
        )

    def forward(self, value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        safe = active.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        scores = self.attention(value).squeeze(-1).masked_fill(
            ~safe, torch.finfo(value.dtype).min
        )
        pooled = (value * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
        pooled = pooled * (~empty).unsqueeze(-1).to(value.dtype)
        return self.classifier(pooled)


class RNNBaseline(nn.Module):
    def __init__(self, kind: str, num_classes: int):
        super().__init__()
        hidden = 384 if kind == "lstm" else 448
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.frame = FrameProjector(256)
        self.rnn = recurrent(
            256,
            hidden,
            num_layers=2,
            dropout=0.12,
            batch_first=True,
            bidirectional=True,
        )
        self.output = nn.Sequential(nn.Linear(hidden * 2, 256), nn.LayerNorm(256))
        self.head = AttentionClassifier(256, num_classes)
        self.config = SimpleNamespace(num_classes=num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value, _ = self.rnn(self.frame(features))
        active = features[:, :, :42, 3].amax(dim=-1) > 0.5
        return self.head(self.output(value), active)


class TemporalResidual(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation),
            nn.Dropout(0.12),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.network(value.transpose(1, 2)).transpose(1, 2)
        return self.norm(value + residual)


class TCNBaseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.frame = FrameProjector(384)
        self.blocks = nn.ModuleList(
            TemporalResidual(384, dilation) for dilation in (1, 2, 4, 8, 16, 32)
        )
        self.output = nn.Sequential(nn.Linear(384, 256), nn.LayerNorm(256))
        self.head = AttentionClassifier(256, num_classes)
        self.config = SimpleNamespace(num_classes=num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.frame(features)
        for block in self.blocks:
            value = block(value)
        active = features[:, :, :42, 3].amax(dim=-1) > 0.5
        return self.head(self.output(value), active)


def transformer_stack(dim: int, heads: int, depth: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=dim * 4,
        dropout=0.12,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)


class TemporalConvAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.network = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.GLU(dim=1),
            nn.Conv1d(256, 256, 7, padding=3, groups=256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Conv1d(256, 256, 1),
            nn.Dropout(0.12),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        adapted = self.network(self.norm(value).transpose(1, 2)).transpose(1, 2)
        return value + adapted


class TransformerBaseline(nn.Module):
    def __init__(self, num_classes: int, *, depth: int = 8, local_conv: bool = False):
        super().__init__()
        self.frame = FrameProjector(256)
        self.local_conv = TemporalConvAdapter() if local_conv else nn.Identity()
        self.position = nn.Parameter(torch.zeros(1, 32, 256))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.encoder = transformer_stack(256, 8, depth)
        self.head = AttentionClassifier(256, num_classes)
        self.config = SimpleNamespace(num_classes=num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.encoder(self.local_conv(self.frame(features)) + self.position)
        active = features[:, :, :42, 3].amax(dim=-1) > 0.5
        return self.head(value, active)


class AnatomicalTokenTransformerBaseline(nn.Module):
    PARTS = {
        "left_hand": (0, 21),
        "right_hand": (21, 42),
        "face": (42, 57),
        "body": (57, 61),
    }

    def __init__(self, num_classes: int):
        super().__init__()
        input_dims = {"left_hand": 264, "right_hand": 264, "face": 165, "body": 44}
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(input_dims[name], 128), nn.LayerNorm(128))
                for name in self.PARTS
            }
        )
        self.part_tokens = nn.Parameter(torch.zeros(1, 1, 4, 128))
        nn.init.trunc_normal_(self.part_tokens, std=0.02)
        self.spatial_encoder = transformer_stack(128, 4, 1)
        self.fusion = nn.Sequential(
            nn.Linear(128, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.12)
        )
        self.position = nn.Parameter(torch.zeros(1, 32, 256))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.encoder = transformer_stack(256, 8, 8)
        self.head = AttentionClassifier(256, num_classes)
        self.config = SimpleNamespace(num_classes=num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        derived = masked_temporal_features(features)
        pairwise = FrameProjector.pairwise(features)
        projected = []
        valid = []
        for name, (start, end) in self.PARTS.items():
            value = derived[:, :, start:end].flatten(2)
            if name in ("left_hand", "right_hand"):
                offset = 0 if name == "left_hand" else 33
                value = torch.cat((value, pairwise[..., offset : offset + 33]), dim=-1)
            projected.append(self.projections[name](value))
            valid.append(features[:, :, start:end, 3].amax(dim=-1) > 0.5)
        value = torch.stack(projected, dim=2) + self.part_tokens
        part_valid = torch.stack(valid, dim=2)
        batch, frames, parts, dim = value.shape
        safe_valid = part_valid.reshape(batch * frames, parts).clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[empty, 0] = True
        value = self.spatial_encoder(
            value.reshape(batch * frames, parts, dim),
            src_key_padding_mask=~safe_valid,
        )
        weights = part_valid.reshape(batch * frames, parts).to(value.dtype)
        value = (value * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        value = self.fusion(value.reshape(batch, frames, dim)) + self.position
        value = self.encoder(value)
        active = features[:, :, :42, 3].amax(dim=-1) > 0.5
        return self.head(value, active)


class PartWiseTransformerBaseline(nn.Module):
    PARTS = {
        "left_hand": (0, 21),
        "right_hand": (21, 42),
        "face": (42, 57),
        "body": (57, 61),
    }

    def __init__(self, num_classes: int):
        super().__init__()
        input_dims = {"left_hand": 264, "right_hand": 264, "face": 165, "body": 44}
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(input_dims[name], 64), nn.LayerNorm(64), nn.Dropout(0.12))
                for name in self.PARTS
            }
        )
        self.part_positions = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1, 32, 64)) for name in self.PARTS}
        )
        for position in self.part_positions.values():
            nn.init.trunc_normal_(position, std=0.02)
        self.part_encoders = nn.ModuleDict(
            {name: transformer_stack(64, 8, 1) for name in self.PARTS}
        )
        self.fusion = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.12)
        )
        self.position = nn.Parameter(torch.zeros(1, 32, 256))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.encoder = transformer_stack(256, 8, 8)
        self.head = AttentionClassifier(256, num_classes)
        self.config = SimpleNamespace(num_classes=num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        derived = masked_temporal_features(features)
        pairwise = FrameProjector.pairwise(features)
        outputs = []
        for name, (start, end) in self.PARTS.items():
            value = derived[:, :, start:end].flatten(2)
            if name in ("left_hand", "right_hand"):
                offset = 0 if name == "left_hand" else 33
                value = torch.cat((value, pairwise[..., offset : offset + 33]), dim=-1)
            value = self.projections[name](value) + self.part_positions[name]
            outputs.append(self.part_encoders[name](value))
        value = self.fusion(torch.cat(outputs, dim=-1)) + self.position
        value = self.encoder(value)
        active = features[:, :, :42, 3].amax(dim=-1) > 0.5
        return self.head(value, active)


class STGCNBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.graph = nn.Conv2d(input_dim, output_dim, 1)
        self.temporal = nn.Conv2d(output_dim, output_dim, (9, 1), padding=(4, 0))
        self.norm = nn.BatchNorm2d(output_dim)
        self.residual = (
            nn.Identity() if input_dim == output_dim else nn.Conv2d(input_dim, output_dim, 1)
        )
        self.dropout = nn.Dropout(0.12)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.residual(value)
        value = torch.einsum("bctn,nm->bctm", value, self.adjacency)
        value = self.temporal(self.graph(value))
        return self.dropout(F.gelu(self.norm(value) + residual))


class STGCNBaseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        adjacency = anatomical_adjacency_v17()
        # The original ST-GCN family uses a compact 64/128-channel schedule. Wider
        # channels make dense T×N temporal convolutions disproportionately slow on MPS.
        widths = (64, 64, 128, 128)
        blocks = []
        input_dim = 11
        for output_dim in widths:
            blocks.append(STGCNBlock(input_dim, output_dim, adjacency))
            input_dim = output_dim
        self.blocks = nn.ModuleList(blocks)
        self.output = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256))
        self.head = AttentionClassifier(256, num_classes)
        self.config = SimpleNamespace(num_classes=num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        presence = features[..., 3] > 0.5
        value = masked_temporal_features(features).permute(0, 3, 1, 2)
        for block in self.blocks:
            value = block(value)
        node_weights = presence.to(value.dtype).unsqueeze(1)
        value = (value * node_weights).sum(dim=-1) / node_weights.sum(dim=-1).clamp_min(1.0)
        value = self.output(value.transpose(1, 2))
        active = features[:, :, :42, 3].amax(dim=-1) > 0.5
        return self.head(value, active)


def refresh_recurrent_packing(model: nn.Module) -> None:
    """Refresh backend-packed RNN weights after loading a different state dict."""
    for module in model.modules():
        if isinstance(module, nn.RNNBase):
            module.flatten_parameters()


def build_model(family: str, num_classes: int) -> nn.Module:
    if family == "bilstm":
        return RNNBaseline("lstm", num_classes)
    if family == "bigru":
        return RNNBaseline("gru", num_classes)
    if family == "tcn":
        return TCNBaseline(num_classes)
    if family == "transformer":
        return TransformerBaseline(num_classes)
    if family == "compact_transformer":
        return TransformerBaseline(num_classes, depth=4)
    if family == "conv_transformer":
        return TransformerBaseline(num_classes, local_conv=True)
    if family == "anatomical_token_transformer":
        return AnatomicalTokenTransformerBaseline(num_classes)
    if family == "partwise_transformer":
        return PartWiseTransformerBaseline(num_classes)
    if family == "stgcn":
        return STGCNBaseline(num_classes)
    if family == "squeezeformer":
        return SLTStage1V17(
            Stage1V17Config(
                num_classes=num_classes,
                temporal_encoder="partwise_global",
                part_depth=1,
            )
        )
    raise ValueError(f"unknown family: {family}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    confusion = np.zeros((model.config.num_classes, model.config.num_classes), dtype=np.int64)
    loss_sum = total = top5_correct = 0
    for features, targets in loader:
        features = features.to(device)
        targets_device = targets.to(device)
        logits = model(features)
        loss_sum += float(F.cross_entropy(logits, targets_device).detach().cpu()) * len(targets)
        predictions = logits.argmax(dim=1).detach().cpu().numpy()
        top5 = logits.topk(5, dim=1).indices.detach().cpu().numpy()
        targets_numpy = targets.numpy()
        np.add.at(confusion, (targets_numpy, predictions), 1)
        top5_correct += int((top5 == targets_numpy[:, None]).any(axis=1).sum())
        total += len(targets)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "loss": loss_sum / total,
        "top1": 100.0 * float(true_positive.sum()) / total,
        "top5": 100.0 * top5_correct / total,
        "macro_f1": 100.0 * float(f1.mean()),
        "samples": total,
    }


def make_loaders(seed: int, batch_size: int):
    expected_schema = extractor_schema_fingerprint("apple")
    primary = Citizen100V17Dataset(
        ROOT / "data/local/citizen100_v17/landmarks",
        "train",
        ROOT / "active/v17/citizen100_manifest.json",
        ROOT / "data/local/citizen100_v17/rejections.csv",
        expected_schema=expected_schema,
    )
    validation = Citizen100V17Dataset(
        ROOT / "data/local/citizen100_v17/landmarks",
        "val",
        ROOT / "active/v17/citizen100_manifest.json",
        ROOT / "data/local/citizen100_v17/rejections.csv",
        expected_schema=expected_schema,
    )
    supplement = SemLexSupplementV17Dataset(
        ROOT / "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17",
        ROOT / "data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json",
        primary.label_to_index,
        expected_schema=expected_schema,
    )
    training = ConcatDataset((primary, supplement))
    targets, sources = dataset_targets_and_sources(training)
    weights, summary = class_source_balanced_weights(
        targets, sources, primary.num_classes, {"citizen": 0.5, "semlex": 0.5}
    )
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(training),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(training, batch_size=batch_size, sampler=sampler, num_workers=0),
        DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0),
        primary.num_classes,
        {"train_samples": len(training), "validation_samples": len(validation), "sampling": summary},
    )


def train_family(
    family: str,
    *,
    output_root: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_loader, validation_loader, num_classes, data_summary = make_loaders(seed, batch_size)
    model = build_model(family, num_classes).to(device)
    validation_model = build_model(family, num_classes).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.03)
    warmup = min(8, epochs)

    def schedule(epoch_index: int) -> float:
        if epoch_index < warmup:
            return float(epoch_index + 1) / warmup
        progress = (epoch_index - warmup) / max(epochs - warmup, 1)
        return 0.02 + 0.98 * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = ExponentialMovingAverage(model, 0.999)
    best_top1 = -1.0
    stale = 0
    best_state = None
    best_metrics = None
    best_epoch = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = seen = 0.0
        started = time.monotonic()
        for features, targets in train_loader:
            features = augment_v17(features.to(device))
            targets = targets.to(device)
            loss = F.cross_entropy(model(features), targets, label_smoothing=0.10)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            train_loss += float(loss.detach().cpu()) * len(targets)
            seen += len(targets)
        ema_state = {key: value.detach().cpu() for key, value in ema.shadow.items()}
        validation_model.load_state_dict(ema_state)
        refresh_recurrent_packing(validation_model)
        metrics = evaluate(validation_model, validation_loader, torch.device("cpu"))
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss / seen,
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        print(
            f"{family} epoch={epoch:03d} top1={metrics['top1']:.2f} "
            f"top5={metrics['top5']:.2f} loss={metrics['loss']:.4f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            best_metrics = metrics
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in ema_state.items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None or best_metrics is None or best_epoch is None:
        raise RuntimeError(f"{family} produced no checkpoint")
    family_root = output_root / family
    family_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"family": family, "state_dict": best_state, "num_classes": num_classes},
        family_root / "best_model.pth",
    )
    restored = load_baseline(family, num_classes, output_root)
    restored_metrics = evaluate(restored, validation_loader, torch.device("cpu"))
    if abs(restored_metrics["top1"] - best_metrics["top1"]) > 1e-9:
        raise RuntimeError(
            f"{family} checkpoint validation drift: selected={best_metrics['top1']:.8f}, "
            f"restored={restored_metrics['top1']:.8f}"
        )
    result = {
        "family": family,
        "display_name": DISPLAY_NAMES[family],
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": best_metrics,
        "data": data_summary,
        "history": history,
        "test_accessed": False,
    }
    (family_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    del model, validation_model, restored
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


@torch.no_grad()
def benchmark_latency(model: nn.Module, sample: torch.Tensor) -> dict[str, float]:
    model = model.cpu().eval()
    torch.set_num_threads(1)
    for _ in range(20):
        model(sample)
    timings = []
    for _ in range(300):
        started = time.perf_counter_ns()
        model(sample)
        timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {
        "median_ms": statistics.median(timings),
        "p90_ms": float(np.percentile(timings, 90)),
        "iterations": len(timings),
    }


def load_baseline(family: str, num_classes: int, output_root: Path) -> nn.Module:
    checkpoint = torch.load(
        output_root / family / "best_model.pth", map_location="cpu", weights_only=False
    )
    model = build_model(family, num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    refresh_recurrent_packing(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    output_root = OUTPUT / "smoke" if args.smoke else OUTPUT
    output_root.mkdir(parents=True, exist_ok=True)
    epochs = 2 if args.smoke else args.epochs
    patience = 2 if args.smoke else args.patience
    results = []
    if args.aggregate_only:
        for family in families:
            result_path = output_root / family / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"missing completed family result: {result_path}")
            result = json.loads(result_path.read_text())
            result["display_name"] = DISPLAY_NAMES[family]
            results.append(result)
    else:
        for family in families:
            results.append(
                train_family(
                    family,
                    output_root=output_root,
                    device=device,
                    epochs=epochs,
                    patience=patience,
                    batch_size=args.batch_size,
                    seed=args.seed,
                )
            )

    _, validation_loader, num_classes, _ = make_loaders(args.seed, args.batch_size)
    sample = next(iter(validation_loader))[0][:1]
    combined = []
    for result in results:
        model = load_baseline(result["family"], num_classes, output_root)
        result["latency"] = benchmark_latency(model, sample)
        combined.append(result)
    protocol = {
        "format": "slt_stage1_family_benchmark_v17",
        "training": "same 2,863-sample balanced 100-gloss training corpus",
        "validation": "same 378-clip signer-disjoint validation split",
        "maximum_epochs": epochs,
        "early_stopping_patience": patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "optimizer": "AdamW(lr=3e-4, weight_decay=0.03)",
        "schedule": "8-epoch warmup then cosine decay to 0.02x",
        "objective": "cross-entropy with label smoothing 0.10",
        "selection": "highest validation top-1 using EMA weights",
        "reported_validation_backend": "PyTorch CPU from the persisted selected checkpoint",
        "augmentation": "augment_v17 with identical defaults",
        "latency": "PyTorch CPU, one thread, batch one, 20 warmups, 300 iterations",
        "training_device": str(device),
        "host": platform.platform(),
        "torch_version": torch.__version__,
        "test_accessed": False,
    }
    output = {"protocol": protocol, "results": combined}
    target = output_root / "result.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"result": str(target), "families": len(combined)}, indent=2))


if __name__ == "__main__":
    main()
