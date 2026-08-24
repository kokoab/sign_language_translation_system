#!/usr/bin/env python3
"""Train the v17 sign-specialized MoViNet fusion model on CUDA or Apple Metal.

This runner uses the converted Kinetics-600 MoViNet-A0 streaming weights from
Atze00/MoViNet-pytorch.  The streaming 2+1D backbone is the faithful mobile form of
MoViNet and, unlike TensorFlow Model Garden's grouped Conv3D graph, can train on Apple
Metal.  The classifier keeps anatomical left/right/union streams, box trajectories,
missing-view masks, and the frozen Apple landmark logits as separate evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time

# MoViNet contains AvgPool3D in its squeeze/excitation path. PyTorch 2.8 executes
# that small operation on CPU while all convolutions remain on Metal.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from active.v17.movinet_data_v17 import (
        FRAMES,
        VIEWS,
        augment_sign_views,
        decode_crop_archive,
        load_aligned_records,
    )
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    from .movinet_data_v17 import (
        FRAMES,
        VIEWS,
        augment_sign_views,
        decode_crop_archive,
        load_aligned_records,
    )


LOG = logging.getLogger("stage1_movinet_torch_v17")
NUM_CLASSES = 100
LANDMARK_DIM = 256
BACKBONE_DIM = 480
PINNED_MOVINET_COMMIT = "c2d1edf48fc6c5259707f9d833f22171b4f63493"
PINNED_STREAM_WEIGHT_SHA256 = "447c0554daa6bebdcf6fc69b2651b25b29cc69e003da4e6ff56f9a2488f403cf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_subset(records, count: int):
    remaining = {target: count for target in range(NUM_CLASSES)}
    selected = []
    for record in records:
        if remaining[record.target] > 0:
            selected.append(record)
            remaining[record.target] -= 1
    if any(remaining.values()):
        raise ValueError("balanced subset exceeds available examples")
    return selected


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but PyTorch found no CUDA GPU")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but PyTorch found no Metal GPU")
    device = torch.device(requested)
    LOG.info("using training device: %s", device)
    return device


def verify_movinet_source(root: Path, weights: Path) -> str:
    root = root.resolve()
    if not (root / "movinets" / "models.py").is_file():
        raise FileNotFoundError(f"MoViNet-pytorch source is missing under {root}")
    if not weights.is_file():
        raise FileNotFoundError(f"pretrained MoViNet weights are missing: {weights}")
    weight_sha = sha256_file(weights)
    if weight_sha != PINNED_STREAM_WEIGHT_SHA256:
        raise RuntimeError(
            "pretrained streaming weight checksum mismatch: "
            f"expected {PINNED_STREAM_WEIGHT_SHA256}, got {weight_sha}"
        )
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_MOVINET_COMMIT:
        raise RuntimeError(
            f"MoViNet-pytorch source commit mismatch: expected {PINNED_MOVINET_COMMIT}, got {commit}"
        )
    return commit


def build_backbone(root: Path, weights: Path) -> nn.Module:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from movinets import MoViNet
    from movinets.config import _C

    backbone = MoViNet(
        _C.MODEL.MoViNetA0,
        causal=True,
        pretrained=False,
        num_classes=600,
        conv_type="2plus1d",
        tf_like=True,
    )
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch only.
        state = torch.load(weights, map_location="cpu")
    backbone.load_state_dict(state, strict=True)
    backbone.classifier = nn.Identity()
    return backbone


class SignMoViNetFusion(nn.Module):
    """Shared streaming MoViNet plus identity-preserving Apple residual fusion."""

    def __init__(self, backbone: nn.Module, dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.backbone = backbone
        self.view_embedding = nn.Parameter(torch.empty(VIEWS, 32))
        nn.init.normal_(self.view_embedding, std=0.02)
        self.box_projection = nn.Sequential(
            nn.Linear(FRAMES * 5, 96),
            nn.GELU(),
            nn.LayerNorm(96),
        )
        self.view_projection = nn.Sequential(
            nn.Linear(BACKBONE_DIM + 96 + 32, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )
        self.view_attention = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.visual_classifier = nn.Linear(dim, NUM_CLASSES)
        self.landmark_projection = nn.Sequential(nn.LayerNorm(LANDMARK_DIM), nn.Linear(LANDMARK_DIM, dim), nn.GELU())
        self.fusion_hidden = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Linear(dim * 4, 1)
        nn.init.constant_(self.fusion_gate.bias, -2.0)
        self.residual_classifier = nn.Linear(dim, NUM_CLASSES)
        nn.init.zeros_(self.residual_classifier.weight)
        nn.init.zeros_(self.residual_classifier.bias)

    def forward(self, pixels, valid, boxes, landmark_features, base_logits):
        batch = pixels.shape[0]
        # [B,T,V,H,W,C] -> [B*V,C,T,H,W], preserving time within each view.
        videos = pixels.permute(0, 2, 5, 1, 3, 4).reshape(
            batch * VIEWS, 3, FRAMES, pixels.shape[3], pixels.shape[4]
        )
        self.backbone.clean_activation_buffers()
        visual = self.backbone(videos).reshape(batch, VIEWS, BACKBONE_DIM)

        valid_by_view = valid.permute(0, 2, 1)
        stream_valid = valid_by_view.any(dim=-1)
        box_sequence = boxes.permute(0, 2, 1, 3) * valid_by_view.unsqueeze(-1)
        geometry = torch.cat(
            (box_sequence.reshape(batch, VIEWS, FRAMES * 4), valid_by_view.float()), dim=-1
        )
        geometry = self.box_projection(geometry)
        view_ids = self.view_embedding.unsqueeze(0).expand(batch, -1, -1)
        tokens = self.view_projection(torch.cat((visual, geometry, view_ids), dim=-1))
        attention = self.view_attention(tokens).squeeze(-1)
        attention = attention.masked_fill(~stream_valid, -1e4)
        weights = torch.softmax(attention, dim=1)
        visual_feature = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        visual_logits = self.visual_classifier(visual_feature)

        landmark = self.landmark_projection(landmark_features)
        cross_modal = torch.cat(
            (landmark, visual_feature, landmark * visual_feature, torch.abs(landmark - visual_feature)),
            dim=-1,
        )
        hidden = self.fusion_hidden(cross_modal)
        gate = torch.sigmoid(self.fusion_gate(cross_modal))
        fused_logits = base_logits + gate * self.residual_classifier(hidden)
        return {"fused_logits": fused_logits, "visual_logits": visual_logits, "gate": gate}


def iter_batches(records, resolution, batch_size, training, seed):
    rng = np.random.default_rng(seed)
    order = np.arange(len(records))
    if training:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunks = []
        for index in order[start:start + batch_size]:
            record = records[int(index)]
            pixels, valid, boxes = decode_crop_archive(record.crop_path, resolution)
            if training:
                pixels, valid, boxes = augment_sign_views(pixels, valid, boxes, rng)
            chunks.append((pixels, valid, boxes, record.landmark_feature, record.base_logits, record.target))
        yield (
            torch.from_numpy(np.stack([item[0] for item in chunks])).float().div_(255.0),
            torch.from_numpy(np.stack([item[1] for item in chunks])),
            torch.from_numpy(np.stack([item[2] for item in chunks])),
            torch.from_numpy(np.stack([item[3] for item in chunks])),
            torch.from_numpy(np.stack([item[4] for item in chunks])),
            torch.tensor([item[5] for item in chunks], dtype=torch.long),
        )


def to_device(batch, device):
    return tuple(value.to(device, non_blocking=False) for value in batch)


def confusion_metrics(targets, logits):
    targets = np.concatenate(targets).astype(np.int64)
    logits = np.concatenate(logits)
    predictions = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "top1": 100 * float((predictions == targets).mean()),
        "top5": 100 * float((top5 == targets[:, None]).any(axis=1).mean()),
        "macro_f1": 100 * float(f1.mean()),
        "samples": int(len(targets)),
    }


@torch.inference_mode()
def evaluate(model, records, args, device):
    model.eval()
    targets, fused_logits, visual_logits, gates = [], [], [], []
    for batch_index, batch in enumerate(
        iter_batches(records, args.resolution, args.batch_size, False, args.seed)
    ):
        if args.max_val_batches and batch_index >= args.max_val_batches:
            break
        pixels, valid, boxes, landmark, base, target = to_device(batch, device)
        output = model(pixels, valid, boxes, landmark, base)
        targets.append(target.cpu().numpy())
        fused_logits.append(output["fused_logits"].cpu().numpy())
        visual_logits.append(output["visual_logits"].cpu().numpy())
        gates.append(output["gate"].cpu().numpy())
    return {
        "fused": confusion_metrics(targets, fused_logits),
        "visual": confusion_metrics(targets, visual_logits),
        "mean_gate": float(np.concatenate(gates).mean()),
    }


def save_best(model, output, epoch, phase, metrics, args, source_commit):
    torch.save(model.state_dict(), output / "best.pt")
    metadata = {
        "format": "slt_stage1_sign_movinet_stream_fusion_v17",
        "epoch": epoch,
        "phase": phase,
        "validation_metrics": metrics,
        "movinet_source_commit": source_commit,
        "pretrained_stream_weight_sha256": PINNED_STREAM_WEIGHT_SHA256,
        "resolution": args.resolution,
        "frames": FRAMES,
        "views": ["left", "right", "union"],
        "backbone": "MoViNet-A0-stream-2plus1d",
        "warmup_fusion_frozen": True,
        "joint_head_lr": args.joint_head_lr,
        "backbone_batchnorm_frozen": not args.update_backbone_bn,
        "test_evaluated": False,
    }
    (output / "best_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run(args):
    if args.split_test:
        raise ValueError("the official Citizen test split is frozen and unavailable")
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_commit = verify_movinet_source(args.movinet_root, args.pretrained)
    train_records = load_aligned_records(args.crop_root, args.landmark_train, "train")
    val_records = load_aligned_records(args.crop_root, args.landmark_val, "val")
    if args.smoke:
        train_records = balanced_subset(train_records, 1)
        val_records = balanced_subset(val_records, 1)
        if not args.max_train_batches:
            args.max_train_batches = 1
        if not args.max_val_batches:
            args.max_val_batches = 1

    backbone = build_backbone(args.movinet_root, args.pretrained)
    model = SignMoViNetFusion(backbone, dim=args.dim, dropout=args.dropout).to(device)
    args.output.mkdir(parents=True, exist_ok=True)

    sample = to_device(
        next(iter(iter_batches(train_records[:1], args.resolution, 1, False, args.seed))), device
    )
    with torch.inference_mode():
        initial = model(*sample[:-1])
    if not torch.equal(initial["fused_logits"], sample[4]):
        raise RuntimeError("fusion must begin as the exact Apple baseline")

    initial_metrics = evaluate(model, val_records, args, device)
    best = initial_metrics["fused"]["top1"]
    history = []
    epoch_number = 0
    save_best(model, args.output, 0, "initial_apple_baseline", initial_metrics, args, source_commit)
    LOG.info(
        "initial Apple baseline fused=%.2f visual=%.2f samples=%d",
        initial_metrics["fused"]["top1"],
        initial_metrics["visual"]["top1"],
        initial_metrics["fused"]["samples"],
    )

    phases = [
        ("warmup", args.warmup_epochs, False, args.head_lr),
        ("joint_finetune", args.finetune_epochs, True, args.backbone_lr),
    ]
    if args.smoke:
        phases = [("smoke", 1, False, args.head_lr)]

    joint_stale = 0
    for phase_name, phase_epochs, train_backbone, learning_rate in phases:
        for parameter in model.parameters():
            parameter.requires_grad = True
        if not train_backbone:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
            # Warm up only the RGB evidence path. Keeping fusion untouched preserves
            # the useful initial gate and prevents the strong Apple logits from
            # teaching the gate to suppress RGB before RGB has learned the task.
            for module in (
                model.landmark_projection,
                model.fusion_hidden,
                model.fusion_gate,
                model.residual_classifier,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False

        if train_backbone:
            backbone_parameters = list(model.backbone.parameters())
            head_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("backbone.")
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_parameters, "lr": learning_rate},
                    {"params": head_parameters, "lr": args.joint_head_lr},
                ],
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=learning_rate,
                weight_decay=args.weight_decay,
            )
        for _ in range(phase_epochs):
            epoch_number += 1
            started = time.monotonic()
            model.train()
            if not train_backbone:
                model.backbone.eval()
            elif not args.update_backbone_bn:
                # Citizen100 is too small for reliable MoViNet BatchNorm running-stat
                # updates. Train convolution and affine parameters while preserving
                # the pretrained Kinetics population statistics.
                for module in model.backbone.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            losses = []
            batches = iter_batches(
                train_records, args.resolution, args.batch_size, True, args.seed + epoch_number
            )
            for batch_index, batch in enumerate(batches):
                if args.max_train_batches and batch_index >= args.max_train_batches:
                    break
                pixels, valid, boxes, landmark, base, targets = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(pixels, valid, boxes, landmark, base)
                fused_loss = F.cross_entropy(
                    output["fused_logits"], targets, label_smoothing=args.label_smoothing
                )
                visual_loss = F.cross_entropy(
                    output["visual_logits"], targets, label_smoothing=args.label_smoothing
                )
                loss = fused_loss + args.visual_loss_weight * visual_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
                )
                optimizer.step()
                losses.append((float(loss.detach()), float(fused_loss.detach()), float(visual_loss.detach())))
                if args.log_every and (batch_index + 1) % args.log_every == 0:
                    LOG.info(
                        "epoch=%d phase=%s batch=%d loss=%.4f",
                        epoch_number,
                        phase_name,
                        batch_index + 1,
                        float(np.mean(losses, axis=0)[0]),
                    )

            metrics = evaluate(model, val_records, args, device)
            means = np.mean(losses, axis=0)
            row = {
                "epoch": epoch_number,
                "phase": phase_name,
                "train_loss": float(means[0]),
                "train_fused_loss": float(means[1]),
                "train_visual_loss": float(means[2]),
                **metrics,
                "seconds": time.monotonic() - started,
            }
            history.append(row)
            (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            LOG.info(
                "epoch=%d phase=%s loss=%.4f fused=%.2f visual=%.2f gate=%.3f seconds=%.1f",
                epoch_number,
                phase_name,
                row["train_loss"],
                metrics["fused"]["top1"],
                metrics["visual"]["top1"],
                metrics["mean_gate"],
                row["seconds"],
            )

            score = metrics["fused"]["top1"]
            if score > best:
                best = score
                joint_stale = 0
                save_best(model, args.output, epoch_number, phase_name, metrics, args, source_commit)
            elif phase_name == "joint_finetune":
                joint_stale += 1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch_number,
                    "phase": phase_name,
                    "best_validation_top1": best,
                    "joint_stale": joint_stale,
                },
                args.output / "last.pt",
            )
            if phase_name == "warmup":
                torch.save(model.state_dict(), args.output / "warmup.pt")
            if not args.smoke and phase_name == "joint_finetune" and joint_stale >= args.patience:
                LOG.info("early stopping after %d stale joint epochs", joint_stale)
                break
        if not args.smoke and phase_name == "joint_finetune" and joint_stale >= args.patience:
            break

    try:
        best_state = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover
        best_state = torch.load(args.output / "best.pt", map_location=device)
    model.load_state_dict(best_state)
    final = evaluate(model, val_records, args, device)
    result = {
        "best_validation_top1": best,
        "best_checkpoint_metrics": final,
        "initial_validation_metrics": initial_metrics,
        "epochs_completed": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "backbone_parameters": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "device": str(device),
        "movinet_source_commit": source_commit,
        "pretrained_stream_weight_sha256": PINNED_STREAM_WEIGHT_SHA256,
        "warmup_fusion_frozen": True,
        "joint_head_lr": args.joint_head_lr,
        "backbone_batchnorm_frozen": not args.update_backbone_bn,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/local/citizen100_v17/hand_rgb"))
    parser.add_argument("--landmark-train", type=Path, default=Path("artifacts/generated/fusion_v17/landmark_train.npz"))
    parser.add_argument("--landmark-val", type=Path, default=Path("artifacts/generated/fusion_v17/landmark_val.npz"))
    parser.add_argument("--movinet-root", type=Path, default=Path("artifacts/model_assets/movinet/movinet_pytorch_v17"))
    parser.add_argument("--pretrained", type=Path, default=Path("artifacts/model_assets/movinet/movinet_pytorch_v17/weights/modelA0_stream_statedict_v3"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17_sign_movinet_stream_fusion"))
    parser.add_argument("--resolution", type=int, default=172)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--joint-head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--visual-loss-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument(
        "--update-backbone-bn",
        action="store_true",
        help="update pretrained MoViNet BatchNorm running statistics during joint training",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-batches", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--split-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
