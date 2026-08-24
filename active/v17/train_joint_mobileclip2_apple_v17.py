#!/usr/bin/env python3
"""Jointly train late MobileCLIP2 video features and Apple residual fusion.

Unlike the earlier frozen-feature diagnostic, gradients from the fused Apple/RGB loss
update MobileCLIP's late visual projection, the sign-specific temporal head, and the
fusion head in one optimizer step. Early FastViT spatial maps remain cached, so this is
joint late-visual co-training rather than full pixel-to-logit fine-tuning.
"""

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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_fusion_features_v17 import item_id
    from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
    from active.v17.model_hand_mobileclip2_v17 import (
        HandMobileCLIP2Stage1Config,
        HandMobileCLIP2Stage1V17,
    )
    from active.v17.train_feature_fusion_v17 import GatedFeatureResidual
    from active.v17.train_stage_1_hand_mobileclip2_v17 import supervised_contrastive
    from active.v17.train_stage_1_hand_spatial_mobileclip2_v17 import (
        HandSpatialDataset,
        SpatialTemporalMobileCLIP2V17,
        augment,
        evaluate as evaluate_visual,
    )
else:
    from .extract_fusion_features_v17 import item_id
    from .extract_mobileclip2_v17 import build_encoder, select_device
    from .model_hand_mobileclip2_v17 import (
        HandMobileCLIP2Stage1Config,
        HandMobileCLIP2Stage1V17,
    )
    from .train_feature_fusion_v17 import GatedFeatureResidual
    from .train_stage_1_hand_mobileclip2_v17 import supervised_contrastive
    from .train_stage_1_hand_spatial_mobileclip2_v17 import (
        HandSpatialDataset,
        SpatialTemporalMobileCLIP2V17,
        augment,
        evaluate as evaluate_visual,
    )


LOG = logging.getLogger("joint_mobileclip2_apple_v17")


class JointSpatialDataset(Dataset):
    def __init__(self, spatial_dataset: HandSpatialDataset, landmark_cache: Path):
        self.spatial = spatial_dataset
        with np.load(landmark_cache, allow_pickle=False) as payload:
            if str(payload["mode"]) != "landmark" or str(payload["split"]) != spatial_dataset.split:
                raise ValueError("landmark cache mode/split mismatch")
            lookup = {str(value): index for index, value in enumerate(payload["item_ids"])}
            features = payload["features"].astype(np.float32)
            logits = payload["logits"].astype(np.float32)
            targets = payload["targets"].astype(np.int64)
        spatial_ids = [item_id(path) for path in spatial_dataset.files]
        if set(spatial_ids) != set(lookup):
            raise ValueError("spatial and landmark item IDs differ")
        order = np.asarray([lookup[value] for value in spatial_ids])
        if not np.array_equal(spatial_dataset.targets.numpy(), targets[order]):
            raise ValueError("spatial and landmark targets differ")
        self.landmark_features = torch.from_numpy(features[order])
        self.base_logits = torch.from_numpy(logits[order])
        self.targets = spatial_dataset.targets
        self.files = spatial_dataset.files

    def __len__(self):
        return len(self.spatial)

    def __getitem__(self, index):
        maps, valid, boxes, target = self.spatial[index]
        return (
            maps,
            valid,
            boxes,
            self.landmark_features[index],
            self.base_logits[index],
            target,
        )


class JointMobileCLIP2AppleV17(nn.Module):
    def __init__(self, visual_model: SpatialTemporalMobileCLIP2V17):
        super().__init__()
        self.visual_model = visual_model
        self.fusion = GatedFeatureResidual(
            dim=visual_model.temporal_head.config.dim,
            classes=visual_model.temporal_head.config.num_classes,
        )

    def forward(self, maps, valid, boxes, landmark_features, base_logits):
        visual_features = self.visual_model.forward_features(maps, valid, boxes)
        visual_logits = self.visual_model.temporal_head.classifier(visual_features)
        fused_logits = self.fusion(landmark_features, visual_features, base_logits)
        return fused_logits, visual_logits, visual_features


def metrics_from_logits(logits, targets):
    predictions = logits.argmax(1)
    top5 = logits.topk(5, dim=1).indices
    classes = logits.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets.numpy(), predictions.numpy()), 1)
    true_positive = np.diag(confusion).astype(float)
    precision = true_positive / np.maximum(confusion.sum(0), 1)
    recall = true_positive / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "top1": 100 * float((predictions == targets).float().mean()),
        "top5": 100 * float((top5 == targets[:, None]).any(1).float().mean()),
        "macro_f1": 100 * float(f1.mean()),
        "samples": int(len(targets)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    fused_all = []
    visual_all = []
    base_all = []
    target_all = []
    for maps, valid, boxes, landmark, base, targets in loader:
        fused, visual, _ = model(
            maps.to(device).float(),
            valid.to(device),
            boxes.to(device),
            landmark.to(device),
            base.to(device),
        )
        if device.type == "mps":
            torch.mps.synchronize()
        fused_all.append(fused.cpu())
        visual_all.append(visual.cpu())
        base_all.append(base)
        target_all.append(targets)
    targets = torch.cat(target_all)
    return {
        "fused": metrics_from_logits(torch.cat(fused_all), targets),
        "visual": metrics_from_logits(torch.cat(visual_all), targets),
        "apple": metrics_from_logits(torch.cat(base_all), targets),
    }


def make_visual_model(device, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_hand_spatial_mobileclip2_v17":
        raise ValueError("spatial MobileCLIP checkpoint format mismatch")
    clip, _ = build_encoder(device)
    temporal = HandMobileCLIP2Stage1V17(
        HandMobileCLIP2Stage1Config(**checkpoint["model_config"])
    )
    model = SpatialTemporalMobileCLIP2V17(
        clip.visual.trunk.final_conv,
        clip.visual.trunk.head,
        temporal,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    train_spatial = HandSpatialDataset(
        args.data_root, "train", args.manifest, args.rejections
    )
    val_spatial = HandSpatialDataset(
        args.data_root, "val", args.manifest, args.rejections
    )
    train_set = JointSpatialDataset(train_spatial, args.landmark_train)
    val_set = JointSpatialDataset(val_spatial, args.landmark_val)
    if args.smoke:
        train_indices = train_spatial.balanced_subset(1).indices
        val_indices = val_spatial.balanced_subset(1).indices
        train_set = torch.utils.data.Subset(train_set, train_indices)
        val_set = torch.utils.data.Subset(val_set, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    model = JointMobileCLIP2AppleV17(
        make_visual_model(device, args.spatial_checkpoint)
    ).to(device)
    first = next(iter(val_loader))
    with torch.no_grad():
        fused, _, _ = model(
            first[0].to(device).float(),
            first[1].to(device),
            first[2].to(device),
            first[3].to(device),
            first[4].to(device),
        )
    if not torch.equal(fused.cpu(), first[4]):
        raise RuntimeError("joint fusion must start as the exact Apple baseline")

    visual_parameters = list(model.visual_model.final_conv.parameters()) + list(
        model.visual_model.visual_head.parameters()
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": visual_parameters, "lr": args.visual_lr},
            {"params": model.visual_model.temporal_head.parameters(), "lr": args.head_lr},
            {"params": model.fusion.parameters(), "lr": args.fusion_lr},
        ],
        weight_decay=args.weight_decay,
    )
    epochs = 1 if args.smoke else args.epochs
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        started = time.monotonic()
        for batch_index, (maps, valid, boxes, landmark, base, targets) in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            maps, valid, boxes = augment(
                maps.to(device).float(), valid.to(device), boxes.to(device)
            )
            landmark = landmark.to(device)
            base = base.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            fused, visual, visual_features = model(
                maps, valid, boxes, landmark, base
            )
            loss = (
                F.cross_entropy(fused, targets, label_smoothing=args.label_smoothing)
                + args.visual_loss_weight
                * F.cross_entropy(visual, targets, label_smoothing=args.label_smoothing)
                + args.contrastive_weight
                * supervised_contrastive(visual_features, targets)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            seen += len(targets)

        result = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            **result,
            "seconds": time.monotonic() - started,
        }
        history.append(row)
        LOG.info(
            "epoch=%d loss=%.4f apple=%.2f visual=%.2f fused=%.2f seconds=%.1f",
            epoch,
            row["train_loss"],
            result["apple"]["top1"],
            result["visual"]["top1"],
            result["fused"]["top1"],
            row["seconds"],
        )
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if result["fused"]["top1"] > best:
            best = result["fused"]["top1"]
            stale = 0
            torch.save(
                {
                    "format": "slt_stage1_joint_mobileclip2_apple_v17",
                    "epoch": epoch,
                    "validation_metrics": result,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in model.state_dict().items()
                    },
                    "test_evaluated": False,
                    "scope": "joint late-visual co-training; early FastViT maps cached",
                },
                args.output / "best_model.pth",
            )
        else:
            stale += 1
        if stale >= args.patience:
            break

    result = {
        "best_validation_top1": best,
        "epochs_completed": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/local/citizen100_v17/hand_spatial_mobileclip2_s0"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--landmark-train", type=Path, default=Path("artifacts/generated/fusion_v17/landmark_train.npz"))
    parser.add_argument("--landmark-val", type=Path, default=Path("artifacts/generated/fusion_v17/landmark_val.npz"))
    parser.add_argument("--spatial-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_hand_mobileclip2_spatial/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17_joint_mobileclip2_apple"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--visual-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--fusion-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--visual-loss-weight", type=float, default=0.35)
    parser.add_argument("--contrastive-weight", type=float, default=0.03)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
