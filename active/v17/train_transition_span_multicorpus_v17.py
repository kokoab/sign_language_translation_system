#!/usr/bin/env python3
"""Train source-balanced transition timing on genuine continuous trajectories."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import copy
import gc
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_span_v17 import (
    TransitionSpanPredictorV17,
    TransitionSpanV17Config,
    endpoint_only_context,
    kinematic_span,
)
from active.v17.train_transition_inpainter_multicorpus_v17 import (
    manifest_signers,
    weighted_sampler,
)
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    discover_signers,
    landmark_tree_fingerprint,
    sha256,
)


LOG = logging.getLogger("train_transition_span_multicorpus_v17")


class TransitionSpanDataset(Dataset):
    """Expose all 4–12 frame elapsed spans without revealing the gap width."""

    def __init__(self, root: Path, signers: set[str]):
        self.base = TransitionWindowDataset(
            root, signers, seed=6701, fixed_masks=True
        )
        self.minimum_span = 4
        self.maximum_span = 12
        self.examples_per_window = self.maximum_span - self.minimum_span + 1

    def __len__(self) -> int:
        return len(self.base) * self.examples_per_window

    def __getitem__(self, index: int) -> dict[str, object]:
        base_index, offset = divmod(index, self.examples_per_window)
        row = self.base[base_index]
        features = row["features"]
        span = self.minimum_span + offset
        start = (features.shape[0] - span) // 2
        stop = start + span
        side = 8
        context = torch.cat((
            features[start - side:start], features[stop:stop + side]
        ), dim=0)
        if context.shape[0] != side * 2:
            raise RuntimeError("transition timing context is incomplete")
        return {
            "context": context,
            "target_class": span - self.minimum_span,
            "target_span": span,
            "signer": row["signer"],
            "item": row["item"],
        }


def classification_metrics(
    predicted: torch.Tensor, target: torch.Tensor, minimum_span: int = 4
) -> dict[str, float | int]:
    predicted = predicted.to(torch.long)
    target = target.to(torch.long)
    classes = int(max(predicted.max(), target.max()).item()) - minimum_span + 1
    f1 = []
    for span in range(minimum_span, minimum_span + classes):
        true_positive = ((predicted == span) & (target == span)).sum()
        false_positive = ((predicted == span) & (target != span)).sum()
        false_negative = ((predicted != span) & (target == span)).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        f1.append(float(2 * true_positive / denominator.clamp_min(1)))
    absolute = (predicted - target).abs()
    return {
        "examples": len(target),
        "accuracy": float((predicted == target).float().mean()),
        "mae_frames": float(absolute.float().mean()),
        "within_one_frame": float((absolute <= 1).float().mean()),
        "macro_f1": sum(f1) / len(f1),
    }


@torch.inference_mode()
def evaluate(
    model: TransitionSpanPredictorV17,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    targets = []
    learned = []
    endpoint_only = []
    kinematic = []
    for batch in loader:
        context = batch["context"].to(device)
        targets.append(batch["target_span"])
        learned.append(model(context).argmax(dim=1).cpu() + 4)
        endpoint_only.append(
            model(endpoint_only_context(context)).argmax(dim=1).cpu() + 4
        )
        kinematic.append(kinematic_span(context).cpu())
    target = torch.cat(targets)
    learned_predictions = torch.cat(learned)
    fixed_predictions = torch.full_like(target, 8)
    metrics = {
        "learned": classification_metrics(learned_predictions, target),
        "endpoint_only_ablation": classification_metrics(
            torch.cat(endpoint_only), target
        ),
        "kinematic_distance_over_speed": classification_metrics(
            torch.cat(kinematic), target
        ),
        "fixed_eight_frames": classification_metrics(fixed_predictions, target),
    }
    fixed_mae = metrics["fixed_eight_frames"]["mae_frames"]
    metrics["learned_relative_mae_improvement_vs_fixed"] = (
        fixed_mae - metrics["learned"]["mae_frames"]
    ) / fixed_mae
    metrics["learned_relative_mae_improvement_vs_kinematic"] = (
        metrics["kinematic_distance_over_speed"]["mae_frames"]
        - metrics["learned"]["mae_frames"]
    ) / metrics["kinematic_distance_over_speed"]["mae_frames"]
    metrics["style_context_mae_gain_vs_endpoint_only"] = (
        metrics["endpoint_only_ablation"]["mae_frames"]
        - metrics["learned"]["mae_frames"]
    )
    return metrics


def selection_score(how2sign: dict[str, object], web: dict[str, object]) -> float:
    return 0.5 * (
        float(how2sign["learned_relative_mae_improvement_vs_fixed"])
        + float(web["learned_relative_mae_improvement_vs_fixed"])
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    device_name = (
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)

    how2sign_all = {
        signer for signer in discover_signers(args.how2sign_root)
        if signer.startswith("how2sign:")
    }
    if args.held_out_how2sign not in how2sign_all:
        raise ValueError("held-out How2Sign signer is absent")
    how2sign_train = how2sign_all - {args.held_out_how2sign}
    manifest_train, manifest_validation = manifest_signers(args.web_manifest)
    available_web = discover_signers(args.web_root)
    web_train = manifest_train & available_web
    web_validation = manifest_validation & available_web
    if len(web_train) < 80 or len(web_validation) < 16:
        raise ValueError("web voice breadth floor failed")

    train_how2sign = TransitionSpanDataset(args.how2sign_root, how2sign_train)
    train_web = TransitionSpanDataset(args.web_root, web_train)
    validation_how2sign = TransitionSpanDataset(
        args.how2sign_root, {args.held_out_how2sign}
    )
    validation_web = TransitionSpanDataset(args.web_root, web_validation)
    combined = ConcatDataset((train_how2sign, train_web))
    sampler = weighted_sampler(
        len(train_how2sign), len(train_web), args.web_probability,
        args.samples_per_epoch or len(combined), args.seed,
    )
    train_loader = DataLoader(
        combined, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    validation_loaders = {
        "how2sign": DataLoader(
            validation_how2sign, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        ),
        "youtube_asl": DataLoader(
            validation_web, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        ),
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = TransitionSpanPredictorV17(TransitionSpanV17Config(
        dim=args.dim, depth=args.depth, heads=args.heads, dropout=args.dropout
    )).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    initial = {
        name: evaluate(model, loader, device)
        for name, loader in validation_loaders.items()
    }
    best_metrics = initial
    best_score = selection_score(initial["how2sign"], initial["youtube_asl"])
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = [{"epoch": 0, "validation": initial, "selection_score": best_score}]
    patience = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in train_loader:
            context = batch["context"].to(device)
            target = batch["target_class"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(context)
            loss = F.cross_entropy(
                logits, target, label_smoothing=args.label_smoothing
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite transition timing loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(context)
            seen += len(context)
        metrics = {
            name: evaluate(model, loader, device)
            for name, loader in validation_loaders.items()
        }
        score = selection_score(metrics["how2sign"], metrics["youtube_asl"])
        history.append({
            "epoch": epoch, "train_loss": total / seen,
            "validation": metrics, "selection_score": score,
        })
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "epoch=%d loss=%.5f h2s_mae=%.3f web_mae=%.3f score=%.4f best=%d",
            epoch, total / seen,
            metrics["how2sign"]["learned"]["mae_frames"],
            metrics["youtube_asl"]["learned"]["mae_frames"],
            score, best_epoch,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break

    how2sign_count, how2sign_hash = landmark_tree_fingerprint(args.how2sign_root)
    web_count, web_hash = landmark_tree_fingerprint(args.web_root)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_transition_span_predictor_v17",
        "version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": best_state,
        "seed": args.seed,
        "epoch": best_epoch,
        "held_out_signer": args.held_out_how2sign,
        "how2sign_train_signers": sorted(how2sign_train),
        "youtube_asl_train_voice_proxies": sorted(web_train),
        "youtube_asl_validation_voice_proxies": sorted(web_validation),
        "web_probability": args.web_probability,
        "validation_metrics": best_metrics,
        "how2sign_landmark_archive_count": how2sign_count,
        "how2sign_landmark_tree_sha256": how2sign_hash,
        "youtube_asl_landmark_archive_count": web_count,
        "youtube_asl_landmark_tree_sha256": web_hash,
        "youtube_asl_manifest_sha256": sha256(args.web_manifest),
        "test_evaluated": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    checkpoint_path = args.output / "best_model.pth"
    torch.save(checkpoint, checkpoint_path)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selected_epoch": best_epoch,
        "selection_score": best_score,
        "validation": best_metrics,
        "web_probability": args.web_probability,
        "how2sign_train_windows": len(train_how2sign) // 9,
        "youtube_asl_train_windows": len(train_web) // 9,
        "training_examples": len(combined),
        "seconds": time.monotonic() - started,
        "claim_boundary": (
            "self-supervised elapsed-span recovery on genuine trajectories is a "
            "timing proxy, not a human preference or semantic timing result"
        ),
        "test_evaluated": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--how2sign-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--web-root", type=Path,
        default=Path("data/local/youtube_asl_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--web-manifest", type=Path,
        default=Path("active/v17/youtube_asl_transition_manifest_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/transition_span_multicorpus_v17_h8"),
    )
    parser.add_argument("--held-out-how2sign", default="how2sign:8")
    parser.add_argument("--web-probability", type=float, default=0.10)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12701)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
