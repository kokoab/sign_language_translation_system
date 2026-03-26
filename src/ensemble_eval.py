"""
SLT Stage 1 — Multi-Stream Ensemble Evaluation & Weight Optimization
Loads checkpoints from multiple streams, optimizes fusion weights on val set,
evaluates on test set, and saves per-class accuracy + confusion matrix.
"""

import os, sys, json, argparse, itertools
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from train_stage_1 import (
    SLTStage1, AngleStreamModel, DSGCNEncoder, SignDataset,
    compute_bone_features, STREAM_CHANNELS,
)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("Ensemble")


def load_stream_model(stream_name, ckpt_path, device):
    """Load a trained stream model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    if stream_name in ('angle', 'angle_motion'):
        model = AngleStreamModel(
            num_classes=ckpt['num_classes'],
            in_features=43,
            d_model=ckpt.get('d_model', 192),
            nhead=ckpt.get('nhead', 8),
            num_transformer_layers=ckpt.get('num_transformer_layers', 4),
        )
    else:
        model = SLTStage1(
            num_classes=ckpt['num_classes'],
            in_channels=ckpt.get('in_channels', 16),
            d_model=ckpt.get('d_model', 384),
            nhead=ckpt.get('nhead', 8),
            num_transformer_layers=ckpt.get('num_transformer_layers', 6),
        )

    # Prefer EMA weights if available
    if ckpt.get('ema_shadow'):
        model.load_state_dict(ckpt['ema_shadow'], strict=False)
        log.info(f"  Loaded EMA weights for {stream_name}")
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

    model.to(device).eval()
    return model, ckpt


def transform_input(x, stream_name, geo_encoder=None):
    """Apply stream-specific input transform to a batch."""
    if stream_name in ('angle', 'angle_motion') and geo_encoder is not None:
        xyz = x[..., :3]
        face_mask = x[:, :, 42:47, 9:10]
        geo = geo_encoder._compute_geo_features(xyz, face_mask)
        angles = geo[..., 34:]
        mask_scalar = x[:, :, :42, 9].max(dim=2).values.unsqueeze(-1)
        out = torch.cat([angles, mask_scalar], dim=-1)
        if stream_name == 'angle_motion':
            motion = torch.zeros_like(out)
            motion[:, 1:-1] = (out[:, 2:] - out[:, :-2]) / 2.0
            motion[:, 0] = motion[:, 1]
            motion[:, -1] = motion[:, -2]
            out = motion
        return out
    elif stream_name in STREAM_CHANNELS and STREAM_CHANNELS[stream_name] is not None:
        return x[..., STREAM_CHANNELS[stream_name]]
    return x  # joint: full 16 channels


@torch.no_grad()
def collect_predictions(models, loader, device, geo_encoder):
    """Run all streams on a dataset, return per-stream softmax probs and targets."""
    all_probs = {name: [] for name in models}
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        all_targets.append(y.cpu())

        for name, model in models.items():
            inp = transform_input(x, name, geo_encoder)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(inp)
            all_probs[name].append(F.softmax(logits, dim=-1).cpu())

    all_probs = {k: torch.cat(v) for k, v in all_probs.items()}
    all_targets = torch.cat(all_targets)
    return all_probs, all_targets


def optimize_weights(all_probs, all_targets, stream_names):
    """Grid search for optimal ensemble weights on validation set."""
    weight_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    best_acc, best_weights = 0.0, {}

    for w in itertools.product(weight_options, repeat=len(stream_names)):
        weighted = sum(all_probs[n] * w[i] for i, n in enumerate(stream_names))
        preds = weighted.argmax(dim=1)
        acc = (preds == all_targets).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_weights = {n: w[i] for i, n in enumerate(stream_names)}

    return best_weights, best_acc


def evaluate_ensemble(all_probs, all_targets, weights, idx_to_label):
    """Evaluate ensemble with given weights. Returns metrics + per-class accuracy + confusion."""
    stream_names = list(weights.keys())
    total_weight = sum(weights.values())
    weighted = sum(all_probs[n] * weights[n] for n in stream_names) / total_weight

    preds = weighted.argmax(dim=1)
    _, top5 = weighted.topk(5, dim=1)

    top1_acc = (preds == all_targets).float().mean().item() * 100
    top5_acc = (top5 == all_targets.unsqueeze(1)).any(dim=1).float().mean().item() * 100

    # Per-class accuracy + confusion matrix
    per_class_correct, per_class_total = Counter(), Counter()
    confusion = {}
    for pred, target in zip(preds.tolist(), all_targets.tolist()):
        true_lbl = idx_to_label.get(str(target), f"UNK_{target}")
        pred_lbl = idx_to_label.get(str(pred), f"UNK_{pred}")
        per_class_total[true_lbl] += 1
        if pred == target:
            per_class_correct[true_lbl] += 1
        if true_lbl not in confusion:
            confusion[true_lbl] = {}
        confusion[true_lbl][pred_lbl] = confusion[true_lbl].get(pred_lbl, 0) + 1

    per_class_acc = {}
    for lbl in sorted(per_class_total.keys()):
        acc = per_class_correct[lbl] / max(per_class_total[lbl], 1) * 100
        per_class_acc[lbl] = {"correct": per_class_correct[lbl], "total": per_class_total[lbl], "acc": round(acc, 1)}

    return {
        "top1": round(top1_acc, 2),
        "top5": round(top5_acc, 2),
        "per_class_acc": per_class_acc,
        "confusion": confusion,
        "weights": weights,
    }


def main():
    parser = argparse.ArgumentParser(description="SLT Stage 1 Ensemble Evaluation")
    parser.add_argument('--streams', type=str, required=True,
                        help='Comma-separated stream:checkpoint pairs, e.g. joint:output/best_model.pth,bone:output_bone/best_model.pth')
    parser.add_argument('--data_dir', default=None, help='Path to ASL_landmarks_float16/')
    parser.add_argument('--manifest', default='manifest.json')
    parser.add_argument('--optimize_weights', action='store_true', help='Run grid search for optimal weights')
    parser.add_argument('--test', action='store_true', help='Evaluate on test set')
    parser.add_argument('--output', default='ensemble_results.json', help='Output file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse stream specifications
    stream_specs = {}
    for spec in args.streams.split(','):
        name, path = spec.strip().split(':')
        stream_specs[name] = path
    log.info(f"Loading {len(stream_specs)} streams: {list(stream_specs.keys())}")

    # Load all models
    models = {}
    idx_to_label = None
    num_classes = None
    for name, path in stream_specs.items():
        model, ckpt = load_stream_model(name, path, device)
        models[name] = model
        if idx_to_label is None:
            idx_to_label = ckpt.get('idx_to_label', {})
            num_classes = ckpt.get('num_classes', 311)
        log.info(f"  {name}: loaded from {path} (d_model={ckpt.get('d_model')}, acc={ckpt.get('best_acc', '?')}%)")

    # Geo encoder for angle streams
    geo_encoder = None
    if any(n in ('angle', 'angle_motion') for n in models):
        geo_encoder = DSGCNEncoder(in_channels=16, d_model=64, nhead=8, num_transformer_layers=1)
        geo_encoder.to(device).eval()

    # Load dataset
    data_dir = args.data_dir
    if data_dir is None:
        for p in ['/workspace/ASL_landmarks_float16', 'ASL_landmarks_float16']:
            if os.path.isdir(p): data_dir = p; break

    with open(args.manifest) as f:
        manifest = json.load(f)

    label_to_idx = {v: int(k) for k, v in manifest.get('idx_to_gloss', idx_to_label).items()}
    if not label_to_idx:
        label_to_idx = {v: int(k) for k, v in idx_to_label.items()}

    from torch.utils.data import DataLoader, Subset
    full_ds = SignDataset(data_path=data_dir, label_to_idx=label_to_idx, manifest=manifest,
                          cache_path=str(Path(data_dir).parent / 'ds_cache_ensemble.pt'))

    # Same split as training
    all_targets = full_ds.targets.numpy()
    indices = list(range(len(full_ds)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=all_targets)
    temp_targets = all_targets[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42, stratify=temp_targets)

    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Collect predictions on val set
    log.info("Collecting predictions on validation set...")
    val_probs, val_targets = collect_predictions(models, val_loader, device, geo_encoder)

    # Individual stream accuracy
    log.info("\n=== Individual Stream Accuracy (Val) ===")
    for name in models:
        preds = val_probs[name].argmax(dim=1)
        acc = (preds == val_targets).float().mean().item() * 100
        log.info(f"  {name:>15s}: {acc:.2f}%")

    # Optimize weights or use uniform
    stream_names = list(models.keys())
    if args.optimize_weights:
        log.info("\nOptimizing ensemble weights (grid search)...")
        weights, val_acc = optimize_weights(val_probs, val_targets, stream_names)
        log.info(f"Best val accuracy: {val_acc*100:.2f}%")
        log.info(f"Optimal weights: {weights}")
    else:
        weights = {n: 1.0 for n in stream_names}
        weighted = sum(val_probs[n] for n in stream_names) / len(stream_names)
        val_acc = (weighted.argmax(dim=1) == val_targets).float().mean().item()
        log.info(f"\nUniform ensemble val accuracy: {val_acc*100:.2f}%")

    # Test evaluation
    if args.test:
        log.info("\nCollecting predictions on test set...")
        test_probs, test_targets = collect_predictions(models, test_loader, device, geo_encoder)
        results = evaluate_ensemble(test_probs, test_targets, weights, idx_to_label)
        log.info(f"\n=== ENSEMBLE TEST RESULTS ===")
        log.info(f"Top-1: {results['top1']}% | Top-5: {results['top5']}%")

        # Per-class tier breakdown
        tiers = {'>90%': 0, '80-90%': 0, '70-80%': 0, '60-70%': 0, '<60%': 0}
        for cls, stats in results['per_class_acc'].items():
            a = stats['acc']
            if a >= 90: tiers['>90%'] += 1
            elif a >= 80: tiers['80-90%'] += 1
            elif a >= 70: tiers['70-80%'] += 1
            elif a >= 60: tiers['60-70%'] += 1
            else: tiers['<60%'] += 1
        log.info(f"Tier breakdown: {tiers}")

        # Bottom 15 classes
        sorted_acc = sorted(results['per_class_acc'].items(), key=lambda x: x[1]['acc'])
        log.info("\nBottom 15 classes:")
        for lbl, stats in sorted_acc[:15]:
            conf = results['confusion'].get(lbl, {})
            misses = sorted(((p, c) for p, c in conf.items() if p != lbl), key=lambda x: -x[1])[:3]
            miss_str = ", ".join(f"{p}({c})" for p, c in misses)
            log.info(f"  {lbl:>20s}: {stats['acc']:5.1f}% ({stats['correct']}/{stats['total']}) -> {miss_str}")

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
