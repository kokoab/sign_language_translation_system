"""
Post-training evaluation for paper charts.
Computes clean train/val/test metrics with model.eval() (no dropout, no augmentation, no ArcFace margin).

Usage:
    python scripts/eval_metrics.py \
        --checkpoint /workspace/output_v15/best_model.pth \
        --data_path /workspace/ASL_landmarks_apple_vision \
        --output /workspace/output_v15/eval_results.json

Outputs:
    - eval_results.json: train/val/test loss, accuracy, precision, recall, F1, top5, per-class accuracy
    - confusion_matrix.json: full NxN confusion matrix
    - per_class_accuracy.json: sorted class-level accuracy
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix as sk_confusion_matrix
)

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from train_stage_1 import SignDataset, compute_bone_features
from torch.utils.data import DataLoader, Subset


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt['num_classes']
    d_model = ckpt.get('d_model', 384)
    model_type = ckpt.get('model_type', '')

    if 'v14' in model_type or 'v15' in model_type:
        from model_v14 import SLTStage1V14
        model = SLTStage1V14(
            num_classes=num_classes, d_model=d_model, use_arcface=True,
        ).to(device)
    else:
        from train_stage_1 import SLTStage1
        model = SLTStage1(
            num_classes=num_classes, d_model=d_model,
        ).to(device)

    if ckpt.get('ema_shadow'):
        state_dict = ckpt['ema_shadow']
    else:
        state_dict = ckpt['model_state_dict']

    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    label_to_idx = ckpt['label_to_idx']
    idx_to_label = ckpt.get('idx_to_label', {str(v): k for k, v in label_to_idx.items()})

    return model, label_to_idx, idx_to_label, ckpt


def evaluate_split(model, loader, device, num_classes, augment_fn=None):
    """Evaluate a data split. Optionally applies augmentation for honest train metrics."""
    total_loss = 0.0
    correct_1, correct_5, total = 0, 0, 0
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            if augment_fn is not None:
                x = augment_fn(x)
            logits = model(x)  # no labels → no ArcFace margin
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * y.size(0)

            preds = logits.argmax(1)
            correct_1 += (preds == y).sum().item()
            _, top5 = logits.topk(5, dim=1)
            correct_5 += (top5 == y.unsqueeze(1)).any(1).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(y.cpu().tolist())

    prec = precision_score(all_true, all_preds, average='weighted', zero_division=0) * 100
    rec = recall_score(all_true, all_preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(all_true, all_preds, average='weighted', zero_division=0) * 100

    prec_macro = precision_score(all_true, all_preds, average='macro', zero_division=0) * 100
    rec_macro = recall_score(all_true, all_preds, average='macro', zero_division=0) * 100
    f1_macro = f1_score(all_true, all_preds, average='macro', zero_division=0) * 100

    return {
        'loss': round(total_loss / max(total, 1), 4),
        'accuracy': round(100 * correct_1 / max(total, 1), 2),
        'top5_accuracy': round(100 * correct_5 / max(total, 1), 2),
        'precision_weighted': round(prec, 2),
        'recall_weighted': round(rec, 2),
        'f1_weighted': round(f1, 2),
        'precision_macro': round(prec_macro, 2),
        'recall_macro': round(rec_macro, 2),
        'f1_macro': round(f1_macro, 2),
        'total_samples': total,
        'all_preds': all_preds,
        'all_true': all_true,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-training evaluation for paper charts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="ASL_landmarks_apple_vision")
    parser.add_argument("--output", default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output dir
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path(args.checkpoint).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model, label_to_idx, idx_to_label, ckpt = load_model(args.checkpoint, device)
    num_classes = len(label_to_idx)
    print(f"Model: {ckpt.get('model_type', 'unknown')}, {num_classes} classes, d_model={ckpt.get('d_model', '?')}")
    print(f"Best epoch: {ckpt.get('epoch', '?')}, Best val acc: {ckpt.get('best_acc', '?'):.2f}%")

    # Load dataset with manifest
    manifest_path = Path(args.data_path) / 'manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)

    cache_path = str(Path(args.data_path) / 'ds_cache_eval.pt')
    full_ds = SignDataset(args.data_path, label_to_idx, manifest=manifest, cache_path=cache_path)

    # Same split as training (random_state=42)
    all_targets = full_ds.targets.cpu().numpy()
    indices = list(range(len(full_ds)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=42,
                                            stratify=all_targets[indices])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=42,
                                          stratify=all_targets[temp_idx])

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    print(f"Split: Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    # Import augmentation for honest train eval
    from train_v15 import online_augment_v15

    # Evaluate all splits
    print("\nEvaluating train set (WITH augmentation — honest train metrics)...")
    train_m = evaluate_split(model, train_loader, device, num_classes, augment_fn=online_augment_v15)
    print(f"  Loss={train_m['loss']} | Acc={train_m['accuracy']}% | Top5={train_m['top5_accuracy']}% | F1={train_m['f1_weighted']}%")

    print("Evaluating train set (clean — no augmentation)...")
    train_clean_m = evaluate_split(model, train_loader, device, num_classes)
    print(f"  Loss={train_clean_m['loss']} | Acc={train_clean_m['accuracy']}% | Top5={train_clean_m['top5_accuracy']}% | F1={train_clean_m['f1_weighted']}%")

    print("Evaluating val set...")
    val_m = evaluate_split(model, val_loader, device, num_classes)
    print(f"  Loss={val_m['loss']} | Acc={val_m['accuracy']}% | Top5={val_m['top5_accuracy']}% | F1={val_m['f1_weighted']}%")

    print("Evaluating test set...")
    test_m = evaluate_split(model, test_loader, device, num_classes)
    print(f"  Loss={test_m['loss']} | Acc={test_m['accuracy']}% | Top5={test_m['top5_accuracy']}% | F1={test_m['f1_weighted']}%")

    # Per-class accuracy
    per_class = {}
    for i in range(num_classes):
        label = idx_to_label.get(str(i), idx_to_label.get(i, f"UNK_{i}"))
        true_mask = [t == i for t in test_m['all_true']]
        if sum(true_mask) == 0:
            continue
        correct = sum(1 for t, p in zip(test_m['all_true'], test_m['all_preds']) if t == i and p == i)
        total = sum(true_mask)
        per_class[label] = {
            'accuracy': round(100 * correct / total, 1),
            'correct': correct,
            'total': total,
        }

    sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['accuracy'])

    # Confusion matrix (test set)
    cm = sk_confusion_matrix(test_m['all_true'], test_m['all_preds'])
    cm_labels = [idx_to_label.get(str(i), idx_to_label.get(i, f"UNK_{i}")) for i in range(num_classes)]

    # Save results
    results = {
        'model_type': ckpt.get('model_type', 'unknown'),
        'best_epoch': ckpt.get('epoch', None),
        'd_model': ckpt.get('d_model', None),
        'num_classes': num_classes,
        'train_augmented': {k: v for k, v in train_m.items() if k not in ['all_preds', 'all_true']},
        'train_clean': {k: v for k, v in train_clean_m.items() if k not in ['all_preds', 'all_true']},
        'val': {k: v for k, v in val_m.items() if k not in ['all_preds', 'all_true']},
        'test': {k: v for k, v in test_m.items() if k not in ['all_preds', 'all_true']},
        'bottom_15_classes': [{'class': k, **v} for k, v in sorted_classes[:15]],
        'top_15_classes': [{'class': k, **v} for k, v in sorted_classes[-15:]],
    }

    with open(out_dir / 'eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'eval_results.json'}")

    # Save per-class accuracy
    with open(out_dir / 'per_class_accuracy.json', 'w') as f:
        json.dump(dict(sorted_classes), f, indent=2)
    print(f"Saved: {out_dir / 'per_class_accuracy.json'}")

    # Save confusion matrix
    cm_data = {
        'labels': cm_labels,
        'matrix': cm.tolist(),
    }
    with open(out_dir / 'confusion_matrix_full.json', 'w') as f:
        json.dump(cm_data, f, indent=2)
    print(f"Saved: {out_dir / 'confusion_matrix_full.json'}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY (model.eval, no ArcFace margin)")
    print(f"{'='*60}")
    print(f"{'':15s} {'Loss':>8s} {'Acc':>8s} {'Top5':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s}")
    for name, m in [('Train (aug)', train_m), ('Train (clean)', train_clean_m), ('Val', val_m), ('Test', test_m)]:
        print(f"{name:15s} {m['loss']:8.4f} {m['accuracy']:7.2f}% {m['top5_accuracy']:7.2f}% "
              f"{m['precision_weighted']:7.2f}% {m['recall_weighted']:7.2f}% {m['f1_weighted']:7.2f}%")

    print(f"\nBottom 10 classes (test):")
    for cls, stats in sorted_classes[:10]:
        print(f"  {cls:20s}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\nTop 10 classes (test):")
    for cls, stats in sorted_classes[-10:]:
        print(f"  {cls:20s}: {stats['accuracy']:5.1f}% ({stats['correct']}/{stats['total']})")


if __name__ == "__main__":
    main()
