"""
Data cleaning using Confident Learning.
Uses a trained model to find mislabeled samples in the dataset.

Usage:
    python scripts/clean_labels.py \
        --checkpoint /workspace/output_v2/best_model.pth \
        --data_path ASL_landmarks_rtmlib \
        --output cleaned_manifest.json \
        --remove_pct 3.0

What it does:
1. Loads the trained model
2. Runs inference on ALL training samples (with no augmentation)
3. Identifies samples where the model confidently disagrees with the label
4. Outputs a cleaned manifest with suspicious samples removed
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

# Fake mediapipe
import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from train_stage_1 import SLTStage1, DSGCNEncoder, ClassifierHead, compute_bone_features


def load_model(checkpoint_path, device):
    """Load trained Stage 1 model from checkpoint. Supports v12/v14/v15."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt['num_classes']
    d_model = ckpt.get('d_model', 384)
    model_type = ckpt.get('model_type', '')

    if 'v14' in model_type or 'v15' in model_type:
        from model_v14 import SLTStage1V14
        model = SLTStage1V14(
            num_classes=num_classes, d_model=d_model,
            use_arcface=True,
        ).to(device)
    else:
        nhead = ckpt.get('nhead', 8)
        num_layers = ckpt.get('num_transformer_layers', 6)
        model = SLTStage1(
            num_classes=num_classes, d_model=d_model,
            nhead=nhead, num_transformer_layers=num_layers,
        ).to(device)

    # Try EMA weights first (better quality)
    if ckpt.get('ema_shadow'):
        state_dict = ckpt['ema_shadow']
    else:
        state_dict = ckpt['model_state_dict']

    # Strip compiled prefix if present
    clean_sd = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '')
        clean_sd[k] = v

    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    label_to_idx = ckpt['label_to_idx']
    idx_to_label = ckpt.get('idx_to_label', {v: k for k, v in label_to_idx.items()})

    return model, label_to_idx, idx_to_label


def get_predictions(model, data_path, manifest, label_to_idx, device, batch_size=256):
    """Run model on all samples, return predictions and confidences."""
    print("Loading all samples...")
    filenames = []
    labels = []
    features = []

    for fname, label in manifest.items():
        if label not in label_to_idx:
            continue
        fpath = os.path.join(data_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            arr = np.load(fpath).astype(np.float32)
            if arr.shape not in [(32, 47, 10), (32, 61, 10)]:
                continue
            filenames.append(fname)
            labels.append(label_to_idx[label])
            features.append(arr)
        except:
            continue

    print(f"Loaded {len(features)} samples")

    # Compute bone features
    print("Computing bone features...")
    all_data = np.stack(features)  # [N, 32, 47, 10]
    all_data_torch = torch.from_numpy(all_data).float()
    all_data_16ch = compute_bone_features(all_data_torch)  # [N, 32, 47, 16]

    # Run inference in batches
    print("Running inference...")
    all_probs = []
    all_preds = []
    all_confs = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            batch = all_data_16ch[start:end].to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())

            if (start // batch_size) % 20 == 0:
                print(f"  [{start}/{len(features)}]")

    all_probs = np.concatenate(all_probs, axis=0)  # [N, num_classes]
    labels_arr = np.array(labels)
    preds_arr = np.array(all_preds)
    confs_arr = np.array(all_confs)

    return filenames, labels_arr, preds_arr, confs_arr, all_probs


def find_issues(filenames, labels, preds, confs, probs, manifest, label_to_idx, idx_to_label, remove_pct=3.0):
    """Find mislabeled samples using confident learning heuristics."""
    N = len(filenames)
    num_classes = probs.shape[1]

    # Method 1: High-confidence disagreement
    # Model is confident (>0.8) AND disagrees with label
    high_conf_disagree = (confs > 0.8) & (preds != labels)
    n_high_conf = high_conf_disagree.sum()
    print(f"\nHigh-confidence disagreements (conf>0.8): {n_high_conf} ({100*n_high_conf/N:.1f}%)")

    # Method 2: Self-confidence score (Northcutt et al.)
    # For each sample, get the model's probability for the given label
    self_conf = np.array([probs[i, labels[i]] for i in range(N)])
    # Low self-confidence = model doesn't think this label is right
    # Sort by ascending self-confidence (most suspicious first)
    suspicious_indices = np.argsort(self_conf)

    # Method 3: Cross-class confusion
    # Samples where P(predicted_class) >> P(given_label)
    margin = confs - self_conf
    high_margin = margin > 0.5  # predicted class much more likely than given label
    n_high_margin = high_margin.sum()
    print(f"High-margin disagreements (margin>0.5): {n_high_margin} ({100*n_high_margin/N:.1f}%)")

    # Combine: rank by self-confidence (lowest = most suspicious)
    n_remove = int(remove_pct / 100 * N)
    issues_to_remove = set(suspicious_indices[:n_remove])

    # Print examples
    print(f"\nTop 20 most suspicious samples:")
    print(f"{'Filename':60s} {'Given':20s} {'Predicted':20s} {'P(given)':>10s} {'P(pred)':>10s}")
    print("-" * 125)
    for i in suspicious_indices[:20]:
        given = idx_to_label.get(labels[i], idx_to_label.get(str(labels[i]), f"UNK_{labels[i]}"))
        predicted = idx_to_label.get(preds[i], idx_to_label.get(str(preds[i]), f"UNK_{preds[i]}"))
        print(f"{filenames[i]:60s} {given:20s} {predicted:20s} {self_conf[i]:10.4f} {confs[i]:10.4f}")

    # Class-level analysis
    print(f"\nClasses with most suspicious samples (in top {remove_pct}%):")
    issue_labels = Counter()
    for i in issues_to_remove:
        given = idx_to_label.get(labels[i], idx_to_label.get(str(labels[i]), f"UNK"))
        issue_labels[given] += 1
    for label, count in issue_labels.most_common(20):
        total_in_class = sum(1 for l in labels if idx_to_label.get(l, idx_to_label.get(str(l), "")) == label)
        print(f"  {label:20s}: {count}/{total_in_class} suspicious ({100*count/max(total_in_class,1):.1f}%)")

    return issues_to_remove, filenames, self_conf


def main():
    parser = argparse.ArgumentParser(description="Clean dataset labels using confident learning")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_path", default="ASL_landmarks_rtmlib", help="Path to .npy files")
    parser.add_argument("--output", default="ASL_landmarks_rtmlib/manifest_cleaned.json", help="Output cleaned manifest")
    parser.add_argument("--remove_pct", type=float, default=3.0, help="Percentage of most suspicious samples to remove")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, label_to_idx, idx_to_label = load_model(args.checkpoint, device)
    print(f"Model loaded: {len(label_to_idx)} classes")

    # Load manifest
    manifest_path = os.path.join(args.data_path, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} entries")

    # Get predictions
    filenames, labels, preds, confs, probs = get_predictions(
        model, args.data_path, manifest, label_to_idx, device, args.batch_size)

    # Overall accuracy on full dataset
    acc = (preds == labels).mean()
    print(f"\nModel accuracy on full dataset: {100*acc:.2f}%")

    # Find issues
    issues, filenames, self_conf = find_issues(
        filenames, labels, preds, confs, probs, manifest, label_to_idx, idx_to_label, args.remove_pct)

    # Create cleaned manifest
    cleaned_manifest = {}
    removed_count = 0
    for i, fname in enumerate(filenames):
        if i in issues:
            removed_count += 1
            continue
        cleaned_manifest[fname] = manifest[fname]

    # Add files that weren't in the prediction set (couldn't load, etc.)
    # These are kept as-is
    predicted_files = set(filenames)
    for fname, label in manifest.items():
        if fname not in predicted_files and fname not in cleaned_manifest:
            cleaned_manifest[fname] = label

    print(f"\n{'='*60}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Original manifest:  {len(manifest)} samples")
    print(f"Removed:            {removed_count} samples ({100*removed_count/len(manifest):.1f}%)")
    print(f"Cleaned manifest:   {len(cleaned_manifest)} samples")
    print(f"Classes:            {len(set(cleaned_manifest.values()))}")

    # Save
    with open(args.output, 'w') as f:
        json.dump(cleaned_manifest, f, indent=2)
    print(f"Saved to: {args.output}")

    # Also save the full analysis for review
    analysis_path = args.output.replace('.json', '_analysis.json')
    analysis = {
        'total_samples': len(filenames),
        'removed': removed_count,
        'remove_pct': args.remove_pct,
        'model_accuracy': float(acc),
        'suspicious_samples': [
            {
                'filename': filenames[i],
                'given_label': idx_to_label.get(labels[i], idx_to_label.get(str(labels[i]), f"UNK_{labels[i]}")),
                'predicted_label': idx_to_label.get(preds[i], idx_to_label.get(str(preds[i]), f"UNK_{preds[i]}")),
                'self_confidence': float(self_conf[i]),
                'pred_confidence': float(confs[i]),
            }
            for i in sorted(issues, key=lambda x: self_conf[x])[:100]
        ]
    }
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_path}")


if __name__ == "__main__":
    main()
