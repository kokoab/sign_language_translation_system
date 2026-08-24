"""
Clean v16 dataset using Run 1 checkpoint (96.01% val, 5ch, d=256).
Runs locally on Mac. Outputs cleaned manifest_v16_files.json.

Uses self-confidence ranking (Northcutt et al.):
  - For each sample, get P(given_label) from model
  - Rank by ascending self-confidence
  - Remove bottom N% (most suspicious)

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE conda run -n sign_ai python scripts/clean_v16_local.py
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))
from model_v16 import SLTStage1V16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="src_v16/output_v16_d384/best_model.pth")
    parser.add_argument("--data_path", default="src_v16/ASL_landmarks_v16")
    parser.add_argument("--manifest", default="models/manifest_v16_files.json")
    parser.add_argument("--output", default="src_v16/manifest_v16_files_cleaned.json")
    parser.add_argument("--remove_pct", type=float, default=3.0, help="Bottom N%% by self-confidence")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"  Model: d={ckpt['d_model']}, in_ch={ckpt['in_channels']}, classes={ckpt['num_classes']}")
    print(f"  Val acc: {ckpt['best_acc']:.2f}%")

    model = SLTStage1V16(
        num_classes=ckpt['num_classes'],
        in_channels=ckpt['in_channels'],  # 5
        dim=ckpt['d_model'],              # 256
        use_pairwise=False,
        use_angles=False,
    ).to(device)

    # Load EMA weights if available, else model weights
    sd = ckpt.get('ema_shadow', ckpt['model_state_dict'])
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"  WARNING: missing keys: {missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys: {unexpected}")
    model.eval()

    label_to_idx = ckpt['label_to_idx']
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} files")

    # Load all samples (5ch only — model expects 5ch)
    print("Loading samples...")
    filenames = []
    labels = []
    arrays = []
    skipped = 0

    for fname, label in manifest.items():
        if label not in label_to_idx:
            skipped += 1
            continue
        fpath = os.path.join(args.data_path, fname)
        if not os.path.exists(fpath):
            skipped += 1
            continue
        try:
            arr = np.load(fpath).astype(np.float32)
            # Only use first 5 channels (model was trained on 5ch)
            if arr.shape[-1] > 5:
                arr = arr[..., :5]
            if arr.shape != (32, 61, 5):
                skipped += 1
                continue
            filenames.append(fname)
            labels.append(label_to_idx[label])
            arrays.append(arr)
        except:
            skipped += 1
            continue

        if len(filenames) % 10000 == 0:
            print(f"  Loaded {len(filenames)}...")

    print(f"Loaded: {len(filenames)} | Skipped: {skipped}")

    # Run inference in batches
    print("Running inference...")
    all_self_conf = []  # P(given_label) for each sample
    all_pred_conf = []  # max P for each sample
    all_preds = []      # predicted class for each sample

    with torch.no_grad():
        for start in range(0, len(arrays), args.batch_size):
            end = min(start + args.batch_size, len(arrays))
            batch = torch.from_numpy(np.stack(arrays[start:end])).to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)

            batch_labels = torch.tensor(labels[start:end], device=device)
            self_conf = probs.gather(1, batch_labels.unsqueeze(1)).squeeze(1)
            pred_conf, pred_cls = probs.max(dim=-1)

            all_self_conf.extend(self_conf.cpu().tolist())
            all_pred_conf.extend(pred_conf.cpu().tolist())
            all_preds.extend(pred_cls.cpu().tolist())

            if (start // args.batch_size) % 50 == 0:
                print(f"  [{start}/{len(arrays)}]")

    self_conf_arr = np.array(all_self_conf)
    pred_conf_arr = np.array(all_pred_conf)
    preds_arr = np.array(all_preds)
    labels_arr = np.array(labels)

    # Model accuracy on this data
    acc = (preds_arr == labels_arr).mean()
    print(f"\nModel accuracy on full dataset: {100*acc:.2f}%")

    # Rank by self-confidence (lowest = most suspicious)
    rank = np.argsort(self_conf_arr)
    n_remove = int(args.remove_pct / 100 * len(filenames))

    # Print top 30 most suspicious
    print(f"\nTop 30 most suspicious samples:")
    print(f"{'Filename':50s} {'Given':20s} {'Predicted':20s} {'P(given)':>10s} {'P(pred)':>10s}")
    print("-" * 115)
    for i in rank[:30]:
        given = idx_to_label[labels_arr[i]]
        predicted = idx_to_label[preds_arr[i]]
        print(f"{filenames[i]:50s} {given:20s} {predicted:20s} {self_conf_arr[i]:10.4f} {pred_conf_arr[i]:10.4f}")

    # Class breakdown of suspicious samples
    remove_set = set(rank[:n_remove].tolist())
    print(f"\nClasses with most removals (top 15):")
    class_removed = Counter()
    class_total = Counter()
    for i, fname in enumerate(filenames):
        label_name = idx_to_label[labels_arr[i]]
        class_total[label_name] += 1
        if i in remove_set:
            class_removed[label_name] += 1
    for label, count in class_removed.most_common(15):
        total = class_total[label]
        print(f"  {label:20s}: {count}/{total} removed ({100*count/total:.1f}%)")

    # Also count high-confidence disagreements
    high_conf_disagree = (pred_conf_arr > 0.8) & (preds_arr != labels_arr)
    print(f"\nHigh-confidence disagreements (conf>0.8): {high_conf_disagree.sum()} ({100*high_conf_disagree.sum()/len(filenames):.1f}%)")

    # Build cleaned manifest
    remove_filenames = set(filenames[i] for i in rank[:n_remove])
    cleaned = {fname: label for fname, label in manifest.items() if fname not in remove_filenames}

    print(f"\n{'='*60}")
    print(f"CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Original:  {len(manifest)} samples")
    print(f"Removed:   {n_remove} samples ({args.remove_pct}%)")
    print(f"Cleaned:   {len(cleaned)} samples")
    print(f"Classes:   {len(set(cleaned.values()))}")

    # Save
    with open(args.output, 'w') as f:
        json.dump(cleaned, f, indent=2)
    print(f"\nSaved to: {args.output}")

    # Also save analysis
    analysis_path = args.output.replace('.json', '_analysis.json')
    analysis = {
        'model_accuracy': float(acc),
        'total_samples': len(filenames),
        'removed': n_remove,
        'remove_pct': args.remove_pct,
        'high_conf_disagree': int(high_conf_disagree.sum()),
        'suspicious_samples': [
            {
                'filename': filenames[i],
                'given': idx_to_label[labels_arr[i]],
                'predicted': idx_to_label[preds_arr[i]],
                'p_given': float(self_conf_arr[i]),
                'p_pred': float(pred_conf_arr[i]),
            }
            for i in rank[:200]
        ]
    }
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis: {analysis_path}")


if __name__ == "__main__":
    main()
