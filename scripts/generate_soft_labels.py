"""
Generate soft labels from SAM ensemble for knowledge distillation.
Runs all 4 SAM stream models on the training data and averages their softmax outputs.

Usage:
    python scripts/generate_soft_labels.py \
        --joint /workspace/sam_joint/best_model.pth \
        --bone /workspace/sam_bone/best_model.pth \
        --joint_motion /workspace/sam_jm/best_model.pth \
        --bone_motion /workspace/sam_bm/best_model.pth \
        --data_path ASL_landmarks_sam27 \
        --output /workspace/sam_soft_labels.pt
"""
import os, sys, json, argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# Add SAM to path
for p in ['/workspace/SLT/SAM-SLR-v2/SL-GCN', 'SAM-SLR-v2/SL-GCN']:
    if os.path.isdir(p):
        sys.path.insert(0, p)
        break

SAM_BONE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (7, 8), (7, 9), (7, 11), (7, 13), (7, 15),
    (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (17, 19), (17, 21), (17, 23), (17, 25),
    (19, 20), (21, 22), (23, 24), (25, 26),
    (5, 7), (6, 17),
]
MAX_FRAME = 150


def load_sam_model(ckpt_path, device):
    from model.decouple_gcn_attn import Model as SLGCN
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt['num_classes']
    model = SLGCN(num_class=num_classes, num_point=27, num_person=1,
                  graph='graph.sign_27.Graph',
                  graph_args={'labeling_mode': 'spatial'},
                  groups=16, block_size=41, in_channels=3).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt.get('label_to_idx', {})


def prepare_input(joint_data, stream, window_size):
    """Prepare one sample for SAM inference."""
    is_vector = stream in ('bone', 'joint_motion', 'bone_motion')

    if stream == 'joint':
        data = joint_data.copy()
    elif stream == 'bone':
        data = np.zeros_like(joint_data)
        for v1, v2 in SAM_BONE_PAIRS:
            data[:, v2, :] = joint_data[:, v2, :] - joint_data[:, v1, :]
    elif stream == 'joint_motion':
        data = np.zeros_like(joint_data)
        data[:-1] = joint_data[1:] - joint_data[:-1]
    elif stream == 'bone_motion':
        bone = np.zeros_like(joint_data)
        for v1, v2 in SAM_BONE_PAIRS:
            bone[:, v2, :] = joint_data[:, v2, :] - joint_data[:, v1, :]
        data = np.zeros_like(bone)
        data[:-1] = bone[1:] - bone[:-1]

    # Pad
    T = data.shape[0]
    if T < MAX_FRAME:
        rest = MAX_FRAME - T
        num_reps = int(np.ceil(rest / max(T, 1)))
        pad = np.concatenate([data for _ in range(num_reps)], axis=0)[:rest]
        data = np.concatenate([data, pad], axis=0)
    else:
        data = data[:MAX_FRAME]

    # Format [3, T, 27, 1]
    data = data.transpose(2, 0, 1)
    data = np.expand_dims(data, axis=-1)

    # Crop to window
    if data.shape[1] > window_size:
        data = data[:, :window_size, :, :]

    # Normalize
    if is_vector:
        data[0, :, 0, :] -= data[0, :, 0, 0].mean()
        data[1, :, 0, :] -= data[1, :, 0, 0].mean()
    else:
        data[0, :, :, :] -= data[0, :, 0, 0].mean()
        data[1, :, :, :] -= data[1, :, 0, 0].mean()

    return data.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--joint', required=True)
    parser.add_argument('--bone', required=True)
    parser.add_argument('--joint_motion', default=None)
    parser.add_argument('--bone_motion', default=None)
    parser.add_argument('--data_path', default='ASL_landmarks_sam27')
    parser.add_argument('--output', default='/workspace/sam_soft_labels.pt')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load manifest
    with open(os.path.join(args.data_path, 'manifest.json')) as f:
        manifest = json.load(f)
    unique_labels = sorted(set(manifest.values()))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Classes: {num_classes}, Samples: {len(manifest)}")

    # Load available models (motion streams optional)
    streams = {
        'joint': {'path': args.joint, 'window': 120, 'weight': 1.0},
        'bone': {'path': args.bone, 'window': 100, 'weight': 0.9},
    }
    if args.joint_motion and os.path.exists(args.joint_motion):
        streams['joint_motion'] = {'path': args.joint_motion, 'window': 100, 'weight': 0.5}
    if args.bone_motion and os.path.exists(args.bone_motion):
        streams['bone_motion'] = {'path': args.bone_motion, 'window': 100, 'weight': 0.5}

    models = {}
    for name, cfg in streams.items():
        print(f"Loading {name}...")
        models[name], _ = load_sam_model(cfg['path'], device)

    # Process all samples
    filenames = sorted(manifest.keys())
    all_soft_labels = torch.zeros(len(filenames), num_classes)
    total_weight = sum(cfg['weight'] for cfg in streams.values())

    for stream_name, cfg in streams.items():
        model = models[stream_name]
        window = cfg['window']
        weight = cfg['weight']
        print(f"\nGenerating soft labels: {stream_name} (weight={weight})")

        for start in range(0, len(filenames), args.batch_size):
            end = min(start + args.batch_size, len(filenames))
            batch_data = []
            for i in range(start, end):
                fname = filenames[i]
                fpath = os.path.join(args.data_path, fname)
                if not os.path.exists(fpath):
                    batch_data.append(np.zeros((3, window, 27, 1), dtype=np.float32))
                    continue
                joint = np.load(fpath).astype(np.float32)
                inp = prepare_input(joint, stream_name, window)
                batch_data.append(inp)

            batch = torch.from_numpy(np.stack(batch_data)).float().to(device)
            with torch.no_grad():
                logits = model(batch, keep_prob=1.0)
                probs = F.softmax(logits, dim=-1).cpu()
            all_soft_labels[start:end] += probs * weight

            if (start // args.batch_size) % 50 == 0:
                print(f"  [{start}/{len(filenames)}]")

    # Normalize by total weight
    all_soft_labels /= total_weight

    # Compute sample weights from SAM confidence
    hard_labels = torch.tensor([label_to_idx.get(manifest[f], 0) for f in filenames])
    hard_preds = all_soft_labels.argmax(dim=1)
    # Confidence = probability assigned to the TRUE label
    true_label_conf = all_soft_labels.gather(1, hard_labels.unsqueeze(1)).squeeze(1)  # [N]
    # Sample weights: confident + correct = 1.5, uncertain = 1.0, confident + wrong = 0.3 (likely mislabeled)
    correct_mask = (hard_preds == hard_labels).float()
    sample_weights = torch.where(
        correct_mask.bool(),
        0.8 + 0.7 * true_label_conf,   # correct: 0.8-1.5 (higher confidence = higher weight)
        torch.clamp(0.5 - 0.2 * (1.0 - true_label_conf), min=0.1),  # wrong: 0.1-0.5 (downweight)
    )
    # Flag likely mislabeled: SAM confident but predicts different class
    max_conf = all_soft_labels.max(dim=1).values
    mislabel_mask = (~correct_mask.bool()) & (max_conf > 0.8)
    sample_weights[mislabel_mask] = 0.05  # nearly zero-out likely mislabeled

    n_mislabeled = mislabel_mask.sum().item()
    n_correct = correct_mask.sum().item()
    print(f"\nSample weighting: {n_correct:.0f} correct, {n_mislabeled:.0f} likely mislabeled (weight=0.05)")

    # Curriculum difficulty: entropy of soft label distribution (high entropy = hard)
    entropy = -(all_soft_labels * (all_soft_labels + 1e-8).log()).sum(dim=1)  # [N]
    # Normalize to [0, 1] range
    difficulty = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-8)

    # Save
    fname_to_idx = {fname: i for i, fname in enumerate(filenames)}
    torch.save({
        'soft_labels': all_soft_labels,  # [N, num_classes]
        'sample_weights': sample_weights,  # [N] per-sample importance weight
        'difficulty': difficulty,  # [N] curriculum difficulty (0=easy, 1=hard)
        'filenames': filenames,
        'fname_to_idx': fname_to_idx,
        'label_to_idx': label_to_idx,
        'num_classes': num_classes,
    }, args.output)
    print(f"\nSaved soft labels: {all_soft_labels.shape} to {args.output}")

    # Quick sanity check
    acc = correct_mask.mean().item() * 100
    print(f"SAM ensemble accuracy on full dataset: {acc:.2f}%")
    print(f"Mean sample weight: {sample_weights.mean():.3f} (min={sample_weights.min():.3f}, max={sample_weights.max():.3f})")
    print(f"Mean difficulty: {difficulty.mean():.3f}")


if __name__ == "__main__":
    main()
