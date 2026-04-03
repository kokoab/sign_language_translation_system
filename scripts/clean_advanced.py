"""
Advanced data cleaning: coordinate outliers, jittery keypoints, distribution outliers.
Runs on all .npy files and flags/removes bad samples.

Usage:
    python scripts/clean_advanced.py \
        --data_path ASL_landmarks_rtmlib \
        --output ASL_landmarks_rtmlib/manifest_deep_cleaned.json \
        --z_threshold 3.0
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# Fake mediapipe
import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions


def compute_sample_stats(arr):
    """Compute quality metrics for a single [32, N, 10] sample (N=47 or 61)."""
    xyz = arr[:, :42, :3]  # hands only, XYZ (same indices for both formats)

    # 1. Hand motion — total movement across frames
    hand_motion = np.abs(np.diff(xyz, axis=0)).sum()

    # 2. Jitter — high frequency noise (second derivative magnitude)
    if arr.shape[0] > 2:
        accel = np.diff(np.diff(xyz, axis=0), axis=0)
        jitter = np.abs(accel).mean()
    else:
        jitter = 0.0

    # 3. Coordinate range — how spread out the landmarks are
    coord_range = xyz.max() - xyz.min()

    # 4. Mean position — centroid of all hand landmarks across all frames
    mean_pos = xyz.mean(axis=(0, 1))  # [3]

    # 5. Temporal consistency — average frame-to-frame distance
    if arr.shape[0] > 1:
        frame_diffs = np.sqrt(((xyz[1:] - xyz[:-1]) ** 2).sum(axis=-1))  # [T-1, 42]
        temporal_consistency = frame_diffs.mean()
    else:
        temporal_consistency = 0.0

    # 6. Hand symmetry — difference between left and right hand motion
    l_motion = np.abs(np.diff(xyz[:, :21, :], axis=0)).sum()
    r_motion = np.abs(np.diff(xyz[:, 21:42, :], axis=0)).sum()

    # 7. Feature vector for distribution comparison — flatten mean XYZ per joint
    feature_vec = xyz.mean(axis=0).flatten()  # [42*3] = [126]

    return {
        'hand_motion': float(hand_motion),
        'jitter': float(jitter),
        'coord_range': float(coord_range),
        'mean_pos': mean_pos.tolist(),
        'temporal_consistency': float(temporal_consistency),
        'l_motion': float(l_motion),
        'r_motion': float(r_motion),
        'feature_vec': feature_vec,
    }


def main():
    parser = argparse.ArgumentParser(description="Advanced data cleaning")
    parser.add_argument("--data_path", default="ASL_landmarks_rtmlib")
    parser.add_argument("--output", default="ASL_landmarks_rtmlib/manifest_deep_cleaned.json")
    parser.add_argument("--z_threshold", type=float, default=3.0, help="Z-score threshold for outlier detection")
    parser.add_argument("--jitter_pct", type=float, default=2.0, help="Percent of most jittery samples to remove")
    args = parser.parse_args()

    # Load manifest
    manifest_path = os.path.join(args.data_path, 'manifest.json')
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} entries")

    # Compute stats for all samples
    print("Computing per-sample quality metrics...")
    sample_stats = {}
    class_features = defaultdict(list)  # label -> list of feature vectors
    class_filenames = defaultdict(list)  # label -> list of filenames

    loaded = 0
    for fname, label in manifest.items():
        fpath = os.path.join(args.data_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            arr = np.load(fpath).astype(np.float32)
            if arr.shape not in [(32, 47, 10), (32, 61, 10)]:
                continue
        except:
            continue

        stats = compute_sample_stats(arr)
        sample_stats[fname] = stats
        class_features[label].append(stats['feature_vec'])
        class_filenames[label].append(fname)

        loaded += 1
        if loaded % 5000 == 0:
            print(f"  Processed {loaded}/{len(manifest)}...")

    print(f"Computed stats for {len(sample_stats)} samples")

    # === CLEANING PASS 1: Coordinate outliers (per-class z-score) ===
    print(f"\n=== Pass 1: Coordinate outliers (z-score > {args.z_threshold}) ===")
    coord_outliers = set()

    for label in sorted(class_features.keys()):
        vecs = np.array(class_features[label])  # [N, 126]
        fnames = class_filenames[label]

        if len(vecs) < 5:
            continue

        # Z-score per feature dimension
        mean = vecs.mean(axis=0)
        std = vecs.std(axis=0) + 1e-8
        z_scores = np.abs((vecs - mean) / std)  # [N, 126]

        # Max z-score across all dimensions for each sample
        max_z = z_scores.max(axis=1)  # [N]

        for i, fname in enumerate(fnames):
            if max_z[i] > args.z_threshold:
                coord_outliers.add(fname)

    print(f"  Found {len(coord_outliers)} coordinate outliers")

    # === CLEANING PASS 2: Jittery keypoints (high-frequency noise) ===
    print(f"\n=== Pass 2: Jittery samples (top {args.jitter_pct}%) ===")
    jitter_values = [(fname, stats['jitter']) for fname, stats in sample_stats.items()]
    jitter_values.sort(key=lambda x: -x[1])

    n_jittery = int(len(jitter_values) * args.jitter_pct / 100)
    jittery_samples = set(fname for fname, _ in jitter_values[:n_jittery])

    print(f"  Found {len(jittery_samples)} jittery samples")
    if jitter_values:
        print(f"  Jitter range: {jitter_values[-1][1]:.6f} (best) to {jitter_values[0][1]:.6f} (worst)")
        print(f"  Threshold: {jitter_values[n_jittery-1][1]:.6f}")

    # === CLEANING PASS 3: Distribution outliers (per-class Mahalanobis-like distance) ===
    print(f"\n=== Pass 3: Distribution outliers (per-class) ===")
    dist_outliers = set()

    for label in sorted(class_features.keys()):
        vecs = np.array(class_features[label])
        fnames = class_filenames[label]

        if len(vecs) < 10:
            continue

        # Use L2 distance from class centroid
        centroid = vecs.mean(axis=0)
        distances = np.sqrt(((vecs - centroid) ** 2).sum(axis=1))

        # Remove samples > 3 std from centroid
        mean_dist = distances.mean()
        std_dist = distances.std() + 1e-8

        for i, fname in enumerate(fnames):
            z = (distances[i] - mean_dist) / std_dist
            if z > args.z_threshold:
                dist_outliers.add(fname)

    print(f"  Found {len(dist_outliers)} distribution outliers")

    # === CLEANING PASS 4: Low motion samples ===
    print(f"\n=== Pass 4: Low motion samples ===")
    low_motion = set()
    motion_values = [(fname, stats['hand_motion']) for fname, stats in sample_stats.items()]
    motion_values.sort(key=lambda x: x[1])

    # Bottom 1% by motion
    n_low = max(1, int(len(motion_values) * 0.01))
    for fname, motion in motion_values[:n_low]:
        low_motion.add(fname)

    print(f"  Found {len(low_motion)} low-motion samples")

    # === Combine all flagged samples ===
    all_flagged = coord_outliers | jittery_samples | dist_outliers | low_motion

    # Count overlaps
    print(f"\n=== Overlap analysis ===")
    print(f"  Coordinate outliers only: {len(coord_outliers - jittery_samples - dist_outliers - low_motion)}")
    print(f"  Jittery only: {len(jittery_samples - coord_outliers - dist_outliers - low_motion)}")
    print(f"  Distribution outliers only: {len(dist_outliers - coord_outliers - jittery_samples - low_motion)}")
    print(f"  Low motion only: {len(low_motion - coord_outliers - jittery_samples - dist_outliers)}")
    print(f"  In multiple categories: {len(all_flagged) - len(coord_outliers - jittery_samples - dist_outliers - low_motion) - len(jittery_samples - coord_outliers - dist_outliers - low_motion) - len(dist_outliers - coord_outliers - jittery_samples - low_motion) - len(low_motion - coord_outliers - jittery_samples - dist_outliers)}")

    # Per-class impact
    print(f"\n=== Per-class impact (classes losing >5% samples) ===")
    class_impact = Counter()
    class_total = Counter()
    for fname in sample_stats:
        label = manifest.get(fname, "UNK")
        class_total[label] += 1
        if fname in all_flagged:
            class_impact[label] += 1

    for label in sorted(class_total.keys()):
        removed = class_impact.get(label, 0)
        total = class_total[label]
        pct = 100 * removed / max(total, 1)
        if pct > 5:
            print(f"  {label}: {removed}/{total} ({pct:.1f}%)")

    # Create cleaned manifest
    cleaned_manifest = {}
    removed_count = 0
    for fname, label in manifest.items():
        if fname in all_flagged:
            removed_count += 1
            continue
        cleaned_manifest[fname] = label

    print(f"\n{'='*60}")
    print(f"ADVANCED CLEANING SUMMARY")
    print(f"{'='*60}")
    print(f"Original:           {len(manifest)} samples")
    print(f"Coordinate outliers: {len(coord_outliers)}")
    print(f"Jittery samples:    {len(jittery_samples)}")
    print(f"Distribution outliers: {len(dist_outliers)}")
    print(f"Low motion:         {len(low_motion)}")
    print(f"Total flagged:      {len(all_flagged)} (unique)")
    print(f"Removed:            {removed_count} ({100*removed_count/len(manifest):.1f}%)")
    print(f"Cleaned:            {len(cleaned_manifest)} samples")
    print(f"Classes:            {len(set(cleaned_manifest.values()))}")

    with open(args.output, 'w') as f:
        json.dump(cleaned_manifest, f, indent=2)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
