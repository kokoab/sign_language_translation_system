"""
Comprehensive quality audit of extracted .npy files.
Checks for: jitter, hand swaps, frozen frames, bone length violations,
anatomical issues, duplicates, outliers, and motion range.

Usage:
    python3 src/verify_extraction_quality.py
    python3 src/verify_extraction_quality.py --samples 500
"""

import numpy as np
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from pathlib import Path



# Hand skeleton: (parent, child) pairs for bone length checks
# MediaPipe hand landmark indices within each 21-point hand
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


def check_bone_consistency(xyz, hand_offset, mask_frames):
    """Check if bone lengths are consistent across frames.
    Returns coefficient of variation (std/mean) — should be <0.15 for stable detection."""
    if len(mask_frames) < 3:
        return 0.0, []

    bone_cvs = []
    bad_bones = []
    for parent, child in HAND_BONES:
        p_idx = hand_offset + parent
        c_idx = hand_offset + child
        lengths = []
        for t in mask_frames:
            bone_len = np.linalg.norm(xyz[t, p_idx] - xyz[t, c_idx])
            if bone_len > 1e-6:
                lengths.append(bone_len)
        if len(lengths) >= 3:
            cv = np.std(lengths) / (np.mean(lengths) + 1e-8)
            bone_cvs.append(cv)
            if cv > 0.3:
                bad_bones.append((parent, child, round(cv, 3)))

    avg_cv = np.mean(bone_cvs) if bone_cvs else 0.0
    return avg_cv, bad_bones


def check_anatomical(xyz, hand_offset, t):
    """Check for anatomical violations at a single frame.
    Returns list of violations."""
    violations = []

    # Check: fingertip shouldn't be closer to wrist than its MCP joint
    wrist = xyz[t, hand_offset]
    for finger_start, tip_idx in [(5, 8), (9, 12), (13, 16), (17, 20)]:
        mcp = xyz[t, hand_offset + finger_start]
        tip = xyz[t, hand_offset + tip_idx]
        wrist_to_mcp = np.linalg.norm(mcp - wrist)
        wrist_to_tip = np.linalg.norm(tip - wrist)
        if wrist_to_mcp > 1e-6 and wrist_to_tip < wrist_to_mcp * 0.3:
            violations.append(f"finger_{tip_idx}_collapsed")

    return violations


def audit_sample(filepath):
    """Comprehensive audit of a single .npy file."""
    arr = np.load(filepath)  # [32, 47, 10]
    issues = []
    metrics = {}

    xyz = arr[:, :, :3]
    vel = arr[:, :, 3:6]
    mask = arr[:, :, 9]
    T = arr.shape[0]

    # === 1. Mask coverage ===
    l_hand_mask = mask[:, 0:21].max(axis=1)
    r_hand_mask = mask[:, 21:42].max(axis=1)
    best_coverage = max(l_hand_mask.mean(), r_hand_mask.mean())
    metrics['coverage'] = round(float(best_coverage), 3)

    if best_coverage < 0.5:
        issues.append("low_coverage")

    # === 2. Jitter (position jumps) ===
    hand_xyz = xyz[:, :42, :]
    frame_diffs = np.linalg.norm(np.diff(hand_xyz, axis=0), axis=-1)
    max_jump = float(frame_diffs.max())
    metrics['max_jump'] = round(max_jump, 3)

    if max_jump > 2.0:
        issues.append("extreme_jitter")
    elif max_jump > 1.0:
        issues.append("high_jitter")

    # === 3. Frozen frames ===
    frozen_count = 0
    for t in range(T - 1):
        if np.allclose(xyz[t, :42], xyz[t+1, :42], atol=1e-6):
            frozen_count += 1
    metrics['frozen_frames'] = frozen_count

    if frozen_count > 15:
        issues.append("mostly_frozen")
    elif frozen_count > 8:
        issues.append("partially_frozen")

    # === 4. Hand swap detection ===
    l_wrist_x = xyz[:, 0, 0]
    r_wrist_x = xyz[:, 21, 0]
    if l_hand_mask.sum() > 0 and r_hand_mask.sum() > 0:
        crossings = 0
        for t in range(T - 1):
            if l_hand_mask[t] > 0 and r_hand_mask[t] > 0:
                if l_hand_mask[t+1] > 0 and r_hand_mask[t+1] > 0:
                    diff_now = l_wrist_x[t] - r_wrist_x[t]
                    diff_next = l_wrist_x[t+1] - r_wrist_x[t+1]
                    if diff_now * diff_next < 0 and abs(diff_now) > 0.1:
                        crossings += 1
        if crossings > 2:
            issues.append("possible_hand_swap")

    # === 5. All zeros ===
    if np.abs(xyz[:, :42]).max() < 1e-6:
        issues.append("all_zeros")

    # === 6. Bone length consistency ===
    l_active_frames = [t for t in range(T) if l_hand_mask[t] > 0]
    r_active_frames = [t for t in range(T) if r_hand_mask[t] > 0]

    l_bone_cv, l_bad_bones = check_bone_consistency(xyz, 0, l_active_frames)
    r_bone_cv, r_bad_bones = check_bone_consistency(xyz, 21, r_active_frames)
    worst_bone_cv = max(l_bone_cv, r_bone_cv)
    metrics['bone_cv'] = round(float(worst_bone_cv), 3)

    if worst_bone_cv > 0.4:
        issues.append("severe_bone_instability")
    elif worst_bone_cv > 0.25:
        issues.append("bone_instability")

    # === 7. Anatomical violations ===
    anat_violations = 0
    for t in range(T):
        if l_hand_mask[t] > 0:
            anat_violations += len(check_anatomical(xyz, 0, t))
        if r_hand_mask[t] > 0:
            anat_violations += len(check_anatomical(xyz, 21, t))
    metrics['anatomical_violations'] = anat_violations

    if anat_violations > 20:
        issues.append("many_anatomical_violations")
    elif anat_violations > 8:
        issues.append("some_anatomical_violations")

    # === 8. Motion range ===
    # A valid sign should have some movement — completely static = suspicious
    hand_motion = 0.0
    for hand_start in [0, 21]:
        hand_active = l_active_frames if hand_start == 0 else r_active_frames
        if len(hand_active) >= 2:
            wrist_trajectory = xyz[hand_active, hand_start, :]
            total_dist = np.sum(np.linalg.norm(np.diff(wrist_trajectory, axis=0), axis=-1))
            hand_motion = max(hand_motion, total_dist)
    metrics['motion_range'] = round(float(hand_motion), 3)

    if hand_motion < 0.05 and best_coverage > 0.3:
        issues.append("no_motion")

    # === 9. Spatial sanity ===
    # After normalization, landmarks should be within a reasonable range
    hand_positions = xyz[:, :42, :]
    active_positions = hand_positions[hand_positions.any(axis=-1)]
    if len(active_positions) > 0:
        max_dist = np.abs(active_positions).max()
        metrics['max_spatial'] = round(float(max_dist), 3)
        if max_dist > 10.0:
            issues.append("spatial_outlier")
    else:
        metrics['max_spatial'] = 0.0

    # === 10. Velocity sanity ===
    if best_coverage > 0.5:
        vel_magnitude = np.linalg.norm(vel[:, :42, :], axis=-1)
        max_vel = float(vel_magnitude.max())
        metrics['max_velocity'] = round(max_vel, 3)
        if max_vel > 5.0:
            issues.append("extreme_velocity")
    else:
        metrics['max_velocity'] = 0.0

    # === 11. Fingerprint for duplicate detection ===
    # Hash of downsampled wrist trajectory
    wrist_sig = np.concatenate([xyz[::4, 0, :], xyz[::4, 21, :]])
    fingerprint = hash(wrist_sig.tobytes())
    metrics['fingerprint'] = fingerprint

    # === 12. Motion signature for variant detection ===
    # Captures the overall motion pattern of the sign
    # Used to cluster different sign variants within the same class
    sig = []
    for hand_start in [0, 21]:
        wrist = xyz[:, hand_start, :2]      # wrist XY trajectory
        fingertip_avg = xyz[:, hand_start+8, :2]  # middle fingertip as shape proxy

        # Direction of motion (normalized displacement over time)
        if np.linalg.norm(wrist[-1] - wrist[0]) > 0.01:
            direction = (wrist[-1] - wrist[0]) / (np.linalg.norm(wrist[-1] - wrist[0]) + 1e-8)
        else:
            direction = np.array([0.0, 0.0])
        sig.extend(direction.tolist())

        # Average hand height (Y position)
        sig.append(float(wrist[:, 1].mean()))

        # Motion spread (how much the hand moves)
        sig.append(float(np.std(wrist[:, 0])))
        sig.append(float(np.std(wrist[:, 1])))

        # Hand openness (avg distance from wrist to middle fingertip)
        hand_spread = np.linalg.norm(fingertip_avg - wrist[:, :2], axis=-1).mean()
        sig.append(float(hand_spread))

    metrics['motion_signature'] = sig  # 12-dim vector per sample

    return {
        "metrics": metrics,
        "issues": issues,
    }


def detect_duplicates(results_by_file):
    """Find near-duplicate files based on wrist trajectory fingerprints."""
    fingerprints = defaultdict(list)
    for fname, result in results_by_file.items():
        fp = result['metrics'].get('fingerprint', 0)
        fingerprints[fp].append(fname)

    duplicates = {fp: files for fp, files in fingerprints.items() if len(files) > 1}
    return duplicates


def detect_sign_variants(results_by_file, min_samples=10):
    """Detect different sign variants within the same class.
    Groups samples by motion pattern to find classes with multiple distinct signs.
    Returns dict of {class: [{variant_id, files, center, size}, ...]}
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    class_sigs = defaultdict(list)
    class_files = defaultdict(list)

    for fname, result in results_by_file.items():
        label = fname.rsplit('_', 2)[0]
        sig = result['metrics'].get('motion_signature', [])
        if len(sig) == 12:
            class_sigs[label].append(sig)
            class_files[label].append(fname)

    variant_report = {}
    for label in sorted(class_sigs.keys()):
        sigs = np.array(class_sigs[label])
        files = class_files[label]

        if len(sigs) < min_samples:
            continue

        # Normalize features
        scaler = StandardScaler()
        sigs_norm = scaler.fit_transform(sigs)

        # Cluster with DBSCAN (finds arbitrary-shaped clusters, handles noise)
        clustering = DBSCAN(eps=1.2, min_samples=max(3, len(sigs) // 10)).fit(sigs_norm)
        labels = clustering.labels_

        unique_labels = set(labels)
        if len(unique_labels) <= 2:  # 1 cluster + noise, or just noise — normal
            continue

        # Multiple clusters found — different sign variants
        clusters = []
        for cl in sorted(unique_labels):
            if cl == -1:
                cluster_name = "noise/outliers"
            else:
                cluster_name = f"variant_{cl}"
            member_files = [files[i] for i, l in enumerate(labels) if l == cl]
            cluster_center = sigs[labels == cl].mean(axis=0).tolist() if cl != -1 else []
            clusters.append({
                "variant": cluster_name,
                "count": len(member_files),
                "files": member_files[:5],  # sample files
                "pct": round(100 * len(member_files) / len(files), 1),
            })

        variant_report[label] = {
            "total_samples": len(files),
            "num_variants": len([c for c in clusters if c["variant"] != "noise/outliers"]),
            "clusters": sorted(clusters, key=lambda x: -x["count"]),
        }

    return variant_report


def detect_class_outliers(results_by_file):
    """Find samples that are statistical outliers within their class."""
    class_features = defaultdict(list)
    class_files = defaultdict(list)

    for fname, result in results_by_file.items():
        label = fname.rsplit('_', 2)[0]
        m = result['metrics']
        feature_vec = [m['coverage'], m['max_jump'], m['motion_range'], m['bone_cv']]
        class_features[label].append(feature_vec)
        class_files[label].append(fname)

    outliers = []
    for label in class_features:
        features = np.array(class_features[label])
        if len(features) < 10:
            continue
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        z_scores = np.abs((features - mean) / std)
        max_z = z_scores.max(axis=1)

        for i, z in enumerate(max_z):
            if z > 3.0:
                outliers.append((class_files[label][i], label, round(float(z), 2)))

    return sorted(outliers, key=lambda x: -x[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="ASL_landmarks_float16", help="Directory with .npy files")
    parser.add_argument("--samples", type=int, default=0, help="Number of random samples (0 = all)")
    args = parser.parse_args()

    npy_dir = Path(args.dir)
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])

    if args.samples > 0 and args.samples < len(files):
        import random
        random.seed(42)
        files = random.sample(files, args.samples)

    print(f"Auditing {len(files)} files from {npy_dir}...\n")

    results_by_file = {}
    class_issues = defaultdict(lambda: defaultdict(int))
    class_counts = defaultdict(int)
    issue_counts = defaultdict(int)
    flagged_files = []

    for i, f in enumerate(files):
        label = f.rsplit('_', 2)[0]
        result = audit_sample(npy_dir / f)
        results_by_file[f] = result
        class_counts[label] += 1

        if result["issues"]:
            flagged_files.append((f, result))
            for issue in result["issues"]:
                issue_type = issue.split(" ")[0]
                issue_counts[issue_type] += 1
                class_issues[label][issue_type] += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(files)}...", end="\r")

    # === Summary ===
    total = len(files)
    flagged = len(flagged_files)
    clean = total - flagged

    print(f"\n{'='*60}")
    print(f"QUALITY AUDIT RESULTS")
    print(f"{'='*60}")
    print(f"Total files:  {total}")
    print(f"Clean:        {clean} ({100*clean/total:.1f}%)")
    print(f"Flagged:      {flagged} ({100*flagged/total:.1f}%)")

    print(f"\n--- Issue Breakdown ---")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count} ({100*count/total:.1f}%)")

    # === Duplicate Detection ===
    print(f"\n--- Duplicate Detection ---")
    duplicates = detect_duplicates(results_by_file)
    total_dupes = sum(len(v) - 1 for v in duplicates.values())
    print(f"  Duplicate groups: {len(duplicates)}")
    print(f"  Total duplicate files: {total_dupes}")
    if duplicates:
        for fp, dup_files in list(duplicates.items())[:5]:
            print(f"    Group: {', '.join(dup_files[:3])}{'...' if len(dup_files) > 3 else ''}")

    # === Class Outliers ===
    print(f"\n--- Class Outliers (z-score > 3) ---")
    outliers = detect_class_outliers(results_by_file)
    print(f"  Total outliers: {len(outliers)}")
    for fname, label, z in outliers[:10]:
        print(f"    {fname} (class={label}, z={z})")

    # === Per-Metric Distribution ===
    print(f"\n--- Metric Distributions ---")
    coverages = [r['metrics']['coverage'] for r in results_by_file.values()]
    bone_cvs = [r['metrics']['bone_cv'] for r in results_by_file.values()]
    motions = [r['metrics']['motion_range'] for r in results_by_file.values()]
    jumps = [r['metrics']['max_jump'] for r in results_by_file.values()]

    for name, vals in [('coverage', coverages), ('bone_cv', bone_cvs),
                        ('motion_range', motions), ('max_jump', jumps)]:
        arr = np.array(vals)
        print(f"  {name:15s}: mean={arr.mean():.3f}, std={arr.std():.3f}, "
              f"min={arr.min():.3f}, p25={np.percentile(arr,25):.3f}, "
              f"median={np.median(arr):.3f}, p75={np.percentile(arr,75):.3f}, max={arr.max():.3f}")

    # === Worst Classes ===
    print(f"\n--- Top 20 Most Flagged Classes ---")
    class_flag_rate = {}
    for label in class_counts:
        total_issues = sum(class_issues[label].values())
        rate = total_issues / class_counts[label]
        class_flag_rate[label] = rate

    for label, rate in sorted(class_flag_rate.items(), key=lambda x: -x[1])[:20]:
        if rate > 0:
            details = ", ".join(f"{k}:{v}" for k, v in class_issues[label].items())
            n = class_counts[label]
            print(f"  {label} ({n} files): {rate*100:.0f}% flagged — {details}")

    # === Sign Variant Detection ===
    print(f"\n--- Sign Variant Detection ---")
    print("  (Classes with multiple distinct sign patterns — possible wrong variants)")
    try:
        variants = detect_sign_variants(results_by_file)
        if variants:
            print(f"  Classes with multiple variants: {len(variants)}")
            for label, info in sorted(variants.items(), key=lambda x: -x[1]['num_variants']):
                print(f"\n  {label} ({info['total_samples']} samples, {info['num_variants']} variants):")
                for cl in info['clusters']:
                    sample_files = ', '.join(os.path.basename(f) for f in cl['files'][:3])
                    print(f"    {cl['variant']}: {cl['count']} samples ({cl['pct']}%) — e.g. {sample_files}")
        else:
            print("  No multi-variant classes detected (all classes have consistent motion patterns)")
    except ImportError:
        print("  Skipped — requires scikit-learn: pip install scikit-learn")

    # === Quality Tiers ===
    print(f"\n--- Quality Tiers ---")
    tier_a = sum(1 for r in results_by_file.values()
                 if not r['issues'] and r['metrics']['coverage'] > 0.85 and r['metrics']['bone_cv'] < 0.15)
    tier_b = sum(1 for r in results_by_file.values()
                 if not r['issues'] and (r['metrics']['coverage'] <= 0.85 or r['metrics']['bone_cv'] >= 0.15))
    tier_c = sum(1 for r in results_by_file.values()
                 if r['issues'] and not any(i in ['extreme_jitter', 'severe_bone_instability',
                    'all_zeros', 'spatial_outlier', 'many_anatomical_violations'] for i in r['issues']))
    tier_d = total - tier_a - tier_b - tier_c

    print(f"  Tier A (high quality):    {tier_a:5d} ({100*tier_a/total:.1f}%) — clean, high coverage, stable bones")
    print(f"  Tier B (usable):          {tier_b:5d} ({100*tier_b/total:.1f}%) — clean but lower coverage or bone variance")
    print(f"  Tier C (noisy but usable):{tier_c:5d} ({100*tier_c/total:.1f}%) — minor issues, can train with downweighting")
    print(f"  Tier D (problematic):     {tier_d:5d} ({100*tier_d/total:.1f}%) — severe issues, consider excluding")

    # === Save Results ===
    output = {
        "summary": {
            "total": total, "clean": clean, "flagged": flagged,
            "clean_pct": round(100*clean/total, 1),
            "duplicates": total_dupes,
            "outliers": len(outliers),
            "tiers": {"A": tier_a, "B": tier_b, "C": tier_c, "D": tier_d},
        },
        "issue_counts": dict(issue_counts),
        "metric_distributions": {
            "coverage": {"mean": round(np.mean(coverages), 3), "std": round(np.std(coverages), 3)},
            "bone_cv": {"mean": round(np.mean(bone_cvs), 3), "std": round(np.std(bone_cvs), 3)},
            "motion_range": {"mean": round(np.mean(motions), 3), "std": round(np.std(motions), 3)},
            "max_jump": {"mean": round(np.mean(jumps), 3), "std": round(np.std(jumps), 3)},
        },
        "outliers": [(f, l, z) for f, l, z in outliers[:50]],
        "sign_variants": {k: {kk: vv for kk, vv in v.items() if kk != 'clusters' or True}
                          for k, v in variants.items()} if 'variants' in dir() and variants else {},
        "flagged_files": [(f, r['issues'], {k: v for k, v in r['metrics'].items()
                          if k not in ('fingerprint', 'motion_signature')})
                          for f, r in flagged_files[:200]],
    }
    out_path = npy_dir.parent / "quality_audit.json"
    with open(out_path, 'w') as fp:
        json.dump(output, fp, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
