#!/usr/bin/env python3
"""
validate_manifest.py — Post-extraction manifest sanity checks.

Run after extract.py to catch:
  1. Labels that are also LABEL_ALIASES keys (alias wasn't applied)
  2. Labels with fewer than N samples (too few to train reliably)
  3. .npy files on disk missing from manifest (orphans)
  4. Manifest entries whose .npy files don't exist (stale entries)

Usage:
    python src/validate_manifest.py [--landmarks-dir ASL_landmarks_float16] [--min-samples 5]
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Import LABEL_ALIASES from extract.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import LABEL_ALIASES


def main():
    ap = argparse.ArgumentParser(description="Post-extraction manifest validation")
    ap.add_argument("--landmarks-dir", default="ASL_landmarks_float16")
    ap.add_argument("--min-samples", type=int, default=5,
                    help="Warn if any label has fewer than this many samples")
    args = ap.parse_args()

    landmarks_dir = Path(args.landmarks_dir)
    manifest_path = landmarks_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"FATAL: {manifest_path} not found. Run extraction first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Manifest: {len(manifest)} entries")
    errors = 0

    # Check 1: No label should be a key in LABEL_ALIASES
    labels = set(manifest.values())
    alias_leaks = labels & set(LABEL_ALIASES.keys())
    if alias_leaks:
        print(f"\nFAIL: {len(alias_leaks)} labels are LABEL_ALIASES keys (alias was NOT applied):")
        for lbl in sorted(alias_leaks):
            count = sum(1 for v in manifest.values() if v == lbl)
            print(f"  {lbl} ({count} files) -> should be '{LABEL_ALIASES[lbl]}'")
        errors += len(alias_leaks)
    else:
        print("PASS: No label is a LABEL_ALIASES key (all aliases applied correctly)")

    # Check 2: Label distribution — flag low-sample classes
    label_counts = Counter(manifest.values())
    low_sample = [(lbl, cnt) for lbl, cnt in label_counts.items() if cnt < args.min_samples]
    if low_sample:
        print(f"\nWARN: {len(low_sample)} labels have fewer than {args.min_samples} samples:")
        for lbl, cnt in sorted(low_sample, key=lambda x: x[1]):
            print(f"  {lbl}: {cnt} samples")
    else:
        print(f"PASS: All labels have >= {args.min_samples} samples")

    # Check 3: Orphan .npy files (on disk but not in manifest)
    disk_files = {f for f in os.listdir(landmarks_dir) if f.endswith('.npy')}
    manifest_files = set(manifest.keys())
    orphans = disk_files - manifest_files
    if orphans:
        print(f"\nWARN: {len(orphans)} .npy files on disk but NOT in manifest (orphans):")
        for f in sorted(orphans)[:20]:
            print(f"  {f}")
        if len(orphans) > 20:
            print(f"  ... and {len(orphans) - 20} more")
    else:
        print("PASS: No orphan .npy files")

    # Check 4: Stale manifest entries (in manifest but missing on disk)
    stale = manifest_files - disk_files
    if stale:
        print(f"\nWARN: {len(stale)} manifest entries have no .npy file on disk (stale):")
        for f in sorted(stale)[:20]:
            print(f"  {f}")
        if len(stale) > 20:
            print(f"  ... and {len(stale) - 20} more")
    else:
        print("PASS: All manifest entries have corresponding .npy files")

    # Print full label distribution (ascending)
    print(f"\nLabel distribution ({len(label_counts)} classes):")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: x[1]):
        print(f"  {lbl}: {cnt}")

    print(f"\nTotal: {len(manifest)} files, {len(label_counts)} classes")
    if errors:
        print(f"\n{errors} ERRORS found. Fix before training.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED. Ready for training.")


if __name__ == "__main__":
    main()
