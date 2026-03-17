#!/usr/bin/env python3
"""
Merge letter+number folders (e.g. "A 00001", "B 00002") into single-letter folders (A, B)
inside raw_videos/ALPHABETS. Copies videos with renumbering, removes the numbered folders.
Leaves originals (A, B, ...) intact and adds videos from the numbered folders.

Usage:
  python scripts/merge_alphabet_numbered_folders.py
  python scripts/merge_alphabet_numbered_folders.py --source data/raw_videos/ALPHABETS
"""

import argparse
import re
import shutil
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
# Matches "A 00001", "B 00002", etc. (single letter + space + digits)
NUMBERED_PATTERN = re.compile(r"^([A-Za-z])\s+\d+$")


def main():
    parser = argparse.ArgumentParser(
        description="Merge letter+number folders (e.g. A 00001) into single-letter folders (A)."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw_videos/ALPHABETS"),
        help="Folder containing A, A 00001, B, B 00002, etc.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying or deleting",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"Source not found: {source}")
        return 1

    to_merge = []
    for d in source.iterdir():
        if not d.is_dir():
            continue
        m = NUMBERED_PATTERN.match(d.name)
        if m:
            letter = m.group(1).upper()
            letter_dir = source / letter
            if letter_dir.exists() and letter_dir.is_dir():
                to_merge.append((d, letter_dir, letter))
            else:
                print(f"  Skip {d.name}: no target folder {letter}/")

    if not to_merge:
        print("No letter+number folders found to merge.")
        return 0

    print(f"Will merge {len(to_merge)} numbered folders into single-letter folders:\n")

    for numbered_dir, letter_dir, letter in sorted(to_merge, key=lambda x: x[0].name):
        videos = [f for f in numbered_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
        if not videos:
            print(f"  {numbered_dir.name}/ -> {letter}/ (no videos, will remove empty)")
            continue

        print(f"  {numbered_dir.name}/ -> {letter}/ (+{len(videos)} videos)")

        if args.dry_run:
            continue

        for vid in sorted(videos):
            dst = letter_dir / vid.name
            shutil.copy2(vid, dst)

        shutil.rmtree(numbered_dir)
        print(f"    Merged and removed {numbered_dir.name}/")

    if args.dry_run:
        print("\nDry run. Use without --dry-run to apply.")
    else:
        print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
