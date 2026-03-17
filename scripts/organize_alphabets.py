#!/usr/bin/env python3
"""
Move alphabet folders (A–Z) from merged output into a single ALPHABETS subfolder.

Usage:
  python scripts/organize_alphabets.py
  python scripts/organize_alphabets.py --source data/raw_videos_merged --folder ALPHABETS

Default: source=data/raw_videos_merged, subfolder=ALPHABETS
"""

import argparse
from pathlib import Path

ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def main():
    parser = argparse.ArgumentParser(
        description="Put alphabet folders (A–Z) into a single subfolder."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw_videos_merged"),
        help="Merged folder containing label subfolders",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="ALPHABETS",
        help="Name of subfolder for alphabets (default: ALPHABETS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without moving",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"Source not found: {source}")
        return 1

    alphabet_dirs = [d for d in source.iterdir() if d.is_dir() and d.name in ALPHABET]
    if not alphabet_dirs:
        print(f"No alphabet folders (A–Z) found in {source}")
        return 0

    dest_dir = source / args.folder

    print(f"Alphabet folders to move: {sorted(d.name for d in alphabet_dirs)}")
    print(f"Destination: {dest_dir}\n")

    if args.dry_run:
        for d in sorted(alphabet_dirs, key=lambda x: x.name):
            print(f"  {d.name}/ -> {dest_dir.name}/{d.name}/")
        print("\nDry run. Use without --dry-run to move.")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)

    for d in sorted(alphabet_dirs, key=lambda x: x.name):
        new_path = dest_dir / d.name
        d.rename(new_path)
        print(f"  Moved {d.name}/ -> {dest_dir.name}/{d.name}/")

    print(f"\nDone. Alphabets are in {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
