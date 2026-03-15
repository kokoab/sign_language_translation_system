#!/usr/bin/env python3
"""
Rename video files in raw_videos/ALPHABETS to signer_NN_MMM.mp4, optionally
merge all same-named letter folders into a single flat structure.

Structure before:
  ALPHABETS/
    SIGNER 1/
      A/
        signer_01_001.mp4
        signer_01_002.mp4
      B/
        ...
    SIGNER 2/
      A/
        signer_02_001.mp4
      ...

With --merge, output:
  ALPHABETS_merged/
    A/
      signer_01_001.mp4
      signer_01_002.mp4
      signer_02_001.mp4
    B/
      ...
"""

import argparse
import re
import shutil
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SIGNER_PATTERN = re.compile(r"SIGNER\s*(\d+)", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(
        description="Rename alphabet videos to signer_NN_MMM.mp4 format."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw_videos/ALPHABETS"),
        help="Root folder (ALPHABETS) containing SIGNER 1, SIGNER 2, ...",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print renames without applying",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all same-named letter folders into output (A/, B/, ...)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder for --merge (default: <source>_merged)",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"Source not found: {source}")
        return 1

    renamed_count = 0

    for signer_dir in sorted(source.iterdir()):
        if not signer_dir.is_dir():
            continue
        m = SIGNER_PATTERN.match(signer_dir.name.strip())
        if not m:
            continue

        signer_num = int(m.group(1))
        prefix = f"signer_{signer_num:02d}_"

        for letter_dir in sorted(signer_dir.iterdir()):
            if not letter_dir.is_dir():
                continue

            videos = sorted(
                f for f in letter_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not videos:
                continue

            temp_renames = []
            for i, vid in enumerate(videos):
                new_name = f"{prefix}{i + 1:03d}{vid.suffix}"
                if vid.name == new_name:
                    continue
                temp_renames.append((vid, new_name))

            if not temp_renames:
                continue

            for i, (vid, new_name) in enumerate(temp_renames):
                temp_name = f"{prefix}tmp_{i:04d}{vid.suffix}"
                temp_path = letter_dir / temp_name

                if args.dry_run:
                    print(f"  {vid.relative_to(source)} -> {letter_dir.name}/{new_name}")
                else:
                    vid.rename(temp_path)
                    temp_renames[i] = (temp_path, new_name)

            if not args.dry_run:
                for temp_path, new_name in temp_renames:
                    temp_path.rename(letter_dir / new_name)
                    renamed_count += 1
            else:
                renamed_count += len(temp_renames)

    print(f"{'Would rename' if args.dry_run else 'Renamed'} {renamed_count} files.")

    if args.merge:
        out_dir = (args.output or source.parent / f"{source.name}_merged").resolve()
        if args.dry_run:
            print(f"\nWould merge into {out_dir}:")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        by_letter = {}
        for signer_dir in sorted(source.iterdir()):
            if not signer_dir.is_dir():
                continue
            if not SIGNER_PATTERN.match(signer_dir.name.strip()):
                continue
            for letter_dir in sorted(signer_dir.iterdir()):
                if not letter_dir.is_dir():
                    continue
                letter = letter_dir.name
                if letter not in by_letter:
                    by_letter[letter] = []
                videos = [
                    f for f in letter_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
                ]
                by_letter[letter].extend(sorted(videos))

        merge_count = 0
        for letter in sorted(by_letter.keys()):
            videos = by_letter[letter]
            if not videos:
                continue
            dst_dir = out_dir / letter
            if not args.dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
            for vid in videos:
                dst = dst_dir / vid.name
                if args.dry_run:
                    print(f"  {letter}/{vid.name} <- {vid.relative_to(source)}")
                else:
                    shutil.copy2(vid, dst)
                merge_count += 1

        print(f"{'Would merge' if args.dry_run else 'Merged'} {merge_count} files -> {out_dir}")

    if args.dry_run:
        print("\nRun without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
