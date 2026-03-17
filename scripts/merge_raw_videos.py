#!/usr/bin/env python3
"""
Merge raw_video folders with identical names from different locations into a new folder.
Leaves originals untouched. Copies all videos, renumbering to avoid collisions.

Usage:
  python scripts/merge_raw_videos.py
  python scripts/merge_raw_videos.py --source data/raw_videos --output data/raw_videos_merged

Default: source=data/raw_videos, output=data/raw_videos_merged
"""

import argparse
from collections import defaultdict
from pathlib import Path
import shutil

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def main():
    parser = argparse.ArgumentParser(
        description="Merge same-named folders from raw_videos into a new output folder."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw_videos"),
        help="Root folder to scan for label folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw_videos_merged"),
        help="Output folder for merged results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print source paths for each label",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()

    if not source.exists():
        print(f"Source not found: {source}")
        return 1

    # Find all folders that contain at least one video file.
    # Map folder_name -> list of (folder_path, video_files)
    by_label: dict[str, list[tuple[Path, list[Path]]]] = defaultdict(list)

    for d in source.rglob("*"):
        if not d.is_dir():
            continue
        videos = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS]
        if not videos:
            continue
        label = d.name.strip()
        if label:
            by_label[label].append((d, sorted(videos)))

    merged_labels = dict(by_label)
    if not merged_labels:
        print("No label folders with videos found.")
        return 0

    multi = [k for k, v in merged_labels.items() if len(v) > 1]
    single = [k for k, v in merged_labels.items() if len(v) == 1]
    print(f"Labels in multiple folders (will merge): {sorted(multi)}")
    print(f"Labels in single folder (will copy): {sorted(single)}")
    print(f"Output: {output}\n")

    if args.dry_run or args.verbose:
        for label, folders in sorted(merged_labels.items()):
            total = sum(len(vids) for _, vids in folders)
            print(f"  {label}: {total} videos from {len(folders)} folders")
            if args.verbose:
                for folder_path, vids in folders:
                    rel = folder_path.relative_to(source)
                    print(f"      <- {rel}")
        if args.dry_run:
            print("\nDry run. Use without --dry-run to copy.")
            return 0

    output.mkdir(parents=True, exist_ok=True)

    for label, folders in sorted(merged_labels.items()):
        out_dir = output / label
        out_dir.mkdir(parents=True, exist_ok=True)
        counter = 1
        ext = ".mp4"

        for folder_path, video_files in folders:
            for vid in video_files:
                new_name = f"{label}_{counter}{vid.suffix}"
                dst = out_dir / new_name
                shutil.copy2(vid, dst)
                print(f"  {vid.relative_to(source)} -> {dst.relative_to(output)}")
                counter += 1

        print(f"  Merged {label}: {counter - 1} videos -> {out_dir}")

    print(f"\nDone. Merged output in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
