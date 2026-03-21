"""
Check extraction status: compare raw videos vs already-extracted .npy files.

Uses the same logic as extract.py for "done" vs "to process":
  - expected_save_name = {label}_{stem}_{md5(filename)[:6]}
  - A video is "extracted" if that base name exists in ASL_landmarks_float16.

Run: python src/check_extraction_status.py [--raw-dir DIR] [--output-dir DIR] [--verbose]
"""

import argparse
import hashlib
import os
from pathlib import Path

# Match extract.py exactly (no heavy imports)
LABEL_ALIASES = {
    "DRIVE": "DRIVE_CAR",
    "CAR": "DRIVE_CAR",
    "HARD": "HARD_DIFFICULT",
    "DIFFICULT": "HARD_DIFFICULT",
    "MAKE": "MAKE_CREATE",
    "CREATE": "MAKE_CREATE",
    "EAT": "EAT_FOOD",
    "FOOD": "EAT_FOOD",
    "ALSO_SAME": "ALSO",
    "SAME": "ALSO",
    "MARKET_STORE": "STORE",
    "MARKET": "STORE",
    "US_WE": "WE",
    "US": "WE",
    "FEW_SEVERAL": "FEW",
    "SEVERAL": "FEW",
    "I_ME": "I",
    "ME": "I",
    "HE_SHE": "HE",
    "SHE": "HE",
}
DEFAULT_RAW_DIR = "data/raw_videos/ASL VIDEOS"
DEFAULT_OUTPUT_DIR = "ASL_landmarks_float16"


def main():
    ap = argparse.ArgumentParser(description="Check extraction status: extracted vs pending videos")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Raw video directory")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Extraction output directory")
    ap.add_argument("--verbose", "-v", action="store_true", help="Show per-folder breakdown")
    args = ap.parse_args()

    raw_root = Path(args.raw_dir)
    out_path = Path(args.output_dir)

    if not raw_root.exists():
        print(f"Error: Raw video dir not found: {raw_root}")
        return 1

    # Build done set (same logic as extract.py)
    done_base_files = set()
    if out_path.exists():
        for f in out_path.iterdir():
            if f.is_file() and f.suffix == ".npy":
                base_name = f.stem
                for suffix in ["_fast", "_slow", "_mirror"]:
                    if suffix in base_name:
                        base_name = base_name.replace(suffix, "")
                        break
                done_base_files.add(base_name)

    # Walk raw videos and classify
    extracted = 0
    pending = 0
    pending_by_folder = []

    for root, _, files in sorted(
        [(r, d, f) for r, d, f in os.walk(raw_root)],
        key=lambda x: x[0],
    ):
        root_path = Path(root)
        label = root_path.name
        label = LABEL_ALIASES.get(label, label)

        folder_extracted = 0
        folder_pending = 0

        for f in files:
            if not f.lower().endswith((".mp4", ".mov")):
                continue
            file_hash = hashlib.md5(f.encode()).hexdigest()[:6]
            stem = Path(f).stem
            expected_save_name = f"{label}_{stem}_{file_hash}"

            if expected_save_name in done_base_files:
                extracted += 1
                folder_extracted += 1
            else:
                pending += 1
                folder_pending += 1

        if args.verbose and (folder_extracted > 0 or folder_pending > 0):
            pending_by_folder.append((str(root_path), label, folder_extracted, folder_pending))

    total = extracted + pending

    print("=" * 60)
    print("EXTRACTION STATUS")
    print("=" * 60)
    print(f"Raw video dir:     {raw_root}")
    print(f"Output dir:        {out_path}")
    print(f"Existing .npy:     {len(done_base_files)} unique base names")
    print("-" * 60)
    print(f"Total videos:      {total:,}")
    print(f"Already extracted: {extracted:,}")
    print(f"Not yet extracted: {pending:,}  <- these will be ADDED when you run extract.py")
    print("=" * 60)

    if args.verbose and pending_by_folder:
        print("\nPer-folder breakdown (folders with videos):")
        print(f"{'Folder':<50} {'Done':>8} {'Pending':>8}")
        print("-" * 68)
        for folder, label, done, pend in sorted(pending_by_folder, key=lambda x: -x[3]):
            short = folder[-47:] if len(folder) > 50 else folder
            print(f"{short:<50} {done:>8,} {pend:>8,}")

    return 0


if __name__ == "__main__":
    exit(main())
