import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DWIGHT_ROOT_DEFAULT = Path("data/raw_videos/dwight")
ASL_ROOT_DEFAULT = Path("data/raw_videos/ASL VIDEOS")

SOURCE_EXTENSIONS = {".mp4"}

SOURCE_TIERS_HINT = [
    "Dwight ASL Datasets",
    "Dwight ASL Datasets 2",
]

# Explicit mapping for leaf folders in dwight that do not exist in ASL VIDEOS.
# (Identity mapping is used for any sign folder that already exists in ASL VIDEOS.)
EXPLICIT_CLASS_MAPPING: Dict[str, str] = {
    "CONFUSED": "CONFUSE",
    "DON_T": "DONT",
    "MARKET": "MARKET_STORE",
    "SAME": "ALSO_SAME",
    "STORE": "MARKET_STORE",
    "WE_US": "US_WE",
}


def _sha_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _is_hidden_path(p: Path) -> bool:
    return any(part.startswith(".") for part in p.parts)


def _count_mp4s_direct(dest_dir: Path) -> int:
    if not dest_dir.is_dir():
        return 0
    c = 0
    for item in dest_dir.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_file() and item.suffix.lower() in SOURCE_EXTENSIONS:
            c += 1
    return c


def _iter_dwight_mp4s(dwight_root: Path) -> Iterable[Path]:
    """
    Recursively yield mp4 files under dwight_root.
    Leaf sign is defined as the *parent directory name* of each mp4.
    """
    # Using rglob over '*' is more robust than '*.mp4' when filenames vary case.
    for p in dwight_root.rglob("*"):
        if not p.is_file():
            continue
        if _is_hidden_path(p):
            continue
        if p.suffix.lower() in SOURCE_EXTENSIONS:
            yield p


def map_sign(source_sign: str, asl_root: Path) -> str:
    dest_dir = asl_root / source_sign
    if dest_dir.is_dir():
        return source_sign
    if source_sign in EXPLICIT_CLASS_MAPPING:
        return EXPLICIT_CLASS_MAPPING[source_sign]
    raise KeyError(
        f"Missing destination sign folder '{source_sign}'. "
        f"No explicit mapping available."
    )


def collision_safe_destination_name(
    src_video: Path,
    dest_dir: Path,
    dest_sign: str,
) -> Path:
    """
    Deterministically rename incoming file if dest already has same name.
    """
    base = src_video.stem
    suffix = src_video.suffix
    h = _sha_short(str(src_video))

    candidate = dest_dir / f"{base}__from_DWIGHT_{dest_sign}__{h}{suffix}"
    if not candidate.exists():
        return candidate

    # Defensive: add an integer counter if the candidate hash still collides.
    stem2 = f"{base}__from_DWIGHT_{dest_sign}__{h}"
    for i in range(1, 1000):
        alt = dest_dir / f"{stem2}__{i}{suffix}"
        if not alt.exists():
            return alt
    raise RuntimeError(f"Could not find a collision-free name for {src_video}")


@dataclass(frozen=True)
class MovePlanItem:
    source_sign: str
    dest_sign: str
    source_video: Path
    dest_video_final: Path
    collision_renamed: bool


def build_move_plan(dwight_root: Path, asl_root: Path) -> Tuple[List[MovePlanItem], Dict[str, int]]:
    plan: List[MovePlanItem] = []
    dest_signs_touched: Set[str] = set()

    for src_video in _iter_dwight_mp4s(dwight_root):
        source_sign = src_video.parent.name
        dest_sign = map_sign(source_sign, asl_root=asl_root)
        dest_dir = asl_root / dest_sign
        if not dest_dir.is_dir():
            # map_sign should have validated, but keep it defensive.
            raise FileNotFoundError(f"Destination folder missing: {dest_dir}")

        dest_candidate = dest_dir / src_video.name
        if not dest_candidate.exists():
            plan.append(
                MovePlanItem(
                    source_sign=source_sign,
                    dest_sign=dest_sign,
                    source_video=src_video,
                    dest_video_final=dest_candidate,
                    collision_renamed=False,
                )
            )
        else:
            renamed = collision_safe_destination_name(
                src_video=src_video,
                dest_dir=dest_dir,
                dest_sign=dest_sign,
            )
            plan.append(
                MovePlanItem(
                    source_sign=source_sign,
                    dest_sign=dest_sign,
                    source_video=src_video,
                    dest_video_final=renamed,
                    collision_renamed=True,
                )
            )

        dest_signs_touched.add(dest_sign)

    dest_before_counts: Dict[str, int] = {}
    for dest_sign in dest_signs_touched:
        dest_before_counts[dest_sign] = _count_mp4s_direct(asl_root / dest_sign)

    return plan, dest_before_counts


def _remove_empty_leaf_dirs(leaf_dirs: Set[Path]) -> int:
    """
    Remove empty directories; do not recurse upward.
    """
    removed = 0
    for d in sorted(leaf_dirs, key=lambda x: str(x), reverse=True):
        if not d.is_dir():
            continue
        non_hidden = [p for p in d.iterdir() if not p.name.startswith(".")]
        if len(non_hidden) == 0:
            try:
                d.rmdir()
                removed += 1
            except OSError:
                pass
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge (MOVE) mp4 videos from dwight dataset into ASL VIDEOS."
    )
    parser.add_argument(
        "--dwight-root",
        default=str(DWIGHT_ROOT_DEFAULT),
        help=f"Path to dwight root (default: {DWIGHT_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--asl-root",
        default=str(ASL_ROOT_DEFAULT),
        help=f"Path to ASL root (default: {ASL_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually MOVE files. Default is dry-run.",
    )
    parser.add_argument(
        "--remove-empty-sources",
        action="store_true",
        help="After execute, remove empty leaf sign directories under dwight.",
    )
    parser.add_argument(
        "--max-sample-print",
        type=int,
        default=20,
        help="How many planned mappings to print in dry-run/sample mode.",
    )
    args = parser.parse_args()

    dwight_root = Path(args.dwight_root)
    asl_root = Path(args.asl_root)

    if not dwight_root.is_dir():
        raise FileNotFoundError(f"Missing dwight root: {dwight_root}")
    if not asl_root.is_dir():
        raise FileNotFoundError(f"Missing ASL destination root: {asl_root}")

    print(f"dwight root: {dwight_root}")
    print(f"ASL root: {asl_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")

    plan_items, dest_before = build_move_plan(dwight_root=dwight_root, asl_root=asl_root)

    total = len(plan_items)
    collisions = sum(1 for x in plan_items if x.collision_renamed)
    touched_dest_signs = sorted(set(x.dest_sign for x in plan_items))
    print(f"\nPlanned moves: {total} mp4 files")
    print(f"Planned collisions renamed: {collisions}")
    print(f"Touched destination signs: {len(touched_dest_signs)}")

    # Per destination summary
    by_dest: Dict[str, List[MovePlanItem]] = {}
    for item in plan_items:
        by_dest.setdefault(item.dest_sign, []).append(item)

    print("\nPer-destination-sign summary:")
    for dest_sign in sorted(by_dest.keys()):
        items = by_dest[dest_sign]
        moved = len(items)
        renamed_collisions = sum(1 for x in items if x.collision_renamed)
        before = dest_before.get(dest_sign, 0)
        sources = sorted(set(x.source_sign for x in items))
        print(
            f"- {dest_sign}: moved={moved}, renamed_collisions={renamed_collisions}, "
            f"before={before}, sources={sources}"
        )

    if not args.execute:
        print("\nSample mappings (first items):")
        for item in plan_items[: max(0, args.max_sample_print)]:
            tag = " (RENAMED)" if item.collision_renamed else ""
            print(f"- {item.source_video} -> {item.dest_video_final}{tag}")
        return

    # Execute MOVE
    leaf_dirs_touched: Set[Path] = set()
    for item in plan_items:
        leaf_dirs_touched.add(item.source_video.parent)
        item.dest_video_final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.source_video), str(item.dest_video_final))

    # Verification: ensure dwight has no remaining mp4
    remaining_mp4s = sum(1 for _ in _iter_dwight_mp4s(dwight_root))
    if remaining_mp4s != 0:
        print(f"\nWARNING: Remaining mp4 files under dwight after move: {remaining_mp4s}")
    else:
        print("\nVerification: dwight has 0 remaining mp4 files.")

    # Verification: destination counts
    print("\nPost-merge destination counts:")
    anomalies = 0
    for dest_sign in sorted(by_dest.keys()):
        before = dest_before.get(dest_sign, 0)
        moved = len(by_dest[dest_sign])
        after = _count_mp4s_direct(asl_root / dest_sign)
        expected_after = before + moved
        ok = (after == expected_after)
        if not ok:
            anomalies += 1
        print(
            f"- {dest_sign}: before={before}, after={after}, expected_after={expected_after}, ok={ok}"
        )

    if anomalies:
        print(f"\nVerification: {anomalies} destination signs had unexpected counts.")
    else:
        print("\nVerification: all touched destination signs match expected counts.")

    if args.remove_empty_sources:
        removed_dirs = _remove_empty_leaf_dirs(leaf_dirs=leaf_dirs_touched)
        print(f"\nRemoved empty source directories under dwight: {removed_dirs}")


if __name__ == "__main__":
    main()

