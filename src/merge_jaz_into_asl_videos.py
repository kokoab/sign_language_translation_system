import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


JAZ_ROOT_DEFAULT = Path("data/raw_videos/Jaz ASL Dataset")
ASL_ROOT_DEFAULT = Path("data/raw_videos/ASL VIDEOS")

SOURCE_TIERS: List[str] = [
    "TIER 1/CORE NOUNS",
    "TIER 1/CORE VERBS",
    "TIER 2/COMMON OBJECTS",
    "TIER 2/VERBS",
]

# Mapping rules (from user answers)
CLASS_MAPPING: Dict[str, str] = {
    "KWESTION": "QUESTION",
    "MARKET": "MARKET_STORE",
    "STORE": "MARKET_STORE",
}


def _sha_short(s: str, n: int = 10) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:n]


def _iter_sign_dirs(jaz_root: Path) -> Iterable[Tuple[str, Path]]:
    """
    Yields (source_sign_name, source_sign_dir).
    The "end folder" structure is: jaz_root/<tier>/<sign>/[videos...]
    """
    for tier_rel in SOURCE_TIERS:
        tier_dir = jaz_root / tier_rel
        if not tier_dir.is_dir():
            continue
        for sign_dir in tier_dir.iterdir():
            if not sign_dir.is_dir():
                continue
            if sign_dir.name.startswith("."):
                continue
            yield sign_dir.name, sign_dir


def _list_mp4s(sign_dir: Path) -> List[Path]:
    mp4s: List[Path] = []
    for p in sign_dir.rglob("*.mp4"):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            mp4s.append(p)
    # Make deterministic
    mp4s.sort(key=lambda x: str(x))
    return mp4s


def _ensure_dest_exists(dest_dir: Path) -> None:
    if not dest_dir.is_dir():
        raise FileNotFoundError(f"Missing destination folder: {dest_dir}")


def _collision_safe_dest_name(
    src_path: Path,
    source_sign: str,
    dest_dir: Path,
) -> str:
    """
    Returns a filename that does not collide in dest_dir.
    """
    base = src_path.stem
    suffix = src_path.suffix  # includes ".mp4"
    # Deterministic hash based on source path (not file contents) for repeatability.
    short_hash = _sha_short(str(src_path))
    candidate = f"{base}__from_JAZ_{source_sign}__{short_hash}{suffix}"
    return candidate


@dataclass(frozen=True)
class MovePlanItem:
    source_sign: str
    dest_sign: str
    source_video: Path
    dest_video_final: Path
    collision_renamed: bool


def build_move_plan(
    jaz_root: Path,
    asl_root: Path,
) -> Tuple[List[MovePlanItem], Dict[str, int], Dict[str, int], List[str]]:
    """
    Precomputes where each mp4 will go.
    Returns:
      - plan items
      - destination_mp4_counts_before (per dest_sign)
      - source_mp4_counts_by_sign (per source_sign)
      - missing_dest_signs
    """
    # Destination mp4 counts "before" (only for destination signs we touch)
    move_items: List[MovePlanItem] = []
    dest_signs_touched: Dict[str, None] = {}
    source_mp4_counts_by_sign: Dict[str, int] = {}

    for source_sign, source_sign_dir in _iter_sign_dirs(jaz_root):
        mp4s = _list_mp4s(source_sign_dir)
        source_mp4_counts_by_sign[source_sign] = len(mp4s)
        if not mp4s:
            continue

        dest_sign = CLASS_MAPPING.get(source_sign, source_sign)
        dest_signs_touched[dest_sign] = None

        dest_dir = asl_root / dest_sign
        _ensure_dest_exists(dest_dir)

        for src in mp4s:
            dest_dir = asl_root / dest_sign
            dest_candidate = dest_dir / src.name

            if not dest_candidate.exists():
                move_items.append(
                    MovePlanItem(
                        source_sign=source_sign,
                        dest_sign=dest_sign,
                        source_video=src,
                        dest_video_final=dest_candidate,
                        collision_renamed=False,
                    )
                )
                continue

            # Collision: rename incoming
            renamed_name = _collision_safe_dest_name(
                src_path=src, source_sign=source_sign, dest_dir=dest_dir
            )
            renamed_candidate = dest_dir / renamed_name

            # Extremely defensive: if hash-based rename still collides, add a counter.
            if renamed_candidate.exists():
                for i in range(1, 1000):
                    alt = dest_dir / f"{Path(renamed_name).stem}__{i}{Path(renamed_name).suffix}"
                    if not alt.exists():
                        renamed_candidate = alt
                        break

            move_items.append(
                MovePlanItem(
                    source_sign=source_sign,
                    dest_sign=dest_sign,
                    source_video=src,
                    dest_video_final=renamed_candidate,
                    collision_renamed=True,
                )
            )

    destination_mp4_counts_before: Dict[str, int] = {}
    missing_dest_signs: List[str] = []
    for dest_sign in sorted(dest_signs_touched.keys()):
        dest_dir = asl_root / dest_sign
        if not dest_dir.is_dir():
            missing_dest_signs.append(dest_sign)
            continue
        destination_mp4_counts_before[dest_sign] = len(_list_mp4s(dest_dir))

    return move_items, destination_mp4_counts_before, source_mp4_counts_by_sign, missing_dest_signs


def apply_move_plan(
    plan_items: List[MovePlanItem],
    execute: bool,
) -> Tuple[int, int]:
    """
    Returns (moved_count, collision_renamed_count)
    """
    moved_count = 0
    collision_renamed_count = 0
    for item in plan_items:
        if item.collision_renamed:
            collision_renamed_count += 1
        if execute:
            item.dest_video_final.parent.mkdir(parents=True, exist_ok=True)
            # MOVE semantics
            shutil.move(str(item.source_video), str(item.dest_video_final))
        moved_count += 1
    return moved_count, collision_renamed_count


def maybe_remove_empty_source_dirs(jaz_root: Path, execute: bool) -> None:
    if not execute:
        return

    # Only remove sign directories that are "empty" except for hidden files.
    for source_sign, source_sign_dir in _iter_sign_dirs(jaz_root):
        if not source_sign_dir.is_dir():
            continue
        non_hidden_files = [p for p in source_sign_dir.iterdir() if not p.name.startswith(".")]
        if not non_hidden_files:
            try:
                source_sign_dir.rmdir()
            except OSError:
                # Ignore if not empty due to race conditions or hidden files.
                pass


def _count_mp4s_in_dest_signs(
    asl_root: Path,
    dest_signs: Iterable[str],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for dest_sign in sorted(set(dest_signs)):
        dest_dir = asl_root / dest_sign
        if not dest_dir.is_dir():
            counts[dest_sign] = -1
            continue
        counts[dest_sign] = len(_list_mp4s(dest_dir))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move mp4 files from Jaz ASL Dataset into existing folders under ASL VIDEOS."
    )
    parser.add_argument(
        "--jaz-root",
        default=str(JAZ_ROOT_DEFAULT),
        help=f"Path to Jaz root (default: {JAZ_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--asl-root",
        default=str(ASL_ROOT_DEFAULT),
        help=f"Path to ASL root (default: {ASL_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--remove-empty-sources",
        action="store_true",
        help="After moving, remove empty source sign folders (ignoring hidden files).",
    )
    args = parser.parse_args()

    jaz_root = Path(args.jaz_root)
    asl_root = Path(args.asl_root)

    if not jaz_root.is_dir():
        raise FileNotFoundError(f"Missing Jaz root folder: {jaz_root}")
    if not asl_root.is_dir():
        raise FileNotFoundError(f"Missing ASL destination root folder: {asl_root}")

    print(f"Jaz root: {jaz_root}")
    print(f"ASL root: {asl_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")

    plan_items, dest_before, source_counts, missing_dest_signs = build_move_plan(
        jaz_root=jaz_root,
        asl_root=asl_root,
    )

    if missing_dest_signs:
        raise RuntimeError(f"Missing destination folders for: {missing_dest_signs}")

    total_mp4s_in_plan = len(plan_items)
    collision_count = sum(1 for x in plan_items if x.collision_renamed)
    print(f"\nPlanned moves: {total_mp4s_in_plan} mp4 files")
    print(f"Planned collisions renamed: {collision_count}")

    # Per destination summary
    by_dest: Dict[str, List[MovePlanItem]] = {}
    for item in plan_items:
        by_dest.setdefault(item.dest_sign, []).append(item)

    print("\nPer-destination-sign summary:")
    for dest_sign in sorted(by_dest.keys()):
        dest_items = by_dest[dest_sign]
        moved = len(dest_items)
        collisions = sum(1 for x in dest_items if x.collision_renamed)
        touched_sources = sorted(set(x.source_sign for x in dest_items))
        before = dest_before.get(dest_sign, 0)
        print(
            f"- {dest_sign}: moved={moved}, renamed_collisions={collisions}, before={before}, sources={touched_sources}"
        )

    if not args.execute:
        # Dry run: show first few file mappings so you can sanity check.
        print("\nSample mappings (first 20):")
        for item in plan_items[:20]:
            rename_note = " (RENAMED)" if item.collision_renamed else ""
            print(f"- {item.source_video} -> {item.dest_video_final}{rename_note}")
        return

    # Execute
    moved_count, renamed_count = apply_move_plan(plan_items, execute=True)
    print(f"\nMoved {moved_count} files (renamed collisions: {renamed_count})")

    # Post-merge verify for touched destination sign folders
    dest_signs_touched = list(by_dest.keys())
    after_counts = _count_mp4s_in_dest_signs(asl_root, dest_signs_touched)
    print("\nPost-merge destination counts:")
    anomalies = 0
    for dest_sign in sorted(dest_signs_touched):
        before = dest_before.get(dest_sign, 0)
        after = after_counts.get(dest_sign, -1)
        expected_after_min = before + len(by_dest[dest_sign])
        # Hash collision renaming may change number of files but should still equal by_dest size.
        ok = after == expected_after_min
        if not ok:
            anomalies += 1
        print(
            f"- {dest_sign}: before={before}, after={after}, expected_after={expected_after_min}, ok={ok}"
        )

    if anomalies:
        print(f"\nVerification: {anomalies} destination signs had unexpected counts.")
    else:
        print("\nVerification: all touched destination signs match expected counts.")

    if args.remove_empty_sources:
        maybe_remove_empty_source_dirs(jaz_root, execute=True)


if __name__ == "__main__":
    main()

