import argparse
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


MARIAH_ROOT_DEFAULT = Path("data/raw_videos/mariah")
ASL_ROOT_DEFAULT = Path("data/raw_videos/ASL VIDEOS")

SOURCE_EXTENSIONS = {".mp4"}

# Mapping rules for mariah leaf sign folder names that don't exist in ASL VIDEOS.
# Default behavior is identity if the destination folder exists in ASL.
REMAP_CLASS: Dict[str, str] = {
    # numbers
    "FIFTHTEEN": "FIFTEEN",
    "FOURTHTEEN": "FOURTEEN",
    # variants
    "HIGH (1)": "HIGH",
    "LOW (1)": "LOW",
    # pronouns/sign variants
    "RIGHT": "RIGHT (ADJECTIVE)",
    "SHE_HE": "HE_SHE",
    "THANK_YOU": "THANKYOU",
    "STORE": "MARKET_STORE",
    "WE_US": "US_WE",
    "KYU": "Q",
    # literal folder named 'c'
    "c": "C",
    # consolidation target
    "HIS_HER": "HIS_HER",
}

SPECIAL_CONSOLIDATION_DEST = "HIS_HER"
SPECIAL_SOURCE_HIS = "HIS"
SPECIAL_SOURCE_HER = "HER"

SOURCE_TIERS_HINT = [
    "Tier 1",
    "Tier 2",
    "Tier 3",
]


def _sha_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _is_hidden_path(p: Path) -> bool:
    return any(part.startswith(".") for part in p.parts)


def _list_mp4s_recursive(root: Path) -> Iterable[Path]:
    """
    Recursively yield .mp4 files under `root`.
    Leaf sign is defined as parent directory name of each mp4.
    """
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_hidden_path(p):
            continue
        if p.suffix.lower() in SOURCE_EXTENSIONS:
            yield p


def _mp4s_in_dir(dir_path: Path) -> List[Path]:
    if not dir_path.is_dir():
        return []
    out: List[Path] = []
    for p in dir_path.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS:
            out.append(p)
    out.sort(key=lambda x: str(x))
    return out


@dataclass(frozen=True)
class MovePlanItem:
    source_video: Path
    dest_sign: str
    dest_video_final: Path
    collision_renamed: bool


def _choose_dest_name(
    src_video: Path,
    dest_dir: Path,
    dest_sign: str,
    source_sign: str,
    taken_names: Set[str],
) -> Tuple[Path, bool]:
    """
    Choose a collision-free destination filename.
    Collision resolution accounts for:
      - existing files already in `dest_dir`
      - filenames already reserved in `taken_names` for items planned earlier
    """
    candidate = dest_dir / src_video.name
    if (not candidate.exists()) and (candidate.name not in taken_names):
        taken_names.add(candidate.name)
        return candidate, False

    base = src_video.stem
    suffix = src_video.suffix
    short_hash = _sha_short(str(src_video))

    # Deterministic first rename attempt.
    renamed = dest_dir / f"{base}__from_MARIAH_{source_sign}__{short_hash}{suffix}"
    if (not renamed.exists()) and (renamed.name not in taken_names):
        taken_names.add(renamed.name)
        return renamed, True

    # Defensive: if hash-based rename still collides, append counter.
    stem2 = f"{base}__from_MARIAH_{source_sign}__{short_hash}"
    for i in range(1, 1000):
        alt = dest_dir / f"{stem2}__{i}{suffix}"
        if (not alt.exists()) and (alt.name not in taken_names):
            taken_names.add(alt.name)
            return alt, True

    raise RuntimeError(f"Could not find collision-free name for {src_video}")


def map_sign(source_sign: str, asl_root: Path) -> str:
    """
    Map a mariah leaf sign to an ASL VIDEOS destination sign folder.
    """
    if source_sign == SPECIAL_CONSOLIDATION_DEST:
        return SPECIAL_CONSOLIDATION_DEST

    if source_sign in REMAP_CLASS:
        dest = REMAP_CLASS[source_sign]
        return dest

    dest_dir = asl_root / source_sign
    if dest_dir.is_dir():
        return source_sign

    # For mariah, only known non-existing signs should hit here.
    raise KeyError(f"Missing destination sign folder '{source_sign}' in ASL VIDEOS.")


def build_move_plan(mariah_root: Path, asl_root: Path) -> Tuple[List[MovePlanItem], Dict[str, int]]:
    """
    Builds a collision-safe move plan across:
      - consolidation moves from ASL HIS/HER -> ASL HIS_HER
      - mariah moves into their mapped destination folders
    """
    # Ensure destination for consolidation exists logically (created in execute)
    touched_dest_signs: Set[str] = set()
    plan: List[MovePlanItem] = []

    # Initialize taken_names per destination sign based on current filesystem.
    # We'll also reserve names as we build the plan.
    taken_names: Dict[str, Set[str]] = {}

    def taken_for(dest_sign: str) -> Set[str]:
        if dest_sign not in taken_names:
            dest_dir = asl_root / dest_sign
            names: Set[str] = set()
            if dest_dir.is_dir():
                for f in _mp4s_in_dir(dest_dir):
                    names.add(f.name)
            taken_names[dest_sign] = names
        return taken_names[dest_sign]

    # 1) Consolidate existing ASL HIS/HER -> ASL HIS_HER
    his_dir = asl_root / SPECIAL_SOURCE_HIS
    her_dir = asl_root / SPECIAL_SOURCE_HER
    dest_dir = asl_root / SPECIAL_CONSOLIDATION_DEST

    # Note: In dry-run we still include these moves so totals are visible.
    for src_video in _mp4s_in_dir(his_dir):
        item_dest, renamed = _choose_dest_name(
            src_video=src_video,
            dest_dir=dest_dir,
            dest_sign=SPECIAL_CONSOLIDATION_DEST,
            source_sign=SPECIAL_SOURCE_HIS,
            taken_names=taken_for(SPECIAL_CONSOLIDATION_DEST),
        )
        plan.append(
            MovePlanItem(
                source_video=src_video,
                dest_sign=SPECIAL_CONSOLIDATION_DEST,
                dest_video_final=item_dest,
                collision_renamed=renamed,
            )
        )
        touched_dest_signs.add(SPECIAL_CONSOLIDATION_DEST)

    for src_video in _mp4s_in_dir(her_dir):
        item_dest, renamed = _choose_dest_name(
            src_video=src_video,
            dest_dir=dest_dir,
            dest_sign=SPECIAL_CONSOLIDATION_DEST,
            source_sign=SPECIAL_SOURCE_HER,
            taken_names=taken_for(SPECIAL_CONSOLIDATION_DEST),
        )
        plan.append(
            MovePlanItem(
                source_video=src_video,
                dest_sign=SPECIAL_CONSOLIDATION_DEST,
                dest_video_final=item_dest,
                collision_renamed=renamed,
            )
        )
        touched_dest_signs.add(SPECIAL_CONSOLIDATION_DEST)

    # 2) Move mariah mp4 files into ASL VIDEOS (including HIS_HER)
    # Deterministic ordering to make plan stable.
    mariah_videos = sorted(list(_list_mp4s_recursive(mariah_root)), key=lambda p: str(p))
    for src_video in mariah_videos:
        source_sign = src_video.parent.name
        dest_sign = map_sign(source_sign, asl_root=asl_root)
        dest_dir = asl_root / dest_sign

        item_dest, renamed = _choose_dest_name(
            src_video=src_video,
            dest_dir=dest_dir,
            dest_sign=dest_sign,
            source_sign=source_sign,
            taken_names=taken_for(dest_sign),
        )

        plan.append(
            MovePlanItem(
                source_video=src_video,
                dest_sign=dest_sign,
                dest_video_final=item_dest,
                collision_renamed=renamed,
            )
        )
        touched_dest_signs.add(dest_sign)

    # Destination counts before execution (direct mp4 items in folder)
    dest_before: Dict[str, int] = {}
    for dest_sign in touched_dest_signs:
        dest_before[dest_sign] = len(_mp4s_in_dir(asl_root / dest_sign))

    return plan, dest_before


def apply_plan(plan: List[MovePlanItem], execute: bool) -> None:
    if not execute:
        return
    for item in plan:
        item.dest_video_final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.source_video), str(item.dest_video_final))


def _count_remaining_mp4s(root: Path) -> int:
    c = 0
    for p in _list_mp4s_recursive(root):
        c += 1
    return c


def _remove_empty_leaf_dirs_under(root: Path) -> int:
    """
    Remove empty directories directly in the tree (does not remove if not empty).
    """
    removed = 0
    # Remove deepest dirs first.
    all_dirs = [p for p in root.rglob("*") if p.is_dir() and not _is_hidden_path(p)]
    all_dirs.sort(key=lambda p: len(p.parts), reverse=True)
    for d in all_dirs:
        try:
            non_hidden = [x for x in d.iterdir() if not x.name.startswith(".")]
            if len(non_hidden) == 0:
                d.rmdir()
                removed += 1
        except OSError:
            pass
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge (MOVE) mp4 videos from mariah into ASL VIDEOS with remaps and HIS/HER consolidation."
    )
    parser.add_argument(
        "--mariah-root",
        default=str(MARIAH_ROOT_DEFAULT),
        help=f"Path to mariah root (default: {MARIAH_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--asl-root",
        default=str(ASL_ROOT_DEFAULT),
        help=f"Path to ASL destination root (default: {ASL_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually MOVE files. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--remove-empty-sources",
        action="store_true",
        help="After execution, remove empty leaf sign directories under mariah.",
    )
    args = parser.parse_args()

    mariah_root = Path(args.mariah_root)
    asl_root = Path(args.asl_root)

    if not mariah_root.is_dir():
        raise FileNotFoundError(f"Missing mariah root folder: {mariah_root}")
    if not asl_root.is_dir():
        raise FileNotFoundError(f"Missing ASL destination root folder: {asl_root}")

    print(f"mariah root: {mariah_root}")
    print(f"ASL root: {asl_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")

    plan, dest_before = build_move_plan(mariah_root=mariah_root, asl_root=asl_root)

    total = len(plan)
    collisions = sum(1 for x in plan if x.collision_renamed)
    touched = sorted(set(x.dest_sign for x in plan))
    print(f"\nPlanned moves: {total} mp4 files")
    print(f"Planned collisions renamed: {collisions}")
    print(f"Touched destination signs: {len(touched)}")

    by_dest: Dict[str, List[MovePlanItem]] = {}
    for item in plan:
        by_dest.setdefault(item.dest_sign, []).append(item)

    print("\nPer-destination-sign summary:")
    for dest_sign in sorted(by_dest.keys()):
        items = by_dest[dest_sign]
        moved = len(items)
        renamed_collisions = sum(1 for x in items if x.collision_renamed)
        before = dest_before.get(dest_sign, 0)
        sources = sorted(set(x.source_video.parent.name for x in items))
        print(
            f"- {dest_sign}: moved={moved}, renamed_collisions={renamed_collisions}, before={before}, sources={sources}"
        )

    # Dry-run sample
    if not args.execute:
        print("\nSample mappings (first 20):")
        for item in plan[:20]:
            tag = " (RENAMED)" if item.collision_renamed else ""
            print(f"- {item.source_video} -> {item.dest_video_final}{tag}")
        return

    # Execute
    apply_plan(plan, execute=True)

    # Post-move verification
    remaining = _count_remaining_mp4s(mariah_root)
    if remaining != 0:
        print(f"\nWARNING: Remaining mp4 files under mariah after move: {remaining}")
    else:
        print("\nVerification: mariah has 0 remaining mp4 files.")

    print("\nPost-merge destination counts:")
    anomalies = 0
    for dest_sign in sorted(by_dest.keys()):
        before = dest_before.get(dest_sign, 0)
        moved = len(by_dest[dest_sign])
        after = len(_mp4s_in_dir(asl_root / dest_sign))
        expected_after = before + moved
        ok = (after == expected_after)
        if not ok:
            anomalies += 1
        print(f"- {dest_sign}: before={before}, after={after}, expected_after={expected_after}, ok={ok}")

    # Verify HIS/HER consolidation
    his_left = len(_mp4s_in_dir(asl_root / SPECIAL_SOURCE_HIS))
    her_left = len(_mp4s_in_dir(asl_root / SPECIAL_SOURCE_HER))
    print(f"\nConsolidation check: HIS mp4s={his_left}, HER mp4s={her_left}")

    if anomalies:
        print(f"\nVerification: {anomalies} destination signs had unexpected counts.")
    else:
        print("\nVerification: all touched destination signs match expected counts.")

    if args.remove_empty_sources:
        removed = _remove_empty_leaf_dirs_under(mariah_root)
        print(f"\nRemoved empty source directories under mariah: {removed}")


if __name__ == "__main__":
    main()

