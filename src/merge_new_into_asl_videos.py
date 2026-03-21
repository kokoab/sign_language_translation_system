import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


NEW_ROOT_DEFAULT = Path("data/raw_videos/new")
ASL_ROOT_DEFAULT = Path("data/raw_videos/ASL VIDEOS")

SOURCE_EXTENSIONS = {".mp4"}


def _sha_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _is_hidden_path(p: Path) -> bool:
    return any(part.startswith(".") for part in p.parts)


def _iter_mp4s_recursive(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if _is_hidden_path(p):
            continue
        if p.suffix.lower() in SOURCE_EXTENSIONS:
            yield p


def _list_mp4s_direct(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    out: List[Path] = []
    for f in folder.iterdir():
        if f.name.startswith("."):
            continue
        if f.is_file() and f.suffix.lower() in SOURCE_EXTENSIONS:
            out.append(f)
    out.sort(key=lambda x: str(x))
    return out


def _default_dest_sign(source_sign: str, asl_root: Path, mapping: Dict[str, str]) -> str:
    """
    If ASL folder exists for source_sign, keep identity.
    Otherwise, use user-provided mapping.
    """
    cand = asl_root / source_sign
    if cand.is_dir():
        return source_sign

    if source_sign in mapping:
        dest_sign = mapping[source_sign]
        if not (asl_root / dest_sign).is_dir():
            raise FileNotFoundError(
                f"Mapping provided for '{source_sign}' -> '{dest_sign}', "
                f"but destination folder does not exist: {(asl_root / dest_sign)}"
            )
        return dest_sign

    raise KeyError(f"Missing destination folder for '{source_sign}'")


def _choose_dest_filename(
    src_video: Path,
    dest_dir: Path,
    source_sign: str,
    taken_names: Set[str],
) -> Tuple[Path, bool]:
    """
    Collision-safe naming for multiple incoming files.
    Reserves names via taken_names set.
    """
    direct = dest_dir / src_video.name
    if not direct.exists() and direct.name not in taken_names:
        taken_names.add(direct.name)
        return direct, False

    base = src_video.stem
    suffix = src_video.suffix
    short_hash = _sha_short(str(src_video))

    renamed = dest_dir / f"{base}__from_NEW_{source_sign}__{short_hash}{suffix}"
    if not renamed.exists() and renamed.name not in taken_names:
        taken_names.add(renamed.name)
        return renamed, True

    stem2 = f"{base}__from_NEW_{source_sign}__{short_hash}"
    for i in range(1, 1000):
        alt = dest_dir / f"{stem2}__{i}{suffix}"
        if not alt.exists() and alt.name not in taken_names:
            taken_names.add(alt.name)
            return alt, True

    raise RuntimeError(f"Could not find collision-free filename for {src_video}")


@dataclass(frozen=True)
class MovePlanItem:
    source_sign: str
    dest_sign: str
    source_video: Path
    dest_video_final: Path
    collision_renamed: bool


def build_plan(
    new_root: Path,
    asl_root: Path,
    mapping: Dict[str, str],
    require_all_dest_present: bool,
) -> Tuple[List[MovePlanItem], Dict[str, int], List[str]]:
    """
    Returns:
      - plan items
      - destination mp4 count before execution (direct mp4 items)
      - missing destination sign list
    """
    mp4s = sorted(list(_iter_mp4s_recursive(new_root)), key=lambda p: str(p))

    taken_by_dest: Dict[str, Set[str]] = {}
    dest_before: Dict[str, int] = {}
    touched_dest_signs: Set[str] = set()
    missing_signs: Set[str] = set()

    plan: List[MovePlanItem] = []

    def taken_for(dest_sign: str) -> Set[str]:
        if dest_sign not in taken_by_dest:
            dest_dir = asl_root / dest_sign
            taken_names = set([p.name for p in _list_mp4s_direct(dest_dir)])
            taken_by_dest[dest_sign] = taken_names
        return taken_by_dest[dest_sign]

    # Build actual move plan.
    # If a destination sign is missing, we:
    # - in dry-run: record it and skip those videos
    # - in execute: record it and abort by returning an empty plan
    for src_video in mp4s:
        source_sign = src_video.parent.name
        try:
            dest_sign = _default_dest_sign(
                source_sign, asl_root=asl_root, mapping=mapping
            )
        except KeyError:
            missing_signs.add(source_sign)
            continue

        dest_dir = asl_root / dest_sign
        if not dest_dir.is_dir():
            raise FileNotFoundError(f"Destination folder missing: {dest_dir}")

        taken = taken_for(dest_sign)
        dest_final, renamed = _choose_dest_filename(
            src_video=src_video,
            dest_dir=dest_dir,
            source_sign=source_sign,
            taken_names=taken,
        )

        plan.append(
            MovePlanItem(
                source_sign=source_sign,
                dest_sign=dest_sign,
                source_video=src_video,
                dest_video_final=dest_final,
                collision_renamed=renamed,
            )
        )
        touched_dest_signs.add(dest_sign)

    if require_all_dest_present and missing_signs:
        # Execute mode: abort safely (no moves) until the caller provides mappings.
        return [], {}, sorted(missing_signs)

    for dest_sign in touched_dest_signs:
        dest_before[dest_sign] = len(_list_mp4s_direct(asl_root / dest_sign))

    return plan, dest_before, sorted(missing_signs)


def execute_plan(plan: List[MovePlanItem]) -> None:
    for item in plan:
        item.dest_video_final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.source_video), str(item.dest_video_final))


def count_remaining_mp4s(root: Path) -> int:
    c = 0
    for _ in _iter_mp4s_recursive(root):
        c += 1
    return c


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge (MOVE) mp4 videos from data/raw_videos/new into existing ASL VIDEOS folders."
    )
    parser.add_argument("--new-root", default=str(NEW_ROOT_DEFAULT))
    parser.add_argument("--asl-root", default=str(ASL_ROOT_DEFAULT))
    parser.add_argument(
        "--mapping-json",
        default="{}",
        help='JSON dict mapping missing source_sign -> dest_sign, e.g. \'{"OLD":"NEW"}\'',
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="If set, performs the MOVE. In execute mode, missing signs require --mapping-json.",
    )
    parser.add_argument(
        "--remove-empty-sources",
        action="store_true",
        help="After successful execute, remove empty leaf folders under new.",
    )
    args = parser.parse_args()

    new_root = Path(args.new_root)
    asl_root = Path(args.asl_root)

    if not new_root.is_dir():
        raise FileNotFoundError(f"Missing new root: {new_root}")
    if not asl_root.is_dir():
        raise FileNotFoundError(f"Missing ASL destination root: {asl_root}")

    mapping = json.loads(args.mapping_json)
    if not isinstance(mapping, dict):
        raise ValueError("--mapping-json must be a JSON object/dict")

    mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"new root: {new_root}")
    print(f"ASL root: {asl_root}")
    print(f"Mode: {mode}")

    require_all_dest_present = True if args.execute else False
    # In dry-run, we don't require mapping completeness; we only list missing.
    plan, dest_before, missing = build_plan(
        new_root=new_root,
        asl_root=asl_root,
        mapping=mapping,
        require_all_dest_present=require_all_dest_present,
    )

    if args.execute and missing:
        raise RuntimeError(
            "Missing destination folders for execute mode: "
            + ", ".join(missing)
            + ". Provide --mapping-json."
        )

    if missing and not args.execute:
        # Dry-run mode: missing can exist; we skip those videos in the planned move count.
        print("\nMissing destination sign folders (dry-run skips these videos):")
        for s in missing:
            print(f"- {s}")
        print("\nRe-run with --execute and a populated --mapping-json to include them safely.")

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
        moved = len(by_dest[dest_sign])
        renamed_collisions = sum(1 for x in by_dest[dest_sign] if x.collision_renamed)
        before = dest_before.get(dest_sign, 0)
        sources = sorted(set(x.source_sign for x in by_dest[dest_sign]))
        print(
            f"- {dest_sign}: moved={moved}, renamed_collisions={renamed_collisions}, "
            f"before={before}, sources={sources}"
        )

    if not args.execute:
        print("\nSample mappings (first 20):")
        for item in plan[:20]:
            tag = " (RENAMED)" if item.collision_renamed else ""
            print(f"- {item.source_video} -> {item.dest_video_final}{tag}")
        return

    # Execute MOVE
    execute_plan(plan)

    # Post verification
    remaining = count_remaining_mp4s(new_root)
    if remaining != 0:
        print(f"\nWARNING: Remaining mp4 files under new after move: {remaining}")
    else:
        print("\nVerification: new has 0 remaining mp4 files.")

    print("\nPost-merge destination counts:")
    anomalies = 0
    for dest_sign in touched:
        after = len(_list_mp4s_direct(asl_root / dest_sign))
        before = dest_before.get(dest_sign, 0)
        moved = len(by_dest[dest_sign])
        expected_after = before + moved
        ok = after == expected_after
        if not ok:
            anomalies += 1
        print(
            f"- {dest_sign}: before={before}, after={after}, expected_after={expected_after}, ok={ok}"
        )

    if anomalies:
        print(f"\nVerification: {anomalies} destination signs had unexpected counts.")
    else:
        print("\nVerification: all touched destination signs match expected counts.")

    # Optional cleanup of empty source folders
    if args.remove_empty_sources:
        # Remove only empty leaf parent dirs of mp4s.
        leaf_dirs: Set[Path] = set()
        for src_video in plan:
            leaf_dirs.add(src_video.source_video.parent)
        removed = 0
        for d in sorted(leaf_dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                non_hidden = [x for x in d.iterdir() if not x.name.startswith(".")]
            except FileNotFoundError:
                continue
            if len(non_hidden) == 0:
                try:
                    d.rmdir()
                    removed += 1
                except OSError:
                    pass
        print(f"\nRemoved empty source directories under new: {removed}")


if __name__ == "__main__":
    main()

