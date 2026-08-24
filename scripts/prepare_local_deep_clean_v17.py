#!/usr/bin/env python3
"""Prepare the v16 deep-cleaned local corpus for a separate v17 experiment.

The historical manifest is a filename-to-label ledger over v16 features. This script
resolves every selected entry back to its raw video, restricts it to the current v17
vocabulary, and creates content-grouped train/validation/test splits. Exact raw-video
or historical-feature duplicates are kept in one split. Signer overlap is explicitly
allowed because the local files do not contain trustworthy signer identifiers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
TRAIN_TIER = "owner_approved_v16_deep_clean"
TRACEABLE_V16_ALIASES = {
    # The v16 extractor collapsed these source folders. Prefix checks below recover
    # only the requested source side of each merged class.
    ("EAT_FOOD", "EAT_"): "EAT",
    ("MAKE_CREATE", "MAKE_"): "MAKE",
    ("ALSO", "ALSO_SAME_"): "SAME",
    ("HOUSE", "HOUSE_"): "HOME",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def content_group_splits(
    rows: list[dict[str, object]], seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Assign duplicate-connected rows together and quarantine label conflicts."""
    disjoint = DisjointSet(len(rows))
    seen_raw: dict[str, int] = {}
    seen_feature: dict[str, int] = {}
    for index, row in enumerate(rows):
        for value, lookup in (
            (str(row["raw_sha256"]), seen_raw),
            (str(row["source_feature_sha256"]), seen_feature),
        ):
            previous = lookup.setdefault(value, index)
            disjoint.union(index, previous)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[disjoint.find(index)].append(index)

    admitted: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    for indices in grouped.values():
        labels = {str(rows[index]["canonical_label"]) for index in indices}
        if len(labels) != 1:
            for index in indices:
                quarantined.append(
                    {**rows[index], "quarantine_reason": "duplicate_content_label_conflict"}
                )
            continue
        label = next(iter(labels))
        group_identity = "|".join(
            sorted(
                {
                    *(str(rows[index]["raw_sha256"]) for index in indices),
                    *(str(rows[index]["source_feature_sha256"]) for index in indices),
                }
            )
        )
        value = int.from_bytes(
            hashlib.sha256(f"{seed}:{label}:{group_identity}".encode()).digest()[:8],
            "big",
        ) / float(2**64)
        split = "train" if value < 0.70 else "val" if value < 0.85 else "test"
        group_id = hashlib.sha256(group_identity.encode()).hexdigest()
        for index in indices:
            admitted.append(
                {
                    **rows[index],
                    "local_split": split,
                    "duplicate_group_sha256": group_id,
                    "duplicate_group_size": len(indices),
                }
            )
    return admitted, quarantined


def safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"conflicting symlink: {destination}")
        return
    if destination.exists():
        raise ValueError(f"refusing to overwrite: {destination}")
    destination.symlink_to(source.resolve())


def raw_video_index(raw_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for class_root in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        for path in sorted(class_root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            key = f"{class_root.name}_{path.stem}.npy"
            if key in output:
                raise ValueError(f"duplicate historical raw key: {key}")
            output[key] = path
    return output


def count_by_class(rows: Iterable[dict[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["canonical_label"]) for row in rows).items()))


def resolve_current_label(
    source_feature_name: str,
    source_label: str,
    current_labels: set[str],
) -> tuple[str | None, str]:
    if source_label in current_labels:
        return source_label, "exact_v16_canonical_label"
    for (old_label, required_prefix), current_label in TRACEABLE_V16_ALIASES.items():
        if source_label == old_label and source_feature_name.startswith(required_prefix):
            if current_label not in current_labels:
                raise ValueError(f"traceable alias target is not current: {current_label}")
            return current_label, "traceable_v16_source_folder_alias"
    return None, "not_in_current_vocabulary"


def write_manifest(
    path: Path,
    split: str,
    rows: list[dict[str, object]],
    common: dict[str, object],
) -> None:
    split_rows = sorted(
        (row for row in rows if row["local_split"] == split),
        key=lambda row: (str(row["canonical_label"]), str(row["item_id"])),
    )
    if split == "train":
        split_eligibility = "train_only_after_human_review"
    elif split == "val":
        split_eligibility = "validation_nonsigner_disjoint_user_approved"
    else:
        split_eligibility = "local_test_nonsigner_disjoint_unused"
    payload = {
        "format": "slt_v17_local_deep_clean_v1",
        "split": split,
        "split_eligibility": split_eligibility,
        "selected_clips": len(split_rows),
        "selected_classes": len({row["canonical_label"] for row in split_rows}),
        "class_counts": count_by_class(split_rows),
        "signer_disjoint": False,
        "signer_overlap_user_approved": True,
        "signers": [],
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        **common,
        "videos": split_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, object]:
    source_manifest = json.loads(args.clean_manifest.read_text(encoding="utf-8"))
    current_manifest = json.loads(args.v17_manifest.read_text(encoding="utf-8"))
    current_items = {
        str(item["canonical_label"]): item for item in current_manifest["classes"]
    }
    raw_index = raw_video_index(args.raw_root)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for source_feature_name, label_value in source_manifest.items():
        source_label = str(label_value)
        label, label_lineage = resolve_current_label(
            str(source_feature_name), source_label, set(current_items)
        )
        if label is None:
            continue
        raw_path = raw_index.get(str(source_feature_name))
        source_feature_path = args.source_feature_root / str(source_feature_name)
        if raw_path is None or not source_feature_path.is_file():
            missing.append(str(source_feature_name))
            continue
        item_id = Path(str(source_feature_name)).stem
        manifest_item = current_items[label]
        rows.append(
            {
                "canonical_label": label,
                "class_index": int(manifest_item["class_index"]),
                "citizen_raw_gloss": str(manifest_item["citizen_raw_gloss"]),
                "citizen_asl_lex_code": str(manifest_item["citizen_asl_lex_code"]),
                "variant_alignment": (
                    "canonical_text_matches_pinned_raw"
                    if label == str(manifest_item["citizen_raw_gloss"])
                    else "current_v16_canonical_lineage_variant_unverified"
                ),
                "item_id": item_id,
                "source_v16_label": source_label,
                "source_label_lineage": label_lineage,
                "source_feature_name": str(source_feature_name),
                "source_feature_path": str(source_feature_path),
                "source_feature_sha256": sha256_file(source_feature_path),
                "source_raw_path": str(raw_path),
                "raw_sha256": sha256_file(raw_path),
                "consensus_tier": TRAIN_TIER,
                "training_eligible": True,
                "validation_eligible": True,
            }
        )
    if missing:
        raise FileNotFoundError(f"failed to resolve {len(missing)} rows; first={missing[:3]}")
    admitted, quarantined = content_group_splits(rows, args.seed)

    for row in admitted:
        split = str(row["local_split"])
        label = str(row["canonical_label"])
        item_id = str(row["item_id"])
        source = Path(str(row["source_raw_path"]))
        staged = args.output_root / "raw" / split / label / f"{item_id}{source.suffix.lower()}"
        feature = args.output_root / "landmarks" / split / label / f"{item_id}.v17.npz"
        row["raw_path"] = str(staged)
        row["feature_path"] = str(feature)
        row["training_eligible"] = split == "train"
        row["validation_eligible"] = split == "val"
        if args.materialize_symlinks:
            safe_link(source, staged)

    common = {
        "source_name": "local_v16_deep_clean",
        "source_clean_manifest": str(args.clean_manifest),
        "source_clean_manifest_sha256": sha256_file(args.clean_manifest),
        "source_clean_manifest_clips": len(source_manifest),
        "source_clean_manifest_classes": len(set(map(str, source_manifest.values()))),
        "source_feature_root": str(args.source_feature_root),
        "v17_manifest": str(args.v17_manifest),
        "v17_manifest_sha256": sha256_file(args.v17_manifest),
        "split_policy": "duplicate-connected content groups; sha256(seed:label:group); 70/15/15",
        "split_seed": args.seed,
        "checkpoint_selection_policy": "Citizen official validation top1 only; local validation diagnostic",
        "local_label_approval": "project_owner_approved_current_vocabulary_reuse",
        "known_limitation": (
            "local signer IDs and ASL-LEX variant IDs are absent; signer overlap is allowed "
            "and non-text-equal variants remain explicitly marked"
        ),
    }
    for split in ("train", "val", "test"):
        write_manifest(args.output_root / f"{split}_manifest.json", split, admitted, common)
    quarantine_payload = {
        "format": "slt_v17_local_deep_clean_quarantine_v1",
        "rows": quarantined,
        "count": len(quarantined),
    }
    (args.output_root / "quarantine.json").write_text(
        json.dumps(quarantine_payload, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "format": "slt_v17_local_deep_clean_preparation_v1",
        "source_clean_clips": len(source_manifest),
        "source_clean_classes": len(set(map(str, source_manifest.values()))),
        "current_vocabulary_overlap_clips_before_dedup_conflict": len(rows),
        "current_vocabulary_overlap_classes": len({row["canonical_label"] for row in rows}),
        "admitted_clips": len(admitted),
        "quarantined_conflicting_duplicate_clips": len(quarantined),
        "split_counts": Counter(str(row["local_split"]) for row in admitted),
        "split_class_counts": {
            split: len(
                {row["canonical_label"] for row in admitted if row["local_split"] == split}
            )
            for split in ("train", "val", "test")
        },
        "variant_alignment_counts": Counter(
            str(row["variant_alignment"]) for row in admitted
        ),
        "materialized_symlinks": args.materialize_symlinks,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
    }
    serializable = json.loads(json.dumps(summary, default=dict))
    (args.output_root / "preparation_summary.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    return serializable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=Path("active/v16/manifest_v16_files_deep_cleaned.json"),
    )
    parser.add_argument(
        "--source-feature-root", type=Path, default=Path("src_v16/ASL_landmarks_v16")
    )
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw_videos/ASL VIDEOS")
    )
    parser.add_argument(
        "--v17-manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/local_deep_clean_v17")
    )
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--materialize-symlinks", action="store_true")
    return parser


def main() -> None:
    summary = build(build_parser().parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
