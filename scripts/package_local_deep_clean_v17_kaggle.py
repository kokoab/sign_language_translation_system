#!/usr/bin/env python3
"""Package only finalized local train/validation v17 features for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


ARCHIVE_NAME = "local_deep_clean_v17_trainval_v1.tar.gz"
ARCHIVE_ROOT = PurePosixPath("local_deep_clean_v17_trainval_v1")
DATASET_ID = "kokoab/slt-v17-local-deep-clean-trainval-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(members: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(members):
        relative_path = path.relative_to(Path.cwd()) if path.is_absolute() else path
        relative = relative_path.as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def load_final_manifest(path: Path, split: str) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "slt_v17_local_deep_clean_final_v1"
        or manifest.get("split") != split
        or manifest.get("extraction_complete") is not True
        or manifest.get("citizen_test_accessed") is not False
        or manifest.get("semlex_test_accessed") is not False
        or int(manifest.get("selected_classes", -1)) != 94
    ):
        raise ValueError(f"invalid finalized {split} manifest")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or int(manifest.get("selected_clips", -1)) != len(videos):
        raise ValueError(f"invalid finalized {split} row count")
    for row in videos:
        eligibility_valid = (
            row.get("training_eligible") is True
            and row.get("validation_eligible") is False
            if split == "train"
            else row.get("training_eligible") is False
            and row.get("validation_eligible") is True
        )
        if row.get("local_split") != split or not eligibility_valid:
            raise ValueError(f"{split} manifest contains a split-contract violation")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/local/local_deep_clean_v17")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/generated/kaggle_local_deep_clean_v17_dataset_v1"),
    )
    args = parser.parse_args()
    manifests = {
        split: args.root / f"{split}_final_manifest.json"
        for split in ("train", "val")
    }
    loaded = {
        split: load_final_manifest(path, split)
        for split, path in manifests.items()
    }
    members: list[Path] = []
    for split, manifest in loaded.items():
        for row in manifest["videos"]:
            feature = args.root / "landmarks" / split / str(row["canonical_label"]) / f"{row['item_id']}.v17.npz"
            if not feature.is_file() or feature.is_symlink():
                raise FileNotFoundError(f"missing regular finalized feature: {feature}")
            members.append(feature)
    members.extend(manifests.values())
    members.append(args.root / "finalization_summary.json")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as bundle:
        for member in sorted(members):
            relative = (
                member.relative_to(Path.cwd()) if member.is_absolute() else member
            )
            if "test" in relative.parts or member.is_symlink():
                raise ValueError(f"forbidden package member: {relative}")
            bundle.add(member, arcname=str(ARCHIVE_ROOT / relative), recursive=False)
    package_manifest = {
        "format": "slt_v17_local_deep_clean_kaggle_package_v1",
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_root": str(ARCHIVE_ROOT),
        "tree_sha256": tree_sha256(members),
        "member_count": len(members),
        "train_clips": loaded["train"]["selected_clips"],
        "validation_clips": loaded["val"]["selected_clips"],
        "classes": 94,
        "train_manifest_sha256": sha256_file(manifests["train"]),
        "validation_manifest_sha256": sha256_file(manifests["val"]),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_included": False,
    }
    (args.output_dir / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "SLT v17 Local Deep Clean Trainval v1",
                "id": DATASET_ID,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(package_manifest, indent=2))


if __name__ == "__main__":
    main()
