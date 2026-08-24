#!/usr/bin/env python3
"""Build the private code overlay and Kaggle kernel for the local-data challenger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


CODE_DATASET_ID = "kokoab/slt-v17-stage1-local-deep-clean-code-v1"
LOCAL_DATASET_ID = "kokoab/slt-v17-local-deep-clean-trainval-v1"
BASE_DATASET_ID = "kokoab/slt-v17-stage1-challengers-v2"
KERNEL_ID = "kokoab/slt-v17-stage1-local-deep-clean-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("artifacts/generated/kaggle_local_deep_clean_v17_dataset_v1"),
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path("artifacts/generated/kaggle_stage1_local_deep_clean_code_v1"),
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        default=Path("artifacts/generated/kaggle_stage1_local_deep_clean_kokoab_v1"),
    )
    args = parser.parse_args()
    package_manifest_path = args.package_dir / "PACKAGE_MANIFEST.json"
    package = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    if (
        package.get("format") != "slt_v17_local_deep_clean_kaggle_package_v1"
        or package.get("classes") != 94
        or package.get("citizen_test_accessed") is not False
        or package.get("semlex_test_accessed") is not False
        or package.get("local_test_included") is not False
    ):
        raise ValueError("invalid local train/validation package manifest")
    archive = args.package_dir / str(package["archive"])
    if sha256_file(archive) != package["archive_sha256"]:
        raise ValueError("local package archive SHA-256 mismatch")

    build = args.code_dir / "build"
    build.mkdir(parents=True, exist_ok=True)
    overlay_sources = (
        Path("active/v17/model_v17.py"),
        Path("active/v17/train_stage_1_v17.py"),
    )
    overlay_hashes: dict[str, str] = {}
    for source in overlay_sources:
        destination = build / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        overlay_hashes[source.as_posix()] = sha256_file(destination)
    run_config = {
        "format": "slt_v17_local_deep_clean_kaggle_run_v1",
        "local_archive": package["archive"],
        "local_archive_sha256": package["archive_sha256"],
        "local_tree_sha256": package["tree_sha256"],
        "local_train_manifest_sha256": package["train_manifest_sha256"],
        "local_val_manifest_sha256": package["validation_manifest_sha256"],
        "local_train_clips": package["train_clips"],
        "local_validation_clips": package["validation_clips"],
        "local_classes": package["classes"],
        "overlay_files": overlay_hashes,
        "source_probabilities": {"citizen": 0.34, "semlex": 0.33, "local": 0.33},
        "checkpoint_selection_metric": "citizen_official_validation_top1",
        "promotion_gate": {
            "citizen_validation_top1_correct_minimum": 361,
            "semlex_validation_top1_correct_minimum": 839,
            "local_validation_top1_correct_minimum": 1804,
            "local_validation_total": 2896,
            "local_validation_frozen_baseline_correct": 1803,
            "citizen_roll_stress_worst_angle_top1_correct_minimum": 348,
            "all_four_required": True,
        },
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_included": False,
    }
    (build / "local_deep_clean_run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "title": "SLT v17 Stage1 Local Deep Clean Code v1",
        "id": CODE_DATASET_ID,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (build / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    args.code_dir.mkdir(parents=True, exist_ok=True)
    (args.code_dir / "dataset-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    args.kernel_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        "active/v17/kaggle_stage1_local_deep_clean_runner_v17.py",
        args.kernel_dir / "run.py",
    )
    kernel_metadata = {
        "id": KERNEL_ID,
        "title": "SLT v17 Stage1 Local Deep Clean v1",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [BASE_DATASET_ID, LOCAL_DATASET_ID, CODE_DATASET_ID],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (args.kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "code_build": str(build),
                "kernel_dir": str(args.kernel_dir),
                "run_config_sha256": sha256_file(
                    build / "local_deep_clean_run_config.json"
                ),
                "runner_sha256": sha256_file(args.kernel_dir / "run.py"),
                "overlay_files": overlay_hashes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
