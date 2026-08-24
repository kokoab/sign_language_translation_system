#!/usr/bin/env python3
"""CUDA-only Citizen + SemLex + finalized local deep-clean v17 challenger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

import torch


BASE_ARCHIVE_NAME = "stage1_v17_challengers_trainval_v2.tar.gz"
BASE_ARCHIVE_SHA256 = "8a41f26d3393e388d176cf7648b4c3b33797d80de013fa286883508df6c82b79"
BASE_ROOT_NAME = "stage1_v17_challengers_trainval_v2"
BASE_TREE_SHA256 = "990c6045244b00f409d735808b57025132653fe43a37073c34aa4e93ae96fad2"
LOCAL_ROOT_NAME = "local_deep_clean_v17_trainval_v1"
WORK_ROOT = Path("/kaggle/working/SLT")
LOCAL_STAGING = Path("/kaggle/working/local_deep_clean_input")
OUTPUT = Path("/kaggle/working/stage1_v17_local_deep_clean_v1")
OVERLAY_FILES = (
    "active/v17/model_v17.py",
    "active/v17/train_stage_1_v17.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root)
        if path.is_symlink() or "test" in relative.parts:
            raise RuntimeError(f"forbidden dataset member: {relative}")
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def safely_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "test" in path.parts
                or member.issym()
                or member.islnk()
            ):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        bundle.extractall(destination)


def resolve_base() -> None:
    extracted = [
        path
        for path in Path("/kaggle/input").rglob(BASE_ROOT_NAME)
        if path.is_dir()
    ]
    if len(extracted) == 1:
        if tree_sha256(extracted[0]) != BASE_TREE_SHA256:
            raise RuntimeError("extracted base training tree SHA-256 mismatch")
        shutil.copytree(extracted[0], WORK_ROOT, dirs_exist_ok=True)
        return
    if extracted:
        raise RuntimeError(f"expected one extracted base tree, found {extracted}")
    archives = [
        path
        for path in Path("/kaggle/input").rglob(BASE_ARCHIVE_NAME)
        if path.is_file()
    ]
    if len(archives) != 1 or sha256_file(archives[0]) != BASE_ARCHIVE_SHA256:
        raise RuntimeError(f"missing or mismatched base archive: {archives}")
    safely_extract(archives[0], WORK_ROOT)
    if tree_sha256(WORK_ROOT) != BASE_TREE_SHA256:
        raise RuntimeError("base training tree SHA-256 mismatch after extraction")


def load_verified_run_config() -> tuple[dict[str, object], Path]:
    candidates: list[tuple[dict[str, object], Path]] = []
    for config_path in Path("/kaggle/input").rglob("local_deep_clean_run_config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        root = config_path.parent
        if all((root / relative).is_file() for relative in OVERLAY_FILES):
            candidates.append((config, root))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one complete run configuration, found {candidates}")
    config, root = candidates[0]
    if (
        config.get("format") != "slt_v17_local_deep_clean_kaggle_run_v1"
        or config.get("citizen_test_accessed") is not False
        or config.get("semlex_test_accessed") is not False
        or config.get("local_test_included") is not False
        or config.get("checkpoint_selection_metric")
        != "citizen_official_validation_top1"
        or set(config.get("overlay_files", {})) != set(OVERLAY_FILES)
    ):
        raise RuntimeError("invalid local deep-clean run configuration")
    for relative, expected in config["overlay_files"].items():
        source = root / relative
        if source.is_symlink() or sha256_file(source) != expected:
            raise RuntimeError(f"overlay SHA-256 mismatch: {relative}")
    return config, root


def resolve_local(config: dict[str, object]) -> None:
    expected_archive = str(config["local_archive"])
    expected_archive_sha = str(config["local_archive_sha256"])
    expected_tree_sha = str(config["local_tree_sha256"])
    extracted = [
        path
        for path in Path("/kaggle/input").rglob(LOCAL_ROOT_NAME)
        if path.is_dir()
    ]
    if len(extracted) == 1:
        local_root = extracted[0]
    elif extracted:
        raise RuntimeError(f"expected one expanded local tree, found {extracted}")
    else:
        archives = [
            path
            for path in Path("/kaggle/input").rglob(expected_archive)
            if path.is_file()
        ]
        if len(archives) != 1 or sha256_file(archives[0]) != expected_archive_sha:
            raise RuntimeError(f"missing or mismatched local archive: {archives}")
        safely_extract(archives[0], LOCAL_STAGING)
        local_root = LOCAL_STAGING / LOCAL_ROOT_NAME
    if tree_sha256(local_root) != expected_tree_sha:
        raise RuntimeError("local train/validation tree SHA-256 mismatch")
    shutil.copytree(local_root, WORK_ROOT, dirs_exist_ok=True)
    for split in ("train", "val"):
        manifest = WORK_ROOT / "data/local/local_deep_clean_v17" / f"{split}_final_manifest.json"
        if sha256_file(manifest) != config[f"local_{split}_manifest_sha256"]:
            raise RuntimeError(f"copied local {split} manifest SHA-256 mismatch")


def apply_overlay(config: dict[str, object], overlay_root: Path) -> None:
    for relative, expected in config["overlay_files"].items():
        destination = WORK_ROOT / relative
        shutil.copy2(overlay_root / relative, destination)
        if sha256_file(destination) != expected:
            raise RuntimeError(f"copied overlay SHA-256 mismatch: {relative}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA allocation required")
    config, overlay_root = load_verified_run_config()
    resolve_base()
    try:
        resolve_local(config)
        apply_overlay(config, overlay_root)
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "run_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(
            [
                sys.executable,
                "active/v17/train_stage_1_v17.py",
                "--data-root", "data/local/citizen100_v17/landmarks",
                "--manifest", "active/v17/citizen100_manifest.json",
                "--rejections", "data/local/citizen100_v17/rejections.csv",
                "--supplement-root", "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17",
                "--supplement-manifest", "data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json",
                "--approve-supplement",
                "--local-root", "data/local/local_deep_clean_v17/landmarks/train",
                "--local-manifest", "data/local/local_deep_clean_v17/train_final_manifest.json",
                "--local-tiers", "owner_approved_v16_deep_clean",
                "--approve-local-supplement",
                "--local-validation-root", "data/local/local_deep_clean_v17/landmarks/val",
                "--local-validation-manifest", "data/local/local_deep_clean_v17/val_final_manifest.json",
                "--sampling", "class_source_balanced",
                "--source-probabilities", "citizen=.34,semlex=.33,local=.33",
                "--temporal-encoder", "partwise_global",
                "--part-depth", "1",
                "--full-roll-probability", "0.35",
                "--maximum-roll-degrees", "180",
                "--mild-roll-degrees", "12",
                "--epochs", "160",
                "--patience", "30",
                "--batch-size", "64",
                "--workers", "2",
                "--seed", "1701",
                "--device", "cuda",
                "--output", str(OUTPUT),
            ],
            cwd=WORK_ROOT,
            check=True,
        )
    finally:
        shutil.rmtree(WORK_ROOT, ignore_errors=True)
        shutil.rmtree(LOCAL_STAGING, ignore_errors=True)


if __name__ == "__main__":
    main()
