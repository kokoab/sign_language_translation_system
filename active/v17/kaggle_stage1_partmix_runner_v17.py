#!/usr/bin/env python3
"""Fail-closed Kaggle runner for the controlled v17 PartMix experiment."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

import torch


ARCHIVE_NAME = "stage1_v17_partmix_trainval.tar.gz"
ARCHIVE_SHA256 = "2c257e0bbc8fc5e198b445edbc472f7b44e9959e3ec60f8d08baf21bb16e9321"
WORK_ROOT = Path("/kaggle/working/SLT")
OUTPUT = Path("/kaggle/working/stage1_v17_partmix_p50")


def find_archive() -> Path:
    matches = list(Path("/kaggle/input").glob(f"*/{ARCHIVE_NAME}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {ARCHIVE_NAME}, found {matches}")
    return matches[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_verified(archive: Path) -> None:
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise RuntimeError("training archive SHA-256 mismatch")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if "/test/" in f"/{member.name}/":
                raise RuntimeError(f"test artifact is forbidden: {member.name}")
        bundle.extractall(WORK_ROOT)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA allocation required; torch={torch.__version__}, "
            f"cuda={torch.version.cuda}, devices={torch.cuda.device_count()}"
        )
    archive = find_archive()
    extract_verified(archive)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    command = [
        sys.executable,
        "active/v17/train_stage_1_v17.py",
        "--data-root", "data/local/citizen100_v17/landmarks",
        "--manifest", "active/v17/citizen100_manifest.json",
        "--rejections", "data/local/citizen100_v17/rejections.csv",
        "--supplement-root",
        "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17",
        "--supplement-manifest",
        "data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json",
        "--approve-supplement",
        "--sampling", "class_source_balanced",
        "--partmix-probability", "0.5",
        "--epochs", "160",
        "--patience", "30",
        "--batch-size", "64",
        "--workers", "2",
        "--seed", "1701",
        "--device", "cuda",
        "--output", str(OUTPUT),
    ]
    subprocess.run(command, cwd=WORK_ROOT, check=True)


if __name__ == "__main__":
    main()
