#!/usr/bin/env python3
"""CUDA-only Kaggle runner for isolated v17 Stage-1 challengers."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

import torch


ARCHIVE_NAME = "stage1_v17_challengers_trainval_v2.tar.gz"
ARCHIVE_SHA256 = "8a41f26d3393e388d176cf7648b4c3b33797d80de013fa286883508df6c82b79"
EXTRACTED_ROOT_NAME = "stage1_v17_challengers_trainval_v2"
TREE_SHA256 = "990c6045244b00f409d735808b57025132653fe43a37073c34aa4e93ae96fad2"
WORK_ROOT = Path("/kaggle/working/SLT")
EXPERIMENT_ARGS = {
    "partmix": ["--partmix-probability", "0.5"],
    "partwise": ["--temporal-encoder", "partwise_global", "--part-depth", "1"],
    "bone": ["--bone-features"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root)
        if path.is_symlink() or "test" in relative.parts:
            raise RuntimeError(f"forbidden dataset member: {relative}")
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_repo_root() -> Path:
    extracted = [
        path for path in Path("/kaggle/input").rglob(EXTRACTED_ROOT_NAME)
        if path.is_dir()
    ]
    if len(extracted) == 1:
        if tree_sha256(extracted[0]) != TREE_SHA256:
            raise RuntimeError("extracted training tree SHA-256 mismatch")
        return extracted[0]
    if extracted:
        raise RuntimeError(f"expected one extracted training root, found {extracted}")

    matches = [
        path for path in Path("/kaggle/input").rglob(ARCHIVE_NAME)
        if path.is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {ARCHIVE_NAME}, found {matches}")
    archive = matches[0]
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
    if tree_sha256(WORK_ROOT) != TREE_SHA256:
        raise RuntimeError("extracted training tree SHA-256 mismatch")
    return WORK_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_ARGS), default="partmix")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA allocation required; torch={torch.__version__}, "
            f"cuda={torch.version.cuda}, devices={torch.cuda.device_count()}"
        )
    repo_root = resolve_repo_root()
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
        "--epochs", "160",
        "--patience", "30",
        "--batch-size", "64",
        "--workers", "2",
        "--seed", "1701",
        "--device", "cuda",
        "--output", f"/kaggle/working/stage1_v17_{args.experiment}",
        *EXPERIMENT_ARGS[args.experiment],
    ]
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
