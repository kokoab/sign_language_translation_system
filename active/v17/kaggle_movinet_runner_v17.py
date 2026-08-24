#!/usr/bin/env python3
"""Fail-closed Kaggle CUDA launcher for the v17 MoViNet experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


ARCHIVE_NAME = "movinet_v17_trainval.tar"
ARCHIVE_SHA256 = "699834265f70ae6226b4692a0058b7c1ef2ea325d935941bbdf608af2b9c8bab"
OFFLINE_WHEEL_NAME = "tf_models_official-2.20.0-py2.py3-none-any.whl"
OFFLINE_WHEEL_SHA256 = "0bf173f9ea83e1a83f983b35fe515e9d9bfd294b27538f6e8806afeb5a069150"
ARCHIVED_V17_INIT_SHA256 = "79f9bd0758fdc626fcdbc57e674afa9e154601049e3f4b959b61adba803bbc2c"
PART_SHA256 = {
    "movinet_v17_trainval.tar.part-00": "a342ec1f4093c34315955cbe096628ddf15e8449e3abb4273693567d69637a4d",
    "movinet_v17_trainval.tar.part-01": "f3261b4112802c8464aa03654664ea7f02cca41c21da79f6ab3df808141c3625",
    "movinet_v17_trainval.tar.part-02": "69efb3110c22219cb2ade8fbb8a8fb6cdbda64d371329528d4e43be26199cd43",
    "movinet_v17_trainval.tar.part-03": "27c0fea2cdb32625154c40adaa792e655acea371686080c36f57a4df202ee97a",
    "movinet_v17_trainval.tar.part-04": "6a5dd9f87c06aece8d9aecf95f5d985ac244c34431ac9b93aa9867e51eb2c12d",
    "movinet_v17_trainval.tar.part-05": "f1b742de2e0f7d3a50cd7b65591629c948ed32961ef386b868baac467a2736c3",
    "movinet_v17_trainval.tar.part-06": "b524b8ccf64c5ee9e575557ae65dfd57edc32ee54229a80ac872543d235d1512",
    "movinet_v17_trainval.tar.part-07": "c05aeaf2f9f502b944c9763883e0099b8a855b7097560cc375594c183f7e661b",
}
WORKSPACE = Path("/kaggle/working/SLT")
OUTPUT = Path("/kaggle/working/stage1_v17_sign_movinet_fusion")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_archive() -> Path:
    # Kaggle currently mounts CLI-attached private datasets below an additional
    # owner/dataset directory in some runtimes. Recurse from the read-only input
    # root instead of assuming the classic one-level mount layout.
    matches = list(Path("/kaggle/input").rglob(ARCHIVE_NAME))
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise RuntimeError(f"expected at most one {ARCHIVE_NAME}, found {matches}")

    parts = sorted(Path("/kaggle/input").rglob(f"{ARCHIVE_NAME}.part-*"))
    expected_names = list(PART_SHA256)
    if [part.name for part in parts] != expected_names:
        raise RuntimeError(
            f"multipart bundle mismatch: {[part.name for part in parts]} != {expected_names}"
        )
    assembled = Path("/kaggle/working") / ARCHIVE_NAME
    with assembled.open("wb") as destination:
        for part in parts:
            actual = sha256_file(part)
            if actual != PART_SHA256[part.name]:
                raise RuntimeError(f"checksum mismatch for {part.name}: {actual}")
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    destination.write(chunk)
    return assembled


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r") as payload:
        for member in payload.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive path: {member.name}")
        payload.extractall(destination)


def run(command: list[str], *, cwd: Path | None = None, env=None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def probe_preinstalled_environment() -> bool:
    """Log Kaggle image versions and return whether all training imports work."""
    probe = r'''import importlib.metadata as metadata
import json
import platform
packages = [
    "tensorflow", "tf-keras", "tf-models-official", "tensorflow-datasets",
    "tensorflow-metadata", "protobuf", "numpy", "scipy", "opencv-python-headless",
]
versions = {}
for package in packages:
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
print(json.dumps({"python": platform.python_version(), "packages": versions}, indent=2))
import cv2
import numpy
import scipy
import tensorflow as tf
import tf_keras
from official.projects.movinet.modeling import movinet, movinet_model
gpus = [device.name for device in tf.config.list_physical_devices("GPU")]
print(json.dumps({"tensorflow": tf.__version__, "gpus": gpus}, indent=2))
if not gpus:
    raise RuntimeError("TensorFlow found no CUDA GPU")
'''
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi is unavailable in this Kaggle allocation", flush=True)
    completed = subprocess.run([sys.executable, "-c", probe], check=False)
    return completed.returncode == 0


def neutralize_archived_package_initializer() -> None:
    """Avoid an unrelated schema import absent from the minimal training archive."""
    initializer = WORKSPACE / "active/v17/__init__.py"
    actual = sha256_file(initializer)
    if actual != ARCHIVED_V17_INIT_SHA256:
        raise RuntimeError(f"unexpected archived v17 initializer checksum: {actual}")
    initializer.write_text(
        '"""Minimal Kaggle training package; extraction/schema exports are not bundled."""\n'
    )


def main() -> None:
    started = time.time()
    archive = find_archive()
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != ARCHIVE_SHA256:
        raise RuntimeError(
            f"training bundle checksum mismatch: {actual_sha256} != {ARCHIVE_SHA256}"
        )
    safe_extract(archive, WORKSPACE)
    neutralize_archived_package_initializer()

    if probe_preinstalled_environment():
        print("Using complete preinstalled Kaggle training environment", flush=True)
    else:
        print("Preinstalled environment incomplete; installing checksum-pinned offline wheel", flush=True)
        wheel_matches = list(Path("/kaggle/input").rglob(OFFLINE_WHEEL_NAME))
        if len(wheel_matches) != 1:
            raise RuntimeError(f"expected exactly one offline Model Garden wheel, found {wheel_matches}")
        wheel = wheel_matches[0]
        wheel_sha256 = sha256_file(wheel)
        if wheel_sha256 != OFFLINE_WHEEL_SHA256:
            raise RuntimeError(
                f"offline wheel checksum mismatch: {wheel_sha256} != {OFFLINE_WHEEL_SHA256}"
            )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-cache-dir",
                "--no-deps",
                str(wheel),
            ]
        )
        if not probe_preinstalled_environment():
            raise RuntimeError("offline Model Garden wheel installed but required imports still fail")

    environment = os.environ.copy()
    environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    command = [
        sys.executable,
        "active/v17/train_stage_1_movinet_v17.py",
        "--device",
        "cuda",
        "--batch-size",
        "4",
        "--warmup-epochs",
        "5",
        "--finetune-epochs",
        "35",
        "--patience",
        "8",
        "--output",
        str(OUTPUT),
    ]
    run(command, cwd=WORKSPACE, env=environment)

    result_path = OUTPUT / "result.json"
    if not result_path.is_file():
        raise RuntimeError("training exited without result.json")
    result = json.loads(result_path.read_text())
    if result.get("test_evaluated") is not False:
        raise RuntimeError("test isolation invariant was violated")
    manifest = {
        "format": "slt_v17_movinet_kaggle_run",
        "archive_sha256": actual_sha256,
        "runner": "official Model Garden MoViNet-A0 on CUDA",
        "train_split_only": True,
        "validation_split_only": True,
        "test_evaluated": False,
        "elapsed_seconds": time.time() - started,
        "result": result,
    }
    (OUTPUT / "kaggle_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
