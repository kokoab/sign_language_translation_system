#!/usr/bin/env python3
"""Cache frozen selected Stage-1 temporal features for Stage-2 CTC training."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    FrozenUnifiedTemporalEncoderV17,
    load_frozen_unified_stage1,
)
from active.v17.schema_stage2_hand_mobileclip2_v17 import (
    Stage2HandMobileCLIP2V17Config,
    schema_fingerprint as hand_schema_fingerprint,
)
from scripts.encode_stage2_hand_mobileclip2_v17 import output_path as hand_output_path


LOG = logging.getLogger("cache_stage2_frozen_features_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_path(output_root: Path, crop_root: Path, crop_path: Path) -> Path:
    relative = crop_path.relative_to(crop_root)
    stem = crop_path.name.removesuffix(".stage2_rgb_v17.npz")
    return output_root / relative.parent / f"{stem}.stage2_frozen_v17.npz"


def save(path: Path, features, target_indices, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frozen_features=features,
        window_mask=np.ones(features.shape[0], dtype=np.bool_),
        target_indices=target_indices,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        if not 0 < args.mps_memory_fraction <= 0.25:
            raise ValueError("MPS memory fraction must be in (0, 0.25]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    landmark, hand, fusion, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    encoder = FrozenUnifiedTemporalEncoderV17(landmark, hand, fusion).to(device).eval()
    checkpoint_sha = sha256(args.stage1_checkpoint)
    files = sorted(args.crop_root.glob("*/*/*.stage2_rgb_v17.npz"))
    if args.limit:
        files = files[: args.limit]
    written = skipped = failed = 0
    failures = []
    peak_current = peak_driver = 0
    started = time.monotonic()
    expected_hand_schema = hand_schema_fingerprint(Stage2HandMobileCLIP2V17Config())
    for index, crop_path in enumerate(files, start=1):
        destination = cache_path(args.output_root, args.crop_root, crop_path)
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if metadata.get("stage1_checkpoint_sha256") != checkpoint_sha:
                raise ValueError(f"{destination}: selected Stage-1 checkpoint mismatch")
            skipped += 1
            continue
        try:
            hand_path = hand_output_path(args.hand_root, args.crop_root, crop_path)
            if not hand_path.exists():
                raise FileNotFoundError(hand_path)
            with np.load(crop_path, allow_pickle=False) as crop_payload:
                crop_metadata = json.loads(str(crop_payload["metadata_json"]))
                landmarks = crop_payload["landmarks"].astype(np.float32)
                targets = crop_payload["target_indices"].astype(np.int64)
            with np.load(hand_path, allow_pickle=False) as hand_payload:
                hand_metadata = json.loads(str(hand_payload["metadata_json"]))
                if hand_metadata.get("schema_fingerprint") != expected_hand_schema:
                    raise ValueError(f"{hand_path}: hand embedding schema mismatch")
                if hand_metadata.get("source_item_id") != crop_metadata.get("source_item_id"):
                    raise ValueError("crop/hand source mismatch")
                embeddings = hand_payload["embeddings"].astype(np.float32)
                valid = hand_payload["valid"].astype(np.bool_)
                boxes = hand_payload["boxes_normalized"].astype(np.float32)
            with torch.inference_mode():
                value = encoder(
                    torch.from_numpy(landmarks).unsqueeze(0).to(device),
                    torch.from_numpy(embeddings).unsqueeze(0).to(device),
                    torch.from_numpy(valid).unsqueeze(0).to(device),
                    torch.from_numpy(boxes).unsqueeze(0).to(device),
                )[0].float().cpu().numpy()
                if device.type == "mps":
                    torch.mps.synchronize()
            if value.shape != (landmarks.shape[0], 32, FROZEN_TEMPORAL_FEATURE_DIM):
                raise RuntimeError(f"unexpected frozen feature shape {value.shape}")
            if not np.isfinite(value).all():
                raise RuntimeError("non-finite frozen feature")
            metadata = {
                "format": "slt_stage2_frozen_temporal_features_v17",
                "format_version": 1,
                "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
                "stage1_checkpoint_sha256": checkpoint_sha,
                "training_manifest_sha256": crop_metadata["training_manifest_sha256"],
                "source_item_id": crop_metadata["source_item_id"],
                "source": crop_metadata["source"],
                "role": crop_metadata["role"],
                "video_sha256": crop_metadata["video_sha256"],
                "target_sequence": crop_metadata["target_sequence"],
                "window_count": int(value.shape[0]),
                "feature_dim": FROZEN_TEMPORAL_FEATURE_DIM,
                "citizen_test_accessed": False,
                "semlex_test_accessed": False,
                "local_test_accessed": False,
            }
            save(destination, value.astype(np.float16), targets, metadata)
            written += 1
            del value, landmarks, embeddings, valid, boxes
        except Exception as exc:
            failed += 1
            failures.append({"path": crop_path.as_posix(), "error": str(exc)})
            LOG.exception("failed %s", crop_path)
        finally:
            gc.collect()
            if device.type == "mps" and index % 8 == 0:
                torch.mps.empty_cache()
        if device.type == "mps":
            peak_current = max(peak_current, int(torch.mps.current_allocated_memory()))
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if index == 1 or index % 25 == 0 or index == len(files):
            LOG.info("%d/%d written=%d skipped=%d failed=%d elapsed=%.1fs", index, len(files), written, skipped, failed, time.monotonic() - started)
    result = {
        "source_archives": len(files),
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
        "stage1_checkpoint_sha256": checkpoint_sha,
        "feature_dim": FROZEN_TEMPORAL_FEATURE_DIM,
        "device": str(device),
        "mps_memory_fraction": args.mps_memory_fraction if device.type == "mps" else None,
        "peak_mps_current_bytes": peak_current,
        "peak_mps_driver_bytes": peak_driver,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    parser.add_argument("--hand-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument(
        "--stage1-checkpoint", type=Path,
        default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_frozen_features/cache.json"),
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
