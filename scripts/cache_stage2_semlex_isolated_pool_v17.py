#!/usr/bin/env python3
"""Cache exact-variant SemLex-train signer trajectories for Stage-2 synthesis."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.10")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.05")

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


LOG = logging.getLogger("cache_stage2_semlex_isolated_pool_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") != "train_only" or int(manifest.get("selected_signers", 0)) != 32:
        raise ValueError("SemLex manifest is not the frozen 32-signer training selection")
    checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    if sorted(label_to_index.values()) != list(range(100)):
        raise ValueError("selected Stage-1 label map is not the locked 100 classes")

    rows = []
    for row in manifest["videos"]:
        label = str(row["canonical_label"])
        if label not in label_to_index or row.get("semlex_split") != "train":
            continue
        video_id = str(row["semlex_video_id"])
        landmark = args.landmark_root / label / f"{video_id}.v17.npz"
        hand = args.hand_root / label / f"{video_id}.hand_mobileclip2_v17.npz"
        if not landmark.exists() or not hand.exists():
            raise FileNotFoundError(f"missing SemLex multimodal input for {label}/{video_id}")
        rows.append((label, label_to_index[label], str(row["semlex_signer_id"]), video_id, landmark, hand))
    if len(rows) != int(manifest["selected_clips"]):
        raise ValueError(f"SemLex row mismatch: {len(rows)} != {manifest['selected_clips']}")

    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    landmark_model, hand_model, fusion, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    encoder = FrozenUnifiedTemporalEncoderV17(landmark_model, hand_model, fusion).to(device).eval()
    features = np.zeros((len(rows), 32, FROZEN_TEMPORAL_FEATURE_DIM), dtype=np.float16)
    targets = np.zeros(len(rows), dtype=np.int64)
    signer_ids = []
    item_ids = []
    observed_frames = np.zeros(len(rows), dtype=np.int16)
    started = time.monotonic()
    peak_driver = 0
    for index, (label, target, signer, video_id, landmark_path, hand_path) in enumerate(rows):
        with np.load(landmark_path, allow_pickle=False) as payload:
            landmarks = payload["features"].astype(np.float32)
            landmark_metadata = json.loads(str(payload["metadata_json"]))
        with np.load(hand_path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            valid = payload["valid"].astype(np.bool_)
            boxes = payload["boxes_normalized"].astype(np.float32)
        with torch.inference_mode():
            value = encoder(
                torch.from_numpy(landmarks).reshape(1, 1, 32, 61, 5).to(device),
                torch.from_numpy(embeddings).reshape(1, 1, 16, 3, 512).to(device),
                torch.from_numpy(valid).reshape(1, 1, 16, 3).to(device),
                torch.from_numpy(boxes).reshape(1, 1, 16, 3, 4).to(device),
            )[0, 0].float().cpu().numpy()
            if device.type == "mps":
                torch.mps.synchronize()
        features[index] = value.astype(np.float16)
        targets[index] = target
        signer_ids.append(f"semlex:{signer}")
        item_ids.append(f"{label}/{video_id}")
        observed_frames[index] = max(
            4, min(32, int(landmark_metadata.get("source_frames_processed", 32)))
        )
        del landmarks, embeddings, valid, boxes, value
        if (index + 1) % 16 == 0:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        if device.type == "mps":
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(rows):
            LOG.info("%d/%d elapsed=%.1fs", index + 1, len(rows), time.monotonic() - started)

    metadata = {
        "format": "slt_stage2_semlex_train_isolated_pool_v17",
        "format_version": 1,
        "source_split": "semlex_official_train_only",
        "manifest": args.manifest.as_posix(),
        "manifest_sha256": sha256(args.manifest),
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "items": len(rows),
        "signers": len(set(signer_ids)),
        "classes": len(set(targets.tolist())),
        "item_ids": item_ids,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frozen_features=features,
        target_indices=targets,
        signer_ids=np.asarray(signer_ids),
        observed_frames=observed_frames,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(args.output)
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "items": len(rows),
        "signers": len(set(signer_ids)),
        "classes": len(set(targets.tolist())),
        "device": str(device),
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
    parser.add_argument("--manifest", type=Path, default=Path("data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json"))
    parser.add_argument("--landmark-root", type=Path, default=Path("data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"))
    parser.add_argument("--hand-root", type=Path, default=Path("data/local/hand_mobileclip2_supplements_v17/semlex"))
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("data/local/stage2_v17_synthetic/semlex_train_isolated_pool.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_multivoice/semlex_pool.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.08)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
