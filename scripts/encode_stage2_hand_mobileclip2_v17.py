#!/usr/bin/env python3
"""Encode one bounded shard of Stage-2 hand crops with MobileCLIP2-S0."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_rgb_v17 import decode_packed_crops
from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
from active.v17.schema_hand_rgb_v17 import CROP_SIZE
from active.v17.schema_stage2_features_v17 import Stage2FeatureV17Config, schema_fingerprint as crop_schema_fingerprint
from active.v17.schema_stage2_hand_mobileclip2_v17 import (
    Stage2HandMobileCLIP2V17Config,
    schema_fingerprint,
    schema_payload,
)


LOG = logging.getLogger("encode_stage2_hand_mobileclip2_v17")


def output_path(output_root: Path, crop_root: Path, crop_path: Path) -> Path:
    relative = crop_path.relative_to(crop_root)
    stem = crop_path.name.removesuffix(".stage2_rgb_v17.npz")
    return output_root / relative.parent / f"{stem}.stage2_hand_mobileclip2_v17.npz"


def encode_one(
    path: Path, model, preprocess, device: torch.device, image_batch_size: int,
    expected_crop_schema: str,
):
    with np.load(path, allow_pickle=False) as payload:
        crop_metadata = json.loads(str(payload["metadata_json"]))
        if crop_metadata.get("schema_fingerprint") != expected_crop_schema:
            raise ValueError(f"{path}: Stage-2 crop schema mismatch")
        offsets = payload["hand_jpeg_offsets"]
        valid = payload["hand_valid"].astype(np.bool_)
        boxes = payload["hand_boxes_normalized"].astype(np.float16)
        windows = offsets.shape[0]
        flat_offsets = offsets.reshape(windows * 16, 3, 2)
        crops = decode_packed_crops(payload["hand_jpeg_blob"], flat_offsets, CROP_SIZE)
    flat_valid = valid.reshape(windows * 16, 3)
    embeddings = np.zeros((windows * 16, 3, 512), dtype=np.float32)
    locations = np.argwhere(flat_valid)
    dtype = next(model.visual.parameters()).dtype
    with torch.inference_mode():
        for start in range(0, len(locations), image_batch_size):
            batch_locations = locations[start : start + image_batch_size]
            tensors = torch.stack(
                [preprocess(Image.fromarray(crops[int(frame), int(view)])) for frame, view in batch_locations]
            ).to(device=device, dtype=dtype)
            encoded = model.encode_image(tensors, normalize=True).float().cpu().numpy()
            for value, (frame, view) in zip(encoded, batch_locations):
                embeddings[int(frame), int(view)] = value
            del tensors, encoded
            if device.type == "mps":
                torch.mps.synchronize()
    embeddings = embeddings.reshape(windows, 16, 3, 512)
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"{path}: non-finite embeddings")
    metadata = {
        "schema_fingerprint": schema_fingerprint(Stage2HandMobileCLIP2V17Config()),
        "schema": schema_payload(Stage2HandMobileCLIP2V17Config()),
        "crop_schema_fingerprint": crop_metadata["schema_fingerprint"],
        "crop_archive": path.as_posix(),
        "training_manifest_sha256": crop_metadata["training_manifest_sha256"],
        "source_item_id": crop_metadata["source_item_id"],
        "source": crop_metadata["source"],
        "role": crop_metadata["role"],
        "video_sha256": crop_metadata["video_sha256"],
        "target_sequence": crop_metadata["target_sequence"],
        "window_count": windows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    del crops, locations, flat_valid
    gc.collect()
    return embeddings.astype(np.float16), valid, boxes, metadata


def save(path: Path, embeddings, valid, boxes, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        valid=valid,
        boxes_normalized=boxes,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    files = sorted(args.crop_root.glob("*/*/*.stage2_rgb_v17.npz"))
    stop = args.file_stop or len(files)
    if args.file_start < 0 or stop <= args.file_start or stop > len(files):
        raise ValueError(f"invalid shard [{args.file_start}, {stop}) for {len(files)} files")
    files = files[args.file_start:stop]
    device = select_device(args.device)
    if device.type == "mps":
        if not 0 < args.mps_memory_fraction <= 0.25:
            raise ValueError("MPS memory fraction must be in (0, 0.25]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    model, preprocess = build_encoder(device, "fp32")
    expected = schema_fingerprint(Stage2HandMobileCLIP2V17Config())
    written = skipped = 0
    peak_current = peak_driver = 0
    started = time.monotonic()
    for index, crop_path in enumerate(files, start=1):
        destination = output_path(args.output_root, args.crop_root, crop_path)
        if destination.exists() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if metadata.get("schema_fingerprint") != expected:
                raise ValueError(f"{destination}: embedding schema mismatch")
            skipped += 1
            continue
        embeddings, valid, boxes, metadata = encode_one(
            crop_path, model, preprocess, device, args.image_batch_size,
            crop_schema_fingerprint(
                Stage2FeatureV17Config(maximum_source_frames=args.maximum_source_frames)
            ),
        )
        save(destination, embeddings, valid, boxes, metadata)
        written += 1
        del embeddings, valid, boxes, metadata
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
            peak_current = max(peak_current, int(torch.mps.current_allocated_memory()))
            peak_driver = max(peak_driver, int(torch.mps.driver_allocated_memory()))
        if index == 1 or index % 8 == 0 or index == len(files):
            LOG.info("%d/%d written=%d skipped=%d elapsed=%.1fs", index, len(files), written, skipped, time.monotonic() - started)
    return {
        "file_start": args.file_start,
        "file_stop": stop,
        "files": len(files),
        "written": written,
        "skipped": skipped,
        "device": str(device),
        "image_batch_size": args.image_batch_size,
        "mps_memory_fraction": args.mps_memory_fraction if device.type == "mps" else None,
        "maximum_source_frames": args.maximum_source_frames,
        "peak_mps_current_bytes": peak_current,
        "peak_mps_driver_bytes": peak_driver,
        "schema_fingerprint": expected,
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-stop", type=int, default=0)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.08)
    parser.add_argument("--maximum-source-frames", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
