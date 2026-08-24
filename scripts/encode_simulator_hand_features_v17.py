#!/usr/bin/env python3
"""Encode a small simulator hand-crop suite with the bounded MobileCLIP2 image tower."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_mobileclip2_v17 import encode_archive_batch
from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
from active.v17.schema_hand_mobileclip2_v17 import HandMobileCLIP2V17Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.hand_rgb_v17.npz"))
    if len(paths) != 8:
        raise ValueError(f"expected eight simulator crop archives, found {len(paths)}")
    device = select_device(args.device)
    if device.type == "mps":
        if not 0 < args.mps_memory_fraction <= 0.25:
            raise ValueError("MPS memory fraction must be in (0, 0.25]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    model, preprocess = build_encoder(device)
    config = HandMobileCLIP2V17Config()
    encoded = encode_archive_batch(
        paths, model, preprocess, device, config,
        image_batch_size=args.image_batch_size,
    )
    rows = []
    for path, (embeddings, valid, boxes, metadata) in zip(paths, encoded):
        output = path.with_name(
            path.name.removesuffix(".hand_rgb_v17.npz") + ".hand.f32"
        )
        combined = np.concatenate((
            embeddings.astype("<f4", copy=False).reshape(-1),
            valid.astype("<f4", copy=False).reshape(-1),
            boxes.astype("<f4", copy=False).reshape(-1),
        ))
        combined.tofile(output)
        rows.append({
            "cropArchive": str(path),
            "output": str(output),
            "sha256": sha256_file(output),
            "floatCount": int(combined.size),
            "validViews": int(valid.sum()),
            "sourceItemID": metadata["source_item_id"],
        })
    result = {
        "format": "slt_v17_simulator_hand_features",
        "device": str(device),
        "clips": len(rows),
        "embeddingShape": [1, 16, 3, 512],
        "validShape": [1, 16, 3],
        "boxShape": [1, 16, 3, 4],
        "rows": rows,
        "testAccessed": False,
    }
    (args.input_dir / "hand_feature_encoding_result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
