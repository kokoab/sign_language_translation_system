#!/usr/bin/env python3
"""Encode Apple-guided high-resolution hand crops with frozen MobileCLIP2-S0."""

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
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_hand_rgb_v17 import decode_packed_crops
    from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
    from active.v17.schema_hand_mobileclip2_v17 import (
        HandMobileCLIP2V17Config,
        schema_fingerprint,
        schema_payload,
    )
    from active.v17.schema_hand_rgb_v17 import CROP_SIZE, HandRGBV17Config, schema_fingerprint as crop_schema_fingerprint
else:
    from .extract_hand_rgb_v17 import decode_packed_crops
    from .extract_mobileclip2_v17 import build_encoder, select_device
    from .schema_hand_mobileclip2_v17 import HandMobileCLIP2V17Config, schema_fingerprint, schema_payload
    from .schema_hand_rgb_v17 import CROP_SIZE, HandRGBV17Config, schema_fingerprint as crop_schema_fingerprint


LOG = logging.getLogger("hand_mobileclip2_v17")
ALLOWED_SPLITS = ("train", "val")


def load_crop_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        if metadata.get("schema_fingerprint") != crop_schema_fingerprint(HandRGBV17Config()):
            raise ValueError(f"{path}: hand RGB crop schema mismatch")
        crops = decode_packed_crops(payload["jpeg_blob"], payload["jpeg_offsets"], CROP_SIZE)
        valid = payload["valid"].astype(np.bool_)
        boxes = payload["boxes_normalized"].astype(np.float32)
    return crops, valid, boxes, metadata


def encode_archive(path: Path, model, preprocess, device: torch.device, config: HandMobileCLIP2V17Config):
    return encode_archive_batch(
        [path], model, preprocess, device, config, image_batch_size=32
    )[0]


def encode_archive_batch(
    paths: list[Path],
    model,
    preprocess,
    device: torch.device,
    config: HandMobileCLIP2V17Config,
    *,
    image_batch_size: int = 128,
):
    """Encode several archives per accelerator submission without changing outputs."""
    if not paths or image_batch_size < 1:
        raise ValueError("archive paths and a positive image batch are required")
    records = [load_crop_archive(path) for path in paths]
    embeddings = [
        np.zeros(
            (config.sequence_length, len(config.views), config.embedding_dim),
            dtype=np.float32,
        )
        for _ in paths
    ]
    locations: list[tuple[int, int, int]] = []
    images: list[np.ndarray] = []
    for archive_index, (crops, valid, _, _) in enumerate(records):
        for frame_index, view_index in np.argwhere(valid):
            locations.append((archive_index, int(frame_index), int(view_index)))
            images.append(crops[frame_index, view_index])
    with torch.inference_mode():
        for start in range(0, len(images), image_batch_size):
            stop = min(start + image_batch_size, len(images))
            tensors = torch.stack(
                [preprocess(Image.fromarray(image)) for image in images[start:stop]]
            ).to(
                device=device,
                dtype=next(model.visual.parameters()).dtype,
            )
            encoded = model.encode_image(tensors, normalize=True).float().cpu().numpy()
            for value, (archive_index, frame_index, view_index) in zip(
                encoded, locations[start:stop]
            ):
                embeddings[archive_index][frame_index, view_index] = value
            del tensors, encoded
            if device.type == "mps":
                torch.mps.synchronize()
    output = []
    for path, embedding, (_, valid, boxes, crop_metadata) in zip(
        paths, embeddings, records
    ):
        if not np.isfinite(embedding).all():
            raise RuntimeError(f"{path}: non-finite hand embeddings")
        metadata = {
            "schema_fingerprint": schema_fingerprint(config),
            "crop_schema_fingerprint": crop_metadata["schema_fingerprint"],
            "crop_archive": str(path),
            "video_path": crop_metadata["video_path"],
            "orientation": crop_metadata["orientation"],
            "source": crop_metadata.get("source", "citizen"),
            "source_item_id": crop_metadata.get(
                "source_item_id", path.name.removesuffix(".hand_rgb_v17.npz")
            ),
            "canonical_label": crop_metadata.get("canonical_label", path.parent.name),
            "selection_manifest_sha256": crop_metadata.get("selection_manifest_sha256"),
            "split": crop_metadata.get("split"),
            "training_eligible": crop_metadata.get("training_eligible"),
            "test_accessed": False,
        }
        output.append(
            (embedding.astype(np.float16), valid, boxes.astype(np.float16), metadata)
        )
    # Decoded RGB arrays are by far the largest host allocation.  Drop every
    # reference before returning so long resumable runs stay bounded.
    del records, images, locations, embeddings
    gc.collect()
    return output


def save_archive(path: Path, embeddings, valid, boxes, metadata, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings,
        valid=valid,
        boxes_normalized=boxes,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        schema_json=np.array(json.dumps(schema_payload(config), sort_keys=True)),
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.split not in ALLOWED_SPLITS:
        raise ValueError("the Citizen test split is sealed")
    config = HandMobileCLIP2V17Config()
    device = select_device(args.device)
    model, preprocess = build_encoder(device)
    files = sorted((args.crop_root / args.split).glob("*/*.hand_rgb_v17.npz"))
    if args.limit:
        files = files[:args.limit]
    started = time.monotonic()
    written = skipped = 0
    expected = schema_fingerprint(config)
    for index, crop_path in enumerate(files, start=1):
        relative = crop_path.relative_to(args.crop_root / args.split)
        stem = crop_path.name.removesuffix(".hand_rgb_v17.npz")
        output_path = args.output_root / args.split / relative.parent / f"{stem}.hand_mobileclip2_v17.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if metadata.get("schema_fingerprint") != expected:
                raise ValueError(f"{output_path}: existing schema mismatch")
            skipped += 1
            continue
        embeddings, valid, boxes, metadata = encode_archive(
            crop_path, model, preprocess, device, config
        )
        save_archive(output_path, embeddings, valid, boxes, metadata, config)
        written += 1
        if index == 1 or index % 25 == 0 or index == len(files):
            LOG.info(
                "%s %d/%d written=%d skipped=%d elapsed=%.1fs",
                args.split, index, len(files), written, skipped, time.monotonic() - started,
            )
    return {
        "split": args.split,
        "clips": len(files),
        "written": written,
        "skipped": skipped,
        "device": str(device),
        "schema_fingerprint": expected,
        "seconds": time.monotonic() - started,
        "test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument("--crop-root", type=Path, default=Path("data/local/citizen100_v17/hand_rgb"))
    parser.add_argument("--output-root", type=Path, default=Path("data/local/citizen100_v17/hand_mobileclip2_s0"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
