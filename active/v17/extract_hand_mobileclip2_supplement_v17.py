#!/usr/bin/env python3
"""Encode frozen MobileCLIP2 features for hand RGB supplements/diagnostics."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_mobileclip2_v17 import encode_archive_batch, save_archive
from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
from active.v17.schema_hand_mobileclip2_v17 import (
    HandMobileCLIP2V17Config,
    schema_fingerprint,
)


LOG = logging.getLogger("hand_mobileclip2_supplement_v17")


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.archive_batch_size < 1 or args.image_batch_size < 1:
        raise ValueError("archive/image batch sizes must be positive")
    config = HandMobileCLIP2V17Config()
    device = select_device(args.device)
    if device.type == "mps":
        if not 0.0 < args.mps_memory_fraction <= 0.25:
            raise ValueError("MPS memory fraction must be in (0, 0.25]")
        # Fail this worker with an allocation error well before it can pressure the
        # host's unified memory.  A supervisor also recycles the Metal context after
        # a small deterministic file shard.
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    model, preprocess = build_encoder(device, args.model_precision)
    is_nontraining_diagnostic = args.source in (
        "semlex_val", "local_audit", "local_deep_clean_val"
    )
    source_root = args.crop_root if is_nontraining_diagnostic else args.crop_root / args.source
    all_files = sorted(source_root.glob("*/*.hand_rgb_v17.npz"))
    files = all_files
    if args.limit:
        files = files[:args.limit]
    start = args.file_start
    stop = args.file_stop if args.file_stop else len(files)
    if start < 0 or stop <= start or stop > len(files):
        raise ValueError(
            f"invalid file shard [{start}, {stop}) for {len(files)} source files"
        )
    files = files[start:stop]
    if not files:
        raise ValueError(f"no hand RGB supplements under {source_root}")
    expected = schema_fingerprint(config)
    written = skipped = 0
    peak_mps_current = peak_mps_driver = 0
    started = time.monotonic()
    pending: list[tuple[Path, Path]] = []
    for crop_path in files:
        relative = crop_path.relative_to(source_root)
        item_id = crop_path.name.removesuffix(".hand_rgb_v17.npz")
        output_parent = (
            args.output_root / relative.parent
            if is_nontraining_diagnostic
            else args.output_root / args.source / relative.parent
        )
        output_path = output_parent / f"{item_id}.hand_mobileclip2_v17.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
            if (
                metadata.get("schema_fingerprint") != expected
                or metadata.get("source") != args.source
                or metadata.get("test_accessed") is not False
                or (
                    is_nontraining_diagnostic
                    and (
                        metadata.get("split") != (
                            "val_domain_diagnostic"
                            if args.source == "semlex_val"
                            else "validation_nonsigner_disjoint_user_approved"
                            if args.source == "local_deep_clean_val"
                            else "train_only_review_diagnostic"
                        )
                        or metadata.get("training_eligible") is not False
                    )
                )
            ):
                raise ValueError(f"existing supplement embedding mismatch: {output_path}")
            skipped += 1
            continue
        pending.append((crop_path, output_path))
    for start in range(0, len(pending), args.archive_batch_size):
        batch = pending[start:start + args.archive_batch_size]
        batch_length = len(batch)
        values = encode_archive_batch(
            [crop_path for crop_path, _ in batch],
            model,
            preprocess,
            device,
            config,
            image_batch_size=args.image_batch_size,
        )
        for (crop_path, output_path), (embeddings, valid, boxes, metadata) in zip(
            batch, values
        ):
            if metadata.get("source") != args.source:
                raise ValueError(f"crop source mismatch: {crop_path}")
            save_archive(output_path, embeddings, valid, boxes, metadata, config)
            written += 1
        completed = skipped + written
        if device.type == "mps":
            peak_mps_current = max(
                peak_mps_current, int(torch.mps.current_allocated_memory())
            )
            peak_mps_driver = max(
                peak_mps_driver, int(torch.mps.driver_allocated_memory())
            )
        del values, batch
        if completed % 25 < batch_length or completed == len(files):
            gc.collect()
        if (
            written == batch_length
            or completed % 25 < batch_length
            or completed == len(files)
        ):
            LOG.info(
                "%s %d/%d written=%d skipped=%d elapsed=%.1fs",
                args.source, completed, len(files), written, skipped,
                time.monotonic() - started,
            )
    return {
        "source": args.source, "clips": len(files), "written": written,
        "skipped": skipped, "device": str(device),
        "file_start": start, "file_stop": stop,
        "total_source_files": len(all_files),
        "schema_fingerprint": expected, "seconds": time.monotonic() - started,
        "encoder_compute_precision": args.model_precision,
        "mps_memory_fraction": (
            args.mps_memory_fraction if device.type == "mps" else None
        ),
        "peak_mps_current_bytes": peak_mps_current,
        "peak_mps_driver_bytes": peak_mps_driver,
        "split": (
            "val_domain_diagnostic" if args.source == "semlex_val"
            else "validation_nonsigner_disjoint_user_approved"
            if args.source == "local_deep_clean_val"
            else "train_only_review_diagnostic" if args.source == "local_audit"
            else "train_only"
        ),
        "training_eligible": not is_nontraining_diagnostic,
        "test_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True,
        choices=(
            "semlex", "local_tier_a", "local_deep_clean",
            "semlex_val", "local_audit", "local_deep_clean_val",
        )
    )
    parser.add_argument(
        "--crop-root", type=Path,
        default=Path("data/local/hand_rgb_supplements_v17"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("data/local/hand_mobileclip2_supplements_v17"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-precision", choices=("fp32", "fp16"), default="fp32"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--archive-batch-size", type=int, default=1)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-stop", type=int, default=0)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.08)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
