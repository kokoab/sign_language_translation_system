#!/usr/bin/env python3
"""Run local hand-MobileCLIP encoding in short, memory-capped MPS workers."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time


LOG = logging.getLogger("bounded_hand_mobileclip2_v17")
DIAGNOSTIC_SOURCES = {"local_deep_clean_val", "semlex_val", "local_audit"}


def source_root(args: argparse.Namespace) -> Path:
    return (
        args.crop_root
        if args.source in DIAGNOSTIC_SOURCES
        else args.crop_root / args.source
    )


def output_path(args: argparse.Namespace, crop_path: Path) -> Path:
    relative = crop_path.relative_to(source_root(args))
    item_id = crop_path.name.removesuffix(".hand_rgb_v17.npz")
    parent = (
        args.output_root / relative.parent
        if args.source in DIAGNOSTIC_SOURCES
        else args.output_root / args.source / relative.parent
    )
    return parent / f"{item_id}.hand_mobileclip2_v17.npz"


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.chunk_size < 1 or args.image_batch_size < 1:
        raise ValueError("chunk/image batch sizes must be positive")
    if args.device != "mps":
        raise ValueError("this supervisor exists specifically for capped MPS workers")
    files = sorted(source_root(args).glob("*/*.hand_rgb_v17.npz"))
    if not files:
        raise ValueError(f"no crop archives under {source_root(args)}")

    script = Path(__file__).resolve().parents[1] / "active/v17/extract_hand_mobileclip2_supplement_v17.py"
    started = time.monotonic()
    launched = skipped_complete = 0
    shard_reports: list[dict[str, object]] = []
    for start in range(0, len(files), args.chunk_size):
        stop = min(start + args.chunk_size, len(files))
        if all(output_path(args, path).exists() for path in files[start:stop]):
            skipped_complete += stop - start
            continue
        if args.max_workers and launched >= args.max_workers:
            break
        command = [
            sys.executable, str(script),
            "--source", args.source,
            "--crop-root", str(args.crop_root),
            "--output-root", str(args.output_root),
            "--device", "mps",
            "--model-precision", "fp32",
            "--archive-batch-size", "1",
            "--image-batch-size", str(args.image_batch_size),
            "--file-start", str(start),
            "--file-stop", str(stop),
            "--mps-memory-fraction", str(args.mps_memory_fraction),
        ]
        LOG.info("worker %d shard=[%d,%d)", launched + 1, start, stop)
        completed = subprocess.run(
            command, check=False, text=True, capture_output=True,
            env={
                **__import__("os").environ,
                # PyTorch defaults the adaptive-commit (low) ratio above our
                # deliberately small per-process ceiling, which is invalid. Keep
                # both Metal watermarks at/below the requested process cap.
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": str(args.mps_memory_fraction),
                "PYTORCH_MPS_LOW_WATERMARK_RATIO": str(
                    max(0.01, args.mps_memory_fraction / 2.0)
                ),
            },
        )
        if completed.returncode:
            raise RuntimeError(
                f"bounded MPS worker failed for shard [{start},{stop})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        report = json.loads(completed.stdout)
        shard_reports.append(report)
        launched += 1
        LOG.info(
            "worker %d wrote=%s skipped=%s peak_driver=%.1f MiB",
            launched, report["written"], report["skipped"],
            int(report["peak_mps_driver_bytes"]) / (1024 * 1024),
        )

    remaining = sum(not output_path(args, path).exists() for path in files)
    result = {
        "source": args.source,
        "source_archives": len(files),
        "completed_archives": len(files) - remaining,
        "remaining_archives": remaining,
        "workers_launched": launched,
        "already_complete_in_skipped_shards": skipped_complete,
        "chunk_size": args.chunk_size,
        "image_batch_size": args.image_batch_size,
        "mps_memory_fraction": args.mps_memory_fraction,
        "peak_mps_driver_bytes": max(
            (int(report["peak_mps_driver_bytes"]) for report in shard_reports),
            default=0,
        ),
        "seconds": time.monotonic() - started,
        "test_accessed": False,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True,
        choices=("local_deep_clean", "local_deep_clean_val"),
    )
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="mps", choices=("mps",))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.08)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--report", type=Path)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
