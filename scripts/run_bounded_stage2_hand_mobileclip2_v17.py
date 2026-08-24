#!/usr/bin/env python3
"""Encode all Stage-2 hand crops in short sequential memory-capped MPS workers."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.encode_stage2_hand_mobileclip2_v17 import output_path


LOG = logging.getLogger("run_bounded_stage2_hand_mobileclip2_v17")


def run(args: argparse.Namespace) -> dict[str, object]:
    files = sorted(args.crop_root.glob("*/*/*.stage2_rgb_v17.npz"))
    if not files:
        raise ValueError(f"no Stage-2 crop archives under {args.crop_root}")
    worker = Path(__file__).with_name("encode_stage2_hand_mobileclip2_v17.py")
    launched = skipped_complete = 0
    reports = []
    started = time.monotonic()
    for start in range(0, len(files), args.chunk_size):
        stop = min(len(files), start + args.chunk_size)
        if all(output_path(args.output_root, args.crop_root, path).exists() for path in files[start:stop]):
            skipped_complete += stop - start
            continue
        if args.max_workers and launched >= args.max_workers:
            break
        command = [
            sys.executable, str(worker),
            "--crop-root", str(args.crop_root),
            "--output-root", str(args.output_root),
            "--device", "mps",
            "--file-start", str(start),
            "--file-stop", str(stop),
            "--image-batch-size", str(args.image_batch_size),
            "--mps-memory-fraction", str(args.mps_memory_fraction),
            "--maximum-source-frames", str(args.maximum_source_frames),
        ]
        LOG.info("worker %d shard=[%d,%d)", launched + 1, start, stop)
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": str(args.mps_memory_fraction),
                "PYTORCH_MPS_LOW_WATERMARK_RATIO": str(max(0.01, args.mps_memory_fraction / 2)),
            },
        )
        if completed.returncode:
            raise RuntimeError(
                f"Stage-2 MPS worker failed for [{start},{stop})\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        report = json.loads(completed.stdout)
        reports.append(report)
        launched += 1
        LOG.info(
            "worker %d wrote=%d skipped=%d peak_driver=%.1f MiB",
            launched, report["written"], report["skipped"],
            int(report["peak_mps_driver_bytes"]) / (1024 * 1024),
        )
    remaining = sum(
        not output_path(args.output_root, args.crop_root, path).exists() for path in files
    )
    result = {
        "source_archives": len(files),
        "completed_archives": len(files) - remaining,
        "remaining_archives": remaining,
        "workers_launched": launched,
        "already_complete_in_skipped_shards": skipped_complete,
        "chunk_size": args.chunk_size,
        "image_batch_size": args.image_batch_size,
        "mps_memory_fraction": args.mps_memory_fraction,
        "maximum_source_frames": args.maximum_source_frames,
        "peak_mps_driver_bytes": max(
            (int(report["peak_mps_driver_bytes"]) for report in reports), default=0
        ),
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
    parser.add_argument("--output-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.08)
    parser.add_argument("--maximum-source-frames", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_hand_mobileclip2/encoding.json"),
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
