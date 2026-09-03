#!/usr/bin/env python3
"""Measure incremental latency for the experimental Stage-1 evidence CTC head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.model_streaming_stage1_head_v17 import (
    load_streaming_stage1_head_checkpoint,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_streaming_tcn_ctc_v17 import sha256


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, round(0.90 * (len(ordered) - 1)))],
        "maximum_ms": ordered[-1],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    head_checkpoint = torch.load(args.head, map_location="cpu", weights_only=False)
    stage1_path = Path(head_checkpoint["stage1_checkpoint"])
    stage1_checkpoint = torch.load(stage1_path, map_location="cpu", weights_only=False)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    stage1 = SLTStage1V17(Stage1V17Config(**stage1_checkpoint["model_config"]))
    stage1.load_state_dict(stage1_checkpoint["model_state_dict"], strict=True)
    stage1.to(device).eval()
    head = load_streaming_stage1_head_checkpoint(head_checkpoint, device=device)
    features = torch.randn(1, 32, 61, 5, device=device)
    stage1_times = []
    head_times = []
    state = None
    with torch.inference_mode():
        for _ in range(args.warmup):
            evidence = stage1(features)
            _, state = head.stream_step(evidence, state)
        synchronize(device)
        for _ in range(args.iterations):
            started = time.perf_counter()
            evidence = stage1(features)
            synchronize(device)
            stage1_times.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            _, state = head.stream_step(evidence, state)
            synchronize(device)
            head_times.append((time.perf_counter() - started) * 1000)
    result = {
        "format": "slt_streaming_stage1_head_benchmark_v17",
        "device": str(device), "iterations": args.iterations,
        "stage1_window": summary(stage1_times),
        "causal_head_step": summary(head_times),
        "combined_median_ms": statistics.median(stage1_times) + statistics.median(head_times),
        "head_parameters": sum(value.numel() for value in head.parameters()),
        "head_checkpoint_bytes": args.head.stat().st_size,
        "head_checkpoint": str(args.head), "head_sha256": sha256(args.head),
        "stage1_checkpoint": str(stage1_path), "stage1_sha256": sha256(stage1_path),
        "landmark_extraction_excluded": True,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--head", type=Path, default=Path("artifacts/models/streaming_stage1_head_v17_experiment_v3/best_model.pth"))
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/streaming_stage1_head_v17_experiment_v1/latency.json"))
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--warmup", type=int, default=10)
    value.add_argument("--iterations", type=int, default=100)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
