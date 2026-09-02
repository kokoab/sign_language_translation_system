#!/usr/bin/env python3
"""Export and verify the separate v17 Reel emission model as FP16 Core ML."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import coremltools as ct
import numpy as np
import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.export_stage1_coreml_v17 import (
    directory_bytes,
    percentile,
    replace_attention,
    sha256_file,
    tree_sha256,
)
from active.v17.model_reel_emission_v17 import load_reel_emission_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path,
        default=Path("artifacts/models/stage1_v17_reel_emission_v3/best_model.pth"),
        nargs="?",
    )
    parser.add_argument(
        "output", type=Path,
        default=Path("artifacts/coreml/Stage1ReelEmissionV17FP16.mlpackage"),
        nargs="?",
    )
    parser.add_argument(
        "--sample", type=Path,
        default=Path(
            "data/local/citizen100_v17/landmarks/val/HELLO/"
            "020030442376253177-HELLO.v17.npz"
        ),
    )
    parser.add_argument(
        "--parity-root", type=Path,
        default=Path("data/local/citizen100_v17/landmarks/val"),
    )
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if any("test" in {part.casefold() for part in path.parts} for path in (
        args.sample, args.parity_root,
    )):
        raise ValueError("export parity is restricted to non-test samples")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original = load_reel_emission_checkpoint(checkpoint)
    export_model = copy.deepcopy(original)
    replace_attention(export_model)
    export_model.eval()
    with np.load(args.sample, allow_pickle=False) as payload:
        sample_array = payload["features"].astype(np.float32, copy=False)[None]
    sample = torch.from_numpy(sample_array)
    with torch.inference_mode():
        reference = original(sample).numpy()
        replacement = export_model(sample).numpy()
    attention_max_abs = float(np.max(np.abs(reference - replacement)))
    if attention_max_abs > 1e-4:
        raise ValueError(f"manual-attention parity failed: {attention_max_abs}")

    traced = torch.jit.trace(export_model, sample, strict=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="landmarks", shape=(1, 32, 61, 5), dtype=np.float32,
        )],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.ALL)
    input_name = runtime.get_spec().description.input[0].name
    output_name = runtime.get_spec().description.output[0].name
    for _ in range(10):
        runtime.predict({input_name: sample_array})
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        runtime.predict({input_name: sample_array})
        timings.append(1000 * (time.perf_counter() - started))

    threshold = float(
        checkpoint["reel_emission"]["runtime_no_emit_probability_threshold"]
    )
    maximum_abs = 0.0
    top1_mismatches = 0
    decision_mismatches = 0
    paths = sorted(args.parity_root.rglob("*.v17.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            array = payload["features"].astype(np.float32, copy=False)[None]
        with torch.inference_mode():
            expected = original(torch.from_numpy(array)).numpy()
        actual = np.asarray(runtime.predict({input_name: array})[output_name]).reshape(1, 101)
        maximum_abs = max(maximum_abs, float(np.max(np.abs(expected - actual))))
        top1_mismatches += int(expected.argmax() != actual.argmax())
        expected_probability = np.exp(expected - expected.max(axis=1, keepdims=True))
        expected_probability /= expected_probability.sum(axis=1, keepdims=True)
        actual_probability = np.exp(actual - actual.max(axis=1, keepdims=True))
        actual_probability /= actual_probability.sum(axis=1, keepdims=True)
        decision_mismatches += int(
            (expected_probability[0, 100] >= threshold)
            != (actual_probability[0, 100] >= threshold)
        )
    result = {
        "format": "slt_stage1_reel_emission_coreml_export_v17",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "output": str(args.output),
        "package_mib": directory_bytes(args.output) / 2**20,
        "package_tree_sha256": tree_sha256(args.output),
        "manual_attention_max_abs": attention_max_abs,
        "parity_root": str(args.parity_root),
        "parity_samples": len(paths),
        "parity_max_abs": maximum_abs,
        "parity_top1_mismatches": top1_mismatches,
        "parity_emission_decision_mismatches": decision_mismatches,
        "runtime_no_emit_probability_threshold": threshold,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.90),
        "compute_units": "ALL_on_current_Mac_not_iPhone",
        "test_accessed": False,
    }
    report = args.output.parent / f"{args.output.stem}_benchmark.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
