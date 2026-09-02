#!/usr/bin/env python3
"""Export the selected v17 isolated-sign temporal correction to Core ML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import coremltools as ct
import numpy as np
import torch

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from active.v17.export_stage1_coreml_v17 import directory_bytes, sha256_file, tree_sha256
from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    load_stage2_isolated_correction,
)
from active.v17.model_unified_multimodal_v17 import (
    UnifiedFusionHeadV17,
    UnifiedMultimodalV17Config,
)
from active.v17.train_stage_2_isolated_correction_v17 import DOMAINS, stage1_logits


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def run(args: argparse.Namespace) -> dict[str, object]:
    protected = (args.checkpoint, args.stage1_checkpoint, *args.validation_pools)
    if any("test" in {part.lower() for part in path.parts} for path in protected):
        raise ValueError("isolated-correction export is restricted to non-test data")
    model, checkpoint = load_stage2_isolated_correction(args.checkpoint)
    model.eval()
    stage1_checkpoint = torch.load(
        args.stage1_checkpoint, map_location="cpu", weights_only=False
    )
    stage1_head = UnifiedFusionHeadV17(
        UnifiedMultimodalV17Config(**stage1_checkpoint["head_config"])
    )
    stage1_head.load_state_dict(stage1_checkpoint["head_state_dict"], strict=True)
    stage1_head.eval()

    with np.load(args.validation_pools[0], allow_pickle=False) as payload:
        sample_features = payload["frozen_features"][:1].astype(np.float32)
    sample_logits, _, _ = stage1_logits(
        args.unified_cache / "citizen_val.npz", stage1_head, args.batch_size
    )
    sample = (
        torch.from_numpy(sample_features),
        torch.from_numpy(sample_logits[:1].astype(np.float32)),
    )
    traced = torch.jit.trace(model, sample, strict=True)
    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="frozen_features",
                shape=(1, 32, FROZEN_TEMPORAL_FEATURE_DIM),
                dtype=np.float32,
            ),
            ct.TensorType(name="stage1_logits", shape=(1, 100), dtype=np.float32),
        ],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.ALL)
    inputs = [value.name for value in runtime.get_spec().description.input]
    output = runtime.get_spec().description.output[0].name
    provider = dict(zip(inputs, (sample_features, sample_logits[:1].astype(np.float32))))
    for _ in range(args.warmup_iterations):
        runtime.predict(provider)
    timings: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        runtime.predict(provider)
        timings.append((time.perf_counter() - started) * 1000.0)

    domain_metrics: dict[str, object] = {}
    parity_max_abs = 0.0
    parity_prediction_mismatches = 0
    with torch.inference_mode():
        for domain, pool_path in zip(DOMAINS, args.validation_pools):
            with np.load(pool_path, allow_pickle=False) as payload:
                features = payload["frozen_features"].astype(np.float32)
                targets = payload["target_indices"].astype(np.int64)
            base, cache_targets, _ = stage1_logits(
                args.unified_cache / f"{domain}_val.npz", stage1_head, args.batch_size
            )
            if not np.array_equal(targets, cache_targets):
                raise ValueError(f"{domain} validation target order differs")
            correct = 0
            for index in range(len(targets)):
                feature = features[index:index + 1]
                base_value = base[index:index + 1].astype(np.float32)
                expected = model(
                    torch.from_numpy(feature), torch.from_numpy(base_value)
                ).numpy()
                actual = np.asarray(runtime.predict(dict(zip(inputs, (feature, base_value))))[output])
                actual = actual.reshape(expected.shape)
                parity_max_abs = max(
                    parity_max_abs, float(np.max(np.abs(expected - actual)))
                )
                parity_prediction_mismatches += int(
                    expected.argmax(1).item() != actual.argmax(1).item()
                )
                correct += int(actual.argmax(1).item() == int(targets[index]))
            domain_metrics[domain] = {
                "correct": correct,
                "samples": len(targets),
                "accuracy": correct / len(targets),
            }

    result = {
        "format": "slt_stage2_isolated_correction_coreml_export_v17",
        "version": 1,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256_file(args.stage1_checkpoint),
        "coreml_package": args.output.as_posix(),
        "coreml_package_tree_sha256": tree_sha256(args.output),
        "coreml_package_bytes": directory_bytes(args.output),
        "coreml_package_mib": directory_bytes(args.output) / 2**20,
        "format_coreml": f"mlprogram_{args.precision}",
        "minimum_deployment_target": "iOS15",
        "inputs": {
            "frozen_features": [1, 32, FROZEN_TEMPORAL_FEATURE_DIM],
            "stage1_logits": [1, 100],
        },
        "output": [1, 100],
        "blend_weight": checkpoint["blend_weight"],
        "validation": domain_metrics,
        "equal_domain_mean_accuracy": statistics.mean(
            value["accuracy"] for value in domain_metrics.values()
        ),
        "parity_max_abs": parity_max_abs,
        "parity_prediction_mismatches": parity_prediction_mismatches,
        "warmup_iterations": args.warmup_iterations,
        "timed_iterations": args.iterations,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.9),
        "execution_environment": "mac_host_coreml",
        "hardware_performance_claim": False,
        "thermals_interpretable": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_isolated_correction_v1/model.pth"))
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--unified-cache", type=Path, default=Path("artifacts/generated/unified_multimodal_student_v17"))
    parser.add_argument("--validation-pools", nargs=3, type=Path, default=[Path("data/local/stage2_v17_isolated_replay/citizen_validation.npz"), Path("data/local/stage2_v17_isolated_replay/semlex_validation.npz"), Path("data/local/stage2_v17_isolated_replay/local_validation.npz")])
    parser.add_argument("--output", type=Path, default=Path("artifacts/coreml/Stage2IsolatedCorrectionV17FP32.mlpackage"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_coreml_export/isolated_correction_fp32.json"))
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
