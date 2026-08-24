#!/usr/bin/env python3
"""Validate the two-package v17 Stage-2 Core ML pipeline end to end on validation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time

import coremltools as ct
import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.export_stage1_coreml_v17 import tree_sha256
from active.v17.export_stage2_frozen_encoder_coreml_v17 import (
    archive_paths,
    paired_arrays,
)
from active.v17.model_stage2_v17 import load_stage2_context_adapted
from active.v17.train_stage_2_v17 import collapse_ctc, edit_distance


def model_names(model: ct.models.MLModel) -> tuple[list[str], str]:
    spec = model.get_spec().description
    return [value.name for value in spec.input], spec.output[0].name


def decode(logits: np.ndarray, windows: int, tokens_per_window: int) -> list[int]:
    value = np.asarray(logits).reshape(-1, logits.shape[-1])
    return collapse_ctc(value[:windows * tokens_per_window].argmax(-1))


def frozen_metadata(rgb_path: Path, rgb_root: Path, frozen_root: Path):
    relative = rgb_path.relative_to(rgb_root / "validation")
    stem = relative.name.removesuffix(".stage2_rgb_v17.npz")
    path = frozen_root / "validation" / relative.parent / f"{stem}.stage2_frozen_v17.npz"
    with np.load(path, allow_pickle=False) as payload:
        return (
            payload["target_indices"].astype(np.int64).tolist(),
            json.loads(str(payload["metadata_json"])),
        )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def run(args: argparse.Namespace) -> dict[str, object]:
    paths_to_check = (
        args.encoder_package, args.head_package, args.phrase_rgb_root,
        args.context_rgb_root,
    )
    if any("test" in {part.lower() for part in path.parts} for path in paths_to_check):
        raise ValueError("Core ML pipeline evaluation is validation-only")
    encoder = ct.models.MLModel(str(args.encoder_package), compute_units=ct.ComputeUnit.ALL)
    head = ct.models.MLModel(str(args.head_package), compute_units=ct.ComputeUnit.ALL)
    encoder_inputs, encoder_output = model_names(encoder)
    head_inputs, head_output = model_names(head)
    pytorch_head, _ = load_stage2_context_adapted(args.checkpoint)
    pytorch_head.eval()
    tokens_per_window = pytorch_head.base.config.tokens_per_window
    maximum_windows = pytorch_head.base.config.max_windows
    domains = [
        (
            "phrases", args.phrase_rgb_root, args.phrase_hand_root,
            args.phrase_frozen_root,
        ),
        (
            "contextual", args.context_rgb_root, args.context_hand_root,
            args.context_frozen_root,
        ),
    ]
    statistics_by_source = defaultdict(lambda: {"edits": 0, "tokens": 0, "exact": 0, "samples": 0})
    samples = windows_total = decode_mismatches = 0
    max_head_abs = 0.0
    sample_arrays = None
    for _, rgb_root, hand_root, frozen_root in domains:
        for rgb_path in archive_paths(rgb_root):
            arrays, cached, windows = paired_arrays(
                rgb_path, rgb_root=rgb_root, hand_root=hand_root,
                frozen_root=frozen_root, maximum_windows=maximum_windows,
            )
            reference_tokens, metadata = frozen_metadata(rgb_path, rgb_root, frozen_root)
            encoded = np.asarray(
                encoder.predict(dict(zip(encoder_inputs, arrays)))[encoder_output]
            ).reshape(cached.shape)
            head_provider = {
                head_inputs[0]: encoded.astype(np.float32),
                head_inputs[1]: arrays[-1],
            }
            coreml_logits = np.asarray(head.predict(head_provider)[head_output])
            with torch.inference_mode():
                pytorch_logits, _ = pytorch_head(
                    torch.from_numpy(cached), torch.from_numpy(arrays[-1]) > 0.5
                )
            pytorch_value = pytorch_logits.numpy()
            coreml_value = coreml_logits.reshape(pytorch_value.shape)
            max_head_abs = max(
                max_head_abs, float(np.max(np.abs(pytorch_value - coreml_value)))
            )
            hypothesis = decode(coreml_value, windows, tokens_per_window)
            reference_hypothesis = decode(pytorch_value, windows, tokens_per_window)
            decode_mismatches += int(hypothesis != reference_hypothesis)
            source = str(metadata["source"])
            row = statistics_by_source[source]
            edits = edit_distance(reference_tokens, hypothesis)
            row["edits"] += edits
            row["tokens"] += len(reference_tokens)
            row["exact"] += int(reference_tokens == hypothesis)
            row["samples"] += 1
            samples += 1
            windows_total += windows
            sample_arrays = arrays
    if sample_arrays is None:
        raise RuntimeError("no validation samples evaluated")
    provider = dict(zip(encoder_inputs, sample_arrays))
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        encoded = np.asarray(encoder.predict(provider)[encoder_output]).astype(np.float32)
        head.predict({head_inputs[0]: encoded, head_inputs[1]: sample_arrays[-1]})
        timings.append((time.perf_counter() - started) * 1000.0)
    metrics = {
        source: {
            "edits": values["edits"],
            "tokens": values["tokens"],
            "wer": values["edits"] / max(1, values["tokens"]),
            "exact_sequences": values["exact"],
            "samples": values["samples"],
            "sequence_accuracy": values["exact"] / max(1, values["samples"]),
        }
        for source, values in sorted(statistics_by_source.items())
    }
    result = {
        "format": "slt_stage2_coreml_pipeline_validation_v17",
        "version": 1,
        "encoder_package": args.encoder_package.as_posix(),
        "encoder_package_tree_sha256": tree_sha256(args.encoder_package),
        "head_package": args.head_package.as_posix(),
        "head_package_tree_sha256": tree_sha256(args.head_package),
        "validation_samples": samples,
        "validation_windows": windows_total,
        "coreml_vs_pytorch_decode_mismatches": decode_mismatches,
        "coreml_pipeline_vs_cached_pytorch_max_abs": max_head_abs,
        "metrics": metrics,
        "timed_iterations": args.iterations,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.9),
        "execution_environment": "mac_host_coreml",
        "hardware_performance_claim": False,
        "thermals_interpretable": False,
        "mobileclip2_embedding_extractor_in_packages": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-package", type=Path, default=Path("artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage"))
    parser.add_argument("--head-package", type=Path, default=Path("artifacts/coreml/Stage2CompactContextV17FP32.mlpackage"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_compact_context_student_v1/model.pth"))
    parser.add_argument("--phrase-rgb-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    parser.add_argument("--phrase-hand-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    parser.add_argument("--phrase-frozen-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument("--context-rgb-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_multimodal"))
    parser.add_argument("--context-hand-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_hand_mobileclip2"))
    parser.add_argument("--context-frozen-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_coreml_export/pipeline_validation.json"))
    parser.add_argument("--iterations", type=int, default=20)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
