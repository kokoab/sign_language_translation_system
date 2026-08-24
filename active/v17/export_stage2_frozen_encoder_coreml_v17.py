#!/usr/bin/env python3
"""Export the frozen v17 multimodal temporal encoder that feeds Stage 2."""

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
from torch import nn

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.export_stage1_coreml_v17 import (
    directory_bytes,
    replace_attention,
    sha256_file,
    tree_sha256,
)
from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    FrozenUnifiedTemporalEncoderV17,
    load_frozen_unified_stage1,
)


class FrozenEncoderExportWrapper(nn.Module):
    def __init__(self, encoder: FrozenUnifiedTemporalEncoderV17):
        super().__init__()
        self.encoder = encoder

    def forward(self, landmarks, hand_embeddings, hand_valid, hand_boxes, window_mask):
        output = self.encoder(
            landmarks, hand_embeddings, hand_valid > 0.5, hand_boxes
        )
        return output * (window_mask > 0.5).unsqueeze(-1).unsqueeze(-1)


def archive_paths(root: Path) -> list[Path]:
    paths = sorted((root / "validation").glob("*/*.stage2_rgb_v17.npz"))
    if not paths:
        raise ValueError(f"no Stage-2 validation archives under {root}")
    return paths


def paired_arrays(
    rgb_path: Path,
    *,
    rgb_root: Path,
    hand_root: Path,
    frozen_root: Path,
    maximum_windows: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, int]:
    relative = rgb_path.relative_to(rgb_root / "validation")
    stem = relative.name.removesuffix(".stage2_rgb_v17.npz")
    hand_path = (
        hand_root / "validation" / relative.parent
        / f"{stem}.stage2_hand_mobileclip2_v17.npz"
    )
    frozen_path = (
        frozen_root / "validation" / relative.parent
        / f"{stem}.stage2_frozen_v17.npz"
    )
    with np.load(rgb_path, allow_pickle=False) as payload:
        landmarks = payload["landmarks"].astype(np.float32)
    with np.load(hand_path, allow_pickle=False) as payload:
        hand_embeddings = payload["embeddings"].astype(np.float32)
        hand_valid = payload["valid"].astype(np.float32)
        hand_boxes = payload["boxes_normalized"].astype(np.float32)
    with np.load(frozen_path, allow_pickle=False) as payload:
        expected = payload["frozen_features"].astype(np.float32)
    windows = len(landmarks)
    if not 1 <= windows <= maximum_windows:
        raise ValueError(f"{rgb_path}: invalid window count")
    if not (
        len(hand_embeddings) == len(hand_valid) == len(hand_boxes) == len(expected) == windows
    ):
        raise ValueError(f"{rgb_path}: paired archive window mismatch")
    values = (
        np.zeros((1, maximum_windows, 32, 61, 5), np.float32),
        np.zeros((1, maximum_windows, 16, 3, 512), np.float32),
        np.zeros((1, maximum_windows, 16, 3), np.float32),
        np.zeros((1, maximum_windows, 16, 3, 4), np.float32),
        np.zeros((1, maximum_windows), np.float32),
    )
    values[0][0, :windows] = landmarks
    values[1][0, :windows] = hand_embeddings
    values[2][0, :windows] = hand_valid
    values[3][0, :windows] = hand_boxes
    values[4][0, :windows] = 1.0
    padded_expected = np.zeros(
        (1, maximum_windows, 32, FROZEN_TEMPORAL_FEATURE_DIM), np.float32
    )
    padded_expected[0, :windows] = expected
    return values, padded_expected, windows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def run(args: argparse.Namespace) -> dict[str, object]:
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.rgb_root, args.hand_root, args.frozen_root, args.stage1_checkpoint
    )):
        raise ValueError("temporal encoder parity is restricted to non-test data")
    landmark, hand, fusion, checkpoint = load_frozen_unified_stage1(
        args.stage1_checkpoint
    )
    original = FrozenUnifiedTemporalEncoderV17(landmark, hand, fusion).eval()
    export_encoder = copy.deepcopy(original)
    replace_attention(export_encoder)
    wrapper = FrozenEncoderExportWrapper(export_encoder).eval()
    paths = archive_paths(args.rgb_root)
    sample_arrays, sample_expected, _ = paired_arrays(
        paths[0], rgb_root=args.rgb_root, hand_root=args.hand_root,
        frozen_root=args.frozen_root, maximum_windows=args.maximum_windows,
    )
    tensors = tuple(torch.from_numpy(value) for value in sample_arrays)
    with torch.inference_mode():
        reference = FrozenEncoderExportWrapper(original)(*tensors).numpy()
        replacement = wrapper(*tensors).numpy()
    cache_max_abs = float(np.max(np.abs(reference - sample_expected)))
    manual_max_abs = float(np.max(np.abs(reference - replacement)))
    # The cache was produced on MPS then stored as float16; this CPU reference is
    # allowed only the measured low-millithreshold backend/quantization drift.
    if cache_max_abs > args.cache_max_abs_gate:
        raise ValueError(f"frozen feature cache parity failed: {cache_max_abs}")
    if manual_max_abs > 1e-4:
        raise ValueError(f"manual attention parity failed: {manual_max_abs}")
    traced = torch.jit.trace(wrapper, tensors, strict=False)
    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="landmarks", shape=(1, args.maximum_windows, 32, 61, 5), dtype=np.float32),
            ct.TensorType(name="hand_embeddings", shape=(1, args.maximum_windows, 16, 3, 512), dtype=np.float32),
            ct.TensorType(name="hand_valid", shape=(1, args.maximum_windows, 16, 3), dtype=np.float32),
            ct.TensorType(name="hand_boxes", shape=(1, args.maximum_windows, 16, 3, 4), dtype=np.float32),
            ct.TensorType(name="window_mask", shape=(1, args.maximum_windows), dtype=np.float32),
        ],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.ALL)
    input_names = [value.name for value in runtime.get_spec().description.input]
    output_name = runtime.get_spec().description.output[0].name
    provider = dict(zip(input_names, sample_arrays))
    for _ in range(args.warmup_iterations):
        runtime.predict(provider)
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        runtime.predict(provider)
        timings.append((time.perf_counter() - started) * 1000.0)

    coreml_max_abs = 0.0
    cache_parity_max_abs = 0.0
    cache_float16_mismatches = 0
    windows_verified = 0
    for path in paths:
        arrays, expected, windows = paired_arrays(
            path, rgb_root=args.rgb_root, hand_root=args.hand_root,
            frozen_root=args.frozen_root, maximum_windows=args.maximum_windows,
        )
        values = tuple(torch.from_numpy(value) for value in arrays)
        with torch.inference_mode():
            torch_output = FrozenEncoderExportWrapper(original)(*values).numpy()
        coreml_output = np.asarray(
            runtime.predict(dict(zip(input_names, arrays)))[output_name]
        ).reshape(torch_output.shape)
        coreml_max_abs = max(
            coreml_max_abs,
            float(np.max(np.abs(torch_output[:, :windows] - coreml_output[:, :windows]))),
        )
        cache_parity_max_abs = max(
            cache_parity_max_abs,
            float(np.max(np.abs(torch_output[:, :windows] - expected[:, :windows]))),
        )
        cache_float16_mismatches += int(not np.array_equal(
            torch_output[:, :windows].astype(np.float16),
            expected[:, :windows].astype(np.float16),
        ))
        windows_verified += windows
    result = {
        "format": "slt_stage2_frozen_encoder_coreml_export_v17",
        "version": 1,
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256_file(args.stage1_checkpoint),
        "coreml_package": args.output.as_posix(),
        "coreml_package_tree_sha256": tree_sha256(args.output),
        "coreml_package_bytes": directory_bytes(args.output),
        "coreml_package_mib": directory_bytes(args.output) / 2**20,
        "format_coreml": f"mlprogram_{args.precision}",
        "minimum_deployment_target": "iOS15",
        "maximum_windows": args.maximum_windows,
        "output_shape": [1, args.maximum_windows, 32, FROZEN_TEMPORAL_FEATURE_DIM],
        "manual_attention_max_abs": manual_max_abs,
        "frozen_cache_max_abs": cache_parity_max_abs,
        "frozen_cache_float16_mismatches": cache_float16_mismatches,
        "frozen_cache_max_abs_gate": args.cache_max_abs_gate,
        "coreml_max_abs": coreml_max_abs,
        "parity_role": "validation",
        "parity_samples": len(paths),
        "parity_windows": windows_verified,
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
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-checkpoint", type=Path,
        default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"),
    )
    parser.add_argument("--rgb-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    parser.add_argument("--hand-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    parser.add_argument("--frozen-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_coreml_export/frozen_encoder_fp32.json"),
    )
    parser.add_argument("--maximum-windows", type=int, default=8)
    parser.add_argument("--cache-max-abs-gate", type=float, default=0.005)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
