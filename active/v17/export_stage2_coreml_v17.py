#!/usr/bin/env python3
"""Export the exact compact context-adapted v17 Stage-2 graph to Core ML."""

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
import torch.nn.functional as F

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.export_stage1_coreml_v17 import directory_bytes, sha256_file, tree_sha256
from active.v17.model_stage2_v17 import (
    FROZEN_TEMPORAL_FEATURE_DIM,
    load_stage2_context_adapted,
)
from active.v17.train_stage_2_v17 import RealPhraseDataset, collapse_ctc


class ManualMaskedMHA(nn.Module):
    """Traceable batch-first self-attention with Transformer padding-mask parity."""

    def __init__(self, source: nn.MultiheadAttention):
        super().__init__()
        if not source.batch_first or source.embed_dim % source.num_heads:
            raise ValueError("Stage-2 export requires batch-first self-attention")
        self.embed_dim = source.embed_dim
        self.num_heads = source.num_heads
        self.head_dim = source.embed_dim // source.num_heads
        self.batch_first = True
        # Deliberately disable PyTorch's fused encoder fast path so tracing reaches
        # this explicit attention implementation instead of an unsupported op.
        self._qkv_same_embed_dim = False
        self.in_proj_weight = nn.Parameter(source.in_proj_weight.detach().clone())
        self.in_proj_bias = nn.Parameter(source.in_proj_bias.detach().clone())
        self.out_proj = copy.deepcopy(source.out_proj)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if query.data_ptr() != key.data_ptr() or query.data_ptr() != value.data_ptr():
            raise ValueError("Stage-2 export supports self-attention only")
        batch, frames, dimension = query.shape
        projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = projected.chunk(3, dim=-1)
        q = q.reshape(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), -1.0e4
            )
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, frames, dimension)
        output = self.out_proj(output)
        return output, (attention.mean(dim=1) if need_weights else None)


def replace_masked_attention(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, ManualMaskedMHA(child))
        else:
            replace_masked_attention(child)


class Stage2ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, frozen_features, window_mask):
        logits, _ = self.model(frozen_features, window_mask > 0.5)
        return logits


def fixed_input(features: np.ndarray, maximum_windows: int) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 3 or tuple(features.shape[1:]) != (
        32, FROZEN_TEMPORAL_FEATURE_DIM
    ):
        raise ValueError("invalid frozen Stage-2 features")
    if not 1 <= len(features) <= maximum_windows:
        raise ValueError("Stage-2 sample exceeds fixed Core ML window contract")
    value = np.zeros(
        (1, maximum_windows, 32, FROZEN_TEMPORAL_FEATURE_DIM), dtype=np.float32
    )
    mask = np.zeros((1, maximum_windows), dtype=np.float32)
    value[0, :len(features)] = features
    mask[0, :len(features)] = 1.0
    return value, mask


def decoded(logits: np.ndarray, windows: int, tokens_per_window: int) -> list[int]:
    length = windows * tokens_per_window
    return collapse_ctc(np.asarray(logits).reshape(-1, logits.shape[-1])[:length].argmax(-1))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def run(args: argparse.Namespace) -> dict[str, object]:
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.parity_root, args.checkpoint
    )):
        raise ValueError("Stage-2 Core ML parity is restricted to non-test data")
    original, checkpoint = load_stage2_context_adapted(args.checkpoint)
    original.eval()
    export_model = copy.deepcopy(original)
    replace_masked_attention(export_model)
    wrapper = Stage2ExportWrapper(export_model).eval()
    dataset = RealPhraseDataset(args.parity_root, "validation")
    sample = dataset[0]
    sample_arrays = fixed_input(sample.features.astype(np.float32), original.base.config.max_windows)
    sample_tensors = tuple(torch.from_numpy(value) for value in sample_arrays)
    with torch.inference_mode():
        reference, _ = original(sample_tensors[0], sample_tensors[1] > 0.5)
        replacement = wrapper(*sample_tensors)
    manual_max_abs = float((reference - replacement).abs().max())
    if manual_max_abs > 1e-4:
        raise ValueError(f"manual masked-attention parity failed: {manual_max_abs}")
    traced = torch.jit.trace(wrapper, sample_tensors, strict=False)
    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="frozen_features",
                shape=(1, original.base.config.max_windows, 32, FROZEN_TEMPORAL_FEATURE_DIM),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="window_mask",
                shape=(1, original.base.config.max_windows),
                dtype=np.float32,
            ),
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
    timings: list[float] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        runtime.predict(provider)
        timings.append((time.perf_counter() - started) * 1000.0)

    max_abs = 0.0
    decode_mismatches = 0
    exact_tensor_mismatches = 0
    with torch.inference_mode():
        for row in dataset.samples:
            arrays = fixed_input(
                row.features.astype(np.float32), original.base.config.max_windows
            )
            tensors = tuple(torch.from_numpy(value) for value in arrays)
            torch_logits, _ = original(tensors[0], tensors[1] > 0.5)
            torch_value = torch_logits.numpy()
            coreml_value = np.asarray(
                runtime.predict(dict(zip(input_names, arrays)))[output_name]
            ).reshape(torch_value.shape)
            max_abs = max(max_abs, float(np.max(np.abs(torch_value - coreml_value))))
            torch_decoded = decoded(
                torch_value, len(row.features), original.base.config.tokens_per_window
            )
            coreml_decoded = decoded(
                coreml_value, len(row.features), original.base.config.tokens_per_window
            )
            decode_mismatches += int(torch_decoded != coreml_decoded)
            exact_tensor_mismatches += int(not np.array_equal(torch_value, coreml_value))
    result = {
        "format": "slt_stage2_compact_coreml_export_v17",
        "version": 1,
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_format": checkpoint["format"],
        "coreml_package": args.output.as_posix(),
        "coreml_package_tree_sha256": tree_sha256(args.output),
        "coreml_package_bytes": directory_bytes(args.output),
        "coreml_package_mib": directory_bytes(args.output) / 2**20,
        "format_coreml": f"mlprogram_{args.precision}",
        "minimum_deployment_target": "iOS15",
        "inputs": {
            "frozen_features": [1, original.base.config.max_windows, 32, FROZEN_TEMPORAL_FEATURE_DIM],
            "window_mask": [1, original.base.config.max_windows],
        },
        "output": [1, original.base.config.max_windows * original.base.config.tokens_per_window, 101],
        "manual_attention_max_abs": manual_max_abs,
        "parity_role": "validation",
        "parity_samples": len(dataset),
        "parity_max_abs": max_abs,
        "parity_decode_mismatches": decode_mismatches,
        "parity_exact_tensor_mismatches": exact_tensor_mismatches,
        "warmup_iterations": args.warmup_iterations,
        "timed_iterations": args.iterations,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.9),
        "execution_environment": "mac_host_coreml",
        "hardware_performance_claim": False,
        "thermals_interpretable": False,
        "frozen_stage1_temporal_encoder_in_package": False,
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
        "--checkpoint", type=Path,
        default=Path("artifacts/models/stage2_v17_compact_context_student_v1/model.pth"),
    )
    parser.add_argument(
        "--parity-root", type=Path,
        default=Path("data/local/stage2_v17_frozen_features"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/coreml/Stage2CompactContextV17FP32.mlpackage"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_coreml_export/compact_fp32.json"),
    )
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
