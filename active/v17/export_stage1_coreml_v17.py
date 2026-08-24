#!/usr/bin/env python3
"""Export a frozen v17 Stage-1 checkpoint to FP16 Core ML and benchmark it."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config


class ManualMHA(nn.Module):
    """Traceable batch-first self-attention equivalent used only for export."""

    def __init__(self, source: nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = source.embed_dim
        self.num_heads = source.num_heads
        self.head_dim = source.embed_dim // source.num_heads
        self.in_proj_weight = nn.Parameter(source.in_proj_weight.detach().clone())
        self.in_proj_bias = nn.Parameter(source.in_proj_bias.detach().clone())
        self.out_proj = copy.deepcopy(source.out_proj)

    def forward(self, query, key, value, need_weights=False):
        batch, frames, dim = query.shape
        projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = projected.chunk(3, dim=-1)
        q = q.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, frames, self.num_heads, self.head_dim).transpose(1, 2)
        attention = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch, frames, dim)
        return self.out_proj(output), None


def replace_attention(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, ManualMHA(child))
        else:
            replace_attention(child)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(
            "data/local/citizen100_v17/landmarks/val/HELLO/"
            "020030442376253177-HELLO.v17.npz"
        ),
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--parity-root",
        type=Path,
        default=Path("data/local/citizen100_v17/landmarks/val"),
        help="Validation-only v17 tree used for exhaustive PyTorch/Core ML parity",
    )
    args = parser.parse_args()
    if "test" in args.parity_root.parts or "test" in args.sample.parts:
        raise ValueError("Core ML export parity is restricted to non-test samples")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("not a v17 Stage-1 checkpoint")
    original = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    original.load_state_dict(checkpoint["model_state_dict"])
    original.eval()
    export_model = copy.deepcopy(original)
    replace_attention(export_model)
    export_model.eval()

    with np.load(args.sample, allow_pickle=False) as payload:
        sample_array = payload["features"].astype(np.float32, copy=False)[None]
    sample = torch.from_numpy(sample_array)
    with torch.inference_mode():
        reference = original(sample).numpy()
        replacement = export_model(sample).numpy()
    replacement_max_abs = float(np.max(np.abs(reference - replacement)))
    if replacement_max_abs > 1e-4:
        raise ValueError(f"manual attention parity failed: {replacement_max_abs}")

    traced = torch.jit.trace(export_model, sample, strict=False)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="landmarks", shape=(1, 32, 61, 5), dtype=np.float32
            )
        ],
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
    timings: list[float] = []
    prediction = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        prediction = runtime.predict({input_name: sample_array})
        timings.append(1000 * (time.perf_counter() - started))
    coreml_output = np.asarray(prediction[output_name]).reshape(reference.shape)
    parity_paths = sorted(args.parity_root.rglob("*.v17.npz"))
    if not parity_paths:
        raise ValueError(f"no parity archives found under {args.parity_root}")
    parity_max_abs = 0.0
    parity_top1_mismatches = 0
    for path in parity_paths:
        with np.load(path, allow_pickle=False) as payload:
            parity_array = payload["features"].astype(np.float32, copy=False)[None]
        with torch.inference_mode():
            torch_output = original(torch.from_numpy(parity_array)).numpy()
        ml_output = np.asarray(
            runtime.predict({input_name: parity_array})[output_name]
        ).reshape(torch_output.shape)
        parity_max_abs = max(
            parity_max_abs, float(np.max(np.abs(torch_output - ml_output)))
        )
        parity_top1_mismatches += int(torch_output.argmax() != ml_output.argmax())
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "dim": checkpoint["model_config"]["dim"],
        "parameters": original.parameter_count,
        "format": "mlprogram_fp16",
        "minimum_deployment_target": "iOS15",
        "package_bytes": directory_bytes(args.output),
        "package_mib": directory_bytes(args.output) / 2**20,
        "package_tree_sha256": tree_sha256(args.output),
        "manual_attention_max_abs": replacement_max_abs,
        "coreml_max_abs": float(np.max(np.abs(reference - coreml_output))),
        "top1_match": int(reference.argmax()) == int(coreml_output.argmax()),
        "parity_root": str(args.parity_root),
        "parity_samples": len(parity_paths),
        "parity_max_abs": parity_max_abs,
        "parity_top1_mismatches": parity_top1_mismatches,
        "warm_iterations": args.iterations,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.90),
        "compute_units": "ALL_on_current_Mac_not_iPhone",
    }
    result_path = args.output.parent / f"{args.output.stem}_benchmark.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
