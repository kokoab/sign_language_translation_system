#!/usr/bin/env python3
"""Export the exact frozen MobileCLIP2-S0 hand-crop image tower to Core ML."""

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
from PIL import Image
import torch
from torch import nn
from timm.utils.model import reparameterize_model

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from active.v17.export_stage1_coreml_v17 import directory_bytes, sha256_file, tree_sha256
from active.v17.extract_hand_rgb_v17 import decode_packed_crops
from active.v17.extract_mobileclip2_v17 import build_encoder
from active.v17.schema_hand_rgb_v17 import CROP_SIZE
from active.v17.schema_mobileclip2_v17 import (
    CHECKPOINT_SHA256,
    EMBEDDING_DIM,
    INPUT_SIZE,
    MODEL_NAME,
    PRETRAINED_TAG,
)


class NormalizedImageTower(nn.Module):
    def __init__(self, visual: nn.Module):
        super().__init__()
        self.visual = visual

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.visual(image)
        denominator = torch.sqrt(torch.sum(value * value, dim=-1, keepdim=True)).clamp_min(1e-8)
        return value / denominator


def disable_fused_attention(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "fused_attn"):
            child.fused_attn = False


def load_first_valid_crop(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    with np.load(path, allow_pickle=False) as payload:
        crops = decode_packed_crops(
            payload["jpeg_blob"], payload["jpeg_offsets"], CROP_SIZE
        )
        valid = payload["valid"].astype(np.bool_)
    locations = np.argwhere(valid)
    if not len(locations):
        raise ValueError(f"{path}: no valid hand crop")
    frame, view = [int(value) for value in locations[0]]
    return Image.fromarray(crops[frame, view]), (frame, view)


def paired_embedding_path(crop_path: Path, crop_root: Path, embedding_root: Path) -> Path:
    relative = crop_path.relative_to(crop_root)
    stem = relative.name.removesuffix(".hand_rgb_v17.npz")
    return embedding_root / relative.parent / f"{stem}.hand_mobileclip2_v17.npz"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def run(args: argparse.Namespace) -> dict[str, object]:
    forbidden = (args.crop_root, args.embedding_root)
    if any("test" in {part.lower() for part in path.parts} for path in forbidden):
        raise ValueError("MobileCLIP2 Core ML parity is restricted to non-test data")
    source, preprocess = build_encoder(torch.device("cpu"), precision="fp32")
    original = NormalizedImageTower(source.visual).eval()
    export = NormalizedImageTower(
        reparameterize_model(copy.deepcopy(source.visual), inplace=True)
    ).eval()
    disable_fused_attention(export)
    sample_image, _ = load_first_valid_crop(sorted(args.crop_root.glob("*/*.hand_rgb_v17.npz"))[0])
    sample = preprocess(sample_image).unsqueeze(0)
    with torch.inference_mode():
        reference = original(sample).numpy()
        replacement = export(sample).numpy()
    reparameterized_max_abs = float(np.max(np.abs(reference - replacement)))
    if reparameterized_max_abs > args.reparameterized_gate:
        raise ValueError(f"FastViT reparameterization parity failed: {reparameterized_max_abs}")

    traced = torch.jit.trace(export, sample, strict=False)
    converted = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=(
            ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
        ),
        minimum_deployment_target=ct.target.iOS15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.ALL)

    paths = sorted(args.crop_root.glob("*/*.hand_rgb_v17.npz"))
    if args.maximum_parity_samples:
        # Stable class-spanning selection rather than a favorable contiguous prefix.
        positions = np.rint(
            np.linspace(0, len(paths) - 1, min(args.maximum_parity_samples, len(paths)))
        ).astype(int)
        paths = [paths[int(index)] for index in positions]
    max_abs_torch = 0.0
    max_abs_cached = 0.0
    minimum_cosine_torch = 1.0
    minimum_cosine_cached = 1.0
    cached_float16_exact = 0
    timings: list[float] = []
    rows = []
    with torch.inference_mode():
        for path in paths:
            image, (frame, view) = load_first_valid_crop(path)
            tensor = preprocess(image).unsqueeze(0)
            torch_value = original(tensor).numpy().reshape(EMBEDDING_DIM)
            started = time.perf_counter()
            prediction = runtime.predict({"image": image})
            timings.append((time.perf_counter() - started) * 1_000.0)
            coreml_value = np.asarray(prediction["embedding"]).reshape(EMBEDDING_DIM)
            embedding_path = paired_embedding_path(path, args.crop_root, args.embedding_root)
            with np.load(embedding_path, allow_pickle=False) as payload:
                cached_value = payload["embeddings"][frame, view].astype(np.float32)
            torch_abs = float(np.max(np.abs(torch_value - coreml_value)))
            cached_abs = float(np.max(np.abs(cached_value - coreml_value)))
            torch_cosine = float(np.dot(torch_value, coreml_value) / (
                np.linalg.norm(torch_value) * np.linalg.norm(coreml_value)
            ))
            cached_cosine = float(np.dot(cached_value, coreml_value) / (
                np.linalg.norm(cached_value) * np.linalg.norm(coreml_value)
            ))
            max_abs_torch = max(max_abs_torch, torch_abs)
            max_abs_cached = max(max_abs_cached, cached_abs)
            minimum_cosine_torch = min(minimum_cosine_torch, torch_cosine)
            minimum_cosine_cached = min(minimum_cosine_cached, cached_cosine)
            cached_float16_exact += int(np.array_equal(
                cached_value.astype(np.float16), coreml_value.astype(np.float16)
            ))
            rows.append({
                "crop": path.as_posix(),
                "frame": frame,
                "view": view,
                "torch_max_abs": torch_abs,
                "cached_max_abs": cached_abs,
                "torch_cosine": torch_cosine,
                "cached_cosine": cached_cosine,
            })
    if max_abs_torch > args.max_abs_gate or minimum_cosine_torch < args.cosine_gate:
        raise ValueError(
            f"Core ML image tower parity failed: max_abs={max_abs_torch} "
            f"min_cosine={minimum_cosine_torch}"
        )
    result = {
        "format": "slt_mobileclip2_s0_image_coreml_export_v17",
        "version": 1,
        "model_name": MODEL_NAME,
        "pretrained_tag": PRETRAINED_TAG,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "visual_parameters": sum(parameter.numel() for parameter in original.parameters()),
        "input": {"name": "image", "shape": [1, 3, INPUT_SIZE, INPUT_SIZE], "pixel_scale": 1 / 255},
        "output": {"name": "embedding", "shape": [1, EMBEDDING_DIM], "normalized": True},
        "coreml_package": args.output.as_posix(),
        "coreml_package_tree_sha256": tree_sha256(args.output),
        "coreml_package_bytes": directory_bytes(args.output),
        "format_coreml": f"mlprogram_{args.precision}",
        "minimum_deployment_target": "iOS15",
        "reparameterized_max_abs": reparameterized_max_abs,
        "parity_samples": len(paths),
        "parity_max_abs_vs_torch": max_abs_torch,
        "parity_min_cosine_vs_torch": minimum_cosine_torch,
        "parity_max_abs_vs_cached_float16": max_abs_cached,
        "parity_min_cosine_vs_cached_float16": minimum_cosine_cached,
        "cached_float16_exact_samples": cached_float16_exact,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.9),
        "execution_environment": "mac_host_coreml",
        "hardware_performance_claim": False,
        "thermals_interpretable": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop-root", type=Path,
        default=Path("data/local/citizen100_v17/hand_rgb/val"),
    )
    parser.add_argument(
        "--embedding-root", type=Path,
        default=Path("data/local/citizen100_v17/hand_mobileclip2_s0/val"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/mobile_100gloss_v17/mobileclip2_image_fp32.json"),
    )
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    parser.add_argument("--maximum-parity-samples", type=int, default=0)
    parser.add_argument("--reparameterized-gate", type=float, default=1e-4)
    parser.add_argument("--max-abs-gate", type=float, default=2e-4)
    parser.add_argument("--cosine-gate", type=float, default=0.99999)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
