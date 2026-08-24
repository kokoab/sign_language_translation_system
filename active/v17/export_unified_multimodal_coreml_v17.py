#!/usr/bin/env python3
"""Export the selected unified landmark/hand student to one FP16 Core ML graph."""

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

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.export_stage1_coreml_v17 import replace_attention
from active.v17.model_hand_mobileclip2_v17 import (
    HandMobileCLIP2Stage1Config,
    HandMobileCLIP2Stage1V17,
)
from active.v17.model_unified_multimodal_v17 import (
    UnifiedFusionHeadV17,
    UnifiedMultimodalStage1V17,
    UnifiedMultimodalV17Config,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(item).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_model(checkpoint_path: Path) -> tuple[UnifiedMultimodalStage1V17, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_unified_multimodal_v17":
        raise ValueError("not a unified v17 checkpoint")
    landmark = SLTStage1V17(Stage1V17Config(**checkpoint["landmark_model_config"]))
    landmark.load_state_dict(checkpoint["landmark_model_state_dict"], strict=True)
    hand = HandMobileCLIP2Stage1V17(
        HandMobileCLIP2Stage1Config(**checkpoint["hand_model_config"])
    )
    hand.load_state_dict(checkpoint["hand_model_state_dict"], strict=True)
    head = UnifiedFusionHeadV17(UnifiedMultimodalV17Config(**checkpoint["head_config"]))
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    return UnifiedMultimodalStage1V17(landmark, hand, head).eval(), checkpoint


class ExportWrapper(nn.Module):
    def __init__(self, model: UnifiedMultimodalStage1V17):
        super().__init__()
        self.model = model

    def forward(self, landmarks, hand_embeddings, hand_valid, hand_boxes):
        return self.model(landmarks, hand_embeddings, hand_valid > 0.5, hand_boxes)


def load_pair(landmark_path: Path, hand_path: Path) -> tuple[np.ndarray, ...]:
    with np.load(landmark_path, allow_pickle=False) as payload:
        landmarks = payload["features"].astype(np.float32)[None]
    with np.load(hand_path, allow_pickle=False) as payload:
        embeddings = payload["embeddings"].astype(np.float32)[None]
        valid = payload["valid"].astype(np.float32)[None]
        boxes = payload["boxes_normalized"].astype(np.float32)[None]
    return landmarks, embeddings, valid, boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--sample-landmark", type=Path,
        default=Path("data/local/citizen100_v17/landmarks/val/HELLO/020030442376253177-HELLO.v17.npz"),
    )
    parser.add_argument(
        "--sample-hand", type=Path,
        default=Path("data/local/citizen100_v17/hand_mobileclip2_s0/val/HELLO/020030442376253177-HELLO.hand_mobileclip2_v17.npz"),
    )
    parser.add_argument("--parity-landmark-root", type=Path, default=Path("data/local/citizen100_v17/landmarks/val"))
    parser.add_argument("--parity-hand-root", type=Path, default=Path("data/local/citizen100_v17/hand_mobileclip2_s0/val"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float32")
    args = parser.parse_args()
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.sample_landmark, args.sample_hand, args.parity_landmark_root, args.parity_hand_root
    )):
        raise ValueError("Core ML parity is restricted to non-test data")

    model, checkpoint = load_model(args.checkpoint)
    export_model = copy.deepcopy(model)
    replace_attention(export_model)
    wrapper = ExportWrapper(export_model).eval()
    sample_arrays = load_pair(args.sample_landmark, args.sample_hand)
    sample = tuple(torch.from_numpy(value) for value in sample_arrays)
    with torch.inference_mode():
        reference = model(sample[0], sample[1], sample[2] > 0.5, sample[3]).numpy()
        replacement = wrapper(*sample).numpy()
    manual_max_abs = float(np.max(np.abs(reference - replacement)))
    if manual_max_abs > 1e-4:
        raise ValueError(f"manual attention parity failed: {manual_max_abs}")
    traced = torch.jit.trace(wrapper, sample, strict=False)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="landmarks", shape=(1, 32, 61, 5), dtype=np.float32),
            ct.TensorType(name="hand_embeddings", shape=(1, 16, 3, 512), dtype=np.float32),
            ct.TensorType(name="hand_valid", shape=(1, 16, 3), dtype=np.float32),
            ct.TensorType(name="hand_boxes", shape=(1, 16, 3, 4), dtype=np.float32),
        ],
        convert_to="mlprogram",
        compute_precision=(
            ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
        ),
        minimum_deployment_target=ct.target.iOS15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    runtime = ct.models.MLModel(str(args.output), compute_units=ct.ComputeUnit.ALL)
    names = list(runtime.get_spec().description.input)
    input_names = [value.name for value in names]
    output_name = runtime.get_spec().description.output[0].name
    provider = dict(zip(input_names, sample_arrays))
    for _ in range(10):
        runtime.predict(provider)
    timings = []
    prediction = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        prediction = runtime.predict(provider)
        timings.append(1000.0 * (time.perf_counter() - started))
    coreml_output = np.asarray(prediction[output_name]).reshape(reference.shape)

    mismatch = 0
    max_abs = 0.0
    paths = sorted(args.parity_landmark_root.glob("*/*.v17.npz"))
    with torch.inference_mode():
        for landmark_path in paths:
            stem = landmark_path.name.removesuffix(".v17.npz")
            hand_path = args.parity_hand_root / landmark_path.parent.name / f"{stem}.hand_mobileclip2_v17.npz"
            arrays = load_pair(landmark_path, hand_path)
            values = tuple(torch.from_numpy(value) for value in arrays)
            torch_output = model(values[0], values[1], values[2] > 0.5, values[3]).numpy()
            ml_output = np.asarray(runtime.predict(dict(zip(input_names, arrays)))[output_name]).reshape(torch_output.shape)
            max_abs = max(max_abs, float(np.max(np.abs(torch_output - ml_output))))
            mismatch += int(torch_output.argmax() != ml_output.argmax())
    ordered = sorted(timings)
    result = {
        "format": "slt_v17_unified_multimodal_coreml_export",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_seed": checkpoint["seed"],
        "parameters": model.parameter_count,
        "inputs": {
            "landmarks": [1, 32, 61, 5],
            "hand_embeddings": [1, 16, 3, 512],
            "hand_valid": [1, 16, 3],
            "hand_boxes": [1, 16, 3, 4],
        },
        "hand_rgb_embedding_preprocessor_in_graph": False,
        "format_coreml": f"mlprogram_{args.precision}",
        "minimum_deployment_target": "iOS15",
        "package_bytes": directory_bytes(args.output),
        "package_mib": directory_bytes(args.output) / 2**20,
        "package_tree_sha256": tree_sha256(args.output),
        "manual_attention_max_abs": manual_max_abs,
        "sample_coreml_max_abs": float(np.max(np.abs(reference - coreml_output))),
        "sample_top1_match": int(reference.argmax()) == int(coreml_output.argmax()),
        "parity_samples": len(paths),
        "parity_max_abs": max_abs,
        "parity_top1_mismatches": mismatch,
        "warm_iterations": args.iterations,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))],
        "compute_units": "ALL_on_current_Mac_not_iPhone",
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    result_path = args.output.parent / f"{args.output.stem}_benchmark.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
