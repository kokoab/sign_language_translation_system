#!/usr/bin/env python3
"""Export the landmark-only Stage-2 preview as one Core ML package."""

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
from active.v17.export_stage2_coreml_v17 import replace_masked_attention
from active.v17.model_stage2_v17 import (
    Stage2TemporalHeadV17,
    Stage2V17Config,
    load_frozen_unified_stage1,
)
from active.v17.train_stage_2_v17 import collapse_ctc


class LandmarkCascadeCoreML(nn.Module):
    def __init__(self, landmark: nn.Module, head: Stage2TemporalHeadV17):
        super().__init__()
        self.landmark = landmark
        self.head = head

    def forward(self, landmarks, window_mask):
        batch, windows = landmarks.shape[:2]
        tokens, _ = self.landmark.encode(
            landmarks.reshape(batch * windows, 32, 61, 5)
        )
        landmark_logits = self.landmark.classifier(tokens)
        features = torch.cat((tokens, torch.zeros_like(tokens), landmark_logits), -1)
        features = features.reshape(batch, windows, 32, 612)
        features = features * (window_mask > 0.5).unsqueeze(-1).unsqueeze(-1)
        logits, _ = self.head(features, window_mask > 0.5)
        return logits


def raw_rows(root: Path) -> list[Path]:
    rows = sorted((root / "validation").glob("*/*.stage2_rgb_v17.npz"))
    if not rows:
        raise ValueError(f"no validation landmark rows under {root}")
    return rows


def arrays(path: Path, maximum_windows: int) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as payload:
        landmarks = payload["landmarks"].astype(np.float32)
    if not 1 <= len(landmarks) <= maximum_windows:
        raise ValueError(f"{path}: invalid window count")
    padded = np.zeros((1, maximum_windows, 32, 61, 5), np.float32)
    mask = np.zeros((1, maximum_windows), np.float32)
    padded[0, :len(landmarks)] = landmarks
    mask[0, :len(landmarks)] = 1
    return padded, mask, len(landmarks)


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * fraction))]


def run(args: argparse.Namespace) -> dict[str, object]:
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.preview, args.stage1_checkpoint, args.parity_root
    )):
        raise ValueError("Core ML export parity is restricted to validation data")
    payload = torch.load(args.preview, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_landmark_preview_v17":
        raise ValueError("not a landmark Stage-2 preview checkpoint")
    head = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    head.load_state_dict(payload["model_state_dict"], strict=True)
    landmark, _, _, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    original = LandmarkCascadeCoreML(landmark, head).eval()
    export_landmark = copy.deepcopy(landmark)
    export_head = copy.deepcopy(head)
    replace_attention(export_landmark)
    replace_masked_attention(export_head)
    export = LandmarkCascadeCoreML(export_landmark, export_head).eval()
    rows = raw_rows(args.parity_root)
    sample = arrays(rows[0], head.config.max_windows)
    tensors = tuple(torch.from_numpy(value) for value in sample[:2])
    with torch.inference_mode():
        reference = original(*tensors)
        replacement = export(*tensors)
    manual_max_abs = float((reference - replacement).abs().max())
    if manual_max_abs > 1e-4:
        raise ValueError(f"manual attention parity failed: {manual_max_abs}")
    traced = torch.jit.trace(export, tensors, strict=False)
    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="landmarks", shape=(1, head.config.max_windows, 32, 61, 5),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="window_mask", shape=(1, head.config.max_windows),
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
    output_name = runtime.get_spec().description.output[0].name
    for _ in range(args.warmup_iterations):
        runtime.predict({"landmarks": sample[0], "window_mask": sample[1]})
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        runtime.predict({"landmarks": sample[0], "window_mask": sample[1]})
        timings.append((time.perf_counter() - started) * 1000)
    maximum = 0.0
    decode_mismatches = 0
    with torch.inference_mode():
        for path in rows:
            landmarks, mask, windows = arrays(path, head.config.max_windows)
            torch_logits = original(
                torch.from_numpy(landmarks), torch.from_numpy(mask)
            ).numpy()
            coreml_logits = np.asarray(runtime.predict({
                "landmarks": landmarks, "window_mask": mask,
            })[output_name]).reshape(torch_logits.shape)
            maximum = max(maximum, float(np.max(np.abs(torch_logits - coreml_logits))))
            length = windows * head.config.tokens_per_window
            decode_mismatches += int(
                collapse_ctc(torch_logits[0, :length].argmax(-1))
                != collapse_ctc(coreml_logits[0, :length].argmax(-1))
            )
    report = {
        "format": "slt_stage2_landmark_cascade_coreml_v17",
        "version": 1,
        "preview": args.preview.as_posix(),
        "preview_sha256": sha256_file(args.preview),
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256_file(args.stage1_checkpoint),
        "coreml_package": args.output.as_posix(),
        "coreml_package_tree_sha256": tree_sha256(args.output),
        "coreml_package_mib": directory_bytes(args.output) / 2**20,
        "inputs": {
            "landmarks": [1, head.config.max_windows, 32, 61, 5],
            "window_mask": [1, head.config.max_windows],
        },
        "manual_attention_max_abs": manual_max_abs,
        "coreml_max_abs": maximum,
        "parity_samples": len(rows),
        "parity_decode_mismatches": decode_mismatches,
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p90": percentile(timings, 0.9),
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--preview", type=Path, default=Path(
        "artifacts/models/stage2_v17_landmark_cascade_preview_v1/best_model.pth"
    ))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path(
        "artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"
    ))
    value.add_argument("--parity-root", type=Path, default=Path(
        "data/local/stage2_v17_multimodal"
    ))
    value.add_argument("--output", type=Path, default=Path(
        "artifacts/coreml/Stage2LandmarkCascadePreviewV17FP32.mlpackage"
    ))
    value.add_argument("--report", type=Path, default=Path(
        "artifacts/reports/stage2_v17_landmark_cascade_preview_v1/coreml.json"
    ))
    value.add_argument("--precision", choices=("float16", "float32"), default="float32")
    value.add_argument("--warmup-iterations", type=int, default=3)
    value.add_argument("--iterations", type=int, default=20)
    return value


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
