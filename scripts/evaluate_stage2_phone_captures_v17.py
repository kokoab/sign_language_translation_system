#!/usr/bin/env python3
"""Replay saved physical-phone tensors through a v17 Stage-2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.model_stage2_v17 import (
    FrozenUnifiedTemporalEncoderV17,
    load_frozen_unified_stage1,
    load_stage2_model_v17,
)


OTHER_CLASS_INDEX = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collapse(values):
    output = []
    previous = None
    for value in values:
        token = int(value)
        if token != previous and token != 0:
            output.append(token - 1)
        previous = token
    return output


def run(args):
    labels = [row["canonical_label"] for row in json.loads(args.vocabulary.read_text())["classes"]]
    landmark, hand, fusion, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    encoder = FrozenUnifiedTemporalEncoderV17(landmark, hand, fusion).eval()
    head, checkpoint = load_stage2_model_v17(args.stage2_checkpoint)
    head.eval()
    if head.config.num_classes not in (100, 101):
        raise ValueError("phone replay supports the locked vocabulary with optional OTHER")
    expected = tuple(value.upper() for value in args.expected)
    rows = []
    for capture in sorted(path for path in args.captures.iterdir() if path.is_dir()):
        tensors = {
            name: np.load(capture / f"{name}.npy")
            for name in ("model_landmarks", "hand_features", "hand_valid", "hand_boxes", "window_mask")
        }
        with torch.inference_mode():
            frozen = encoder(
                torch.from_numpy(tensors["model_landmarks"]).float(),
                torch.from_numpy(tensors["hand_features"]).float(),
                torch.from_numpy(tensors["hand_valid"]).bool(),
                torch.from_numpy(tensors["hand_boxes"]).float(),
            )
            logits, lengths = head(
                frozen, torch.from_numpy(tensors["window_mask"]).bool()
            )
        classes = collapse(logits[0, :int(lengths[0])].argmax(dim=-1).tolist())
        raw = ["__OTHER__" if value == OTHER_CLASS_INDEX else labels[value] for value in classes]
        clean = tuple(value for value in raw if value != "__OTHER__")
        report = json.loads((capture / "report.json").read_text())
        rows.append({
            "capture_id": capture.name,
            "created_utc": report.get("createdUTC"),
            "deployed_prediction": report.get("predictedGlosses"),
            "raw_prediction": raw,
            "prediction": list(clean),
            "expected": list(expected),
            "exact": clean == expected,
            "window_count": int(tensors["window_mask"].sum()),
        })
    result = {
        "format": "slt_stage2_phone_tensor_replay_v17",
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "stage2_checkpoint": args.stage2_checkpoint.as_posix(),
        "stage2_checkpoint_sha256": sha256(args.stage2_checkpoint),
        "captures": len(rows),
        "exact": sum(row["exact"] for row in rows),
        "expected_sequence": list(expected),
        "rows": rows,
        "development_captures_only": True,
        "physical_device_latency_claim": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--captures", type=Path, default=Path("data/local/stage2_v17_phone_development/captures"))
    value.add_argument("--vocabulary", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    value.add_argument("--stage2-checkpoint", type=Path, required=True)
    value.add_argument("--expected", nargs="+", default=("I", "NEED", "HELP"))
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_phone_tensor_replay/latest.json"))
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
