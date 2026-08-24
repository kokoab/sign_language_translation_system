#!/usr/bin/env python3
"""Compare fixed landmark/RGB late-fusion weights on aligned validation logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.evaluate_stage_1_mobileclip2_v17 import metrics_from_logits


LANDMARK_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)


def standardized(logits: np.ndarray) -> np.ndarray:
    centered = logits - logits.mean(axis=1, keepdims=True)
    return centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-6)


def load_aligned(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return payload["logits"], payload["targets"], payload["item_ids"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmark", type=Path, default=Path("artifacts/reports/stage1_v17_validation/logits.npz"))
    parser.add_argument("--rgb", type=Path, default=Path("artifacts/reports/stage1_v17_mobileclip2_s0_validation/logits.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/stage1_v17_late_fusion_validation.json"))
    args = parser.parse_args()
    landmark, landmark_targets, landmark_ids = load_aligned(args.landmark)
    rgb, rgb_targets, rgb_ids = load_aligned(args.rgb)
    landmark_order = {str(item): index for index, item in enumerate(landmark_ids)}
    if set(map(str, landmark_ids)) != set(map(str, rgb_ids)):
        raise ValueError("landmark/RGB item IDs do not match")
    order = np.asarray([landmark_order[str(item)] for item in rgb_ids])
    landmark = landmark[order]
    landmark_targets = landmark_targets[order]
    if not np.array_equal(landmark_targets, rgb_targets):
        raise ValueError("landmark/RGB target vectors differ")
    landmark = standardized(landmark)
    rgb = standardized(rgb)
    rows = []
    for weight in LANDMARK_WEIGHTS:
        fused = weight * landmark + (1.0 - weight) * rgb
        rows.append({"landmark_weight": weight, **metrics_from_logits(fused, rgb_targets)})
    result = {
        "split": "val",
        "normalization": "per-sample logit z-score",
        "predeclared_landmark_weights": list(LANDMARK_WEIGHTS),
        "results": rows,
        "best_by_top1": max(rows, key=lambda row: (row["top1"], row["macro_f1"])),
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
