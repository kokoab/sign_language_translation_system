#!/usr/bin/env python3
"""Measure v17 landmark availability without treating missing points as observations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_v17 import load_v17_result
    from active.v17.extract_mediapipe_v17 import load_mediapipe_v17_result
    from active.v17.schema_mediapipe_v17 import MediaPipeV17Config
    from active.v17.schema_v17 import NODE_NAMES, V17Config
else:
    from .extract_v17 import load_v17_result
    from .extract_mediapipe_v17 import load_mediapipe_v17_result
    from .schema_mediapipe_v17 import MediaPipeV17Config
    from .schema_v17 import NODE_NAMES, V17Config


def summary(values: list[float]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    percentile_10 = ordered[round(0.10 * (len(ordered) - 1))]
    percentile_90 = ordered[round(0.90 * (len(ordered) - 1))]
    return (
        f"{min(values):.4f} / {percentile_10:.4f} / "
        f"{statistics.median(values):.4f} / {percentile_90:.4f} / {max(values):.4f}"
    )


def audit(
    root: Path, extractor: str = "apple"
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows = []
    node_observed = np.zeros(len(NODE_NAMES), dtype=np.int64)
    node_total = 0
    for path in sorted(root.rglob("*.v17.npz")):
        if extractor == "apple":
            result = load_v17_result(path, V17Config())
        elif extractor == "mediapipe_t50":
            config = MediaPipeV17Config(
                minimum_hand_detection_confidence=0.50,
                minimum_hand_presence_confidence=0.50,
                minimum_hand_tracking_confidence=0.50,
            )
            result = load_mediapipe_v17_result(path, config)
        else:
            raise ValueError("extractor must be apple or mediapipe_t50")
        features = result.features.astype(np.float32)
        presence = features[..., 3] > 0.5
        confidence = features[..., 4]
        left_active = presence[:, :21].any(axis=1)
        right_active = presence[:, 21:42].any(axis=1)
        hand_active = left_active | right_active
        both_active = left_active & right_active
        left_completeness = presence[left_active, :21].mean() if left_active.any() else 0.0
        right_completeness = presence[right_active, 21:42].mean() if right_active.any() else 0.0
        observed_confidence = confidence[presence]
        relative = path.relative_to(root)
        rows.append(
            {
                "feature_path": str(relative),
                "split": relative.parts[0],
                "label": relative.parts[-2],
                "hand_active_output_frames": float(hand_active.mean()),
                "left_active_output_frames": float(left_active.mean()),
                "right_active_output_frames": float(right_active.mean()),
                "both_hands_active_output_frames": float(both_active.mean()),
                "left_joint_completeness_when_active": float(left_completeness),
                "right_joint_completeness_when_active": float(right_completeness),
                "observed_point_confidence_mean": float(observed_confidence.mean()),
                "observed_point_confidence_min": float(observed_confidence.min()),
                "detected_hand_source_frames_after_trim": float(
                    result.diagnostics["observed_hand_frame_fraction"]
                ),
                "detected_hand_source_frames_before_trim": float(
                    result.diagnostics["observed_hand_frame_fraction_before_trim"]
                ),
            }
        )
        node_observed += presence.sum(axis=0)
        node_total += presence.shape[0]
    return rows, node_observed / max(node_total, 1)


def write_outputs(
    root: Path,
    rows: list[dict[str, object]],
    node_coverage: np.ndarray,
    csv_path: Path,
    report_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = (
        "hand_active_output_frames",
        "left_active_output_frames",
        "right_active_output_frames",
        "both_hands_active_output_frames",
        "left_joint_completeness_when_active",
        "right_joint_completeness_when_active",
        "observed_point_confidence_mean",
        "detected_hand_source_frames_before_trim",
        "detected_hand_source_frames_after_trim",
    )
    low = sorted(rows, key=lambda row: float(row["hand_active_output_frames"]))[:20]
    critical = [row for row in rows if float(row["hand_active_output_frames"]) < 0.25]
    class_values: dict[str, list[float]] = {}
    for row in rows:
        class_values.setdefault(str(row["label"]), []).append(
            float(row["hand_active_output_frames"])
        )
    weakest_classes = sorted(
        (
            (label, statistics.median(values), min(values), len(values))
            for label, values in class_values.items()
        ),
        key=lambda item: item[1],
    )[:15]

    lines = [
        "# Citizen100 v17 Landmark Availability Audit",
        "",
        f"- Feature root: `{root}`",
        f"- Archives measured: {len(rows)}",
        f"- Clips below 25% hand-active output frames: {len(critical)}",
        "- Values below are min / p10 / median / p90 / max.",
        "- This audit measures missingness and confidence. Visual overlay review is",
        "  still required to judge geometric placement or semantic correctness.",
        "",
        "## Clip-level metrics",
        "",
        "| Metric | Min / p10 / median / p90 / max |",
        "| --- | --- |",
    ]
    for metric in metrics:
        lines.append(f"| `{metric}` | {summary([float(row[metric]) for row in rows])} |")
    lines.extend(
        [
            "",
            "## Per-node presence",
            "",
            "Presence counts include explicit bounded interpolation, but never Kalman",
            "extrapolation. A missing point remains zero and contributes no motion feature.",
            "",
            "| Node | Presence fraction |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{name}` | {coverage:.4f} |"
        for name, coverage in zip(NODE_NAMES, node_coverage)
    )
    lines.extend(
        [
            "",
            "## Weakest classes by median hand-active output frames",
            "",
            "| Class | Median | Minimum | Clips |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| {label} | {median:.4f} | {minimum:.4f} | {count} |"
        for label, median, minimum, count in weakest_classes
    )
    lines.extend(
        [
            "",
            "## Lowest-coverage clips",
            "",
            "| Feature | Hand-active frames | Left completeness | Right completeness |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        f"| `{row['feature_path']}` | {float(row['hand_active_output_frames']):.4f} | "
        f"{float(row['left_joint_completeness_when_active']):.4f} | "
        f"{float(row['right_joint_completeness_when_active']):.4f} |"
        for row in low
    )
    lines.extend(["", f"Full clip measurements: `{csv_path}`", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--extractor", choices=("apple", "mediapipe_t50"), default="apple"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/reports/citizen100_v17_landmark_quality.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/CITIZEN100_V17_LANDMARK_QUALITY.md"),
    )
    args = parser.parse_args()
    rows, node_coverage = audit(args.root, args.extractor)
    if not rows:
        raise SystemExit("no v17 archives found")
    write_outputs(args.root, rows, node_coverage, args.csv, args.report)
    print(
        json.dumps(
            {
                "archives": len(rows),
                "below_25_percent_hand_active": sum(
                    float(row["hand_active_output_frames"]) < 0.25 for row in rows
                ),
                "median_hand_active": statistics.median(
                    float(row["hand_active_output_frames"]) for row in rows
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
