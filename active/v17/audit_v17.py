#!/usr/bin/env python3
"""Validate and summarize a directory of v17 feature archives."""

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
    from active.v17.schema_v17 import NUM_CHANNELS, NUM_NODES, V17Config
else:
    from .extract_v17 import load_v17_result
    from .schema_v17 import NUM_CHANNELS, NUM_NODES, V17Config


def _number_summary(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"{min(values):.4f} / {statistics.median(values):.4f} / "
        f"{max(values):.4f}"
    )


def audit_directory(root: Path, config: V17Config) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    for path in sorted(root.rglob("*.v17.npz")):
        try:
            result = load_v17_result(path, config)
            features = result.features.astype(np.float32)
            presence = features[..., 3]
            missing = presence == 0
            binary_presence = bool(np.isin(presence, (0.0, 1.0)).all())
            missing_spatial_zero = bool((features[..., :3][missing] == 0).all())
            missing_confidence_zero = bool((features[..., 4][missing] == 0).all())
            chirality = result.diagnostics.get("chirality_observation_counts", {})
            video_path = Path(str(result.metadata.get("video_path", "")))
            relative = path.relative_to(root)
            rows.append(
                {
                    "feature_path": str(relative),
                    "video_path": str(video_path),
                    "split": relative.parts[0] if len(relative.parts) >= 3 else "",
                    "label": relative.parts[-2] if len(relative.parts) >= 2 else "",
                    "orientation": result.metadata.get("orientation", ""),
                    "width": result.metadata.get("oriented_width", 0),
                    "height": result.metadata.get("oriented_height", 0),
                    "source_frames_before_trim": result.metadata.get(
                        "source_frames_before_hand_trim", 0
                    ),
                    "source_frames_after_trim": result.metadata.get(
                        "source_frames_processed", 0
                    ),
                    "elapsed_seconds": result.diagnostics.get("elapsed_seconds", 0.0),
                    "hand_frame_fraction_before_trim": result.diagnostics.get(
                        "observed_hand_frame_fraction_before_trim", 0.0
                    ),
                    "hand_frame_fraction_after_trim": result.diagnostics.get(
                        "observed_hand_frame_fraction", 0.0
                    ),
                    "hand_presence_fraction": result.diagnostics.get(
                        "hand_presence_fraction", 0.0
                    ),
                    "face_presence_fraction": result.diagnostics.get(
                        "face_presence_fraction", 0.0
                    ),
                    "body_presence_fraction": result.diagnostics.get(
                        "body_presence_fraction", 0.0
                    ),
                    "shoulder_coverage": result.diagnostics.get("shoulder_coverage", 0.0),
                    "normalization_source": result.diagnostics.get(
                        "normalization_scale_source", ""
                    ),
                    "chirality_left": chirality.get("left", 0),
                    "chirality_right": chirality.get("right", 0),
                    "chirality_unknown": chirality.get("unknown", 0),
                    "finite": bool(np.isfinite(features).all()),
                    "binary_presence": binary_presence,
                    "missing_spatial_zero": missing_spatial_zero,
                    "missing_confidence_zero": missing_confidence_zero,
                    "valid_shape": features.shape
                    == (config.target_frames, NUM_NODES, NUM_CHANNELS),
                }
            )
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return rows, errors


def write_report(
    root: Path,
    rows: list[dict[str, object]],
    errors: list[str],
    csv_path: Path,
    markdown_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    checks = ("finite", "binary_presence", "missing_spatial_zero", "missing_confidence_zero", "valid_shape")
    passed = sum(all(bool(row[name]) for name in checks) for row in rows)
    orientations: dict[str, int] = {}
    normalization: dict[str, int] = {}
    for row in rows:
        orientation = str(row["orientation"])
        source = str(row["normalization_source"])
        orientations[orientation] = orientations.get(orientation, 0) + 1
        normalization[source] = normalization.get(source, 0) + 1

    metric_names = (
        "elapsed_seconds",
        "hand_frame_fraction_before_trim",
        "hand_frame_fraction_after_trim",
        "hand_presence_fraction",
        "face_presence_fraction",
        "body_presence_fraction",
        "shoulder_coverage",
    )
    metrics = {
        name: [float(row[name]) for row in rows]
        for name in metric_names
    }
    chirality = {
        name: sum(int(row[f"chirality_{name}"]) for row in rows)
        for name in ("left", "right", "unknown")
    }
    status = "PASS" if rows and passed == len(rows) and not errors else "FAIL"
    lines = [
        "# v17 Extractor Audit",
        "",
        f"**Status: {status}**",
        "",
        f"- Input: `{root}`",
        f"- Loaded archives: {len(rows)}",
        f"- Archives passing all invariants: {passed}/{len(rows)}",
        f"- Load/schema errors: {len(errors)}",
        f"- Orientation counts: `{json.dumps(orientations, sort_keys=True)}`",
        f"- Normalization sources: `{json.dumps(normalization, sort_keys=True)}`",
        f"- Chirality observations: `{json.dumps(chirality, sort_keys=True)}`",
        "",
        "## Metrics",
        "",
        "Values are min / median / max across videos.",
        "",
        "| Metric | Min / median / max |",
        "| --- | --- |",
    ]
    for name in metric_names:
        lines.append(f"| `{name}` | {_number_summary(metrics[name])} |")
    lines.extend(
        [
            "",
            "## Enforced invariants",
            "",
            "Every archive must load with the current schema fingerprint, have shape "
            "`[32, 61, 5]`, contain only finite values, use binary presence values, and "
            "keep missing spatial/depth/confidence channels exactly zero.",
            "",
            f"Per-video measurements: `{csv_path}`",
        ]
    )
    if errors:
        lines.extend(["", "## Errors", ""] + [f"- {error}" for error in errors])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v17 feature archives")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--csv", type=Path, default=Path("artifacts/reports/v17_extractor_audit.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/reports/V17_EXTRACTOR_AUDIT.md")
    )
    args = parser.parse_args()
    rows, errors = audit_directory(args.input, V17Config())
    write_report(args.input, rows, errors, args.csv, args.report)
    passed = rows and not errors and all(
        all(bool(row[name]) for name in ("finite", "binary_presence", "missing_spatial_zero", "missing_confidence_zero", "valid_shape"))
        for row in rows
    )
    print(json.dumps({"archives": len(rows), "errors": len(errors), "passed": bool(passed)}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
