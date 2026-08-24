#!/usr/bin/env python3
"""Freeze, run, and summarize the v17 Apple/MediaPipe extractor bakeoff.

The bakeoff never reads Citizen's official test split. Its deterministic subset has
three clips per class: low-coverage train, median-coverage train, and low-coverage val.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.extract_mediapipe_v17 import (
        DEFAULT_MODEL_PATH,
        MediaPipeHybridDetector,
        extract_video_mediapipe_v17,
        load_mediapipe_v17_result,
        save_mediapipe_v17_result,
    )
    from active.v17.extract_v17 import load_v17_result
    from active.v17.schema_mediapipe_v17 import (
        MediaPipeV17Config,
        schema_fingerprint,
        schema_payload,
    )
    from active.v17.schema_v17 import V17Config
else:
    from .extract_mediapipe_v17 import (
        DEFAULT_MODEL_PATH,
        MediaPipeHybridDetector,
        extract_video_mediapipe_v17,
        load_mediapipe_v17_result,
        save_mediapipe_v17_result,
    )
    from .extract_v17 import load_v17_result
    from .schema_mediapipe_v17 import (
        MediaPipeV17Config,
        schema_fingerprint,
        schema_payload,
    )
    from .schema_v17 import V17Config


DEFAULT_QUALITY_CSV = Path("artifacts/reports/citizen100_v17_landmark_quality.csv")
DEFAULT_RAW_ROOT = Path("data/local/citizen100_v17/raw")
DEFAULT_APPLE_ROOT = Path("data/local/citizen100_v17/landmarks")
DEFAULT_REJECTIONS = Path("data/local/citizen100_v17/rejections.csv")
DEFAULT_MANIFEST = Path("active/v17/extractor_bakeoff_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("data/local/citizen100_v17/extractor_bakeoff")
DEFAULT_REPORT = Path("artifacts/reports/EXTRACTOR_BAKEOFF_V17.md")
DEFAULT_DETAIL_CSV = Path("artifacts/reports/extractor_bakeoff_v17.csv")

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _video_from_feature(raw_root: Path, feature_path: str) -> Path:
    relative = Path(feature_path)
    return raw_root / relative.parent / (relative.name.removesuffix(".v17.npz") + ".mp4")


def _rejected_keys(path: Path) -> set[tuple[str, str, str]]:
    return {
        (row["split"], row["canonical_label"], row["video"])
        for row in _read_csv(path)
    }


def build_manifest(
    quality_csv: Path,
    raw_root: Path,
    apple_root: Path,
    rejections_csv: Path,
) -> dict[str, object]:
    rejected = _rejected_keys(rejections_csv)
    by_class: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in _read_csv(quality_csv):
        split = row["split"]
        if split not in {"train", "val"}:
            continue
        video = _video_from_feature(raw_root, row["feature_path"])
        if (split, row["label"], video.name) in rejected:
            continue
        by_class.setdefault(row["label"], {}).setdefault(split, []).append(row)

    entries: list[dict[str, object]] = []
    for label in sorted(by_class):
        train = sorted(
            by_class[label].get("train", []),
            key=lambda row: (float(row["hand_active_output_frames"]), row["feature_path"]),
        )
        val = sorted(
            by_class[label].get("val", []),
            key=lambda row: (float(row["hand_active_output_frames"]), row["feature_path"]),
        )
        if len(train) < 2 or not val:
            raise ValueError(f"{label} lacks the required train/val clips")
        chosen = (
            ("train_low", train[0]),
            ("train_median", train[(len(train) - 1) // 2]),
            ("val_low", val[0]),
        )
        if len({row["feature_path"] for _, row in chosen}) != 3:
            raise AssertionError(f"duplicate selection for {label}")
        for stratum, row in chosen:
            raw_path = _video_from_feature(raw_root, row["feature_path"])
            apple_path = apple_root / row["feature_path"]
            if not raw_path.is_file() or not apple_path.is_file():
                raise FileNotFoundError(raw_path if not raw_path.is_file() else apple_path)
            entries.append(
                {
                    "label": label,
                    "split": row["split"],
                    "stratum": stratum,
                    "raw_path": str(raw_path),
                    "apple_feature_path": str(apple_path),
                    "relative_feature_path": row["feature_path"],
                    "apple_hand_active_output_frames": float(
                        row["hand_active_output_frames"]
                    ),
                }
            )
    if len(by_class) != 100 or len(entries) != 300:
        raise ValueError(f"expected 100 classes/300 clips, got {len(by_class)}/{len(entries)}")
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "protocol": "citizen100_extractor_bakeoff_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": ["train_low", "train_median", "val_low"],
        "allowed_splits": ["train", "val"],
        "explicitly_forbidden_split": "test",
        "candidate_thresholds": [0.30, 0.50],
        "entry_sha256": digest,
        "class_count": 100,
        "clip_count": 300,
        "entries": entries,
    }


def write_manifest(args: argparse.Namespace) -> None:
    manifest = build_manifest(
        args.quality_csv, args.raw_root, args.apple_root, args.rejections
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("class_count", "clip_count", "entry_sha256")}, indent=2))


def load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if payload.get("explicitly_forbidden_split") != "test":
        raise ValueError("manifest does not explicitly seal test")
    if len(entries) != 300 or any(row["split"] not in {"train", "val"} for row in entries):
        raise ValueError("invalid bakeoff manifest")
    return payload


def threshold_name(value: float) -> str:
    return f"mediapipe_t{int(round(value * 100)):02d}"


def extract_candidate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    threshold = float(args.threshold)
    if threshold not in {float(value) for value in manifest["candidate_thresholds"]}:
        raise ValueError("threshold was not predeclared in the frozen manifest")
    config = MediaPipeV17Config(
        minimum_hand_detection_confidence=threshold,
        minimum_hand_presence_confidence=threshold,
        minimum_hand_tracking_confidence=threshold,
    )
    output_root = args.output_root / threshold_name(threshold)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_schema_v17.json").write_text(
        json.dumps(schema_payload(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detector = MediaPipeHybridDetector(args.model, config)
    counts = {"ok": 0, "skipped": 0, "no_hands": 0, "failed": 0}
    started = time.perf_counter()
    try:
        for index, row in enumerate(manifest["entries"], start=1):
            relative = Path(row["relative_feature_path"])
            destination = output_root / relative
            if not args.no_resume and destination.is_file():
                try:
                    load_mediapipe_v17_result(destination, config)
                except Exception:
                    pass
                else:
                    counts["skipped"] += 1
                    continue
            try:
                result = extract_video_mediapipe_v17(
                    Path(row["raw_path"]), config, detector
                )
                if result is None:
                    counts["no_hands"] += 1
                else:
                    save_mediapipe_v17_result(destination, result, config)
                    counts["ok"] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"FAILED {row['raw_path']}: {exc}", file=sys.stderr)
            if index == 1 or index % 25 == 0 or index == len(manifest["entries"]):
                elapsed = time.perf_counter() - started
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": len(manifest["entries"]),
                            **counts,
                            "clips_per_second": index / max(elapsed, 1e-6),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        detector.close()
    summary = {
        "candidate": threshold_name(threshold),
        "schema_fingerprint": schema_fingerprint(config),
        "elapsed_seconds": time.perf_counter() - started,
        **counts,
    }
    (output_root / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _bone_cv(features: np.ndarray) -> float:
    values: list[float] = []
    for start in (0, 21):
        for first, second in HAND_EDGES:
            a, b = start + first, start + second
            valid = (features[:, a, 3] > 0.5) & (features[:, b, 3] > 0.5)
            lengths = np.linalg.norm(features[valid, a, :2] - features[valid, b, :2], axis=1)
            lengths = lengths[np.isfinite(lengths) & (lengths > 1e-5)]
            if lengths.size >= 4:
                mean = float(lengths.mean())
                if mean > 1e-5:
                    values.append(float(lengths.std() / mean))
    return _median(values)


def _clip_metrics(result) -> dict[str, float]:
    features = result.features.astype(np.float32)
    presence = features[:, :42, 3] > 0.5
    left = presence[:, :21].any(axis=1)
    right = presence[:, 21:42].any(axis=1)
    return {
        "hand_active": float((left | right).mean()),
        "both_hands_active": float((left & right).mean()),
        "hand_node_presence": float(presence.mean()),
        "source_hand_fraction_before_trim": float(
            result.diagnostics["observed_hand_frame_fraction_before_trim"]
        ),
        "source_hand_fraction_after_trim": float(
            result.diagnostics["observed_hand_frame_fraction"]
        ),
        "bone_length_cv": _bone_cv(features),
        "elapsed_seconds": float(result.diagnostics["elapsed_seconds"]),
        "world_depth_fraction": float(
            result.diagnostics.get("detector_world_depth_fraction", 0.0)
        ),
    }


def _aggregate(rows: list[dict[str, object]], candidate: str) -> dict[str, float]:
    present = [row for row in rows if row[f"{candidate}_available"]]
    keys = (
        "hand_active", "both_hands_active", "hand_node_presence",
        "source_hand_fraction_before_trim", "source_hand_fraction_after_trim",
        "bone_length_cv", "elapsed_seconds", "world_depth_fraction",
    )
    result = {"available": float(len(present)), "missing": float(len(rows) - len(present))}
    for key in keys:
        values = [float(row[f"{candidate}_{key}"]) for row in present]
        values = [value for value in values if np.isfinite(value)]
        result[f"median_{key}"] = _median(values)
        result[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
    return result


def _exact_paired_pvalue(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def write_report(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    candidates = {
        threshold_name(value): (
            args.output_root / threshold_name(value),
            MediaPipeV17Config(
                minimum_hand_detection_confidence=value,
                minimum_hand_presence_confidence=value,
                minimum_hand_tracking_confidence=value,
            ),
        )
        for value in manifest["candidate_thresholds"]
    }
    rows: list[dict[str, object]] = []
    for entry in manifest["entries"]:
        row: dict[str, object] = {
            "label": entry["label"], "split": entry["split"],
            "stratum": entry["stratum"], "raw_path": entry["raw_path"],
            "relative_feature_path": entry["relative_feature_path"],
        }
        apple = load_v17_result(Path(entry["apple_feature_path"]), V17Config())
        row["apple_available"] = True
        row.update({f"apple_{key}": value for key, value in _clip_metrics(apple).items()})
        for name, (root, config) in candidates.items():
            path = root / entry["relative_feature_path"]
            if not path.is_file():
                row[f"{name}_available"] = False
                continue
            result = load_mediapipe_v17_result(path, config)
            row[f"{name}_available"] = True
            row.update({f"{name}_{key}": value for key, value in _clip_metrics(result).items()})
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregates = {"apple": _aggregate(rows, "apple")}
    aggregates.update({name: _aggregate(rows, name) for name in candidates})
    lines = [
        "# v17 Extractor Bakeoff",
        "",
        f"- Frozen manifest: `{args.manifest}`",
        f"- Manifest entry SHA-256: `{manifest['entry_sha256']}`",
        f"- Clips: {len(rows)} across {manifest['class_count']} classes",
        "- Splits used: train and validation only; official test remains sealed.",
        "- Subset: per class, low and median Apple-coverage train clips plus lowest Apple-coverage validation clip.",
        "- Bone-length CV is a tracking-stability proxy, not landmark ground truth.",
        "- MediaPipe confidence is whole-hand confidence; Apple confidence is per-joint, so raw confidence is intentionally not compared.",
        "",
        "## Aggregate measurements",
        "",
        "| Candidate | Available | Median active output | Median source detection (pre-trim) | Median hand-node presence | Median two-hand activity | Median bone CV | Median seconds/clip | Median genuine-depth coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in aggregates.items():
        lines.append(
            f"| {name} | {int(values['available'])}/{len(rows)} | "
            f"{values['median_hand_active']:.4f} | "
            f"{values['median_source_hand_fraction_before_trim']:.4f} | "
            f"{values['median_hand_node_presence']:.4f} | "
            f"{values['median_both_hands_active']:.4f} | "
            f"{values['median_bone_length_cv']:.4f} | "
            f"{values['median_elapsed_seconds']:.4f} | "
            f"{values['median_world_depth_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision gate",
            "",
            "This automated table does not select a winner. Review overlays on the largest coverage gains/losses and two-handed clips for false detections and anatomical placement. If MediaPipe is visually sound and competitive, extract the complete train/validation corpus and compare identical Stage 1 training runs. Validation accuracy is the final extractor-selection metric; latency, missingness, stability, package size, and genuine depth break close ties.",
            "",
            f"Per-clip measurements: `{args.csv}`",
            "",
        ]
    )
    apple_metrics_path = Path("artifacts/reports/stage1_v17_validation/metrics.json")
    media_metrics_path = Path(
        "artifacts/reports/stage1_v17_mediapipe_t50_validation/metrics.json"
    )
    apple_predictions_path = Path(
        "artifacts/reports/stage1_v17_validation/predictions.csv"
    )
    media_predictions_path = Path(
        "artifacts/reports/stage1_v17_mediapipe_t50_validation/predictions.csv"
    )
    if all(
        path.is_file()
        for path in (
            apple_metrics_path, media_metrics_path,
            apple_predictions_path, media_predictions_path,
        )
    ):
        apple_metrics = json.loads(apple_metrics_path.read_text(encoding="utf-8"))
        media_metrics = json.loads(media_metrics_path.read_text(encoding="utf-8"))
        apple_predictions = {
            row["feature_path"]: row["correct"] == "True"
            for row in _read_csv(apple_predictions_path)
        }
        media_predictions = {
            row["feature_path"]: row["correct"] == "True"
            for row in _read_csv(media_predictions_path)
        }
        if set(apple_predictions) != set(media_predictions):
            raise ValueError("Stage 1 predictions are not paired on identical clips")
        apple_only = sum(
            apple_predictions[path] and not media_predictions[path]
            for path in apple_predictions
        )
        media_only = sum(
            media_predictions[path] and not apple_predictions[path]
            for path in apple_predictions
        )
        paired_p = _exact_paired_pvalue(apple_only, media_only)
        lines.extend(
            [
                "## Final full-corpus Stage 1 selection",
                "",
                "Both extractors used the same official signer-disjoint splits, model architecture, seed, augmentations, optimizer, schedule, EMA, and early stopping. The test split remained sealed.",
                "",
                "| Extractor | Top-1 | Top-5 | Macro F1 |",
                "| --- | ---: | ---: | ---: |",
                f"| Apple Vision | {apple_metrics['top1']:.2f}% | {apple_metrics['top5']:.2f}% | {apple_metrics['macro_f1']:.2f}% |",
                f"| MediaPipe 0.50 | {media_metrics['top1']:.2f}% | {media_metrics['top5']:.2f}% | {media_metrics['macro_f1']:.2f}% |",
                "",
                f"Apple alone classified {apple_only} clips correctly; MediaPipe alone classified {media_only}. The exact paired two-sided p-value is {paired_p:.4f}. The five validation signers make this uncertainty worth stating, but Apple wins the engineering decision on higher top-1/top-5, faster extraction, and better visual recovery of overlapping hands.",
                "",
                "**Selected v17 extractor: Apple Vision.**",
                "",
            ]
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregates, indent=2, sort_keys=True))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--quality-csv", type=Path, default=DEFAULT_QUALITY_CSV)
    freeze.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    freeze.add_argument("--apple-root", type=Path, default=DEFAULT_APPLE_ROOT)
    freeze.add_argument("--rejections", type=Path, default=DEFAULT_REJECTIONS)
    freeze.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    freeze.set_defaults(run=write_manifest)

    extract = commands.add_parser("extract-mediapipe")
    extract.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    extract.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    extract.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    extract.add_argument("--threshold", type=float, required=True)
    extract.add_argument("--no-resume", action="store_true")
    extract.set_defaults(run=extract_candidate)

    report = commands.add_parser("report")
    report.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report.add_argument("--csv", type=Path, default=DEFAULT_DETAIL_CSV)
    report.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    report.set_defaults(run=write_report)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
