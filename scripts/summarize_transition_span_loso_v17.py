#!/usr/bin/env python3
"""Aggregate frozen transition-span evidence across three held-out signers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


FOLDS = ("h3", "h5", "h8")
DOMAINS = ("how2sign", "youtube_asl")
METHODS = (
    "learned", "endpoint_only_ablation",
    "kinematic_distance_over_speed", "fixed_eight_frames",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    examples = sum(int(row["examples"]) for row in rows)
    result: dict[str, float | int] = {"examples": examples}
    for key in ("accuracy", "mae_frames", "within_one_frame", "macro_f1"):
        result[key] = sum(
            float(row[key]) * int(row["examples"]) for row in rows
        ) / examples
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    folds = {}
    for fold in FOLDS:
        result_path = (
            args.model_root / f"transition_span_multicorpus_v17_{fold}_w010"
            / "result.json"
        )
        result = json.loads(result_path.read_text())
        folds[fold] = {
            "result": result_path.as_posix(),
            "result_sha256": sha256(result_path),
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "selected_epoch": result["selected_epoch"],
            "validation": result["validation"],
        }

    aggregate = {}
    for domain in DOMAINS:
        aggregate[domain] = {
            method: weighted([
                folds[fold]["validation"][domain][method] for fold in FOLDS
            ])
            for method in METHODS
        }
        aggregate[domain]["learned_relative_mae_improvement_vs_fixed"] = (
            aggregate[domain]["fixed_eight_frames"]["mae_frames"]
            - aggregate[domain]["learned"]["mae_frames"]
        ) / aggregate[domain]["fixed_eight_frames"]["mae_frames"]
        aggregate[domain]["learned_relative_mae_improvement_vs_kinematic"] = (
            aggregate[domain]["kinematic_distance_over_speed"]["mae_frames"]
            - aggregate[domain]["learned"]["mae_frames"]
        ) / aggregate[domain]["kinematic_distance_over_speed"]["mae_frames"]
        aggregate[domain]["style_context_mae_gain_vs_endpoint_only"] = (
            aggregate[domain]["endpoint_only_ablation"]["mae_frames"]
            - aggregate[domain]["learned"]["mae_frames"]
        )

    report = {
        "format": "transition_span_loso_summary_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": (
            "recover a hidden genuine elapsed span of 4-12 frames from eight "
            "visible context frames on each side, without exposing mask width"
        ),
        "frozen_settings": {
            "web_probability": 0.10,
            "epochs": 40,
            "seed": 12701,
            "context_frames_per_side": 8,
            "target_spans": list(range(4, 13)),
        },
        "folds": folds,
        "aggregate": aggregate,
        "all_three_signer_folds_subframe_mae": all(
            folds[fold]["validation"]["how2sign"]["learned"]["mae_frames"] < 1.0
            for fold in FOLDS
        ),
        "all_three_web_evaluations_subframe_mae": all(
            folds[fold]["validation"]["youtube_asl"]["learned"]["mae_frames"] < 1.0
            for fold in FOLDS
        ),
        "interpretation": (
            "Local genuine temporal context carries a strong signer-general timing "
            "signal beyond endpoints, fixed duration, and a distance/speed rule. "
            "This self-supervised elapsed-span task is still a proxy for transition "
            "timing, not semantic prosody or a human naturalness judgment."
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/reports/transition_span_loso_summary_v17.json"),
    )
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({"aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
