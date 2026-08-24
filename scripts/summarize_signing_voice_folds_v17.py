#!/usr/bin/env python3
"""Aggregate the three frozen signer-held-out signing-voice folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.train_transition_inpainter_v17 import sha256


def weighted(rows, key, weight_key="examples"):
    total = sum(int(row[weight_key]) for row in rows)
    return sum(float(row[key]) * int(row[weight_key]) for row in rows) / total


def run(args):
    reports = [json.loads(path.read_text()) for path in args.inputs]
    if sorted(int(row["fold"]) for row in reports) != [0, 1, 2]:
        raise ValueError("exactly folds 0, 1, and 2 are required")
    if any(
        row.get(flag) for row in reports for flag in (
            "test_evaluated", "citizen_test_accessed", "semlex_test_accessed",
            "local_test_accessed", "held_out_validation_signer_accessed",
        )
    ):
        raise ValueError("a forbidden split-access flag is set")
    metrics = [row["validation"] for row in reports]
    aggregate = {
        "folds": 3,
        "held_out_examples": sum(int(row["examples"]) for row in metrics),
        "held_out_voices": sum(len(row["validation_voices"]) for row in reports),
        "selected_epoch_median": int(statistics.median(row["selected_epoch"] for row in reports)),
        "generated_content_accuracy": weighted(metrics, "generated_content_accuracy"),
        "prototype_content_accuracy": weighted(metrics, "prototype_content_accuracy"),
        "target_content_accuracy": weighted(metrics, "target_content_accuracy"),
        "style_verification_auc": weighted(metrics, "style_verification_auc"),
        "same_voice_cosine": weighted(metrics, "same_voice_cosine"),
        "different_voice_cosine": weighted(metrics, "different_voice_cosine"),
    }
    for term in ("spatial", "velocity", "acceleration"):
        generated = weighted(metrics, f"generated_{term}")
        prototype = weighted(metrics, f"prototype_{term}")
        aggregate[f"generated_{term}"] = generated
        aggregate[f"prototype_{term}"] = prototype
        aggregate[f"relative_{term}_improvement"] = (prototype - generated) / prototype
    result = {
        "format": "slt_signing_voice_signer_disjoint_summary_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folds": [
            {
                "report": path.as_posix(), "report_sha256": sha256(path),
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "fold": row["fold"], "selected_epoch": row["selected_epoch"],
                "validation_voices": row["validation_voices"],
                "validation": row["validation"],
            }
            for path, row in zip(args.inputs, reports)
        ],
        "aggregate": aggregate,
        "final_training_schedule": {
            "epochs": aggregate["selected_epoch_median"],
            "model_selection": False,
            "voices": "all 63 eligible train-only identities",
        },
        "claim_boundary": (
            "signer-held-out landmark reconstruction/content/style verification is not "
            "a fluent-signer naturalness or linguistic-correctness rating"
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/signing_voice_signer_disjoint_summary_v17.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
