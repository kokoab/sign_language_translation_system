#!/usr/bin/env python3
"""Aggregate the frozen three-fold content-gated signing-voice profile evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.train_transition_inpainter_v17 import sha256


def weighted(metrics, key):
    total = sum(int(row["examples"]) for row in metrics)
    return sum(float(row[key]) * int(row["examples"]) for row in metrics) / total


def run(args):
    reports = [json.loads(path.read_text()) for path in args.inputs]
    if sorted(row["fold"] for row in reports) != [0, 1, 2]:
        raise ValueError("exactly folds 0, 1, and 2 are required")
    if any(
        row.get(flag) for row in reports for flag in (
            "test_evaluated", "citizen_test_accessed", "semlex_test_accessed",
            "local_test_accessed", "held_out_validation_signer_accessed",
        )
    ):
        raise ValueError("a forbidden split-access flag is set")
    if {
        (row["latent_dim"], row["curve_strength"], row["adaptive_content_gate"])
        for row in reports
    } != {(16, 0.0, True)}:
        raise ValueError("fold design is not the frozen profile design")
    metrics = [row["validation"] for row in reports]
    aggregate = {
        "folds": 3,
        "held_out_voices": sum(len(row["validation_voices"]) for row in reports),
        "held_out_examples": sum(row["examples"] for row in metrics),
        "generated_content_accuracy": weighted(metrics, "generated_content_accuracy"),
        "prototype_content_accuracy": weighted(metrics, "prototype_content_accuracy"),
        "target_content_accuracy": weighted(metrics, "target_content_accuracy"),
        "style_verification_auc": weighted(metrics, "style_verification_auc"),
    }
    for term in ("spatial", "velocity", "acceleration"):
        generated = weighted(metrics, f"generated_{term}")
        prototype = weighted(metrics, f"prototype_{term}")
        aggregate[f"generated_{term}"] = generated
        aggregate[f"prototype_{term}"] = prototype
        aggregate[f"relative_{term}_improvement"] = (prototype - generated) / prototype
    result = {
        "format": "slt_signing_voice_profile_signer_disjoint_summary_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_design": {
            "latent_dim": 16,
            "profile": "per-node median XYZ signing-space offset",
            "curve_strength": 0.0,
            "content_gate_strengths": [1.0, 0.75, 0.50, 0.40, 0.25, 0.0],
            "content_gate": "select strongest profile retaining requested frozen Stage-1 label",
        },
        "folds": [
            {
                "report": path.as_posix(), "report_sha256": sha256(path),
                "checkpoint": row["checkpoint"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "fold": row["fold"], "validation_voices": row["validation_voices"],
                "validation": row["validation"],
            }
            for path, row in zip(args.inputs, reports)
        ],
        "aggregate": aggregate,
        "claim_boundary": (
            "content-controlled signer-profile evidence is not a fluent Deaf-signer "
            "naturalness or linguistic-correctness judgment"
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/signing_voice_profile_signer_disjoint_summary_v17.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
