#!/usr/bin/env python3
"""Consolidate the train-only How2Sign leave-one-signer-out transition evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    folds = []
    for model_path, audit_path in zip(args.models, args.audits):
        model = json.loads(model_path.read_text())
        audit = json.loads(audit_path.read_text())
        if model["held_out_signer"] != audit["held_out_signer"]:
            raise ValueError("model/audit held-out signer mismatch")
        learned = audit["genuine_vs_learned_discriminator"]
        linear = audit["genuine_vs_linear_discriminator"]
        reconstruction = audit["reconstruction_vs_linear"]
        folds.append({
            "held_out_signer": model["held_out_signer"],
            "validation_windows": model["validation_windows"],
            "held_out_source_clips": audit["held_out_source_clips"],
            "selected_epoch": model["selected_epoch"],
            "checkpoint": model["checkpoint"],
            "checkpoint_sha256": model["checkpoint_sha256"],
            "reconstruction_relative_improvement": reconstruction["relative_improvement"],
            "reconstruction_ci95": [
                reconstruction["ci95_low"], reconstruction["ci95_high"]
            ],
            "windows_improved_fraction": reconstruction["windows_improved_fraction"],
            "learned_discriminator_balanced_accuracy": learned["balanced_accuracy"],
            "learned_discriminator_roc_auc": learned["roc_auc"],
            "linear_discriminator_balanced_accuracy": linear["balanced_accuracy"],
            "linear_discriminator_roc_auc": linear["roc_auc"],
            "model_result": model_path.as_posix(),
            "model_result_sha256": sha256(model_path),
            "naturalness_audit": audit_path.as_posix(),
            "naturalness_audit_sha256": sha256(audit_path),
        })
    weights = [row["validation_windows"] for row in folds]
    total = sum(weights)
    weighted = lambda key: sum(
        row[key] * weight for row, weight in zip(folds, weights)
    ) / total
    report = {
        "format": "transition_inpainter_loso_summary_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "landmark_tree_sha256": args.landmark_tree_sha256,
        "folds": folds,
        "aggregate": {
            "held_out_signers": len(folds),
            "held_out_windows": total,
            "held_out_source_clips": sum(row["held_out_source_clips"] for row in folds),
            "weighted_reconstruction_relative_improvement": weighted(
                "reconstruction_relative_improvement"
            ),
            "macro_reconstruction_relative_improvement": mean(
                row["reconstruction_relative_improvement"] for row in folds
            ),
            "weighted_windows_improved_fraction": weighted("windows_improved_fraction"),
            "weighted_learned_discriminator_balanced_accuracy": weighted(
                "learned_discriminator_balanced_accuracy"
            ),
            "weighted_linear_discriminator_balanced_accuracy": weighted(
                "linear_discriminator_balanced_accuracy"
            ),
            "macro_learned_discriminator_roc_auc": mean(
                row["learned_discriminator_roc_auc"] for row in folds
            ),
            "macro_linear_discriminator_roc_auc": mean(
                row["linear_discriminator_roc_auc"] for row in folds
            ),
        },
        "decision": (
            "deterministic residual inpainting generalizes and is closer than linear, "
            "but remains machine-distinguishable and is not a human-naturalness pass"
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
    root = Path("artifacts/models")
    reports = Path("artifacts/reports")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", default=[
        root / "transition_inpainter_residual_v17_full_h3_how2sign_only/result.json",
        root / "transition_inpainter_residual_v17_full_h5_how2sign_only/result.json",
        root / "transition_inpainter_residual_v17_full_h8_how2sign_only/result.json",
    ])
    parser.add_argument("--audits", type=Path, nargs="+", default=[
        reports / "transition_inpainter_naturalness_v17_full_h3_how2sign_only.json",
        reports / "transition_inpainter_naturalness_v17_full_h5_how2sign_only.json",
        reports / "transition_inpainter_naturalness_v17_full_h8_how2sign_only.json",
    ])
    parser.add_argument(
        "--landmark-tree-sha256",
        default="79cc83c2c2ff711f505b9af1ceca5271386287b676a98737e65e74e07c487806",
    )
    parser.add_argument(
        "--output", type=Path,
        default=reports / "transition_inpainter_loso_summary_v17.json",
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
