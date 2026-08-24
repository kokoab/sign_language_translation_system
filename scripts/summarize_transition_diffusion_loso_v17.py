#!/usr/bin/env python3
"""Consolidate frozen-temperature stochastic transition LOSO evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    audits = [json.loads(path.read_text()) for path in args.audits]
    total = sum(row["paired_windows"] for row in audits)
    folds = []
    for path, row in zip(args.audits, audits):
        folds.append({
            "held_out_signer": row["held_out_signer"],
            "paired_windows": row["paired_windows"],
            "held_out_source_clips": row["held_out_source_clips"],
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "audit": path.as_posix(),
            "audit_sha256": digest(path),
            "temperatures": {
                value: {
                    "reconstruction_relative_improvement_vs_linear": row
                    ["temperatures"][value]["reconstruction_vs_linear"]
                    ["relative_improvement"],
                    "windows_improved_fraction_vs_linear": row
                    ["temperatures"][value]["reconstruction_vs_linear"]
                    ["windows_improved_fraction"],
                    "reconstruction_relative_change_vs_mean": row
                    ["temperatures"][value]["reconstruction_vs_deterministic_mean"]
                    ["relative_improvement"],
                    "discriminator_balanced_accuracy": row["temperatures"][value]
                    ["genuine_vs_generated_discriminator"]["balanced_accuracy"],
                    "discriminator_roc_auc": row["temperatures"][value]
                    ["genuine_vs_generated_discriminator"]["roc_auc"],
                }
                for value in ("0.1", "0.2")
            },
        })
    operating_points = {}
    for temperature in ("0.1", "0.2"):
        def weighted(section: str, key: str) -> float:
            return sum(
                row["paired_windows"]
                * row["temperatures"][temperature][section][key]
                for row in audits
            ) / total
        operating_points[temperature] = {
            "weighted_reconstruction_relative_improvement_vs_linear": weighted(
                "reconstruction_vs_linear", "relative_improvement"
            ),
            "weighted_windows_improved_fraction_vs_linear": weighted(
                "reconstruction_vs_linear", "windows_improved_fraction"
            ),
            "weighted_reconstruction_relative_change_vs_mean": weighted(
                "reconstruction_vs_deterministic_mean", "relative_improvement"
            ),
            "weighted_discriminator_balanced_accuracy": weighted(
                "genuine_vs_generated_discriminator", "balanced_accuracy"
            ),
            "macro_discriminator_roc_auc": mean(
                row["temperatures"][temperature]
                ["genuine_vs_generated_discriminator"]["roc_auc"]
                for row in audits
            ),
        }
    report = {
        "format": "transition_residual_diffusion_loso_summary_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "folds": folds,
        "aggregate": {
            "held_out_signers": len(audits),
            "held_out_windows": total,
            "held_out_source_clips": sum(
                row["held_out_source_clips"] for row in audits
            ),
            "deterministic_mean_weighted_balanced_accuracy": sum(
                row["paired_windows"]
                * row["deterministic_mean_discriminator"]["balanced_accuracy"]
                for row in audits
            ) / total,
            "deterministic_mean_macro_roc_auc": mean(
                row["deterministic_mean_discriminator"]["roc_auc"]
                for row in audits
            ),
            "linear_weighted_balanced_accuracy": sum(
                row["paired_windows"]
                * row["linear_discriminator"]["balanced_accuracy"]
                for row in audits
            ) / total,
            "linear_macro_roc_auc": mean(
                row["linear_discriminator"]["roc_auc"] for row in audits
            ),
            "operating_points": operating_points,
        },
        "decision": (
            "temperature 0.10 is the accuracy/diversity mode; temperature 0.20 is "
            "the stronger diversity mode. Neither passes human naturalness."
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
    reports = Path("artifacts/reports")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, nargs="+", default=[
        reports / "transition_residual_diffusion_v17_h3_epoch10_t01_t02.json",
        reports / "transition_residual_diffusion_v17_h5_epoch10_t01_t02.json",
        reports / "transition_residual_diffusion_v17_h8_smoke_full1786_lowtemp_sweep.json",
    ])
    parser.add_argument(
        "--output", type=Path,
        default=reports / "transition_residual_diffusion_loso_summary_v17.json",
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
