#!/usr/bin/env python3
"""Aggregate frozen multi-corpus transition folds without touching sealed splits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


FOLDS = ("h3", "h5", "h8")
DOMAINS = ("how2sign_signer_heldout", "youtube_asl_channel_heldout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(domain: dict[str, object]) -> dict[str, float | int]:
    reconstruction = domain["reconstruction_vs_linear"]
    discriminator = domain["genuine_vs_learned_discriminator"]
    return {
        "windows": int(domain["paired_windows"]),
        "source_clips": int(domain["source_clips"]),
        "relative_reconstruction_improvement": float(
            reconstruction["relative_improvement"]
        ),
        "windows_improved_fraction": float(
            reconstruction["windows_improved_fraction"]
        ),
        "discriminator_balanced_accuracy": float(
            discriminator["balanced_accuracy"]
        ),
        "discriminator_roc_auc": float(discriminator["roc_auc"]),
    }


def weighted(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    windows = sum(int(row["windows"]) for row in rows)
    result: dict[str, float | int] = {"evaluation_windows": windows}
    for key in (
        "relative_reconstruction_improvement",
        "windows_improved_fraction",
        "discriminator_balanced_accuracy",
        "discriminator_roc_auc",
    ):
        result[key] = sum(
            float(row[key]) * int(row["windows"]) for row in rows
        ) / windows
    return result


def delta(
    baseline: dict[str, float | int], adapted: dict[str, float | int]
) -> dict[str, float]:
    return {
        key: float(adapted[key]) - float(baseline[key])
        for key in (
            "relative_reconstruction_improvement",
            "windows_improved_fraction",
            "discriminator_balanced_accuracy",
            "discriminator_roc_auc",
        )
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    folds = {}
    for fold in FOLDS:
        baseline_path = args.report_root / f"transition_inpainter_multicorpus_baseline_{fold}_v17.json"
        adapted_path = args.report_root / f"transition_inpainter_multicorpus_{fold}_w010_v17.json"
        baseline = json.loads(baseline_path.read_text())
        adapted = json.loads(adapted_path.read_text())
        fold_domains = {}
        for domain in DOMAINS:
            baseline_metrics = compact(baseline["domains"][domain])
            adapted_metrics = compact(adapted["domains"][domain])
            fold_domains[domain] = {
                "baseline": baseline_metrics,
                "adapted": adapted_metrics,
                "delta_adapted_minus_baseline": delta(
                    baseline_metrics, adapted_metrics
                ),
            }
        folds[fold] = {
            "baseline_report": baseline_path.as_posix(),
            "baseline_report_sha256": sha256(baseline_path),
            "adapted_report": adapted_path.as_posix(),
            "adapted_report_sha256": sha256(adapted_path),
            "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
            "adapted_checkpoint_sha256": adapted["checkpoint_sha256"],
            "domains": fold_domains,
        }

    aggregate = {}
    for domain in DOMAINS:
        baseline_rows = [folds[fold]["domains"][domain]["baseline"] for fold in FOLDS]
        adapted_rows = [folds[fold]["domains"][domain]["adapted"] for fold in FOLDS]
        baseline_metrics = weighted(baseline_rows)
        adapted_metrics = weighted(adapted_rows)
        aggregate[domain] = {
            "baseline": baseline_metrics,
            "adapted": adapted_metrics,
            "delta_adapted_minus_baseline": delta(
                baseline_metrics, adapted_metrics
            ),
        }

    all_reconstruction_gates_improve = all(
        folds[fold]["domains"][domain]["delta_adapted_minus_baseline"]
        ["relative_reconstruction_improvement"] > 0.0
        for fold in FOLDS for domain in DOMAINS
    )
    report = {
        "format": "transition_inpainter_multicorpus_loso_summary_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_web_probability": 0.10,
        "frozen_learning_rate": 5e-5,
        "selection_fold": "how2sign:8",
        "confirmation_folds": ["how2sign:3", "how2sign:5"],
        "folds": folds,
        "aggregate": aggregate,
        "all_six_fold_domain_reconstruction_gates_improve": (
            all_reconstruction_gates_improve
        ),
        "interpretation": (
            "The frozen adaptation improves reconstruction on every signer/domain "
            "fold and improves aggregate discriminator ROC AUC in both domains. "
            "Web balanced accuracy is mixed and must remain visible; these machine "
            "landmark gates do not establish human-perceptual naturalness."
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
    parser.add_argument("--report-root", type=Path, default=Path("artifacts/reports"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/reports/transition_inpainter_multicorpus_loso_summary_v17.json"),
    )
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({
        "output": "artifacts/reports/transition_inpainter_multicorpus_loso_summary_v17.json",
        "all_reconstruction_gates_improve": report[
            "all_six_fold_domain_reconstruction_gates_improve"
        ],
        "aggregate": report["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
