#!/usr/bin/env python3
"""Build conservative local-Citizen100 review tiers from two frozen models.

The local corpus has no trustworthy signer or lexical-variant IDs. Model agreement
therefore prioritizes human review; it never approves clips for training or relabels
model disagreements.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


EXACT_TEXT_TIER = "canonical_and_pinned_raw_text_equal"


def parse_bool(value: object) -> bool:
    return str(value).lower() == "true"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_candidate(
    provenance: dict[str, object],
    baseline: dict[str, str],
    challenger: dict[str, str],
    diagnostics: dict[str, object],
) -> str:
    """Return a review tier without treating model output as label truth."""
    if provenance.get("lexical_tier") != EXACT_TEXT_TIER:
        return "quarantine_lexical_mismatch"
    if (
        float(diagnostics["observed_hand_frame_fraction"]) < 0.80
        or float(diagnostics["face_presence_fraction"]) < 0.50
    ):
        return "quarantine_extraction_quality"

    baseline_top1 = parse_bool(baseline["top1_hit"])
    challenger_top1 = parse_bool(challenger["top1_hit"])
    baseline_top5 = parse_bool(baseline["top5_hit"])
    challenger_top5 = parse_bool(challenger["top5_hit"])
    if baseline_top1 and challenger_top1:
        return "tier_a_dual_top1"
    if baseline_top5 and challenger_top5 and (baseline_top1 or challenger_top1):
        return "tier_b_dual_top5_one_top1"
    if baseline_top5 and challenger_top5:
        return "tier_c_dual_top5_only"
    return "quarantine_model_disagreement"


def load_predictions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    output: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row["true_label"]), Path(row["raw_path"]).stem)
        if key in output:
            raise ValueError(f"duplicate prediction key in {path}: {key}")
        output[key] = row
    return output


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("data/local/local_citizen100_quality_audit_q82_cap7"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path(
            "artifacts/reports/local_citizen100_quality_audit/cap7_model_triage"
        ),
    )
    parser.add_argument(
        "--challenger-root",
        type=Path,
        default=Path(
            "artifacts/reports/local_citizen100_quality_audit/"
            "cap7_balanced_model_triage"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/reports/local_citizen100_quality_audit/consensus"
        ),
    )
    args = parser.parse_args()

    source_path = args.audit_root / "candidate_selection.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("training_eligible") is not False:
        raise ValueError("local source must explicitly prohibit unreviewed training")
    provenance = {
        (str(row["canonical_label"]), Path(str(row["raw_path"])).stem): row
        for row in source["videos"]
    }
    if len(provenance) != len(source["videos"]):
        raise ValueError("duplicate canonical-label/raw-stem keys in local provenance")

    baseline = load_predictions(args.baseline_root / "predictions.csv")
    challenger = load_predictions(args.challenger_root / "predictions.csv")
    expected = set(provenance)
    if set(baseline) != expected or set(challenger) != expected:
        raise ValueError("prediction keys do not exactly match local provenance")

    baseline_summary = json.loads(
        (args.baseline_root / "summary.json").read_text(encoding="utf-8")
    )
    challenger_summary = json.loads(
        (args.challenger_root / "summary.json").read_text(encoding="utf-8")
    )

    rows: list[dict[str, object]] = []
    seen_raw_hashes: dict[str, tuple[str, str]] = {}
    for key in sorted(expected):
        item = provenance[key]
        feature_path = Path(baseline[key]["feature_path"])
        if Path(challenger[key]["feature_path"]) != feature_path:
            raise ValueError(f"models used different feature paths for {key}")
        with np.load(feature_path, allow_pickle=False) as payload:
            diagnostics = json.loads(str(payload["diagnostics_json"].item()))
        raw_path = Path(str(item["raw_path"]))
        raw_sha256 = sha256_file(raw_path)
        tier = classify_candidate(item, baseline[key], challenger[key], diagnostics)
        duplicate_of = seen_raw_hashes.get(raw_sha256)
        if duplicate_of is None:
            seen_raw_hashes[raw_sha256] = key
        else:
            tier = "quarantine_exact_duplicate"

        rows.append(
            {
                **item,
                "feature_path": str(feature_path),
                "raw_sha256": raw_sha256,
                "duplicate_of": "" if duplicate_of is None else "/".join(duplicate_of),
                "observed_hand_frame_fraction": float(
                    diagnostics["observed_hand_frame_fraction"]
                ),
                "hand_presence_fraction": float(diagnostics["hand_presence_fraction"]),
                "face_presence_fraction": float(diagnostics["face_presence_fraction"]),
                "body_presence_fraction": float(diagnostics["body_presence_fraction"]),
                "citizen_only_prediction": baseline[key]["predicted_label"],
                "citizen_only_top5": baseline[key]["top5_labels"],
                "citizen_only_true_probability": float(
                    baseline[key]["true_probability"]
                ),
                "balanced_prediction": challenger[key]["predicted_label"],
                "balanced_top5": challenger[key]["top5_labels"],
                "balanced_true_probability": float(
                    challenger[key]["true_probability"]
                ),
                "consensus_tier": tier,
                "training_eligible": False,
            }
        )

    tier_order = {
        "tier_a_dual_top1": 0,
        "tier_b_dual_top5_one_top1": 1,
        "tier_c_dual_top5_only": 2,
        "quarantine_model_disagreement": 3,
        "quarantine_extraction_quality": 4,
        "quarantine_lexical_mismatch": 5,
        "quarantine_exact_duplicate": 6,
    }
    rows.sort(
        key=lambda row: (
            tier_order[str(row["consensus_tier"])],
            str(row["canonical_label"]),
            -min(
                float(row["citizen_only_true_probability"]),
                float(row["balanced_true_probability"]),
            ),
            str(row["raw_path"]),
        )
    )
    counts = Counter(str(row["consensus_tier"]) for row in rows)
    classes_by_tier: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        classes_by_tier[str(row["consensus_tier"])].add(
            str(row["canonical_label"])
        )

    class_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        class_groups[str(row["canonical_label"])].append(row)
    class_rows: list[dict[str, object]] = []
    for label, items in sorted(class_groups.items()):
        per_tier = Counter(str(item["consensus_tier"]) for item in items)
        class_rows.append(
            {
                "canonical_label": label,
                "clips": len(items),
                **{tier: per_tier.get(tier, 0) for tier in tier_order},
                "training_approved": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "consensus_predictions.csv", rows)
    write_csv(args.output_dir / "class_summary.csv", class_rows)

    review_rows = [
        row
        for row in rows
        if row["consensus_tier"]
        in {"tier_a_dual_top1", "tier_b_dual_top5_one_top1"}
    ]
    quarantined = sum(
        count for tier, count in counts.items() if tier.startswith("quarantine_")
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "dual-model local exact-label review shortlist",
        "training_eligible": False,
        "split_eligibility": "train_only_after_ASL_fluent_exact_variant_review",
        "signer_warning": (
            "No trustworthy local signer IDs exist; appearance diversity is not a "
            "signer count."
        ),
        "selection_warning": (
            "Model agreement is correlated screening evidence, not independent label "
            "proof. Disagreements were not relabeled."
        ),
        "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path),
        "baseline_summary": baseline_summary,
        "challenger_summary": challenger_summary,
        "gates": {
            "lexical_tier": EXACT_TEXT_TIER,
            "minimum_observed_hand_frame_fraction": 0.80,
            "minimum_face_presence_fraction": 0.50,
            "accepted_review_tiers": [
                "tier_a_dual_top1",
                "tier_b_dual_top5_one_top1",
            ],
            "exact_raw_sha256_deduplication": True,
        },
        "audited_clips": len(rows),
        "audited_classes": len(class_groups),
        "tier_counts": dict(sorted(counts.items())),
        "tier_class_counts": {
            tier: len(labels) for tier, labels in sorted(classes_by_tier.items())
        },
        "selected_clips": len(review_rows),
        "selected_classes": len(
            {str(row["canonical_label"]) for row in review_rows}
        ),
        "videos": review_rows,
    }
    (args.output_dir / "consensus_review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Local Citizen100 dual-model consensus audit",
        "",
        "**Status:** review shortlist only; no clip was automatically training-approved.",
        "",
        "Both models were trained without these local clips. Agreement screens likely",
        "mismatches but cannot establish signer identity or exact ASL lexical variant.",
        "",
        f"- Audited clips/classes: {len(rows)}/{len(class_groups)}",
        f"- Tier A (both top-1): {counts['tier_a_dual_top1']} clips / "
        f"{len(classes_by_tier['tier_a_dual_top1'])} classes",
        f"- Tier B (both top-5, one top-1): "
        f"{counts['tier_b_dual_top5_one_top1']} clips / "
        f"{len(classes_by_tier['tier_b_dual_top5_one_top1'])} classes",
        f"- Tier C (both top-5 only): {counts['tier_c_dual_top5_only']} clips / "
        f"{len(classes_by_tier['tier_c_dual_top5_only'])} classes",
        f"- Priority human-review pool (A+B): {len(review_rows)} clips / "
        f"{manifest['selected_classes']} classes",
        f"- Quarantined: {quarantined} clips",
        f"- Non-priority retained (Tier C plus quarantine): "
        f"{len(rows) - len(review_rows)} clips",
        "",
        "Tier C remains available for slower manual review. Quarantined clips were",
        "preserved in place and no predicted label was written back to the corpus.",
    ]
    (args.output_dir / "REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "audited_clips": len(rows),
                "unique_raw_files": len(seen_raw_hashes),
                "tier_counts": dict(sorted(counts.items())),
                "priority_review_clips": len(review_rows),
                "priority_review_classes": manifest["selected_classes"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
