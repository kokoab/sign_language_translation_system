#!/usr/bin/env python3
"""Narrow the local q82 audit to the strongest exact-text review candidates.

The output is still not training-approved. It requires exact ASL-LEX review and has
no trustworthy signer IDs; this merely removes weak extraction, variant-name, and
frozen-model mismatch cases before human review.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def passes_review_gate(
    provenance: dict[str, object],
    prediction: dict[str, str],
    class_triage: dict[str, str],
    diagnostics: dict[str, object],
) -> bool:
    return bool(
        passes_clip_review_gate(provenance, prediction, diagnostics)
        and class_triage["triage"]
        == "model_consistent_manual_variant_review_required"
    )


def passes_clip_review_gate(
    provenance: dict[str, object],
    prediction: dict[str, str],
    diagnostics: dict[str, object],
) -> bool:
    """Per-clip evidence pool before the stricter class-consistency screen."""
    return bool(
        provenance["lexical_tier"] == "canonical_and_pinned_raw_text_equal"
        and prediction["top5_hit"] == "True"
        and float(diagnostics["observed_hand_frame_fraction"]) >= 0.80
        and float(diagnostics["face_presence_fraction"]) >= 0.50
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("data/local/local_citizen100_quality_audit_q82"),
    )
    parser.add_argument(
        "--triage-root",
        type=Path,
        default=Path("artifacts/reports/local_citizen100_quality_audit/model_triage"),
    )
    parser.add_argument("--cap-per-class", type=int, default=3)
    parser.add_argument("--materialize-symlinks", action="store_true")
    args = parser.parse_args()

    source = json.loads((args.audit_root / "candidate_selection.json").read_text())
    if source.get("training_eligible") is not False:
        raise ValueError("source audit must explicitly prohibit training use")
    provenance = {
        (str(row["canonical_label"]), Path(str(row["raw_path"])).stem): row
        for row in source["videos"]
    }
    predictions = list(csv.DictReader((args.triage_root / "predictions.csv").open()))
    class_rows = {
        str(row["canonical_label"]): row
        for row in csv.DictReader((args.triage_root / "class_triage.csv").open())
    }
    passing: dict[str, list[dict[str, object]]] = {}
    clip_passing: dict[str, list[dict[str, object]]] = {}
    for prediction in predictions:
        feature_path = Path(prediction["feature_path"])
        key = (str(prediction["true_label"]), Path(prediction["raw_path"]).stem)
        row = provenance[key]
        with np.load(feature_path, allow_pickle=False) as payload:
            diagnostics = json.loads(str(payload["diagnostics_json"].item()))
        merged = {
            **row,
            "feature_path": str(feature_path),
            "observed_hand_frame_fraction": diagnostics["observed_hand_frame_fraction"],
            "hand_presence_fraction": diagnostics["hand_presence_fraction"],
            "face_presence_fraction": diagnostics["face_presence_fraction"],
            "body_presence_fraction": diagnostics["body_presence_fraction"],
            "frozen_top1_hit": prediction["top1_hit"] == "True",
            "frozen_top5_hit": True,
            "training_eligible": False,
        }
        if passes_clip_review_gate(row, prediction, diagnostics):
            clip_passing.setdefault(key[0], []).append(merged)
        if passes_review_gate(row, prediction, class_rows[key[0]], diagnostics):
            passing.setdefault(key[0], []).append(merged)

    def capped(rows_by_label: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
        selected_rows: list[dict[str, object]] = []
        for label in sorted(rows_by_label):
            rows = sorted(
                rows_by_label[label],
                key=lambda row: (
                    bool(row["frozen_top1_hit"]),
                    float(row["observed_hand_frame_fraction"]),
                    float(row["face_presence_fraction"]),
                    float(row["quality_score"]),
                ),
                reverse=True,
            )[: args.cap_per_class]
            selected_rows.extend(rows)
        return selected_rows

    selected = capped(passing)
    clip_pool = capped(clip_passing)

    if args.materialize_symlinks:
        for row in selected:
            source_path = Path(str(row["raw_path"]))
            destination = args.audit_root / "review_raw" / str(row["canonical_label"]) / source_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                if destination.resolve() != source_path.resolve():
                    raise ValueError(f"conflicting symlink: {destination}")
            elif destination.exists():
                raise ValueError(f"refusing to overwrite: {destination}")
            else:
                destination.symlink_to(source_path.resolve())

    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "strongest local exact-text candidates for ASL-LEX human review",
        "training_eligible": False,
        "split_eligibility": "train_only_after_ASL_fluent_exact_variant_review",
        "signer_warning": "appearance diversity is not signer identity",
        "gates": {
            "class_model_triage": "model_consistent_manual_variant_review_required",
            "lexical_tier": "canonical_and_pinned_raw_text_equal",
            "clip_frozen_top5_hit": True,
            "minimum_observed_hand_frame_fraction": 0.80,
            "minimum_face_presence_fraction": 0.50,
            "cap_per_class": args.cap_per_class,
        },
        "selected_clips": len(selected),
        "selected_classes": len(passing),
        "videos": selected,
    }
    output_path = args.audit_root / "review_shortlist.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    clip_output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "per-clip local evidence pool for ASL-fluent human review",
        "training_eligible": False,
        "split_eligibility": "train_only_after_ASL_fluent_exact_variant_review",
        "signer_warning": "appearance diversity is not signer identity",
        "class_level_model_consistency_required": False,
        "gates": {
            "lexical_tier": "canonical_and_pinned_raw_text_equal",
            "clip_frozen_top5_hit": True,
            "minimum_observed_hand_frame_fraction": 0.80,
            "minimum_face_presence_fraction": 0.50,
            "cap_per_class": args.cap_per_class,
        },
        "selected_clips": len(clip_pool),
        "selected_classes": len(clip_passing),
        "videos": clip_pool,
    }
    clip_path = args.audit_root / "clip_review_pool.json"
    clip_path.write_text(json.dumps(clip_output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "strict_output": str(output_path),
                "strict_clips": len(selected),
                "strict_classes": len(passing),
                "clip_pool_output": str(clip_path),
                "clip_pool_clips": len(clip_pool),
                "clip_pool_classes": len(clip_passing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
