#!/usr/bin/env python3
"""Select a quality-ranked, signer-balanced SemLex train-only supplement."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def quality_score(diagnostics: dict[str, object]) -> float:
    return (
        0.40 * float(diagnostics["observed_hand_frame_fraction"])
        + 0.25 * float(diagnostics["hand_presence_fraction"])
        + 0.20 * float(diagnostics["face_presence_fraction"])
        + 0.15 * float(diagnostics["body_presence_fraction"])
    )


def select_balanced(
    rows: list[dict[str, object]],
    class_triage: dict[str, str],
    cap_per_class: int,
    minimum_observed_hand: float = 0.0,
    minimum_hand_presence: float = 0.0,
    minimum_face_presence: float = 0.0,
) -> tuple[list[dict[str, object]], list[str]]:
    excluded = sorted(
        label for label, triage in class_triage.items() if triage == "mismatch_review_priority"
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        label = str(row["canonical_label"])
        if label in excluded:
            continue
        if (
            float(row["observed_hand_frame_fraction"]) < minimum_observed_hand
            or float(row.get("hand_presence_fraction", 1.0)) < minimum_hand_presence
            or float(row["face_presence_fraction"]) < minimum_face_presence
        ):
            continue
        grouped.setdefault(label, []).append(row)
    selected: list[dict[str, object]] = []
    for label in sorted(grouped):
        ranked = sorted(
            grouped[label],
            key=lambda row: (
                float(row["quality_score"]),
                float(row["observed_hand_frame_fraction"]),
                float(row["face_presence_fraction"]),
                str(row["semlex_video_id"]),
            ),
            reverse=True,
        )
        seen_signers: set[str] = set()
        for row in ranked:
            signer = str(row["semlex_signer_id"])
            if signer in seen_signers:
                continue
            selected.append(row)
            seen_signers.add(signer)
            if cap_per_class and len(seen_signers) == cap_per_class:
                break
    return selected, excluded


def safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"conflicting symlink: {destination}")
    elif destination.exists():
        raise ValueError(f"refusing to overwrite: {destination}")
    else:
        destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-root", type=Path, default=Path("data/local/semlex_citizen100_train_audit")
    )
    parser.add_argument(
        "--triage",
        type=Path,
        default=Path(
            "artifacts/reports/semlex_citizen100_train_audit/model_triage/class_triage.csv"
        ),
    )
    parser.add_argument(
        "--cap-per-class",
        type=int,
        default=12,
        help="Maximum distinct signers per class; 0 keeps every quality-passing signer",
    )
    parser.add_argument("--minimum-observed-hand", type=float, default=0.0)
    parser.add_argument("--minimum-hand-presence", type=float, default=0.0)
    parser.add_argument("--minimum-face-presence", type=float, default=0.0)
    parser.add_argument("--output-name", default="balanced_train_candidates.json")
    parser.add_argument("--materialized-prefix", default="balanced")
    parser.add_argument("--materialize-symlinks", action="store_true")
    args = parser.parse_args()
    if args.cap_per_class < 0:
        raise ValueError("cap-per-class must be non-negative")
    for name, value in (
        ("minimum-observed-hand", args.minimum_observed_hand),
        ("minimum-hand-presence", args.minimum_hand_presence),
        ("minimum-face-presence", args.minimum_face_presence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    if Path(args.output_name).name != args.output_name:
        raise ValueError("output-name must be a filename")
    if not args.materialized_prefix or Path(args.materialized_prefix).name != args.materialized_prefix:
        raise ValueError("materialized-prefix must be one directory name")

    provenance = json.loads((args.audit_root / "download_provenance.json").read_text())
    class_triage = {
        row["canonical_label"]: row["triage"]
        for row in csv.DictReader(args.triage.open(encoding="utf-8"))
    }
    rows: list[dict[str, object]] = []
    for source in provenance["videos"]:
        if source.get("semlex_split") != "train":
            raise ValueError("SemLex supplement contains a non-train clip")
        if source.get("semlex_label_type") != "asllex":
            raise ValueError("SemLex supplement contains a non-ASL-LEX label")
        if source.get("asllex_entry_id") != source.get("semlex_label"):
            raise ValueError("SemLex ASL-LEX entry/label mismatch")
        feature_path = (
            args.audit_root
            / "landmarks_v17"
            / str(source["canonical_label"])
            / f"{source['semlex_video_id']}.v17.npz"
        )
        with np.load(feature_path, allow_pickle=False) as payload:
            diagnostics = json.loads(str(payload["diagnostics_json"].item()))
        rows.append(
            {
                **source,
                "feature_path": str(feature_path),
                "observed_hand_frame_fraction": diagnostics["observed_hand_frame_fraction"],
                "hand_presence_fraction": diagnostics["hand_presence_fraction"],
                "face_presence_fraction": diagnostics["face_presence_fraction"],
                "body_presence_fraction": diagnostics["body_presence_fraction"],
                "quality_score": quality_score(diagnostics),
                "training_eligible": False,
            }
        )
    selected, excluded = select_balanced(
        rows,
        class_triage,
        args.cap_per_class,
        args.minimum_observed_hand,
        args.minimum_hand_presence,
        args.minimum_face_presence,
    )
    if args.materialize_symlinks:
        for row in selected:
            label = str(row["canonical_label"])
            raw = Path(str(row["raw_path"]))
            feature = Path(str(row["feature_path"]))
            safe_link(
                raw,
                args.audit_root / f"{args.materialized_prefix}_raw" / label / raw.name,
            )
            safe_link(
                feature,
                args.audit_root
                / f"{args.materialized_prefix}_landmarks_v17"
                / label
                / feature.name,
            )
    eligible_before_cap, _ = select_balanced(
        rows,
        class_triage,
        0,
        args.minimum_observed_hand,
        args.minimum_hand_presence,
        args.minimum_face_presence,
    )
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "quality-ranked signer-balanced SemLex train-only supplement",
        "training_eligible": False,
        "split": "train_only",
        "cap_per_class": args.cap_per_class or None,
        "quality_gates": {
            "minimum_observed_hand_frame_fraction": args.minimum_observed_hand,
            "minimum_hand_presence_fraction": args.minimum_hand_presence,
            "minimum_face_presence_fraction": args.minimum_face_presence,
        },
        "quality_passing_before_cap": len(eligible_before_cap),
        "quality_rejected_or_mismatch_priority": len(rows) - len(eligible_before_cap),
        "exact_raw_sha256_deduplication": len(
            {str(row["sha256"]) for row in selected}
        )
        == len(selected),
        "mismatch_review_classes_excluded": excluded,
        "selected_clips": len(selected),
        "selected_classes": len({row["canonical_label"] for row in selected}),
        "selected_signers": len({row["semlex_signer_id"] for row in selected}),
        "videos": selected,
    }
    path = args.audit_root / args.output_name
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "clips": output["selected_clips"],
                "classes": output["selected_classes"],
                "signers": output["selected_signers"],
                "excluded": excluded,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
