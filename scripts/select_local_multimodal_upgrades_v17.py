#!/usr/bin/env python3
"""Select conservative local upgrades from fixed landmark+hand evidence."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.evaluate_multimodal_ensemble_v17 import normalized_item_id


SOURCE_TIER = "tier_b_dual_top5_one_top1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scores(path: Path, key: str) -> dict[str, tuple[np.ndarray, int]]:
    with np.load(path, allow_pickle=False) as payload:
        scores = payload[key].astype(np.float64)
        targets = payload["targets"].astype(np.int64)
        ids = [normalized_item_id(value) for value in payload["item_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate normalized item ID: {path}")
    return {item: (scores[index], int(targets[index])) for index, item in enumerate(ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--consensus-csv", type=Path,
        default=Path("artifacts/reports/local_citizen100_quality_audit/cap14_exact_consensus/consensus_predictions.csv"),
    )
    parser.add_argument("--landmark-logits", type=Path, required=True)
    parser.add_argument("--hand-logits", type=Path, required=True)
    parser.add_argument("--ensemble-scores", type=Path, required=True)
    parser.add_argument(
        "--hand-crop-root", type=Path,
        default=Path("data/local/local_citizen100_quality_audit_q82_cap14_exact/hand_rgb_v17"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-margin", type=float, default=0.5)
    parser.add_argument("--minimum-observed-hand", type=float, default=0.8)
    parser.add_argument("--minimum-crop-validity", type=float, default=0.8)
    parser.add_argument("--cap-per-class", type=int, default=2)
    args = parser.parse_args()
    if args.cap_per_class < 1:
        raise ValueError("cap-per-class must be positive")
    source_rows = list(csv.DictReader(args.consensus_csv.open(encoding="utf-8")))
    if len(source_rows) != 1021:
        raise ValueError("expected the complete 1,021-clip consensus ledger")
    landmark = load_scores(args.landmark_logits, "logits")
    hand = load_scores(args.hand_logits, "logits")
    ensemble = load_scores(args.ensemble_scores, "scores")
    expected = {
        f"{row['canonical_label']}/{Path(row['raw_path']).stem}" for row in source_rows
    }
    if set(landmark) != expected or set(hand) != expected or set(ensemble) != expected:
        raise ValueError("local modality ledgers are not exactly aligned")

    candidates: list[dict[str, object]] = []
    for row in source_rows:
        if row["consensus_tier"] != SOURCE_TIER:
            continue
        item_id = Path(row["raw_path"]).stem
        key = f"{row['canonical_label']}/{item_id}"
        landmark_scores, target = landmark[key]
        hand_scores, hand_target = hand[key]
        ensemble_scores, ensemble_target = ensemble[key]
        if target != hand_target or target != ensemble_target:
            raise ValueError(f"target mismatch: {key}")
        crop_path = (
            args.hand_crop_root / row["canonical_label"]
            / f"{item_id}.hand_rgb_v17.npz"
        )
        with np.load(crop_path, allow_pickle=False) as payload:
            crop_validity = float(payload["valid"].mean())
        true_score = float(ensemble_scores[target])
        runner_up = float(np.max(np.delete(ensemble_scores, target)))
        margin = true_score - runner_up
        if not (
            int(landmark_scores.argmax()) == target
            and int(hand_scores.argmax()) == target
            and int(ensemble_scores.argmax()) == target
            and margin >= args.minimum_margin
            and float(row["observed_hand_frame_fraction"]) >= args.minimum_observed_hand
            and crop_validity >= args.minimum_crop_validity
        ):
            continue
        candidates.append({
            **row,
            "current_landmark_top1": True,
            "current_hand_top1": True,
            "fixed_landmark_hand_top1": True,
            "fixed_landmark_hand_true_margin": margin,
            "new_hand_crop_validity": crop_validity,
            "upgrade_tier": "tier_a2_four_model_multimodal_consensus",
            "training_eligible": False,
        })
    by_class: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_class[str(row["canonical_label"])].append(row)
    selected: list[dict[str, object]] = []
    for label in sorted(by_class):
        ranked = sorted(
            by_class[label],
            key=lambda row: (
                -float(row["fixed_landmark_hand_true_margin"]), str(row["raw_path"])
            ),
        )
        selected.extend(ranked[:args.cap_per_class])
    selected.sort(key=lambda row: (str(row["canonical_label"]), str(row["raw_path"])))
    if len(candidates) != 25 or len(selected) != 23:
        raise ValueError(
            f"frozen upgrade count changed: candidates={len(candidates)} selected={len(selected)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "selected_clips.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "conservative multimodal local upgrade review shortlist",
        "training_eligible": False,
        "split_eligibility": "train_only_after_ASL_fluent_exact_variant_review",
        "source_warning": (
            "Folder/exact-text identity plus four correlated model screens does not "
            "prove the ASL lexical variant or signer identity."
        ),
        "sources": {
            "consensus_csv": str(args.consensus_csv),
            "consensus_csv_sha256": sha256_file(args.consensus_csv),
            "landmark_logits": str(args.landmark_logits),
            "landmark_logits_sha256": sha256_file(args.landmark_logits),
            "hand_logits": str(args.hand_logits),
            "hand_logits_sha256": sha256_file(args.hand_logits),
            "ensemble_scores": str(args.ensemble_scores),
            "ensemble_scores_sha256": sha256_file(args.ensemble_scores),
        },
        "gates": {
            "source_tier": SOURCE_TIER,
            "current_landmark_top1": True,
            "current_hand_top1": True,
            "fixed_75_25_landmark_hand_top1": True,
            "minimum_fixed_true_margin": args.minimum_margin,
            "minimum_observed_hand_frame_fraction": args.minimum_observed_hand,
            "minimum_new_hand_crop_validity": args.minimum_crop_validity,
            "cap_per_class": args.cap_per_class,
            "face_stream_used": False,
        },
        "uncapped_candidates": len(candidates),
        "selected_clips": len(selected),
        "selected_classes": len(by_class),
        "videos": selected,
    }
    (args.output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Local multimodal upgrade shortlist", "",
        "**Status:** ASL-variant review required; not automatically training-approved.", "",
        f"- Selected: {len(selected)} clips across {len(by_class)} classes",
        f"- Uncapped passing candidates: {len(candidates)}",
        f"- Per-class cap: {args.cap_per_class}",
        "- Evidence: old dual-model Tier B + current landmark top-1 + current hand top-1",
        f"- Minimum fixed 75/25 true-class margin: {args.minimum_margin:.2f}",
        "- Face/lip stream used: no (rejected on this local domain)", "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "selected_clips": len(selected), "selected_classes": len(by_class),
        "output_dir": str(args.output_dir), "training_eligible": False,
    }, indent=2))


if __name__ == "__main__":
    main()
