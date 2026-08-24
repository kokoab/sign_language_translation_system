#!/usr/bin/env python3
"""Evaluate a fixed multimodal score ensemble on aligned validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_item_id(value: object) -> str:
    item = str(value).replace("\\", "/")
    if "/landmarks_v17/" in item:
        item = item.split("/landmarks_v17/", 1)[1]
    elif "/landmarks/" in item:
        item = item.split("/landmarks/", 1)[1]
    for prefix in ("semlex_val/", "local_audit/", "local_deep_clean_val/"):
        if item.startswith(prefix):
            item = item[len(prefix):]
            break
    if item.startswith("val/"):
        item = item[4:]
    for suffix in (
        ".visual_speech_v17.npz",
        ".mouth_rgb_v17.npz",
        ".hand_mobileclip2_v17.npz",
        ".hand_spatial_mobileclip2_v17.npz",
        ".v17.npz",
    ):
        if item.endswith(suffix):
            item = item[: -len(suffix)]
            break
    return item


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def per_sample_zscore(logits: np.ndarray) -> np.ndarray:
    value = logits.astype(np.float64)
    return (value - value.mean(axis=1, keepdims=True)) / np.maximum(
        value.std(axis=1, keepdims=True), 1e-8
    )


def classification_metrics(
    scores: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    predictions = scores.argmax(axis=1)
    top5 = np.argpartition(scores, -5, axis=1)[:, -5:]
    confusion = np.zeros((scores.shape[1], scores.shape[1]), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "top1": 100.0 * float((predictions == targets).mean()),
        "top1_correct": int((predictions == targets).sum()),
        "top5": 100.0 * float((top5 == targets[:, None]).any(axis=1).mean()),
        "macro_f1": 100.0 * float(f1.mean()),
    }


def parse_member(value: str) -> tuple[str, Path, float]:
    fields = value.split("=", 2)
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise argparse.ArgumentTypeError("members must use NAME=NPZ_PATH=WEIGHT")
    try:
        weight = float(fields[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("member weight must be numeric") from error
    if weight < 0.0:
        raise argparse.ArgumentTypeError("member weight must be non-negative")
    return fields[0], Path(fields[1]), weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member", action="append", type=parse_member, required=True,
        help="Repeat NAME=NPZ_PATH=WEIGHT; the first member is the paired baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-normalization",
        choices=("probability", "per_sample_zscore"),
        default="probability",
    )
    parser.add_argument("--split-label", default="val")
    parser.add_argument("--weight-selection-split", default="val")
    args = parser.parse_args()
    if len(args.member) < 2:
        raise ValueError("an ensemble needs at least two members")
    names = [name for name, _, _ in args.member]
    if len(names) != len(set(names)):
        raise ValueError("ensemble member names must be unique")
    raw_weights = np.asarray([weight for _, _, weight in args.member], dtype=np.float64)
    if raw_weights.sum() <= 0.0:
        raise ValueError("ensemble weights must have positive total mass")
    weights = raw_weights / raw_weights.sum()

    reference_targets: np.ndarray | None = None
    reference_ids: list[str] | None = None
    aligned_probabilities: list[np.ndarray] = []
    provenance: list[dict[str, object]] = []
    for (name, path, _), weight in zip(args.member, weights):
        with np.load(path, allow_pickle=False) as payload:
            logits = payload["logits"]
            targets = payload["targets"].astype(np.int64)
            ids = [normalized_item_id(value) for value in payload["item_ids"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate normalized item ID in {name}")
        if reference_ids is None:
            reference_ids = ids
            reference_targets = targets
            order = np.arange(len(ids))
        else:
            lookup = {item: index for index, item in enumerate(ids)}
            if set(lookup) != set(reference_ids):
                raise ValueError(f"item IDs differ for {name}")
            order = np.asarray([lookup[item] for item in reference_ids])
            if not np.array_equal(targets[order], reference_targets):
                raise ValueError(f"targets differ for {name}")
        aligned_probabilities.append(
            probabilities(logits[order])
            if args.score_normalization == "probability"
            else per_sample_zscore(logits[order])
        )
        provenance.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "normalized_weight": float(weight),
            }
        )

    assert reference_targets is not None and reference_ids is not None
    scores = sum(
        weight * member
        for weight, member in zip(weights, aligned_probabilities)
    )
    ensemble_metrics = classification_metrics(scores, reference_targets)
    baseline_metrics = classification_metrics(
        aligned_probabilities[0], reference_targets
    )
    baseline_predictions = aligned_probabilities[0].argmax(axis=1)
    ensemble_predictions = scores.argmax(axis=1)
    baseline_correct = baseline_predictions == reference_targets
    ensemble_correct = ensemble_predictions == reference_targets
    paired = {
        "both_correct": int((baseline_correct & ensemble_correct).sum()),
        "baseline_only": int((baseline_correct & ~ensemble_correct).sum()),
        "ensemble_only": int((~baseline_correct & ensemble_correct).sum()),
        "both_wrong": int((~baseline_correct & ~ensemble_correct).sum()),
    }
    result = {
        "split": args.split_label,
        "score_normalization": args.score_normalization,
        "samples": len(reference_targets),
        "members": provenance,
        "baseline": baseline_metrics,
        "ensemble": ensemble_metrics,
        "paired_vs_first_member": paired,
        "weight_selection_split": args.weight_selection_split,
        "independent_confirmation_required": True,
        "test_evaluated": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output / "ensemble_scores.npz",
        scores=scores.astype(np.float32),
        targets=reference_targets,
        item_ids=np.asarray(reference_ids),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
