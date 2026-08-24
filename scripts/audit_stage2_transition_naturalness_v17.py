#!/usr/bin/env python3
"""Test whether train-only synthetic transitions remain separable from genuine spans.

This is a falsification audit, not a perceptual-naturalness proof.  It compares
label-matched genuine ASLLRP bigrams with signer-voice compositions using only frozen
temporal features.  A strong out-of-fold discriminator is evidence that the current
composer has not replicated the genuine transition distribution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import SparseRandomProjection

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.train_stage_2_v17 import (
    RealPhraseDataset,
    SyntheticCompositionDataset,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signer_map(path: Path) -> dict[str, str]:
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output[str(row["utterance_video_filename"])] = str(row["signer_id"])
    return output


def temporal_descriptor(features: np.ndarray, frames: int = 64) -> np.ndarray:
    stream = features.astype(np.float32).reshape(-1, features.shape[-1])
    positions = np.linspace(0.0, len(stream) - 1, frames, dtype=np.float32)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(stream) - 1)
    weight = (positions - lower)[:, None]
    resized = stream[lower] * (1.0 - weight) + stream[upper] * weight
    velocity = np.diff(resized, axis=0)
    acceleration = np.diff(velocity, axis=0)
    center = resized[frames // 2 - 8:frames // 2 + 8]
    outer = np.concatenate((resized[:8], resized[-8:]), axis=0)
    descriptor = np.concatenate((
        resized.mean(axis=0),
        resized.std(axis=0),
        np.abs(velocity).mean(axis=0),
        velocity.std(axis=0),
        np.abs(acceleration).mean(axis=0),
        np.abs(velocity).max(axis=0),
        resized[-1] - resized[0],
        center.std(axis=0) - outer.std(axis=0),
    ))
    if not np.isfinite(descriptor).all():
        raise ValueError("non-finite transition descriptor")
    return descriptor.astype(np.float32)


def model_for(train_size: int, feature_dim: int):
    components = max(1, min(10, train_size - 2, feature_dim))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=components, whiten=True, random_state=17),
        LogisticRegression(C=1.0, max_iter=2000, random_state=17),
    )


def fixed_random_projection(features: np.ndarray) -> np.ndarray:
    """Apply a label-independent projection once before repeated fold fitting."""
    projector = SparseRandomProjection(
        n_components=min(128, features.shape[1]),
        dense_output=True,
        random_state=1701,
    )
    return np.asarray(projector.fit_transform(features), dtype=np.float32)


def grouped_oof(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    fold_rows = []
    for fold, (train, test) in enumerate(splits):
        model = model_for(len(train), features.shape[1])
        model.fit(features[train], labels[train])
        probabilities[test] = model.predict_proba(features[test])[:, 1]
        fold_rows.append({
            "fold": fold,
            "train_rows": len(train),
            "test_rows": len(test),
            "test_groups": sorted(set(groups[test].tolist())),
        })
    if not np.isfinite(probabilities).all():
        raise RuntimeError("incomplete out-of-fold predictions")
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "rows": len(labels),
        "groups": len(set(groups.tolist())),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "folds": fold_rows,
    }


def paired_rows(
    real: RealPhraseDataset,
    synthetic: SyntheticCompositionDataset,
    signers: dict[str, str],
    *,
    require_same_signer: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    by_pair: dict[tuple[int, ...], list[int]] = defaultdict(list)
    by_pair_signer: dict[tuple[tuple[int, ...], str], list[int]] = defaultdict(list)
    for index, row in enumerate(synthetic.rows):
        if row.get("source") != "synthetic_asllrp_contextual_train":
            continue
        target = tuple(int(value) for value in row["target_indices"])
        if len(target) != 2:
            continue
        voice = str(row["signer_voice_synthesis"]["signer_id"])
        by_pair[target].append(index)
        by_pair_signer[(target, voice)].append(index)

    feature_rows = []
    label_rows = []
    group_rows = []
    metadata = []
    for sample in real.samples:
        if sample.source != "asllrp_contiguous" or len(sample.targets) != 2:
            continue
        video = sample.item_id.split(":")[1]
        signer = signers[video]
        target = tuple(int(value) - 1 for value in sample.targets.tolist())
        candidates = (
            by_pair_signer[(target, signer)] if require_same_signer else by_pair[target]
        )
        if not candidates:
            continue
        synthetic_index = sorted(candidates, key=lambda index: synthetic.rows[index]["sequence_id"])[0]
        composed = synthetic[synthetic_index]
        pair_id = f"{sample.item_id}|{composed.item_id}"
        feature_rows.extend((
            temporal_descriptor(sample.features),
            temporal_descriptor(composed.features),
        ))
        label_rows.extend((1, 0))
        group_rows.extend((pair_id, pair_id))
        metadata.append({
            "pair_id": pair_id,
            "signer": signer,
            "target_indices": list(target),
            "real_item_id": sample.item_id,
            "synthetic_item_id": composed.item_id,
            "synthetic_voice": str(
                synthetic.rows[synthetic_index]["signer_voice_synthesis"]["signer_id"]
            ),
        })
    return (
        np.stack(feature_rows),
        np.asarray(label_rows, dtype=np.int64),
        np.asarray(group_rows),
        metadata,
    )


def paired_group_splits(groups: np.ndarray, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=min(folds, len(set(groups.tolist()))))
    dummy = np.zeros(len(groups), dtype=np.int64)
    return list(splitter.split(dummy, dummy, groups))


def signer_splits(
    groups: np.ndarray, metadata: list[dict[str, Any]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    row_signers = np.asarray([
        row["signer"] for row in metadata for _ in range(2)
    ])
    output = []
    for signer in sorted(set(row_signers.tolist())):
        test = np.flatnonzero(row_signers == signer)
        train = np.flatnonzero(row_signers != signer)
        if len(test) >= 2 and len(set(row_signers[train].tolist())) >= 1:
            output.append((train, test))
    return output


def paired_permutation_p_value(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    observed_auc: float,
    permutations: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(1701)
    unique_groups = sorted(set(groups.tolist()))
    null = []
    for _ in range(permutations):
        permuted = labels.copy()
        for group in unique_groups:
            indices = np.flatnonzero(groups == group)
            if rng.random() < 0.5:
                permuted[indices] = permuted[indices[::-1]]
        null.append(grouped_oof(features, permuted, groups, splits)["roc_auc"])
    return {
        "permutations": permutations,
        "null_mean_auc": float(np.mean(null)),
        "null_std_auc": float(np.std(null)),
        "one_sided_p_value": float(
            (1 + sum(value >= observed_auc for value in null)) / (permutations + 1)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    real = RealPhraseDataset(args.real_root, "train")
    synthetic = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    signers = signer_map(args.span_csv)

    same_x, same_y, same_groups, same_meta = paired_rows(
        real, synthetic, signers, require_same_signer=True
    )
    same_x = fixed_random_projection(same_x)
    same_splits = paired_group_splits(same_groups, args.folds)
    same_result = grouped_oof(same_x, same_y, same_groups, same_splits)
    same_result["matching"] = "exact ordered gloss pair and same ASLLRP signer voice"
    same_result["matched_pairs"] = len(same_meta)
    same_result["signer_counts"] = {
        signer: sum(row["signer"] == signer for row in same_meta)
        for signer in sorted({row["signer"] for row in same_meta})
    }
    same_result["permutation_test"] = paired_permutation_p_value(
        same_x, same_y, same_groups, same_splits,
        same_result["roc_auc"], args.permutations,
    )

    broad_x, broad_y, broad_groups, broad_meta = paired_rows(
        real, synthetic, signers, require_same_signer=False
    )
    broad_x = fixed_random_projection(broad_x)
    broad_splits = signer_splits(broad_groups, broad_meta)
    broad_result = grouped_oof(
        broad_x, broad_y, broad_groups, broad_splits
    )
    broad_result["matching"] = (
        "exact ordered gloss pair; evaluation leaves the genuine signer out"
    )
    broad_result["matched_pairs"] = len(broad_meta)
    broad_result["held_out_signers"] = sorted({row["signer"] for row in broad_meta})

    report = {
        "format": "stage2_transition_naturalness_falsification_audit_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": (
            "Can a held-out linear discriminator distinguish genuine continuous "
            "ASLLRP bigrams from label-matched signer-voice compositions?"
        ),
        "same_signer_paired_cross_validation": same_result,
        "genuine_signer_held_out_cross_validation": broad_result,
        "interpretation": (
            "High held-out discrimination is evidence against distributional realism. "
            "Chance discrimination would still not prove perceptual naturalness."
        ),
        "descriptor": (
            "64-frame resampling; per-channel position, velocity, acceleration, "
            "endpoint, and center-versus-edge statistics; fixed label-independent "
            "128D sparse random projection; train-fold scaling/PCA/logistic"
        ),
        "real_root": args.real_root.as_posix(),
        "synthetic_pool": args.synthetic_pool.as_posix(),
        "synthetic_pool_sha256": sha256(args.synthetic_pool),
        "synthetic_plan": args.synthetic_plan.as_posix(),
        "synthetic_plan_sha256": sha256(args.synthetic_plan),
        "limitations": [
            "Frozen recognition features are not a human perceptual judgment.",
            "The genuine set is small and contains only three training signers.",
            "Corpus/capture differences can contribute to separability.",
            "The audit cannot establish facial grammar or rendered RGB naturalness.",
        ],
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "jonathan_validation_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-root", type=Path,
        default=Path("data/local/stage2_v17_frozen_features"),
    )
    parser.add_argument(
        "--synthetic-pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_replay_pool_v2.npz"),
    )
    parser.add_argument(
        "--synthetic-plan", type=Path,
        default=Path("active/v17/stage2_signer_voice_plan_v17.json"),
    )
    parser.add_argument(
        "--span-csv", type=Path,
        default=Path(
            "artifacts/reports/asllrp_continuous_citizen100_v17/"
            "stage2_contiguous_target_spans.csv"
        ),
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument(
        "--report", type=Path,
        default=Path(
            "artifacts/reports/stage2_v17_transition_naturalness_audit_v1.json"
        ),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
