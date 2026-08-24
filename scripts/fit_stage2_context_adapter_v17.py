#!/usr/bin/env python3
"""Fit a signer-cross-validated linear adapter on train-only ASLLRP temporal features."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_MODES = ("mean", "mean_std", "mean_std_max", "mean_std_max_delta")


def temporal_summary(features: np.ndarray, mode: str) -> np.ndarray:
    value = features.astype(np.float32).reshape(-1, features.shape[-1])
    parts = [value.mean(axis=0)]
    if mode in {"mean_std", "mean_std_max", "mean_std_max_delta"}:
        parts.append(value.std(axis=0))
    if mode in {"mean_std_max", "mean_std_max_delta"}:
        parts.append(value.max(axis=0))
    if mode == "mean_std_max_delta":
        edge = min(8, len(value))
        parts.append(value[-edge:].mean(axis=0) - value[:edge].mean(axis=0))
    if mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {mode}")
    return np.concatenate(parts).astype(np.float32)


def load_rows(root: Path, role: str, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    signer_by_item = {str(row["source_item_id"]): str(row["signer_id"]) for row in manifest["rows"]}
    features = []
    labels = []
    signers = []
    items = []
    raw = []
    for path in sorted(root.glob(f"{role}/*/*.stage2_frozen_v17.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            target = payload["target_indices"].astype(np.int64)
            value = payload["frozen_features"].astype(np.float32)
        if len(target) != 1:
            raise ValueError(f"context adapter requires one target: {path}")
        item = str(metadata["source_item_id"])
        if item not in signer_by_item:
            raise ValueError(f"missing signer metadata for {item}")
        raw.append(value)
        labels.append(int(target[0]))
        signers.append(signer_by_item[item])
        items.append(item)
    return raw, np.asarray(labels), np.asarray(signers), items


def matrix(raw: list[np.ndarray], mode: str) -> np.ndarray:
    return np.stack([temporal_summary(value, mode) for value in raw])


def estimator(alpha: float):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(
            alpha=alpha,
            class_weight="balanced",
            # There are fewer clips than summary dimensions for the richer
            # modes.  The exact ridge solve uses the compact sample-space
            # system and is dramatically faster here than iterative LSQR.
            solver="cholesky",
        ),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    train_raw, train_y, train_signers, train_items = load_rows(
        args.train_root, "train", args.train_manifest
    )
    validation_raw, validation_y, validation_signers, validation_items = load_rows(
        args.validation_root, "validation", args.validation_manifest
    )
    unique_signers = sorted(set(train_signers.tolist()))
    if len(unique_signers) < 3:
        raise ValueError("at least three training signers are required for signer CV")
    candidates = []
    cached = {mode: matrix(train_raw, mode) for mode in FEATURE_MODES}
    for mode in FEATURE_MODES:
        x = cached[mode]
        for alpha in args.alpha_values:
            folds = []
            for signer in unique_signers:
                fit_mask = train_signers != signer
                held_mask = train_signers == signer
                model = estimator(alpha)
                model.fit(x[fit_mask], train_y[fit_mask])
                predictions = model.predict(x[held_mask])
                errors = int(np.sum(predictions != train_y[held_mask]))
                folds.append({
                    "held_out_signer": signer,
                    "samples": int(held_mask.sum()),
                    "errors": errors,
                    "wer": errors / int(held_mask.sum()),
                })
            mean_wer = float(np.mean([fold["wer"] for fold in folds]))
            worst_wer = float(max(fold["wer"] for fold in folds))
            candidates.append({
                "feature_mode": mode,
                "alpha": alpha,
                "mean_signer_wer": mean_wer,
                "worst_signer_wer": worst_wer,
                "folds": folds,
            })
    selected = min(
        candidates,
        key=lambda value: (
            value["mean_signer_wer"], value["worst_signer_wer"],
            value["alpha"], value["feature_mode"],
        ),
    )
    selected_x = cached[selected["feature_mode"]]
    final_model = estimator(float(selected["alpha"]))
    final_model.fit(selected_x, train_y)
    validation_x = matrix(validation_raw, selected["feature_mode"])
    validation_predictions = final_model.predict(validation_x)
    validation_errors = int(np.sum(validation_predictions != validation_y))
    confusion = Counter(
        (int(expected), int(predicted))
        for expected, predicted in zip(validation_y, validation_predictions)
        if expected != predicted
    )
    scaler = final_model.named_steps["standardscaler"]
    classifier = final_model.named_steps["ridgeclassifier"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        feature_mode=np.array(selected["feature_mode"]),
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        coefficients=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        class_indices=classifier.classes_.astype(np.int64),
        train_items=np.asarray(train_items),
    )
    report = {
        "format": "slt_stage2_context_adapter_v17",
        "version": 1,
        "selection_contract": "feature and regularization selected only by leave-one-train-signer-out CV",
        "train_signers": unique_signers,
        "train_samples": len(train_y),
        "train_class_count": len(set(train_y.tolist())),
        "selected": selected,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "validation_signers": sorted(set(validation_signers.tolist())),
        "validation_samples": len(validation_y),
        "validation_errors": validation_errors,
        "validation_wer": validation_errors / len(validation_y),
        "validation_confusion": [
            {"expected_index": expected, "predicted_index": predicted, "count": count}
            for (expected, predicted), count in confusion.most_common()
        ],
        "artifact": args.output.as_posix(),
        "seconds": time.monotonic() - started,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_train_frozen_features"))
    parser.add_argument("--validation-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"))
    parser.add_argument("--train-manifest", type=Path, default=Path("active/v17/stage2_asllrp_segmented_train_manifest_v17.json"))
    parser.add_argument("--validation-manifest", type=Path, default=Path("active/v17/stage2_asllrp_segmented_validation_manifest_v17.json"))
    parser.add_argument(
        "--alpha-values", type=float, nargs="+", default=(0.1, 1.0, 10.0, 100.0, 1000.0)
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_context_adapter_v1/adapter.npz"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/models/stage2_v17_context_adapter_v1/result.json"))
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
