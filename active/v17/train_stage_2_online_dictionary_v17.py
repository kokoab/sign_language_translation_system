#!/usr/bin/env python3
"""Fit a four-domain isolated dictionary for online Stage-2 window voting."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import torch

if __package__ in (None, ""):
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.model_stage2_v17 import Stage2IsolatedCorrectionV17
from active.v17.train_stage_2_isolated_correction_v17 import sha256, summary


def pool(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return summary(payload["frozen_features"]), payload["target_indices"].astype(np.int64)


def tree(root: Path, role: str) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for path in sorted((root / role).glob("*/*.stage2_frozen_v17.npz")):
        with np.load(path, allow_pickle=False) as payload:
            value = payload["frozen_features"].astype(np.float32)
            target = int(payload["target_indices"][0])
        features.append(value)
        targets.extend([target] * len(value))
    if not features:
        raise ValueError(f"no {role} features under {root}")
    return summary(np.concatenate(features)), np.asarray(targets, dtype=np.int64)


def accuracy(classifier, scaler, values, targets) -> dict[str, object]:
    prediction = classifier.predict(scaler.transform(values))
    return {
        "accuracy": float((prediction == targets).mean()),
        "correct": int((prediction == targets).sum()),
        "samples": len(targets),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    checked = (*args.train_pools, *args.validation_pools, args.asllrp_train, args.asllrp_validation)
    if any("test" in {part.lower() for part in path.parts} for path in checked):
        raise ValueError("online dictionary training is restricted to train/validation data")
    domains = ("citizen", "semlex", "local", "asllrp")
    training = [pool(path) for path in args.train_pools]
    validation = [pool(path) for path in args.validation_pools]
    training.append(tree(args.asllrp_train, "train"))
    validation.append(tree(args.asllrp_validation, "validation"))
    values = np.concatenate([value for value, _ in training])
    targets = np.concatenate([target for _, target in training])
    sources = np.concatenate([
        np.full(len(target), index, dtype=np.int64)
        for index, (_, target) in enumerate(training)
    ])
    weights = np.empty(len(targets), dtype=np.float64)
    for source in range(len(domains)):
        source_mask = sources == source
        counts = Counter(targets[source_mask].tolist())
        for target, count in counts.items():
            weights[source_mask & (targets == target)] = 1 / len(domains) / len(counts) / count
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(values)
    classifier = SGDClassifier(
        loss="log_loss", alpha=args.alpha, max_iter=args.max_iterations,
        tol=args.tolerance, random_state=args.seed, n_jobs=-1, average=True,
    ).fit(scaler.transform(values), targets, sample_weight=weights)
    model = Stage2IsolatedCorrectionV17(
        scaler_mean=torch.from_numpy(scaler.mean_.astype(np.float32)),
        scaler_scale=torch.from_numpy(scaler.scale_.astype(np.float32)),
        coefficients=torch.from_numpy(classifier.coef_.astype(np.float32)),
        intercept=torch.from_numpy(classifier.intercept_.astype(np.float32)),
        blend_weight=1.0,
    ).eval()
    metrics = {
        domain: accuracy(classifier, scaler, value, target)
        for domain, (value, target) in zip(domains, validation)
    }
    package = {
        "format": "slt_stage2_isolated_correction_v17",
        "format_version": 1,
        "role": "online_sliding_window_dictionary",
        "summary_mode": "mean_std_max_first_last_delta",
        "scaler_mean": model.scaler_mean.cpu(),
        "scaler_scale": model.scaler_scale.cpu(),
        "coefficients": model.coefficients.cpu(),
        "intercept": model.intercept.cpu(),
        "blend_weight": 1.0,
        "validation_metrics": metrics,
        "training_inputs": [path.as_posix() for path in checked],
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output)
    report = {
        "checkpoint": args.output.as_posix(),
        "checkpoint_sha256": sha256(args.output),
        "validation_metrics": metrics,
        "equal_domain_mean_accuracy": float(np.mean([
            value["accuracy"] for value in metrics.values()
        ])),
        "sklearn_iterations": int(classifier.n_iter_),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--train-pools", type=Path, nargs=3, default=[
        Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
        Path("data/local/stage2_v17_synthetic/semlex_train_isolated_pool.npz"),
        Path("data/local/stage2_v17_isolated_replay/local_train.npz"),
    ])
    value.add_argument("--validation-pools", type=Path, nargs=3, default=[
        Path("data/local/stage2_v17_isolated_replay/citizen_validation.npz"),
        Path("data/local/stage2_v17_isolated_replay/semlex_validation.npz"),
        Path("data/local/stage2_v17_isolated_replay/local_validation.npz"),
    ])
    value.add_argument("--asllrp-train", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_train_frozen_features"))
    value.add_argument("--asllrp-validation", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_online_dictionary_v1/model.pth"))
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_online_dictionary_v1/validation.json"))
    value.add_argument("--alpha", type=float, default=3e-5)
    value.add_argument("--max-iterations", type=int, default=80)
    value.add_argument("--tolerance", type=float, default=1e-4)
    value.add_argument("--seed", type=int, default=1701)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
