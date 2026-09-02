#!/usr/bin/env python3
"""Fit the compact Stage-2 temporal correction used by isolated inference."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import torch

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from active.v17.model_stage2_v17 import Stage2IsolatedCorrectionV17
from active.v17.model_unified_multimodal_v17 import (
    UnifiedFusionHeadV17,
    UnifiedMultimodalV17Config,
)


DOMAINS = ("citizen", "semlex", "local")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(features: np.ndarray) -> np.ndarray:
    value = features.astype(np.float32)
    return np.concatenate((
        value.mean(1), value.std(1), value.max(1),
        value[:, 0], value[:, -1], value[:, -1] - value[:, 0],
    ), axis=1)


def stage1_logits(
    cache: Path, head: UnifiedFusionHeadV17, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache, allow_pickle=False) as payload:
        arrays = {
            key: payload[key].astype(np.float32)
            for key in ("landmark_features", "hand_features", "landmark_logits", "hand_logits")
        }
        targets = payload["targets"].astype(np.int64)
        item_ids = payload["item_ids"].astype(str)
    output = []
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            output.append(head(*[
                torch.from_numpy(arrays[key][start:start + batch_size])
                for key in ("landmark_features", "hand_features", "landmark_logits", "hand_logits")
            ]).numpy())
    return np.concatenate(output), targets, item_ids


def zscore(value: np.ndarray) -> np.ndarray:
    centered = value - value.mean(1, keepdims=True)
    return centered / np.maximum(value.std(1, keepdims=True), 1e-6)


def metrics(logits: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, object]:
    domains = {
        domain: {
            "accuracy": float((value.argmax(1) == target).mean()),
            "correct": int((value.argmax(1) == target).sum()),
            "samples": len(target),
        }
        for domain, value, target in zip(DOMAINS, logits, targets)
    }
    return {
        "domains": domains,
        "equal_domain_mean_accuracy": float(np.mean([
            domains[domain]["accuracy"] for domain in DOMAINS
        ])),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
    head = UnifiedFusionHeadV17(
        UnifiedMultimodalV17Config(**checkpoint["head_config"])
    )
    head.load_state_dict(checkpoint["head_state_dict"], strict=True)
    head.eval()
    train_features, train_targets, source_ids = [], [], []
    validation_features, validation_frozen, validation_targets, validation_stage1 = [], [], [], []
    pool_hashes: dict[str, str] = {}
    for source_id, domain in enumerate(DOMAINS):
        train_pool = getattr(args, f"{domain}_train_pool")
        validation_pool = getattr(args, f"{domain}_validation_pool")
        pool_hashes[f"{domain}_train"] = sha256(train_pool)
        pool_hashes[f"{domain}_validation"] = sha256(validation_pool)
        with np.load(train_pool, allow_pickle=False) as payload:
            train_features.append(summary(payload["frozen_features"]))
            train_targets.append(payload["target_indices"].astype(np.int64))
            pool_ids = (
                payload["item_ids"].astype(str)
                if "item_ids" in payload.files
                else np.asarray(json.loads(str(payload["metadata_json"]))["item_ids"])
            )
        _, cache_targets, cache_ids = stage1_logits(
            args.unified_cache / f"{domain}_train.npz", head, args.batch_size
        )
        if not np.array_equal(pool_ids, cache_ids) or not np.array_equal(
            train_targets[-1], cache_targets
        ):
            raise ValueError(f"{domain} train isolated/unified cache order differs")
        source_ids.append(np.full(len(cache_targets), source_id, dtype=np.int64))
        with np.load(validation_pool, allow_pickle=False) as payload:
            frozen = payload["frozen_features"].astype(np.float32)
            validation_frozen.append(frozen)
            validation_features.append(summary(frozen))
            validation_targets.append(payload["target_indices"].astype(np.int64))
            pool_ids = payload["item_ids"].astype(str)
        base, cache_targets, cache_ids = stage1_logits(
            args.unified_cache / f"{domain}_val.npz", head, args.batch_size
        )
        if not np.array_equal(pool_ids, cache_ids) or not np.array_equal(
            validation_targets[-1], cache_targets
        ):
            raise ValueError(f"{domain} validation isolated/unified cache order differs")
        validation_stage1.append(base)

    values = np.concatenate(train_features)
    targets = np.concatenate(train_targets)
    sources = np.concatenate(source_ids)
    sample_weights = np.empty(len(targets), dtype=np.float64)
    for source_id in range(len(DOMAINS)):
        source_mask = sources == source_id
        counts = Counter(int(value) for value in targets[source_mask])
        for target, count in counts.items():
            sample_weights[source_mask & (targets == target)] = (
                1.0 / len(DOMAINS) / len(counts) / count
            )
    sample_weights *= len(sample_weights) / sample_weights.sum()
    scaler = StandardScaler().fit(values)
    classifier = SGDClassifier(
        loss="log_loss", alpha=args.alpha, max_iter=args.max_iterations,
        tol=args.tolerance, random_state=args.seed, n_jobs=-1, average=True,
    ).fit(scaler.transform(values), targets, sample_weight=sample_weights)
    classifier_logits = [
        classifier.decision_function(scaler.transform(value)).astype(np.float32)
        for value in validation_features
    ]
    stage1_normalized = [zscore(value) for value in validation_stage1]
    classifier_normalized = [zscore(value) for value in classifier_logits]
    candidates = []
    for weight in np.arange(0.0, 1.0001, args.blend_step):
        combined = [
            (1.0 - weight) * base + weight * correction
            for base, correction in zip(stage1_normalized, classifier_normalized)
        ]
        candidates.append({"weight": float(weight), "metrics": metrics(combined, validation_targets)})
    winner = max(
        candidates,
        key=lambda row: (
            row["metrics"]["equal_domain_mean_accuracy"],
            -row["weight"],
        ),
    )
    model = Stage2IsolatedCorrectionV17(
        scaler_mean=torch.from_numpy(scaler.mean_.astype(np.float32)),
        scaler_scale=torch.from_numpy(scaler.scale_.astype(np.float32)),
        coefficients=torch.from_numpy(classifier.coef_.astype(np.float32)),
        intercept=torch.from_numpy(classifier.intercept_.astype(np.float32)),
        blend_weight=float(winner["weight"]),
    ).eval()
    parity_mismatches = 0
    parity_max_abs = 0.0
    with torch.inference_mode():
        for frozen, features, base in zip(
            validation_frozen, validation_features, validation_stage1
        ):
            correction = classifier.decision_function(scaler.transform(features))
            expected_logits = (
                (1.0 - winner["weight"]) * zscore(base)
                + winner["weight"] * zscore(correction)
            )
            actual_logits = model(
                torch.from_numpy(frozen), torch.from_numpy(base.astype(np.float32))
            ).numpy()
            parity_max_abs = max(
                parity_max_abs, float(np.max(np.abs(expected_logits - actual_logits)))
            )
            parity_mismatches += int(
                not np.array_equal(expected_logits.argmax(1), actual_logits.argmax(1))
            )
    package = {
        "format": "slt_stage2_isolated_correction_v17",
        "format_version": 1,
        "summary_mode": "mean_std_max_first_last_delta",
        "scaler_mean": model.scaler_mean.cpu(),
        "scaler_scale": model.scaler_scale.cpu(),
        "coefficients": model.coefficients.cpu(),
        "intercept": model.intercept.cpu(),
        "blend_weight": model.blend_weight,
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "training_pool_sha256": pool_hashes,
        "classifier_validation_metrics": metrics(classifier_logits, validation_targets),
        "stage1_validation_metrics": metrics(validation_stage1, validation_targets),
        "hybrid_validation_metrics": winner["metrics"],
        "blend_candidates": candidates,
        "sklearn_iterations": int(classifier.n_iter_),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(package, temporary)
    temporary.replace(args.output)
    result = {
        "checkpoint": args.output.as_posix(),
        "checkpoint_sha256": sha256(args.output),
        "stage1_validation_metrics": package["stage1_validation_metrics"],
        "classifier_validation_metrics": package["classifier_validation_metrics"],
        "hybrid_validation_metrics": package["hybrid_validation_metrics"],
        "stage2_exceeds_stage1": (
            winner["metrics"]["equal_domain_mean_accuracy"]
            > package["stage1_validation_metrics"]["equal_domain_mean_accuracy"]
        ),
        "selected_blend_weight": winner["weight"],
        "parity_prediction_mismatches": parity_mismatches,
        "parity_max_abs": parity_max_abs,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--citizen-train-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"))
    parser.add_argument("--semlex-train-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/semlex_train_isolated_pool.npz"))
    parser.add_argument("--local-train-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/local_train.npz"))
    parser.add_argument("--citizen-validation-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/citizen_validation.npz"))
    parser.add_argument("--semlex-validation-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/semlex_validation.npz"))
    parser.add_argument("--local-validation-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/local_validation.npz"))
    parser.add_argument("--unified-cache", type=Path, default=Path("artifacts/generated/unified_multimodal_student_v17"))
    parser.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_isolated_correction_v1/model.pth"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_isolated_correction_v1/validation.json"))
    parser.add_argument("--alpha", type=float, default=3e-5)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main() -> None:
    started = time.monotonic()
    result = run(build_parser().parse_args())
    result["seconds"] = time.monotonic() - started
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
