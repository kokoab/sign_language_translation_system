#!/usr/bin/env python3
"""Evaluate a two-checkpoint weight soup on the fixed Citizen validation split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_v17 import (
    Citizen100V17Dataset,
    extractor_schema_fingerprint,
    select_device,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def blend_state_dicts(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    first_weight: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= first_weight <= 1.0:
        raise ValueError("first checkpoint weight must be in [0, 1]")
    if first.keys() != second.keys():
        raise ValueError("checkpoint state dictionaries differ")
    output: dict[str, torch.Tensor] = {}
    for key in first:
        left, right = first[key], second[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"incompatible state tensor: {key}")
        if left.is_floating_point():
            output[key] = left * first_weight + right * (1.0 - first_weight)
        else:
            if not torch.equal(left, right):
                raise ValueError(f"non-floating state differs: {key}")
            output[key] = left.clone()
    return output


@torch.inference_mode()
def predict(
    model: SLTStage1V17, loader: DataLoader, device: torch.device
) -> np.ndarray:
    rows: list[np.ndarray] = []
    model.eval()
    for features, _ in loader:
        logits = model(features.to(device, non_blocking=device.type == "cuda"))
        if device.type == "mps":
            torch.mps.synchronize()
        rows.append(logits.float().cpu().numpy())
    return np.concatenate(rows)


def metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    confusion = np.zeros((logits.shape[1], logits.shape[1]), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "top1": 100.0 * float((predictions == targets).mean()),
        "top5": 100.0 * float((top5 == targets[:, None]).any(axis=1).mean()),
        "macro_f1": 100.0 * float(f1.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument(
        "--weights", default="0,0.25,0.5,0.75,1",
        help="Comma-separated weights for the first checkpoint",
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/local/citizen100_v17/landmarks"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--rejections", type=Path,
        default=Path("data/local/citizen100_v17/rejections.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (args.first, args.second)
    ]
    for checkpoint in checkpoints:
        if checkpoint.get("format") != "slt_stage1_v17":
            raise ValueError("model soup requires v17 Stage 1 checkpoints")
    normalized_configs = [
        asdict(Stage1V17Config(**checkpoint["model_config"]))
        for checkpoint in checkpoints
    ]
    if normalized_configs[0] != normalized_configs[1]:
        raise ValueError("checkpoint model_config values differ")
    for key in ("label_to_index", "manifest_sha256", "schema_fingerprint"):
        if checkpoints[0][key] != checkpoints[1][key]:
            raise ValueError(f"checkpoint {key} values differ")
    if checkpoints[0]["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("checkpoint manifest does not match the current manifest")
    expected_schema = extractor_schema_fingerprint("apple")
    if checkpoints[0]["schema_fingerprint"] != expected_schema:
        raise ValueError("checkpoint schema is not the current Apple v17 schema")

    dataset = Citizen100V17Dataset(
        args.data_root, "val", args.manifest, args.rejections,
        cache=True, expected_schema=expected_schema,
    )
    if checkpoints[0]["label_to_index"] != dataset.label_to_index:
        raise ValueError("checkpoint label mapping differs from validation data")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = select_device(args.device)
    model = SLTStage1V17(Stage1V17Config(**normalized_configs[0])).to(device)
    targets = dataset.targets.numpy()
    weights = [float(value) for value in args.weights.split(",")]
    rows: list[dict[str, float]] = []
    logits_by_weight: dict[float, np.ndarray] = {}
    for weight in weights:
        state = blend_state_dicts(
            checkpoints[0]["model_state_dict"],
            checkpoints[1]["model_state_dict"],
            weight,
        )
        model.load_state_dict(state)
        logits = predict(model, loader, device)
        logits_by_weight[weight] = logits
        rows.append({"first_weight": weight, **metrics(logits, targets)})

    best = max(rows, key=lambda row: (row["top1"], row["macro_f1"]))
    best_weight = float(best["first_weight"])
    result = {
        "split": "val",
        "samples": len(dataset),
        "first_checkpoint": str(args.first),
        "first_sha256": sha256_file(args.first),
        "second_checkpoint": str(args.second),
        "second_sha256": sha256_file(args.second),
        "rows": rows,
        "best": best,
        "single_model_runtime": True,
        "test_evaluated": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output / "best_validation_logits.npz",
        logits=logits_by_weight[best_weight].astype(np.float32),
        targets=targets,
        item_ids=np.asarray(
            [str(path.relative_to(args.data_root)) for path in dataset.files]
        ),
    )
    best_state = blend_state_dicts(
        checkpoints[0]["model_state_dict"],
        checkpoints[1]["model_state_dict"],
        best_weight,
    )
    torch.save(
        {
            "format": "slt_stage1_v17",
            "epoch": None,
            "model_config": normalized_configs[0],
            "model_state_dict": best_state,
            "validation_metrics": {
                key: best[key] for key in ("top1", "top5", "macro_f1")
            },
            "label_to_index": checkpoints[0]["label_to_index"],
            "manifest_sha256": checkpoints[0]["manifest_sha256"],
            "schema_fingerprint": checkpoints[0]["schema_fingerprint"],
            "model_soup_provenance": result,
        },
        args.output / "best_model.pth",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
