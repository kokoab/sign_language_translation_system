#!/usr/bin/env python3
"""Evaluate a frozen v17 checkpoint on the approved local familiar-signer split."""

from __future__ import annotations

import argparse
import csv
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
    LocalValidationV17Dataset,
    extractor_schema_fingerprint,
    select_device,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_metrics(
    logits: np.ndarray, targets: np.ndarray, num_classes: int
) -> dict[str, object]:
    predicted = logits.argmax(axis=1)
    top5 = np.argpartition(logits, -5, axis=1)[:, -5:]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = confusion.sum(axis=1) > 0
    return {
        "samples": len(targets),
        "classes_present": int(present.sum()),
        "top1": 100.0 * float((predicted == targets).mean()),
        "top1_correct": int((predicted == targets).sum()),
        "top5": 100.0 * float((top5 == targets[:, None]).any(axis=1).mean()),
        "top5_correct": int((top5 == targets[:, None]).any(axis=1).sum()),
        "macro_f1_all_100": 100.0 * float(f1.mean()),
        "macro_f1_present_classes": 100.0 * float(f1[present].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/local/local_deep_clean_v17/landmarks/val"),
    )
    parser.add_argument(
        "--local-manifest",
        type=Path,
        default=Path("data/local/local_deep_clean_v17/val_final_manifest.json"),
    )
    parser.add_argument(
        "--citizen-manifest",
        type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mask-local-mouth-nodes", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("not a v17 Stage 1 checkpoint")
    if checkpoint.get("manifest_sha256") != sha256_file(args.citizen_manifest):
        raise ValueError("checkpoint Citizen manifest mismatch")
    expected_schema = extractor_schema_fingerprint("apple")
    if checkpoint.get("schema_fingerprint") != expected_schema:
        raise ValueError("checkpoint v17 schema mismatch")
    provenance = checkpoint.get("training_data_provenance", {})
    if (
        provenance.get("citizen_test_accessed") is not False
        or provenance.get("semlex_test_accessed") is not False
    ):
        raise ValueError("checkpoint does not prove sealed test provenance")

    dataset = LocalValidationV17Dataset(
        args.data_root,
        args.local_manifest,
        checkpoint["label_to_index"],
        cache=not args.no_cache,
        expected_schema=expected_schema,
        mask_mouth_nodes=args.mask_local_mouth_nodes,
    )
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = select_device(args.device)
    model.to(device).eval()
    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for features, _ in DataLoader(dataset, batch_size=args.batch_size, shuffle=False):
            parts.append(model(features.to(device)).float().cpu().numpy())
    logits = np.concatenate(parts)
    targets = dataset.targets.numpy()
    metrics = classification_metrics(logits, targets, dataset.num_classes)
    result = {
        "format": "slt_v17_local_deep_clean_validation_evaluation",
        "split": "local_val_nonsigner_disjoint",
        "signer_disjoint": False,
        "signer_overlap_user_approved": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "local_manifest": str(args.local_manifest),
        "local_manifest_sha256": sha256_file(args.local_manifest),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_mouth_node_policy": (
            "zero_only_mouth_left_mouth_right_upper_lip_lower_lip"
            if args.mask_local_mouth_nodes
            else "unchanged"
        ),
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    item_ids = np.asarray(
        [f"{path.parent.name}/{path.name.removesuffix('.v17.npz')}" for path in dataset.files]
    )
    np.savez_compressed(
        args.output_dir / "logits.npz",
        logits=logits.astype(np.float32),
        targets=targets,
        item_ids=item_ids,
    )
    predicted = logits.argmax(axis=1)
    inverse = {value: key for key, value in checkpoint["label_to_index"].items()}
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("item_id", "true_label", "predicted_label", "correct"))
        for index, item_id in enumerate(item_ids):
            writer.writerow(
                (
                    item_id,
                    inverse[int(targets[index])],
                    inverse[int(predicted[index])],
                    bool(predicted[index] == targets[index]),
                )
            )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
