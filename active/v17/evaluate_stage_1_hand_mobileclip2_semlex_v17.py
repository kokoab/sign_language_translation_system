#!/usr/bin/env python3
"""Evaluate a v17 hand-RGB checkpoint on frozen SemLex validation only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.evaluate_stage_1_mobileclip2_v17 import file_sha256, metrics_from_logits
from active.v17.extract_hand_rgb_semlex_val_v17 import SOURCE, SPLIT, validation_items
from active.v17.model_hand_mobileclip2_v17 import (
    HandMobileCLIP2Stage1Config,
    HandMobileCLIP2Stage1V17,
)
from active.v17.schema_hand_mobileclip2_v17 import (
    HandMobileCLIP2V17Config,
    schema_fingerprint,
)
from active.v17.train_stage_1_hand_mobileclip2_v17 import HandMobileCLIP2Dataset
from active.v17.train_stage_1_mobileclip2_v17 import select_device


class SemLexValidationHandDataset(Dataset):
    def __init__(
        self, root: Path, selection_manifest: Path,
        label_to_index: dict[str, int], *, cache: bool = True,
    ):
        self.root = Path(root)
        self.label_to_index = dict(label_to_index)
        self.expected_schema = schema_fingerprint(HandMobileCLIP2V17Config())
        items, _ = validation_items(Path(selection_manifest))
        self.files = [
            self.root / item.label / f"{item.item_id}.hand_mobileclip2_v17.npz"
            for item in items
        ]
        self.targets = torch.tensor(
            [self.label_to_index[item.label] for item in items], dtype=torch.long
        )
        missing = [path for path in self.files if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])
        self._cache = [self._load(path) for path in self.files] if cache else None

    def _load(self, path: Path):
        values = HandMobileCLIP2Dataset._load(self, path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        if (
            metadata.get("source") != SOURCE
            or metadata.get("split") != SPLIT
            or metadata.get("training_eligible") is not False
            or metadata.get("test_accessed") is not False
        ):
            raise ValueError(f"SemLex validation feature provenance mismatch: {path}")
        return values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        values = self._cache[index] if self._cache is not None else self._load(self.files[index])
        return (*values, self.targets[index])


def present_class_macro_f1(logits: np.ndarray, targets: np.ndarray) -> float:
    predicted = logits.argmax(1)
    classes = logits.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predicted), 1)
    tp = np.diag(confusion).astype(float)
    precision = tp / np.maximum(confusion.sum(0), 1)
    recall = tp / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = confusion.sum(1) > 0
    return float(100 * f1[present].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/hand_mobileclip2_s0"),
    )
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/selection_plan.json"),
    )
    parser.add_argument(
        "--citizen-manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_hand_mobileclip2_v17":
        raise ValueError("not a v17 hand-MobileCLIP2 checkpoint")
    if checkpoint["manifest_sha256"] != file_sha256(args.citizen_manifest):
        raise ValueError("checkpoint Citizen manifest mismatch")
    expected_schema = schema_fingerprint(HandMobileCLIP2V17Config())
    if checkpoint["schema_fingerprint"] != expected_schema:
        raise ValueError("checkpoint hand embedding schema mismatch")
    dataset = SemLexValidationHandDataset(
        args.data_root, args.selection_manifest, checkpoint["label_to_index"]
    )
    device = select_device(args.device)
    model = HandMobileCLIP2Stage1V17(
        HandMobileCLIP2Stage1Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    logits_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for embeddings, valid, boxes, _ in DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        ):
            logits = model(embeddings.to(device), valid.to(device), boxes.to(device))
            if device.type == "mps":
                torch.mps.synchronize()
            logits_batches.append(logits.float().cpu().numpy())
    logits = np.concatenate(logits_batches)
    targets = dataset.targets.numpy()
    metrics = {
        **metrics_from_logits(logits, targets),
        "macro_f1_present_classes": present_class_macro_f1(logits, targets),
        "classes_present": int(len(np.unique(targets))),
        "split": SPLIT,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_eligible": False,
        "test_evaluated": False,
    }
    item_ids = np.asarray([
        f"semlex_val/{path.parent.name}/{path.name.removesuffix('.hand_mobileclip2_v17.npz')}"
        for path in dataset.files
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "logits.npz", logits=logits.astype(np.float32),
        targets=targets, item_ids=item_ids,
    )
    predicted = logits.argmax(1)
    index_to_label = {value: key for key, value in checkpoint["label_to_index"].items()}
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("feature_path", "true_label", "predicted_label", "correct"))
        for index, path in enumerate(dataset.files):
            writer.writerow((
                str(path.relative_to(args.data_root)),
                index_to_label[int(targets[index])],
                index_to_label[int(predicted[index])],
                bool(predicted[index] == targets[index]),
            ))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
