#!/usr/bin/env python3
"""Evaluate a hand-RGB checkpoint as a local-corpus screening model."""

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
from active.v17.local_multimodal_audit_v17 import SOURCE, SPLIT, local_audit_items
from active.v17.model_hand_mobileclip2_v17 import (
    HandMobileCLIP2Stage1Config, HandMobileCLIP2Stage1V17,
)
from active.v17.schema_hand_mobileclip2_v17 import HandMobileCLIP2V17Config, schema_fingerprint
from active.v17.train_stage_1_hand_mobileclip2_v17 import HandMobileCLIP2Dataset
from active.v17.train_stage_1_mobileclip2_v17 import select_device


class LocalAuditHandDataset(Dataset):
    def __init__(self, root, manifest, label_to_index):
        self.root = Path(root)
        self.label_to_index = dict(label_to_index)
        self.expected_schema = schema_fingerprint(HandMobileCLIP2V17Config())
        items, _ = local_audit_items(Path(manifest))
        self.files = [
            self.root / item.label / f"{item.item_id}.hand_mobileclip2_v17.npz"
            for item in items
        ]
        self.targets = torch.tensor(
            [self.label_to_index[item.label] for item in items], dtype=torch.long
        )
        self._cache = [self._load(path) for path in self.files]

    def _load(self, path):
        values = HandMobileCLIP2Dataset._load(self, path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        if (
            metadata.get("source") != SOURCE or metadata.get("split") != SPLIT
            or metadata.get("training_eligible") is not False
            or metadata.get("test_accessed") is not False
        ):
            raise ValueError(f"local hand feature provenance mismatch: {path}")
        return values

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return (*self._cache[index], self.targets[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/local/local_citizen100_quality_audit_q82_cap14_exact/hand_mobileclip2_s0"),
    )
    parser.add_argument(
        "--selection-manifest", type=Path,
        default=Path("data/local/local_citizen100_quality_audit_q82_cap14_exact/candidate_selection.json"),
    )
    parser.add_argument(
        "--citizen-manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_hand_mobileclip2_v17":
        raise ValueError("not a v17 hand-MobileCLIP2 checkpoint")
    if checkpoint["manifest_sha256"] != file_sha256(args.citizen_manifest):
        raise ValueError("checkpoint manifest mismatch")
    dataset = LocalAuditHandDataset(
        args.data_root, args.selection_manifest, checkpoint["label_to_index"]
    )
    model = HandMobileCLIP2Stage1V17(
        HandMobileCLIP2Stage1Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = select_device(args.device)
    model.to(device).eval()
    batches = []
    with torch.inference_mode():
        for embeddings, valid, boxes, _ in DataLoader(dataset, batch_size=args.batch_size):
            batches.append(
                model(embeddings.to(device), valid.to(device), boxes.to(device))
                .float().cpu().numpy()
            )
    logits = np.concatenate(batches)
    targets = dataset.targets.numpy()
    metrics = metrics_from_logits(logits, targets)
    metrics.update({
        "purpose": "local-label screening; not accuracy", "split": SPLIT,
        "training_eligible": False, "test_evaluated": False,
    })
    ids = np.asarray([
        f"local_audit/{path.parent.name}/{path.name.removesuffix('.hand_mobileclip2_v17.npz')}"
        for path in dataset.files
    ])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "logits.npz", logits=logits.astype(np.float32),
        targets=targets, item_ids=ids,
    )
    predicted = logits.argmax(1)
    inverse = {value: key for key, value in checkpoint["label_to_index"].items()}
    with (args.output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("item_id", "true_label", "predicted_label", "correct"))
        for index, item in enumerate(ids):
            writer.writerow((
                item, inverse[int(targets[index])], inverse[int(predicted[index])],
                bool(predicted[index] == targets[index]),
            ))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
