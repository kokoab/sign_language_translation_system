#!/usr/bin/env python3
"""Evaluate frozen visual-speech heads on SemLex validation features."""

from __future__ import annotations

import argparse
import csv
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

from active.v17.evaluate_stage_1_hand_mobileclip2_semlex_v17 import present_class_macro_f1
from active.v17.evaluate_stage_1_mobileclip2_v17 import file_sha256, metrics_from_logits
from active.v17.extract_hand_rgb_semlex_val_v17 import SPLIT
from active.v17.model_visual_speech_v17 import (
    MultiViewVisualSpeechHeadV17, MultiViewVisualSpeechV17Config,
    VisualSpeechTeacherV17, VisualSpeechTeacherV17Config,
)
from active.v17.train_stage_1_v17 import select_device
from active.v17.train_stage_1_visual_speech_features_v17 import FrozenVisualFeatureDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--expected-split", default=SPLIT)
    parser.add_argument(
        "--citizen-manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("manifest_sha256") != file_sha256(args.citizen_manifest):
        raise ValueError("visual-speech checkpoint manifest mismatch")
    datasets = [FrozenVisualFeatureDataset(path, args.expected_split) for path in args.cache]
    for dataset in datasets:
        if dataset.metadata.get("training_eligible") is not False:
            raise ValueError("SemLex visual-speech cache is not evaluation-only")
    reference = datasets[0]
    if checkpoint.get("format") == "slt_stage1_visual_speech_v17":
        if len(datasets) != 1:
            raise ValueError("single-view checkpoint requires one cache")
        expected_view = checkpoint["training_data_provenance"]["view"]
        if reference.metadata.get("view") != expected_view:
            raise ValueError("visual-speech checkpoint/cache view mismatch")
        model = VisualSpeechTeacherV17(
            VisualSpeechTeacherV17Config(**checkpoint["model_config"])
        )
        features, valid = reference.features, reference.valid
    elif checkpoint.get("format") == "slt_stage1_visual_speech_multiview_v17":
        if len(datasets) != 2:
            raise ValueError("multi-view checkpoint requires mouth and lower-face caches")
        if tuple(dataset.metadata.get("view") for dataset in datasets) != ("mouth", "lower_face"):
            raise ValueError("multi-view caches must be mouth then lower_face")
        for dataset in datasets[1:]:
            if not np.array_equal(dataset.item_ids, reference.item_ids) or not torch.equal(dataset.targets, reference.targets):
                raise ValueError("multi-view SemLex caches are not aligned")
        model = MultiViewVisualSpeechHeadV17(
            MultiViewVisualSpeechV17Config(**checkpoint["model_config"])
        )
        features = torch.stack([dataset.features for dataset in datasets], dim=2)
        valid = torch.stack([dataset.valid for dataset in datasets], dim=2)
    else:
        raise ValueError("unsupported visual-speech checkpoint format")
    model.load_state_dict(checkpoint["model_state_dict"])
    device = select_device(args.device)
    model.to(device).eval()
    logits_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            batch_features = features[start:start + args.batch_size].to(device)
            batch_valid = valid[start:start + args.batch_size].to(device)
            logits = (
                model.forward_features(batch_features, batch_valid)
                if checkpoint["format"] == "slt_stage1_visual_speech_v17"
                else model(batch_features, batch_valid)
            )
            logits_batches.append(logits.float().cpu().numpy())
    logits = np.concatenate(logits_batches)
    targets = reference.targets.numpy()
    metrics = {
        **metrics_from_logits(logits, targets),
        "macro_f1_present_classes": present_class_macro_f1(logits, targets),
        "classes_present": int(len(np.unique(targets))),
        "split": args.expected_split,
        "view": reference.metadata.get("view") if len(datasets) == 1 else "mouth_lower_face",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_eligible": False,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "logits.npz", logits=logits.astype(np.float32),
        targets=targets, item_ids=reference.item_ids,
    )
    citizen = json.loads(args.citizen_manifest.read_text(encoding="utf-8"))
    index_to_label = {
        int(row["class_index"]): str(row["canonical_label"])
        for row in citizen["classes"]
    }
    predicted = logits.argmax(1)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("item_id", "true_label", "predicted_label", "correct"))
        for index, item_id in enumerate(reference.item_ids):
            writer.writerow((
                item_id, index_to_label[int(targets[index])],
                index_to_label[int(predicted[index])], bool(predicted[index] == targets[index]),
            ))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
