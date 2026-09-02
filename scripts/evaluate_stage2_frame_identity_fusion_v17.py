#!/usr/bin/env python3
"""Evaluate train-selected frame-level Stage-1 identity fusion for Stage-2 CTC."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.model_stage2_v17 import Stage2TemporalHeadV17
from active.v17.train_stage_2_v17 import collapse_ctc, edit_distance


def load_head(path: Path) -> Stage2TemporalHeadV17:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("identity fusion requires the bare Stage-2 head")
    model = Stage2TemporalHeadV17()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def zscore(value: np.ndarray) -> np.ndarray:
    return (value - value.mean(-1, keepdims=True)) / np.maximum(
        value.std(-1, keepdims=True), 1e-6
    )


def collect(role: str, root: Path, model: Stage2TemporalHeadV17, device) -> list[dict]:
    rows = []
    with torch.inference_mode():
        for path in sorted((root / role).glob("*/*.stage2_frozen_v17.npz")):
            with np.load(path, allow_pickle=False) as payload:
                features = payload["frozen_features"].astype(np.float32)
                reference = payload["target_indices"].astype(np.int64).tolist()
                metadata = json.loads(str(payload["metadata_json"]))
            windows = len(features)
            padded = np.zeros((1, 8, 32, features.shape[-1]), np.float32)
            mask = np.zeros((1, 8), np.bool_)
            padded[0, :windows] = features
            mask[0, :windows] = True
            logits, _ = model(
                torch.from_numpy(padded).to(device), torch.from_numpy(mask).to(device)
            )
            logits = logits[0, : windows * 8].cpu().numpy()
            frame_identity = features[..., -100:].reshape(windows, 8, 4, 100).mean(2)
            rows.append({
                "item_id": metadata["source_item_id"],
                "source": metadata["source"],
                "reference": reference,
                "ctc": logits,
                "identity": frame_identity.reshape(windows * 8, 100),
            })
    return rows


def decode(row: dict, config: dict) -> list[int]:
    logits = row["ctc"].copy()
    logits[:, 1:] += zscore(row["identity"]) * config["identity_weight"]
    logits[:, 0] += config["blank_bias"]
    return collapse_ctc(logits.argmax(-1))


def metrics(rows: list[dict], predictions: list[list[int]]) -> dict:
    domains = defaultdict(lambda: {"edits": 0, "tokens": 0, "exact": 0, "samples": 0})
    for row, prediction in zip(rows, predictions):
        value = domains[row["source"]]
        value["edits"] += edit_distance(row["reference"], prediction)
        value["tokens"] += len(row["reference"])
        value["exact"] += int(row["reference"] == prediction)
        value["samples"] += 1
    result = {}
    for source, value in sorted(domains.items()):
        result[source] = {
            **value,
            "wer": value["edits"] / max(1, value["tokens"]),
            "sequence_accuracy": value["exact"] / max(1, value["samples"]),
        }
    return {
        "domains": result,
        "equal_domain_mean_wer": float(np.mean([x["wer"] for x in result.values()])),
        "equal_domain_mean_sequence_accuracy": float(np.mean([
            x["sequence_accuracy"] for x in result.values()
        ])),
    }


def run(args: argparse.Namespace) -> dict:
    if "test" in {part.lower() for part in args.frozen_root.parts}:
        raise ValueError("test data is forbidden")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    head = load_head(args.phrase_checkpoint).to(device)
    training = collect("train", args.frozen_root, head, device)
    validation = collect("validation", args.frozen_root, head, device)
    candidates = []
    for identity_weight in np.arange(0.0, 2.01, 0.1):
        for blank_bias in np.arange(-1.0, 1.01, 0.1):
                config = {
                    "identity_weight": float(identity_weight),
                    "blank_bias": float(blank_bias),
                }
                predictions = [decode(row, config) for row in training]
                score = metrics(training, predictions)
                candidates.append((
                    score["equal_domain_mean_wer"],
                    -score["equal_domain_mean_sequence_accuracy"],
                    identity_weight,
                    abs(blank_bias),
                    config,
                    score,
                ))
    *_, selected, training_score = min(candidates, key=lambda x: x[:4])
    baseline_config = {"identity_weight": 0.0, "blank_bias": 0.0}
    baseline_predictions = [decode(row, baseline_config) for row in validation]
    routed_predictions = [decode(row, selected) for row in validation]
    report = {
        "format": "slt_stage2_frame_identity_fusion_evaluation_v17",
        "version": 1,
        "selection_role": "train",
        "evaluation_role": "validation",
        "selected_config": selected,
        "training_metrics": training_score,
        "validation_baseline_metrics": metrics(validation, baseline_predictions),
        "validation_fused_metrics": metrics(validation, routed_predictions),
        "validation_rows": [{
            "item_id": row["item_id"], "source": row["source"],
            "reference": row["reference"], "baseline": baseline,
            "fused": fused,
        } for row, baseline, fused in zip(
            validation, baseline_predictions, routed_predictions
        )],
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
    value.add_argument("--frozen-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    value.add_argument("--phrase-checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"))
    value.add_argument("--device", default="auto")
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_frame_identity_fusion_v1/validation.json"))
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({
        "selected": result["selected_config"],
        "baseline": result["validation_baseline_metrics"],
        "fused": result["validation_fused_metrics"],
    }, indent=2))
