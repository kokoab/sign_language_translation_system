#!/usr/bin/env python3
"""Measure exact annotated-core classification before window emission/segmentation."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from active.v17 import train_stage1_window_v17 as training
from active.v17.model_reel_emission_v17 import pool_stage1_encoded
from active.v17.stage1_window_v17 import load_stage1_window_checkpoint

REPORT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts/reports/o5s5_augmented_v17_20260914/combined_supervision.json"
BASE = ROOT / "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"


@torch.inference_mode()
def probe(model, samples, device):
    model = model.to(device).eval()
    rows = []
    for start in range(0, len(samples), 128):
        batch = samples[start:start + 128]
        features = torch.from_numpy(np.stack([row.features for row in batch])).to(device)
        foreground = torch.from_numpy(np.stack([row.foreground for row in batch])).to(device)
        encoded, active = model.encode(features)
        logits = model.classifier(pool_stage1_encoded(model, encoded, foreground & active)).cpu()
        for row, prediction, top5 in zip(batch, logits.argmax(1), logits.topk(5, 1).indices):
            rows.append((row, int(prediction), row.target in top5.tolist()))
    model.cpu()
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row, prediction, top5 in rows:
        groups[f"{row.source}/{row.signer}"].append((row, prediction, top5))
        groups[row.source].append((row, prediction, top5))
        groups["all"].append((row, prediction, top5))
    return {name: {
        "samples": len(group),
        "top1_accuracy": sum(row.target == prediction for row, prediction, _ in group) / len(group),
        "top5_accuracy": sum(top5 for _, _, top5 in group) / len(group),
    } for name, group in sorted(groups.items())}


def main():
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    payload = torch.load(BASE, map_location="cpu", weights_only=False)
    labels = {str(k): int(v) for k, v in payload["label_to_index"].items() if int(v) < 100}
    split = {}
    for role in ("train", "validation"):
        samples, _ = training.load_context_samples(MANIFEST, labels, role)
        split[role] = [row for row in samples if row.category == "context"]
    models = {"original_base": training._load_start(payload).base}
    for epoch in (1, 3, 12):
        checkpoint = ROOT / f"artifacts/models/stage1_window_o5s5_v17_seed17111/epoch_{epoch:02d}.pth"
        models[f"o5s5_epoch_{epoch:02d}"] = load_stage1_window_checkpoint(checkpoint)[0].base
    output = {}
    for name, model in models.items():
        output[name] = {role: summarize(probe(model, samples, device))
                        for role, samples in split.items()}
    assert output["original_base"]["train"]["all"]["samples"] == 6819
    assert output["original_base"]["validation"]["all"]["samples"] == 1649
    (REPORT / "foreground_core_probe.json").write_text(json.dumps({
        "metric": "classification after pooling only exact annotated foreground frames from the full encoded window",
        "models": output, "device": str(device), "protected_test_accessed": False,
        "limitations": "Uses boundary annotations at evaluation time; diagnostic only, not deployable inference.",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
