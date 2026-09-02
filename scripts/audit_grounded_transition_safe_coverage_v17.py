#!/usr/bin/env python3
"""Audit exact-label Citizen coverage after transition-edge preparation."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.signing_voice_phrase_v17 import trim_transition_span
from active.v17.train_signing_voice_v17 import sha256
from scripts.generate_grounded_text_to_sign_v17 import (
    landmark_path,
    load_isolated,
    load_stage1,
)


@torch.inference_mode()
def classify_grouped(model, labels, values, device, batch_size):
    predictions = [None] * len(values)
    grouped = defaultdict(list)
    for index, value in enumerate(values):
        grouped[len(value)].append(index)
    model.to(device)
    for indices in grouped.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            batch = torch.from_numpy(np.stack([values[index] for index in selected])).to(device)
            probability = model(batch).softmax(dim=-1)
            confidence, prediction = probability.max(dim=1)
            for index, label_index, score in zip(
                selected, prediction.cpu().tolist(), confidence.cpu().tolist()
            ):
                predictions[index] = (labels[int(label_index)], float(score))
    return predictions


def run(args):
    rows = []
    original = []
    prepared = []
    with args.provenance.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "train":
                continue
            value = load_isolated(landmark_path(args.isolated_root, row))
            rows.append(row)
            original.append(value)
            prepared.append(trim_transition_span(value))

    model, labels = load_stage1(args.stage1_checkpoint)
    device = torch.device(args.device)
    original_predictions = classify_grouped(
        model, labels, original, device, args.batch_size
    )
    prepared_predictions = classify_grouped(
        model, labels, prepared, device, args.batch_size
    )

    safe_by_signer = defaultdict(set)
    original_correct = prepared_correct = safe_clips = 0
    for row, source_prediction, prepared_prediction in zip(
        rows, original_predictions, prepared_predictions
    ):
        gloss = row["canonical_label"]
        source_ok = source_prediction[0] == gloss
        prepared_ok = prepared_prediction[0] == gloss
        original_correct += source_ok
        prepared_correct += prepared_ok
        safe_clips += source_ok and prepared_ok
        if source_ok and prepared_ok:
            safe_by_signer[row["participant"]].add(gloss)

    glosses = sorted({row["canonical_label"] for row in rows})
    safe_signers = {
        gloss: sorted(
            signer for signer, values in safe_by_signer.items() if gloss in values
        )
        for gloss in glosses
    }
    pair_count = 0
    triple_count = 0
    for first_index, first in enumerate(glosses):
        first_signers = set(safe_signers[first])
        for second_index in range(first_index + 1, len(glosses)):
            second = glosses[second_index]
            shared = first_signers & set(safe_signers[second])
            pair_count += bool(shared)
            for third in glosses[second_index + 1:]:
                triple_count += bool(shared & set(safe_signers[third]))

    counts = [len(safe_signers[gloss]) for gloss in glosses]
    report = {
        "format": "slt_grounded_transition_safe_coverage_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage1_checkpoint": args.stage1_checkpoint.as_posix(),
        "stage1_checkpoint_sha256": sha256(args.stage1_checkpoint),
        "train_clips": len(rows),
        "original_exact_clips": original_correct,
        "transition_prepared_exact_clips": prepared_correct,
        "exact_before_and_after_clips": safe_clips,
        "classes_with_safe_source": sum(bool(value) for value in safe_signers.values()),
        "classes_without_safe_source": [
            gloss for gloss, signers in safe_signers.items() if not signers
        ],
        "safe_signers_per_class_min": min(counts),
        "safe_signers_per_class_median": float(np.median(counts)),
        "safe_same_signer_unordered_pairs": pair_count,
        "all_unordered_pairs": len(glosses) * (len(glosses) - 1) // 2,
        "safe_same_signer_unordered_triples": triple_count,
        "all_unordered_triples": (
            len(glosses) * (len(glosses) - 1) * (len(glosses) - 2) // 6
        ),
        "safe_glosses_by_signer": {
            signer: len(values) for signer, values in sorted(safe_by_signer.items())
        },
        "class_safe_signer_counts": {
            gloss: len(safe_signers[gloss]) for gloss in glosses
        },
        "test_evaluated": False,
        "citizen_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--provenance", type=Path,
        default=Path("data/local/citizen100_v17/provenance.csv"),
    )
    value.add_argument(
        "--isolated-root", type=Path,
        default=Path("data/local/citizen100_v17/landmarks"),
    )
    value.add_argument(
        "--stage1-checkpoint", type=Path,
        default=Path("artifacts/models/stage1_v17_baseline/best_model.pth"),
    )
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    value.add_argument(
        "--output", type=Path,
        default=Path(
            "artifacts/reports/stage2_v17_grounded_transition_safe_coverage_v1/report.json"
        ),
    )
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({
        key: result[key] for key in (
            "train_clips", "exact_before_and_after_clips",
            "classes_with_safe_source", "safe_same_signer_unordered_pairs",
            "safe_same_signer_unordered_triples",
        )
    }, indent=2))
