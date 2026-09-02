#!/usr/bin/env python3
"""Validation-only gate sweep for the landmark-first Stage-2 cascade."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from active.v17.model_stage2_v17 import (
    Stage2TemporalHeadV17,
    Stage2V17Config,
    load_frozen_unified_stage1,
    load_stage2_general_ctc_selector,
)
from active.v17.train_stage_2_landmark_cascade_v17 import (
    LandmarkStage2Preview,
    sha256,
)
from active.v17.train_stage_2_v17 import (
    RealPhraseDataset,
    collate,
    collapse_ctc,
    edit_distance,
)


def greedy_with_confidence(logits: torch.Tensor, length: int) -> tuple[list[int], float]:
    probabilities = logits[:length].softmax(-1)
    path = probabilities.argmax(-1).cpu().numpy()
    tokens = collapse_ctc(path)
    emissions = []
    previous = -1
    for index, value in enumerate(path.tolist()):
        if value and value != previous:
            emissions.append(float(probabilities[index, value]))
        previous = value
    return tokens, min(emissions, default=0.0)


def collect_rows(preview, teacher, loader, device) -> list[dict[str, object]]:
    rows = []
    preview.eval()
    teacher.eval()
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["window_mask"].to(device)
            fast_logits, lengths = preview(features, mask)
            full_logits, full_lengths, _ = teacher.forward_with_selection(features, mask)
            offset = 0
            for index, target_length in enumerate(batch["target_lengths"].tolist()):
                reference = [
                    int(value) - 1 for value in
                    batch["targets"][offset:offset + target_length].tolist()
                ]
                offset += target_length
                fast, confidence = greedy_with_confidence(
                    fast_logits[index], int(lengths[index])
                )
                full, _ = greedy_with_confidence(
                    full_logits[index], int(full_lengths[index])
                )
                rows.append({
                    "source": batch["sources"][index],
                    "item_id": batch["item_ids"][index],
                    "target_sequence": batch["target_sequences"][index],
                    "reference": reference,
                    "fast": fast,
                    "full": full,
                    "fast_confidence": confidence,
                    "windows": int(mask[index].sum()),
                })
    return rows


def metrics(rows, threshold: float) -> dict[str, object]:
    by_domain = defaultdict(lambda: {
        "samples": 0, "tokens": 0, "full_edits": 0, "cascade_edits": 0,
        "fast_used": 0, "fast_full_agreement": 0,
    })
    for row in rows:
        values = by_domain[row["source"]]
        use_fast = row["fast_confidence"] >= threshold
        cascade = row["fast"] if use_fast else row["full"]
        values["samples"] += 1
        values["tokens"] += len(row["reference"])
        values["full_edits"] += edit_distance(row["reference"], row["full"])
        values["cascade_edits"] += edit_distance(row["reference"], cascade)
        values["fast_used"] += int(use_fast)
        values["fast_full_agreement"] += int(use_fast and row["fast"] == row["full"])
    return {
        domain: {
            **values,
            "coverage": values["fast_used"] / values["samples"],
            "agreement_when_fast": (
                values["fast_full_agreement"] / values["fast_used"]
                if values["fast_used"] else None
            ),
        }
        for domain, values in sorted(by_domain.items())
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.preview, args.teacher, args.phrase_root, args.context_validation_root
    )):
        raise ValueError("cascade selection is restricted to validation data")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    payload = torch.load(args.preview, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_landmark_preview_v17":
        raise ValueError("not a landmark Stage-2 preview checkpoint")
    model = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    landmark, _, _, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    preview = LandmarkStage2Preview(model, landmark.classifier).to(device).eval()
    teacher, _ = load_stage2_general_ctc_selector(args.teacher)
    teacher.to(device).eval()
    loaders = (
        DataLoader(
            RealPhraseDataset(args.phrase_root, "validation"),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        ),
        DataLoader(
            RealPhraseDataset(args.context_validation_root, "validation"),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        ),
    )
    rows = []
    for loader in loaders:
        rows.extend(collect_rows(preview, teacher, loader, device))
    candidates = []
    for threshold in np.linspace(0.50, 0.999, 500):
        result = metrics(rows, float(threshold))
        safe = all(
            value["cascade_edits"] <= value["full_edits"]
            for value in result.values()
        )
        candidates.append({
            "threshold": float(threshold),
            "safe": safe,
            "fast_used": sum(value["fast_used"] for value in result.values()),
            "domains": result,
        })
    safe = [row for row in candidates if row["safe"]]
    selected = max(
        safe,
        key=lambda row: (row["fast_used"], -row["threshold"]),
        default=candidates[-1],
    )
    report = {
        "format": "slt_stage2_landmark_cascade_validation_v17",
        "version": 1,
        "preview": args.preview.as_posix(),
        "preview_sha256": sha256(args.preview),
        "teacher": args.teacher.as_posix(),
        "teacher_sha256": sha256(args.teacher),
        "gate": "minimum greedy nonblank emission probability",
        "selected": selected,
        "validation_rows": len(rows),
        "candidate_thresholds": len(candidates),
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--preview", type=Path, default=Path(
        "artifacts/models/stage2_v17_landmark_cascade_preview_v1/best_model.pth"
    ))
    value.add_argument("--teacher", type=Path, default=Path(
        "artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"
    ))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path(
        "artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"
    ))
    value.add_argument("--phrase-root", type=Path, default=Path(
        "data/local/stage2_v17_frozen_features"
    ))
    value.add_argument("--context-validation-root", type=Path, default=Path(
        "data/local/stage2_v17_asllrp_segmented_validation_frozen_features"
    ))
    value.add_argument("--output", type=Path, default=Path(
        "artifacts/reports/stage2_v17_landmark_cascade_preview_v1/validation.json"
    ))
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--device", default="auto")
    return value


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
