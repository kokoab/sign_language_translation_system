#!/usr/bin/env python3
"""Probe whether landmark CTC can safely lock glosses before a phrase ends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import coremltools as ct
import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.geometry_v17 import resample_features
from active.v17.train_stage_2_v17 import collapse_ctc


def confidence(logits: np.ndarray, length: int) -> tuple[list[int], float]:
    value = np.asarray(logits).reshape(-1, logits.shape[-1])[:length]
    shifted = value - value.max(-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(-1, keepdims=True)
    path = probabilities.argmax(-1)
    tokens = collapse_ctc(path)
    emissions = []
    previous = -1
    for index, token in enumerate(path.tolist()):
        if token and token != previous:
            emissions.append(float(probabilities[index, token]))
        previous = token
    return tokens, min(emissions, default=0.0)


def probe_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.ALL)
    output_name = model.get_spec().description.output[0].name
    paths = sorted((args.raw_root / "validation").glob("*/*.stage2_rgb_v17.npz"))
    if not paths:
        raise ValueError("no validation rows")
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            windows = payload["landmarks"].astype(np.float32)
            reference = payload["target_indices"].astype(int).tolist()
            metadata = json.loads(str(payload["metadata_json"]))
        probes = []
        completed = []
        seen = 0
        for window_index, window in enumerate(windows):
            for partial in args.partial_frames:
                value = np.zeros((1, 8, 32, 61, 5), np.float32)
                mask = np.zeros((1, 8), np.float32)
                candidate = resample_features(window[:partial], 32).astype(np.float32)
                context = [*completed, candidate]
                value[0, :len(context)] = context
                mask[0, :len(context)] = 1
                logits = np.asarray(model.predict({
                    "landmarks": value, "window_mask": mask,
                })[output_name])
                tokens, score = confidence(logits, len(context) * 8)
                probes.append({
                    "hypothesis": tokens,
                    "confidence": score,
                    "model_frames_seen": window_index * 32 + partial,
                    "is_phrase_end": (
                        window_index == len(windows) - 1 and partial == 32
                    ),
                })
                seen += 1
            completed.append(window)
        rows.append({
            "source": metadata["source"],
            "item_id": metadata["source_item_id"],
            "reference": reference,
            "target_sequence": metadata["target_sequence"],
            "probes": probes,
            "total_model_frames": len(windows) * 32,
        })
    return rows


def simulate(rows, threshold: float, stable_probes: int) -> dict[str, object]:
    locked_tokens = correct_locks = incorrect_locks = 0
    rows_with_early_lock = rows_with_complete_early_lock = 0
    frame_leads = []
    for row in rows:
        locked = []
        previous = None
        repeated = 0
        early = False
        complete_early = False
        for probe in row["probes"]:
            hypothesis = probe["hypothesis"]
            repeated = repeated + 1 if hypothesis == previous else 1
            stable = repeated >= stable_probes and len(hypothesis) > len(locked)
            extends = hypothesis[:len(locked)] == locked
            if stable and extends and probe["confidence"] >= threshold:
                added = hypothesis[len(locked):]
                for index, token in enumerate(added, start=len(locked)):
                    correct = index < len(row["reference"]) and token == row["reference"][index]
                    correct_locks += int(correct)
                    incorrect_locks += int(not correct)
                locked = list(hypothesis)
                locked_tokens += len(added)
                if not probe["is_phrase_end"]:
                    early = True
                    frame_leads.append(
                        row["total_model_frames"] - probe["model_frames_seen"]
                    )
                    if locked == row["reference"]:
                        complete_early = True
            previous = list(hypothesis)
        rows_with_early_lock += int(early)
        rows_with_complete_early_lock += int(complete_early)
    return {
        "threshold": threshold,
        "stable_probes": stable_probes,
        "locked_tokens": locked_tokens,
        "correct_locks": correct_locks,
        "incorrect_locks": incorrect_locks,
        "rows_with_early_lock": rows_with_early_lock,
        "rows_with_complete_sequence_locked_early": rows_with_complete_early_lock,
        "median_early_lead_model_frames": (
            float(np.median(frame_leads)) if frame_leads else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.model, args.raw_root
    )):
        raise ValueError("early-lock selection is restricted to validation data")
    rows = probe_rows(args)
    candidates = [
        simulate(rows, float(threshold), stable_probes)
        for stable_probes in args.stable_probes
        for threshold in np.linspace(0.50, 0.999, 500)
    ]
    zero_error = [row for row in candidates if row["incorrect_locks"] == 0]
    selected = max(
        zero_error,
        key=lambda row: (
            row["locked_tokens"], -row["stable_probes"], -row["threshold"]
        ),
        default=candidates[-1],
    )
    report = {
        "format": "slt_stage2_landmark_early_lock_validation_v17",
        "version": 1,
        "model": args.model.as_posix(),
        "raw_root": args.raw_root.as_posix(),
        "probe_partial_frames": args.partial_frames,
        "rule": "lock only a strict extension repeated for the selected number of consecutive probes and above confidence threshold",
        "selected": selected,
        "validation_rows": len(rows),
        "selection_warning": "threshold selected and measured on the same development validation rows",
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", type=Path, default=Path(
        "artifacts/coreml/Stage2LandmarkCascadePreviewV17FP32.mlpackage"
    ))
    value.add_argument("--raw-root", type=Path, default=Path(
        "data/local/stage2_v17_multimodal"
    ))
    value.add_argument("--partial-frames", type=int, nargs="+", default=(8, 16, 24, 28, 32))
    value.add_argument("--stable-probes", type=int, nargs="+", default=(2, 3, 4))
    value.add_argument("--output", type=Path, default=Path(
        "artifacts/reports/stage2_v17_landmark_cascade_preview_v1/early_lock.json"
    ))
    return value


def main() -> None:
    print(json.dumps(run(parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
