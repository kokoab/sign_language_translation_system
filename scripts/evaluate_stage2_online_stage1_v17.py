#!/usr/bin/env python3
"""Evaluate Stage 1 as a sliding-window Stage-2 recognizer on non-test caches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.model_stage2_v17 import (
    Stage2TemporalHeadV17,
    Stage2V17Config,
    load_frozen_unified_stage1,
    load_stage2_isolated_correction,
)
from active.v17.model_unified_multimodal_v17 import UnifiedMultimodalStage1V17
from active.v17.train_stage_2_v17 import collapse_ctc, edit_distance


def paired_path(root: Path, role: str, relative: Path, suffix: str) -> Path:
    stem = relative.name.removesuffix(".stage2_rgb_v17.npz")
    return root / role / relative.parent / f"{stem}.{suffix}.npz"


def load_head(path: Path) -> Stage2TemporalHeadV17:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage2_ctc_v17":
        raise ValueError(f"{path}: expected a bare Stage-2 CTC checkpoint")
    model = Stage2TemporalHeadV17(Stage2V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def activity(landmarks: np.ndarray) -> np.ndarray:
    presence = landmarks[:, :, :42, 3].mean(axis=(1, 2))
    xy = landmarks[:, :, :42, :2]
    valid = landmarks[:, :, :42, 3] > 0.5
    pair = valid[:, 1:] & valid[:, :-1]
    motion = np.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)
    motion = (motion * pair).sum(axis=(1, 2)) / np.maximum(pair.sum(axis=(1, 2)), 1)
    return presence + motion


def decode_online(
    logits: np.ndarray,
    blank_probability: np.ndarray,
    activity_score: np.ndarray,
    *,
    confidence_threshold: float,
    margin_threshold: float,
    blank_threshold: float,
    activity_fraction: float,
) -> list[int]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    order = np.argsort(probabilities, axis=-1)
    best = order[:, -1]
    confidence = probabilities[np.arange(len(best)), best]
    margin = confidence - probabilities[np.arange(len(best)), order[:, -2]]
    active_gate = activity_score >= max(float(activity_score.max()) * activity_fraction, 1e-6)
    foreground = (
        active_gate
        & (confidence >= confidence_threshold)
        & (margin >= margin_threshold)
        & (blank_probability <= blank_threshold)
    )
    output: list[int] = []
    previous: int | None = None
    for label, keep in zip(best.tolist(), foreground.tolist()):
        if not keep:
            previous = None
        elif label != previous:
            output.append(int(label))
            previous = int(label)
    if not output:
        score = confidence * (1.0 - blank_probability) * active_gate
        output = [int(best[int(score.argmax())])]
    return output


def collect(
    role: str,
    args: argparse.Namespace,
    stage1: UnifiedMultimodalStage1V17,
    correction,
    phrase_head: Stage2TemporalHeadV17,
    device: torch.device,
) -> list[dict[str, object]]:
    paths = sorted((args.rgb_root / role).glob("*/*.stage2_rgb_v17.npz"))
    rows = []
    with torch.inference_mode():
        for path in paths:
            relative = path.relative_to(args.rgb_root / role)
            hand_path = paired_path(
                args.hand_root, role, relative, "stage2_hand_mobileclip2_v17"
            )
            frozen_path = paired_path(
                args.frozen_root, role, relative, "stage2_frozen_v17"
            )
            with np.load(path, allow_pickle=False) as payload:
                landmarks = payload["landmarks"].astype(np.float32)
                reference_full = payload["target_indices"].astype(np.int64).tolist()
                reference = [value for value in reference_full if value != 100]
                metadata = json.loads(str(payload["metadata_json"]))
            with np.load(hand_path, allow_pickle=False) as payload:
                embeddings = payload["embeddings"].astype(np.float32)
                valid = payload["valid"].astype(np.float32)
                boxes = payload["boxes_normalized"].astype(np.float32)
            with np.load(frozen_path, allow_pickle=False) as payload:
                frozen = payload["frozen_features"].astype(np.float32)
            windows = len(landmarks)
            stage1_logits = stage1(
                torch.from_numpy(landmarks).to(device),
                torch.from_numpy(embeddings).to(device),
                torch.from_numpy(valid > 0.5).to(device),
                torch.from_numpy(boxes).to(device),
            )
            hybrid_logits = correction(
                torch.from_numpy(frozen).to(device), stage1_logits
            ).cpu().numpy()
            padded = np.zeros((1, 8, 32, frozen.shape[-1]), np.float32)
            mask = np.zeros((1, 8), np.bool_)
            padded[0, :windows] = frozen
            mask[0, :windows] = True
            phrase_logits, _ = phrase_head(
                torch.from_numpy(padded).to(device), torch.from_numpy(mask).to(device)
            )
            phrase_probability = phrase_logits[0, : windows * 8].softmax(-1).cpu().numpy()
            blank_probability = phrase_probability[:, 0].reshape(windows, 8).mean(axis=1)
            baseline = [
                value for value in collapse_ctc(phrase_probability.argmax(-1))
                if value != 100
            ]
            rows.append({
                "item_id": metadata["source_item_id"],
                "source": metadata["source"],
                "reference": reference,
                "reference_full": reference_full,
                "stage1_logits": stage1_logits.cpu().numpy(),
                "hybrid_logits": hybrid_logits,
                "blank_probability": blank_probability,
                "activity": activity(landmarks),
                "phrase_baseline": baseline,
            })
    return rows


def metrics(rows: list[dict[str, object]], predictions: list[list[int]]) -> dict[str, object]:
    domains = defaultdict(lambda: {"edits": 0, "tokens": 0, "exact": 0, "samples": 0})
    for row, hypothesis in zip(rows, predictions):
        reference = row["reference"]
        value = domains[row["source"]]
        value["edits"] += edit_distance(reference, hypothesis)
        value["tokens"] += len(reference)
        value["exact"] += int(reference == hypothesis)
        value["samples"] += 1
    result = {
        source: {
            **value,
            "wer": value["edits"] / max(1, value["tokens"]),
            "sequence_accuracy": value["exact"] / max(1, value["samples"]),
        }
        for source, value in sorted(domains.items())
    }
    return {
        "domains": result,
        "equal_domain_mean_wer": float(np.mean([value["wer"] for value in result.values()])),
        "equal_domain_mean_sequence_accuracy": float(np.mean([
            value["sequence_accuracy"] for value in result.values()
        ])),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    guarded = (args.rgb_root, args.hand_root, args.frozen_root)
    if any("test" in {part.lower() for part in path.parts} for path in guarded):
        raise ValueError("online Stage-2 evaluation is restricted to train/validation data")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    landmark, hand, fusion, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    stage1 = UnifiedMultimodalStage1V17(landmark, hand, fusion).to(device).eval()
    correction, _ = load_stage2_isolated_correction(args.correction_checkpoint)
    correction = correction.to(device).eval()
    phrase_head = load_head(args.phrase_checkpoint).to(device).eval()
    training = collect("train", args, stage1, correction, phrase_head, device)
    validation = collect("validation", args, stage1, correction, phrase_head, device)

    candidates = []
    for logits_name in ("stage1_logits", "hybrid_logits"):
        for confidence in (0.0, 0.2, 0.3, 0.4, 0.5):
            for margin in (0.0, 0.05, 0.1, 0.2):
                for blank in (0.4, 0.6, 0.8, 1.0):
                    for activity_fraction in (0.0, 0.1, 0.25, 0.4):
                        config = {
                            "logits": logits_name,
                            "confidence_threshold": confidence,
                            "margin_threshold": margin,
                            "blank_threshold": blank,
                            "activity_fraction": activity_fraction,
                        }
                        predictions = [
                            decode_online(
                                row[logits_name], row["blank_probability"], row["activity"],
                                confidence_threshold=confidence,
                                margin_threshold=margin,
                                blank_threshold=blank,
                                activity_fraction=activity_fraction,
                            )
                            for row in training
                        ]
                        score = metrics(training, predictions)
                        candidates.append((
                            score["equal_domain_mean_wer"],
                            -score["equal_domain_mean_sequence_accuracy"],
                            config,
                            score,
                        ))
    _, _, selected, train_metrics = min(candidates, key=lambda value: value[:2])
    validation_predictions = [
        decode_online(
            row[selected["logits"]], row["blank_probability"], row["activity"],
            confidence_threshold=selected["confidence_threshold"],
            margin_threshold=selected["margin_threshold"],
            blank_threshold=selected["blank_threshold"],
            activity_fraction=selected["activity_fraction"],
        )
        for row in validation
    ]
    baseline_metrics = metrics(
        validation, [row["phrase_baseline"] for row in validation]
    )
    online_metrics = metrics(validation, validation_predictions)
    report = {
        "format": "slt_stage2_online_stage1_evaluation_v17",
        "version": 1,
        "selection_role": "train",
        "evaluation_role": "validation",
        "selected_decoder": selected,
        "training_metrics": train_metrics,
        "phrase_ctc_validation_metrics": baseline_metrics,
        "online_stage1_validation_metrics": online_metrics,
        "validation_rows": [
            {
                "item_id": row["item_id"],
                "source": row["source"],
                "reference": row["reference"],
                "phrase_ctc": row["phrase_baseline"],
                "online_stage1": prediction,
            }
            for row, prediction in zip(validation, validation_predictions)
        ],
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
    value.add_argument("--rgb-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    value.add_argument("--hand-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    value.add_argument("--frozen-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    value.add_argument("--correction-checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_isolated_correction_v1/model.pth"))
    value.add_argument("--phrase-checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"))
    value.add_argument("--device", default="auto")
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_online_stage1/nonoverlap_validation.json"))
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
