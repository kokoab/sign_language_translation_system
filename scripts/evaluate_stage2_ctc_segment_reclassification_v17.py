#!/usr/bin/env python3
"""Evaluate CTC-boundary / isolated-identity routing on non-test Stage-2 data."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

from active.v17.model_stage2_v17 import (
    FrozenUnifiedTemporalEncoderV17,
    Stage2TemporalHeadV17,
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
        raise ValueError("segment reclassification requires the bare Stage-2 head")
    model = Stage2TemporalHeadV17()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def emitted_tokens(path: np.ndarray) -> list[tuple[int, int]]:
    """Return CTC-collapsed one-based token and first emission time."""
    output: list[tuple[int, int]] = []
    previous = -1
    for time, token in enumerate(path.tolist()):
        if token and token != previous:
            output.append((int(token), time))
        previous = int(token)
    return output


def temporal_resample(value: np.ndarray, frames: int, *, nearest: bool = False) -> np.ndarray:
    if len(value) == frames:
        return value.copy()
    if len(value) == 1:
        return np.repeat(value, frames, axis=0)
    positions = np.linspace(0, len(value) - 1, frames)
    if nearest:
        return value[np.rint(positions).astype(np.int64)]
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, len(value) - 1)
    weight = (positions - left).reshape((-1,) + (1,) * (value.ndim - 1))
    return value[left] * (1.0 - weight) + value[right] * weight


def segment_ranges(emissions: list[tuple[int, int]], token_steps: int) -> list[tuple[int, int]]:
    if not emissions:
        return []
    centers = [time + 0.5 for _, time in emissions]
    boundaries = [0.0]
    boundaries.extend((left + right) / 2.0 for left, right in zip(centers, centers[1:]))
    boundaries.append(float(token_steps))
    output = []
    for left, right in zip(boundaries, boundaries[1:]):
        start = max(0, min(token_steps - 1, int(np.floor(left))))
        end = max(start + 1, min(token_steps, int(np.ceil(right))))
        output.append((start, end))
    return output


def slice_and_resample(
    value: np.ndarray, token_range: tuple[int, int], token_steps: int, frames: int,
    *, nearest: bool = False,
) -> np.ndarray:
    start = int(np.floor(token_range[0] / token_steps * len(value)))
    end = int(np.ceil(token_range[1] / token_steps * len(value)))
    start = max(0, min(len(value) - 1, start))
    end = max(start + 1, min(len(value), end))
    return temporal_resample(value[start:end], frames, nearest=nearest).astype(np.float32)


def softmax_confidence(logits: np.ndarray, index: int | None = None) -> tuple[float, float]:
    shifted = logits - logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    ordered = np.sort(probabilities)
    chosen = int(probabilities.argmax()) if index is None else index
    return float(probabilities[chosen]), float(ordered[-1] - ordered[-2])


def collect(role: str, args: argparse.Namespace, models: dict[str, object], device) -> list[dict]:
    rows = []
    rgb_paths = sorted((args.rgb_root / role).glob("*/*.stage2_rgb_v17.npz"))
    with torch.inference_mode():
        for rgb_path in rgb_paths:
            relative = rgb_path.relative_to(args.rgb_root / role)
            hand_path = paired_path(args.hand_root, role, relative, "stage2_hand_mobileclip2_v17")
            frozen_path = paired_path(args.frozen_root, role, relative, "stage2_frozen_v17")
            with np.load(rgb_path, allow_pickle=False) as payload:
                landmarks = payload["landmarks"].astype(np.float32)
                reference = payload["target_indices"].astype(np.int64).tolist()
                metadata = json.loads(str(payload["metadata_json"]))
            with np.load(hand_path, allow_pickle=False) as payload:
                embeddings = payload["embeddings"].astype(np.float32)
                valid = payload["valid"].astype(np.float32)
                boxes = payload["boxes_normalized"].astype(np.float32)
            with np.load(frozen_path, allow_pickle=False) as payload:
                frozen = payload["frozen_features"].astype(np.float32)
            windows = len(landmarks)
            padded = np.zeros((1, 8, 32, frozen.shape[-1]), np.float32)
            mask = np.zeros((1, 8), np.bool_)
            padded[0, :windows] = frozen
            mask[0, :windows] = True
            logits, _ = models["head"](
                torch.from_numpy(padded).to(device), torch.from_numpy(mask).to(device)
            )
            logits = logits[0, : windows * 8].cpu().numpy()
            path = logits.argmax(-1)
            emissions = emitted_tokens(path)
            baseline = collapse_ctc(path)
            replacements = []
            ranges = segment_ranges(emissions, len(path))
            flat_landmarks = landmarks.reshape(windows * 32, 61, 5)
            flat_embeddings = embeddings.reshape(windows * 16, 3, 512)
            flat_valid = valid.reshape(windows * 16, 3)
            flat_boxes = boxes.reshape(windows * 16, 3, 4)
            for (token, time), token_range in zip(emissions, ranges):
                segment_landmarks = slice_and_resample(
                    flat_landmarks, token_range, len(path), 32
                )[None]
                segment_embeddings = slice_and_resample(
                    flat_embeddings, token_range, len(path), 16
                )[None]
                segment_valid = slice_and_resample(
                    flat_valid, token_range, len(path), 16, nearest=True
                )[None]
                segment_boxes = slice_and_resample(
                    flat_boxes, token_range, len(path), 16
                )[None]
                landmark_tensor = torch.from_numpy(segment_landmarks).to(device)
                embedding_tensor = torch.from_numpy(segment_embeddings).to(device)
                valid_tensor = torch.from_numpy(segment_valid > 0.5).to(device)
                box_tensor = torch.from_numpy(segment_boxes).to(device)
                stage1_logits = models["stage1"](
                    landmark_tensor, embedding_tensor, valid_tensor, box_tensor
                )
                frozen_segment = models["encoder"](
                    landmark_tensor[:, None], embedding_tensor[:, None],
                    valid_tensor[:, None], box_tensor[:, None],
                )
                hybrid = models["correction"](frozen_segment, stage1_logits)[0].cpu().numpy()
                stage1 = stage1_logits[0].cpu().numpy()
                ctc_probability = torch.softmax(
                    torch.from_numpy(logits[time]), dim=-1
                ).numpy()
                replacements.append({
                    "token": token,
                    "ctc_confidence": float(ctc_probability[token]),
                    "stage1_prediction": int(stage1.argmax()) + 1,
                    "stage1_confidence": softmax_confidence(stage1)[0],
                    "stage1_margin": softmax_confidence(stage1)[1],
                    "hybrid_prediction": int(hybrid.argmax()) + 1,
                    "hybrid_confidence": softmax_confidence(hybrid)[0],
                    "hybrid_margin": softmax_confidence(hybrid)[1],
                })
            rows.append({
                "item_id": metadata["source_item_id"],
                "source": metadata["source"],
                "reference": reference,
                "baseline": baseline,
                "tokens": replacements,
            })
    return rows


def decode(row: dict, config: dict) -> list[int]:
    output = []
    for token in row["tokens"]:
        name = config["logits"]
        prediction = token[f"{name}_prediction"]
        replace = (
            token[f"{name}_confidence"] >= config["confidence"]
            and token[f"{name}_margin"] >= config["margin"]
            and token[f"{name}_confidence"] - token["ctc_confidence"] >= config["advantage"]
        )
        output.append(prediction if replace else token["token"])
    return output


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
    for path in (args.rgb_root, args.hand_root, args.frozen_root):
        if "test" in {part.lower() for part in path.parts}:
            raise ValueError("test data is forbidden")
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    landmark, hand, fusion, _ = load_frozen_unified_stage1(args.stage1_checkpoint)
    stage1 = UnifiedMultimodalStage1V17(landmark, hand, fusion).to(device).eval()
    encoder = FrozenUnifiedTemporalEncoderV17(landmark, hand, fusion).to(device).eval()
    correction, _ = load_stage2_isolated_correction(args.correction_checkpoint)
    models = {
        "stage1": stage1,
        "encoder": encoder,
        "correction": correction.to(device).eval(),
        "head": load_head(args.phrase_checkpoint).to(device).eval(),
    }
    training = collect("train", args, models, device)
    validation = collect("validation", args, models, device)
    candidates = []
    for name in ("stage1", "hybrid"):
        for confidence in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for margin in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
                for advantage in (-1.0, -0.2, 0.0, 0.1, 0.2, 0.3):
                    config = {
                        "logits": name, "confidence": confidence,
                        "margin": margin, "advantage": advantage,
                    }
                    score = metrics(training, [decode(row, config) for row in training])
                    changes = sum(
                        decode(row, config) != row["baseline"] for row in training
                    )
                    candidates.append((
                        score["equal_domain_mean_wer"],
                        -score["equal_domain_mean_sequence_accuracy"], changes, config, score,
                    ))
    _, _, _, selected, train_score = min(candidates, key=lambda item: item[:3])
    predictions = [decode(row, selected) for row in validation]
    baseline = metrics(validation, [row["baseline"] for row in validation])
    result = {
        "format": "slt_stage2_ctc_segment_reclassification_evaluation_v17",
        "version": 1,
        "selection_role": "train",
        "evaluation_role": "validation",
        "selected_gate": selected,
        "training_metrics": train_score,
        "validation_baseline_metrics": baseline,
        "validation_routed_metrics": metrics(validation, predictions),
        "validation_rows": [{
            "item_id": row["item_id"], "source": row["source"],
            "reference": row["reference"], "baseline": row["baseline"],
            "routed": prediction,
        } for row, prediction in zip(validation, predictions)],
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--rgb-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    value.add_argument("--hand-root", type=Path, default=Path("data/local/stage2_v17_hand_mobileclip2"))
    value.add_argument("--frozen-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    value.add_argument("--correction-checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_isolated_correction_v1/model.pth"))
    value.add_argument("--phrase-checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"))
    value.add_argument("--device", default="auto")
    value.add_argument("--mps-memory-fraction", type=float, default=0.12)
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_ctc_segment_reclassification_v1/validation.json"))
    return value


if __name__ == "__main__":
    output = run(parser().parse_args())
    print(json.dumps({
        "gate": output["selected_gate"],
        "baseline": output["validation_baseline_metrics"],
        "routed": output["validation_routed_metrics"],
    }, indent=2))
