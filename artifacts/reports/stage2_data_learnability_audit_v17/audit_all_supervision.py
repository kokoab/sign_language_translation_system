#!/usr/bin/env python3
"""Audit every frozen Stage-1-window supervision sample and its actual training exposure."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from active.v17 import train_stage1_window_v17 as training
from active.v17.stage1_window_v17 import load_stage1_window_checkpoint


REPORT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts/reports/o5s5_augmented_v17_20260914/combined_supervision.json"
BASE = ROOT / "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"
MODELS = ROOT / "artifacts/models/stage1_window_o5s5_v17_seed17111"
PRIOR = ROOT / "artifacts/models/stage1_window_v17_seed17111"


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def describe(values):
    values = list(values)
    return {
        "count": len(values), "min": min(values) if values else None,
        "p10": percentile(values, 10), "median": percentile(values, 50),
        "p90": percentile(values, 90), "max": max(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
    }


def item_id(identity):
    for marker in (":center:", ":schedule:", ":background:"):
        if marker in identity:
            return identity.split(marker, 1)[0]
    raise ValueError(identity)


def interval_for(sample, manifest_row, labels):
    known = [x for x in manifest_row["intervals"] if x.get("label") in labels]
    if ":center:" in sample.identity:
        position = int(sample.identity.split(":center:", 1)[1].split(":", 1)[0])
        return known[position]
    if ":schedule:" in sample.identity:
        end = float(sample.identity.rsplit(":", 1)[1])
        center = end - sample.duration_seconds / 2
        candidates = [x for x in manifest_row["intervals"]
                      if float(x["start_seconds"]) <= center <= float(x["end_seconds"])]
        candidates = [x for x in candidates if x.get("label") in labels]
        assert len(candidates) == 1
        return candidates[0]
    return None


def sample_row(sample, role, manifest_rows, labels):
    row = manifest_rows[item_id(sample.identity)]
    interval = interval_for(sample, row, labels)
    present = sample.features[:, :42, 3] > .5
    left = present[:, :21].any(1)
    right = present[:, 21:].any(1)
    consecutive = present[1:] & present[:-1]
    delta = np.linalg.norm(np.diff(sample.features[:, :42, :2], axis=0), axis=-1)
    motion = delta[consecutive]
    return {
        "role": role, "source": sample.source, "signer": sample.signer,
        "item_id": item_id(sample.identity), "identity": sample.identity,
        "category": sample.category, "target": sample.target,
        "schedule": sample.schedule, "window_seconds": sample.duration_seconds,
        "target_seconds": None if interval is None else float(interval["end_seconds"]) - float(interval["start_seconds"]),
        "foreground_fraction": float(sample.foreground.mean()),
        "hand_frame_fraction": float(present.any(1).mean()),
        "both_hand_frame_fraction": float((left & right).mean()),
        "mean_joint_motion": float(motion.mean()) if len(motion) else 0.0,
        "feature_sha256": hashlib.sha256(sample.features.tobytes()).hexdigest(),
    }


@torch.inference_mode()
def predict(model, samples, device):
    model = model.to(device).eval()
    prediction, known_prediction, top5, confidence = [], [], [], []
    for start in range(0, len(samples), 128):
        features = torch.from_numpy(np.stack([x.features for x in samples[start:start + 128]])).to(device)
        logits = model(features).cpu()
        target = torch.tensor([x.target for x in samples[start:start + 128]])
        known = logits[:, :100]
        prediction.extend(logits.argmax(1).tolist())
        known_prediction.extend(known.argmax(1).tolist())
        top5.extend((known.topk(5, 1).indices == target[:, None]).any(1).tolist())
        confidence.extend(logits.softmax(1).max(1).values.tolist())
    model.cpu()
    return prediction, known_prediction, top5, confidence


def accuracy_rows(name, rows, predictions, known_predictions, top5, confidence):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        keys = {
            "all": "all",
            "role": row["role"], "source": row["source"], "category": row["category"],
            "role/source/category": f'{row["role"]}/{row["source"]}/{row["category"]}',
            "source/category/schedule/window":
                f'{row["source"]}/{row["category"]}/{row["schedule"]}/{row["window_seconds"]:.2f}',
        }
        for kind, value in keys.items():
            groups[(kind, value)].append(i)
    output = []
    for (kind, value), indices in sorted(groups.items()):
        context = [i for i in indices if rows[i]["target"] < 100]
        background = [i for i in indices if rows[i]["target"] == 100]
        output.append({
            "model": name, "group_kind": kind, "group": value, "samples": len(indices),
            "context_samples": len(context), "background_samples": len(background),
            "full_accuracy": sum(predictions[i] == rows[i]["target"] for i in indices) / len(indices),
            "known_top1_accuracy": (sum(known_predictions[i] == rows[i]["target"] for i in context) / len(context)) if context else None,
            "known_top5_accuracy": (sum(top5[i] for i in context) / len(context)) if context else None,
            "context_no_emit_rate": (sum(predictions[i] == 100 for i in context) / len(context)) if context else None,
            "background_no_emit_rate": (sum(predictions[i] == 100 for i in background) / len(background)) if background else None,
            "mean_confidence": float(np.mean([confidence[i] for i in indices])),
        })
    return output


@torch.inference_mode()
def isolated_accuracy(model, replay, device):
    model = model.to(device).eval()
    correct = total = 0
    by_source = Counter()
    counts = Counter()
    for start in range(0, len(replay), 128):
        batch = replay[start:start + 128]
        features = torch.from_numpy(np.stack([training.load_features(path) for path, _, _ in batch])).to(device)
        target = torch.tensor([target for _, target, _ in batch])
        prediction = model(features).cpu().argmax(1)
        for (_, _, source), ok in zip(batch, prediction == target):
            counts[source] += 1
            by_source[source] += int(ok)
        correct += int((prediction == target).sum())
        total += len(batch)
    model.cpu()
    return {"correct": correct, "total": total, "accuracy": correct / total,
            "by_source": {source: {"correct": by_source[source], "total": count,
                                   "accuracy": by_source[source] / count}
                          for source, count in counts.items()}}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_payload = torch.load(BASE, map_location="cpu", weights_only=False)
    labels = {str(k): int(v) for k, v in base_payload["label_to_index"].items() if int(v) < 100}
    label_names = {v: k for k, v in labels.items()}
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_rows = {str(row["source_item_id"]): row for row in payload["rows"]}

    samples, rows, audits = [], [], {}
    split_samples = {}
    for role in ("train", "validation"):
        split_samples[role], audits[role] = training.load_context_samples(MANIFEST, labels, role)
        samples.extend(split_samples[role])
        rows.extend(sample_row(sample, role, manifest_rows, labels) for sample in split_samples[role])
    assert len(samples) == len(rows) == 8978
    assert Counter(row["category"] for row in rows) == {"context": 8468, "background": 510}

    duplicate_labels = defaultdict(set)
    for row in rows:
        duplicate_labels[row["feature_sha256"]].add(row["target"])
    conflicts = {key: sorted(value) for key, value in duplicate_labels.items() if len(value) > 1}

    train_replay = training._replay_rows(ROOT / "data/local/citizen100_v17/landmarks/train", labels, "citizen", 10)
    train_replay += training._replay_rows(ROOT / "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17", labels, "semlex", 10)
    validation_replay = training._replay_rows(ROOT / "data/local/citizen100_v17/landmarks/val", labels, "citizen", 10000)
    validation_replay += training._replay_rows(ROOT / "data/local/semlex_citizen100_val_audit/landmarks_v17", labels, "semlex", 10000)
    train_data = training._Rows(train_replay, split_samples["train"])
    generator = torch.Generator().manual_seed(17111)
    exposure = Counter()
    for _ in range(12):
        for index in training.epoch_indices(train_data.keys, generator):
            if index >= len(train_replay):
                exposure[index - len(train_replay)] += 1
    train_rows = rows[:len(split_samples["train"])]
    for index, row in enumerate(train_rows):
        row["training_draws"] = exposure[index]
    for row in rows[len(train_rows):]:
        row["training_draws"] = None

    models = {"original_base": training._load_start(base_payload).base}
    for epoch in range(1, 13):
        models[f"o5s5_epoch_{epoch:02d}"] = load_stage1_window_checkpoint(MODELS / f"epoch_{epoch:02d}.pth")[0]
    for epoch in (1, 4):
        models[f"prior_no_o5s5_epoch_{epoch:02d}"] = load_stage1_window_checkpoint(PRIOR / f"epoch_{epoch:02d}.pth")[0]

    summaries, predictions = [], {}
    isolated = {}
    for name, model in models.items():
        pred, known, top5, confidence = predict(model, samples, device)
        predictions[name] = pred
        summaries.extend(accuracy_rows(name, rows, pred, known, top5, confidence))
        isolated[name] = {
            "train": isolated_accuracy(model.base if hasattr(model, "base") else model, train_replay, device),
            "validation": isolated_accuracy(model.base if hasattr(model, "base") else model, validation_replay, device),
        }
        if name in {"original_base", "o5s5_epoch_01", "o5s5_epoch_03", "o5s5_epoch_12"}:
            for index, row in enumerate(rows):
                row[f"prediction_{name}"] = pred[index]

    candidate_names = [f"o5s5_epoch_{epoch:02d}" for epoch in range(1, 13)]
    for index, row in enumerate(rows):
        row["correct_any_candidate_epoch"] = any(predictions[name][index] == row["target"] for name in candidate_names)
        row["target_label"] = "NO_EMIT" if row["target"] == 100 else label_names[row["target"]]

    exposure_summary = {}
    for source in sorted({row["source"] for row in train_rows if row["category"] == "context"}):
        subset = [row for row in train_rows if row["source"] == source and row["category"] == "context"]
        exposure_summary[source] = {
            "windows": len(subset), "draws": sum(row["training_draws"] for row in subset),
            "never_drawn": sum(row["training_draws"] == 0 for row in subset),
            "never_drawn_fraction": sum(row["training_draws"] == 0 for row in subset) / len(subset),
            "draws_per_window": describe(row["training_draws"] for row in subset),
        }

    event_rows = []
    for row in payload["rows"]:
        intervals = sorted(row["intervals"], key=lambda x: float(x["start_seconds"]))
        for index, interval in enumerate(intervals):
            label = interval.get("label")
            if label not in labels:
                continue
            start, end = float(interval["start_seconds"]), float(interval["end_seconds"])
            event_rows.append({
                "role": row["role"], "source": row["source"], "signer": row["signer_id"],
                "item_id": row["source_item_id"], "label": label,
                "duration_seconds": end - start,
                "previous_gap_seconds": None if index == 0 else start - float(intervals[index - 1]["end_seconds"]),
                "next_gap_seconds": None if index + 1 == len(intervals) else float(intervals[index + 1]["start_seconds"]) - end,
                "all_signs_annotated": row["all_signs_annotated"],
            })

    duration_summary = {}
    for key in sorted({(row["role"], row["source"]) for row in event_rows}):
        subset = [row for row in event_rows if (row["role"], row["source"]) == key]
        values = [row["duration_seconds"] for row in subset]
        duration_summary["/".join(key)] = {
            **describe(values),
            "under_0_27_fraction": sum(value < .27 for value in values) / len(values),
            "under_0_53_fraction": sum(value < .53 for value in values) / len(values),
        }

    learnability = {}
    for key in sorted({(row["role"], row["source"], row["category"]) for row in rows}):
        subset = [i for i, row in enumerate(rows) if (row["role"], row["source"], row["category"]) == key]
        learnability["/".join(key)] = {
            "samples": len(subset),
            "correct_in_any_of_12_epochs": sum(rows[i]["correct_any_candidate_epoch"] for i in subset),
            "correct_in_any_of_12_epochs_fraction": sum(rows[i]["correct_any_candidate_epoch"] for i in subset) / len(subset),
            "mean_foreground_fraction": float(np.mean([rows[i]["foreground_fraction"] for i in subset])),
            "mean_hand_frame_fraction": float(np.mean([rows[i]["hand_frame_fraction"] for i in subset])),
            "mean_both_hand_frame_fraction": float(np.mean([rows[i]["both_hand_frame_fraction"] for i in subset])),
        }

    output = {
        "format": "slt_stage2_all_supervision_learnability_audit_v17", "version": 1,
        "manifest": str(MANIFEST.relative_to(ROOT)), "manifest_sha256": training._sha256(MANIFEST),
        "device": str(device), "audits": audits, "total_samples": len(samples),
        "context_samples": 8468, "background_samples": 510,
        "exact_feature_conflicts": conflicts, "exposure": exposure_summary,
        "event_duration_seconds": duration_summary, "learnability": learnability,
        "isolated": isolated, "model_summaries": summaries,
        "protected_test_accessed": False,
        "limitations": [
            "O5S5 annotations are positive-only; unannotated surrounding signing cannot be scored as background or full-narrative WER.",
            "Overlapping windows are correlated and are not independent sign examples.",
            "Correct-in-any-epoch is a diagnostic upper envelope selected after observing the same samples, not deployable accuracy.",
        ],
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "audit.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_csv(REPORT / "all_window_predictions.csv", rows)
    write_csv(REPORT / "event_durations.csv", event_rows)
    write_csv(REPORT / "model_group_metrics.csv", summaries)
    print(json.dumps({"samples": len(samples), "models": len(models), "device": str(device),
                      "exposure": exposure_summary, "duration": duration_summary}, indent=2))


if __name__ == "__main__":
    main()
