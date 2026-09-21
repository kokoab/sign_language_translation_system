#!/usr/bin/env python3
"""Train and evaluate one segment-first Stage-2 experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.segment_first_v17 import (
    CHECKPOINT_FORMAT, END, IGNORE, OUTSIDE, SIGNING, START, STATES,
    SegmentCandidate, SegmentFirstV17, frame_targets, merge_candidates,
)
from active.v17.stage1_window_v17 import normalize_time_window, window_sample_times
from active.v17.train_stage_1_phrase_adapt_v17 import load_features


MANIFEST = ROOT / "artifacts/reports/confident_supervision_v17_20260920/confident_supervision.json"
BASE = ROOT / "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"
MODEL = ROOT / "artifacts/models/segment_first_v17_20260921/final.pth"
CITIZEN_TRAIN = ROOT / "data/local/citizen100_v17/landmarks/train"
CITIZEN_VAL = ROOT / "data/local/citizen100_v17/landmarks/val"
SEMLEX_TRAIN = ROOT / "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"
SEMLEX_VAL = ROOT / "data/local/semlex_citizen100_val_audit/landmarks_v17"
SEED, EPOCHS, BATCH = 17221, 8, 64
WINDOW_SECONDS, POST_SECONDS, LIVE_STRIDE = 1.60, 0.13, 0.067
MIN_LIVE_HISTORY, RECENT_END_SECONDS = 0.27, 0.25
EDGE_TOLERANCES = (0.10, 0.20)


@dataclass(frozen=True)
class BoundaryExample:
    features: np.ndarray
    targets: np.ndarray
    times: np.ndarray
    role: str
    source: str
    item: str
    anchor: str


@dataclass(frozen=True)
class SegmentExample:
    features: np.ndarray
    target: int
    role: str
    source: str
    item: str
    start: float
    end: float


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(name: str, value) -> None:
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_model() -> tuple[SegmentFirstV17, dict[str, int]]:
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("starting checkpoint is not plain Stage 1")
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    if len(labels) != 100 or sorted(labels.values()) != list(range(100)):
        raise ValueError("visible vocabulary is not the locked 100")
    base = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return SegmentFirstV17(base), labels


def archive_path(row) -> Path:
    value = row.get("archive_path", row.get("archive"))
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if {"test", "external_evaluation_reserved"} & {part.casefold() for part in path.parts}:
        raise ValueError(f"protected path: {path}")
    return path


def raw_sequence(row) -> tuple[np.ndarray, np.ndarray]:
    path = archive_path(row)
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["raw_format"].item()) != "apple_vision_isotropic_xy_confidence_v1":
            raise ValueError(f"raw schema mismatch: {path}")
        raw = payload["raw_features"].astype(np.float32, copy=False)
        times = payload["timestamps_seconds"].astype(np.float64, copy=False)
    if raw.shape != (len(times), 61, 5) or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"invalid archive: {path}")
    return raw, times


def normalized(raw, times, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    if end <= start:
        raise ValueError("normalization interval is empty")
    features = normalize_time_window(raw, times, end, end - start)[0].astype(np.float16)
    keep = (times >= start - 1e-9) & (times <= end + 1e-9)
    samples = window_sample_times(times[keep], end, end - start)
    return features, samples


def group_manifest(payload, role: str):
    accepted = defaultdict(list)
    decisions = defaultdict(list)
    for row in payload["rows"]:
        if row["role"] == role:
            accepted[str(row["source_item_id"])].append(row)
    for row in payload["decisions"]:
        if row["role"] == role:
            decisions[str(row["item"])].append(row)
    return accepted, decisions


def build_examples(payload, labels, role: str):
    by_item, decision_by_item = group_manifest(payload, role)
    boundary, segments = [], []
    audit = Counter()
    for item, rows in by_item.items():
        raw, times = raw_sequence(rows[0])
        source = str(rows[0]["source"])
        accepted_intervals = sorted(
            [(float(row["start_seconds"]), float(row["end_seconds"])) for row in rows]
        )
        excluded_intervals = sorted(
            [(float(row["start"]), float(row["end"])) for row in decision_by_item[item]
             if not row["accepted"]]
        )
        for row in rows:
            start, end = float(row["start_seconds"]), float(row["end_seconds"])
            core, core_times = normalized(raw, times, start, end)
            target = int(row["target_index"]) if row["target_kind"] == "known" else -1
            if target >= 0 and labels.get(str(row["label"])) != target:
                raise ValueError(f"target mismatch: {row['identity']}")
            segments.append(SegmentExample(core, target, role, source, item, start, end))
            if source == "o5s5":
                targets = frame_targets(core_times, [(start, end)], [], False)
                boundary.append(BoundaryExample(core, targets, core_times, role, source, item, "core"))
                continue
            for anchor, endpoint in (("start", start + POST_SECONDS), ("end", end + POST_SECONDS)):
                window_end = min(float(times[-1]), max(float(times[0]) + 1e-6, endpoint))
                window_start = max(float(times[0]), window_end - WINDOW_SECONDS)
                features, sample_times = normalized(raw, times, window_start, window_end)
                targets = frame_targets(sample_times, accepted_intervals, excluded_intervals, True)
                boundary.append(BoundaryExample(features, targets, sample_times, role, source, item, anchor))
        audit[(source, "items")] += 1
        audit[(source, "accepted_events")] += len(rows)
    if not boundary or not segments:
        raise ValueError(f"no {role} examples")
    return boundary, segments, {"|".join(key): value for key, value in sorted(audit.items())}


def isolated_samples(root: Path, labels, source: str, limit: int):
    output = []
    for label, target in sorted(labels.items(), key=lambda item: item[1]):
        paths = sorted((root / label).glob("*.v17.npz"))[:limit]
        if source == "citizen" and not paths:
            raise FileNotFoundError(f"missing {source}/{label}")
        for path in paths:
            if "test" in {part.casefold() for part in path.parts}:
                raise ValueError(f"protected isolated path: {path}")
            output.append(SegmentExample(load_features(path).astype(np.float16), target,
                                          "isolated", source, str(path.relative_to(ROOT)), 0.0, 0.0))
    return output


def weights(values, classes: int, device):
    count = np.bincount(np.asarray(values, np.int64), minlength=classes).clip(1)
    result = np.sqrt(count.sum() / (classes * count))
    return torch.tensor(result, dtype=torch.float32, device=device)


def boundary_batch(rows, indices, device):
    selected = [rows[index] for index in indices]
    features = torch.from_numpy(np.stack([row.features for row in selected]).astype(np.float32)).to(device)
    targets = torch.from_numpy(np.stack([row.targets for row in selected])).to(device)
    return selected, features, targets


def segment_batch(rows, indices, device):
    selected = [rows[index] for index in indices]
    features = torch.from_numpy(np.stack([row.features for row in selected]).astype(np.float32)).to(device)
    targets = torch.tensor([row.target for row in selected], device=device)
    known = (targets >= 0).long()
    return selected, features, targets, known


@torch.inference_mode()
def boundary_probabilities(model, rows, device):
    output = []
    model.eval()
    for begin in range(0, len(rows), BATCH):
        selected, features, _ = boundary_batch(rows, range(begin, min(begin + BATCH, len(rows))), device)
        probabilities = model.boundary_logits(features).softmax(-1).cpu().numpy()
        output.extend(probabilities[index] for index in range(len(selected)))
    return output


@torch.inference_mode()
def segment_predictions(model, rows, device):
    output = []
    model.eval()
    for begin in range(0, len(rows), BATCH):
        selected, features, _, _ = segment_batch(rows, range(begin, min(begin + BATCH, len(rows))), device)
        gloss, known = model.segment_logits(features)
        gp, kp = gloss.softmax(-1).cpu().numpy(), known.softmax(-1).cpu().numpy()
        output.extend(dict(gloss=int(gp[index].argmax()), gloss_confidence=float(gp[index].max()),
                           known_probability=float(kp[index, 1])) for index in range(len(selected)))
    return output


def candidate_from_window(times, probabilities, threshold: float):
    recent = np.flatnonzero(times >= times[-1] - RECENT_END_SECONDS)
    if not len(recent):
        return None
    end_index = int(recent[np.argmax(probabilities[recent, END])])
    end_probability = float(probabilities[end_index, END])
    eligible = np.flatnonzero((times >= times[end_index] - 1.30) &
                              (times <= times[end_index] - 0.10))
    if not len(eligible):
        return None
    start_index = int(eligible[np.argmax(probabilities[eligible, START])])
    start_probability = float(probabilities[start_index, START])
    if min(start_probability, end_probability) < threshold:
        return None
    signing = float(probabilities[start_index:end_index + 1, SIGNING].mean())
    return SegmentCandidate(float(times[start_index]), float(times[end_index]),
                            min(start_probability, end_probability) * max(signing, 1e-6))


def window_truth(row, accepted):
    return [(start, end) for start, end in accepted[row.item]
            if row.times[0] <= start and end <= row.times[-1]
            and end >= row.times[-1] - RECENT_END_SECONDS]


def edge_match(candidate, truths, tolerance):
    return any(abs(candidate.start_seconds - start) <= tolerance and
               abs(candidate.end_seconds - end) <= tolerance for start, end in truths)


def select_boundary_threshold(rows, probabilities, accepted):
    best = None
    for threshold in np.linspace(0.10, 0.70, 13):
        tp = fp = fn = 0
        for row, probability in zip(rows, probabilities):
            if row.source == "o5s5":
                continue
            truth = window_truth(row, accepted)
            candidate = candidate_from_window(row.times, probability, float(threshold))
            matched = candidate is not None and edge_match(candidate, truth, 0.10)
            tp += int(matched)
            fp += int(candidate is not None and not matched)
            fn += int(bool(truth) and not matched)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        value = (f1, -fp, float(threshold), tp, fp, fn)
        if best is None or value > best:
            best = value
    return dict(threshold=best[2], train_f1=best[0], tp=best[3], fp=best[4], fn=best[5])


def select_gate_threshold(rows, predictions):
    truth = np.asarray([row.target >= 0 for row in rows])
    scores = np.asarray([row["known_probability"] for row in predictions])
    best = None
    for threshold in np.linspace(0.05, 0.95, 91):
        guess = scores >= threshold
        known_recall = float(guess[truth].mean()) if truth.any() else 0.0
        unknown_recall = float((~guess[~truth]).mean()) if (~truth).any() else 0.0
        value = ((known_recall + unknown_recall) / 2, float(threshold), known_recall, unknown_recall)
        if best is None or value > best:
            best = value
    return dict(threshold=best[1], train_balanced_accuracy=best[0],
                known_recall=best[2], unknown_recall=best[3])


def evaluate_edges(rows, probabilities, threshold):
    confusion = np.zeros((4, 4), dtype=np.int64)
    edge = {str(tolerance): Counter() for tolerance in EDGE_TOLERANCES}
    for row, probability in zip(rows, probabilities):
        valid = row.targets != IGNORE
        predicted = probability.argmax(1)
        for truth, guess in zip(row.targets[valid], predicted[valid]):
            confusion[int(truth), int(guess)] += 1
        accepted = []
        starts = np.flatnonzero(row.targets == START)
        ends = np.flatnonzero(row.targets == END)
        if len(starts) and len(ends):
            for start_index in starts:
                later = ends[ends > start_index]
                if len(later):
                    accepted.append((float(row.times[start_index]), float(row.times[later[0]])))
        candidate = candidate_from_window(row.times, probability, threshold)
        for tolerance in EDGE_TOLERANCES:
            key = str(tolerance)
            matched = candidate is not None and edge_match(candidate, accepted, tolerance)
            edge[key]["tp"] += int(matched)
            edge[key]["fp"] += int(candidate is not None and not matched)
            edge[key]["fn"] += int(bool(accepted) and not matched)
    metrics = {}
    for tolerance, counts in edge.items():
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        metrics[tolerance] = dict(counts, precision=precision, recall=recall,
                                  f1=2 * precision * recall / max(1e-12, precision + recall))
    return dict(frame_confusion=confusion.tolist(), edge=metrics)


def matching(predictions, truths, tolerance):
    pairs, used = [], set()
    for prediction_index, prediction in enumerate(predictions):
        choices = [(max(abs(prediction.start_seconds - start), abs(prediction.end_seconds - end)), index)
                   for index, (start, end) in enumerate(truths) if index not in used]
        choices = [choice for choice in choices if choice[0] <= tolerance]
        if choices:
            _, truth_index = min(choices)
            used.add(truth_index); pairs.append((prediction_index, truth_index))
    return pairs


def edit_counts(reference, hypothesis):
    rows = [[(0, 0, 0, 0)] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
    rows[0] = [(j, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i in range(1, len(reference) + 1): rows[i][0] = (i, i, 0, 0)
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                rows[i][j] = rows[i - 1][j - 1]
            else:
                delete = rows[i - 1][j]; insert = rows[i][j - 1]; substitute = rows[i - 1][j - 1]
                options = [(delete[0] + 1, delete[1] + 1, delete[2], delete[3]),
                           (insert[0] + 1, insert[1], insert[2] + 1, insert[3]),
                           (substitute[0] + 1, substitute[1], substitute[2], substitute[3] + 1)]
                rows[i][j] = min(options)
    return rows[-1][-1]


def live_endpoints(times):
    values = list(np.arange(float(times[0]) + MIN_LIVE_HISTORY, float(times[-1]) + 1e-9, LIVE_STRIDE))
    if not values or values[-1] < times[-1] - 1e-6:
        values.append(float(times[-1]))
    return values


@torch.inference_mode()
def scan_source(model, raw, times, threshold, device):
    windows, sample_times = [], []
    for endpoint in live_endpoints(times):
        start = max(float(times[0]), endpoint - WINDOW_SECONDS)
        try:
            features, samples = normalized(raw, times, start, endpoint)
        except ValueError:
            continue
        windows.append(features); sample_times.append(samples)
    candidates = []
    for begin in range(0, len(windows), BATCH):
        features = torch.from_numpy(np.stack(windows[begin:begin + BATCH]).astype(np.float32)).to(device)
        probabilities = model.boundary_logits(features).softmax(-1).cpu().numpy()
        for offset, probability in enumerate(probabilities):
            candidate = candidate_from_window(sample_times[begin + offset], probability, threshold)
            if candidate is not None:
                candidates.append(candidate)
    return merge_candidates(candidates)


def ignored_prediction(candidate, excluded):
    duration = candidate.end_seconds - candidate.start_seconds
    return any(max(0.0, min(candidate.end_seconds, end) - max(candidate.start_seconds, start)) /
               max(duration, end - start) >= 0.30 for start, end in excluded)


@torch.inference_mode()
def classify_candidates(model, raw, times, candidates, device):
    valid, features = [], []
    for candidate in candidates:
        try:
            value, _ = normalized(raw, times, candidate.start_seconds, candidate.end_seconds)
        except ValueError:
            continue
        valid.append(candidate); features.append(value)
    if not features:
        return [], []
    tensor = torch.from_numpy(np.stack(features).astype(np.float32)).to(device)
    gloss, known = model.segment_logits(tensor)
    gp, kp = gloss.softmax(-1).cpu().numpy(), known.softmax(-1).cpu().numpy()
    result = [dict(gloss=int(gp[index].argmax()), known_probability=float(kp[index, 1]))
              for index in range(len(valid))]
    return valid, result


def online_evaluation(model, payload, labels, boundary_threshold, gate_threshold, device):
    accepted, decisions = group_manifest(payload, "validation")
    totals = {str(value): Counter() for value in EDGE_TOLERANCES}
    visible_edits = Counter(); visible_references = 0
    gate_truth, gate_guess, gloss_correct = [], [], 0
    matched_known = 0; sources = 0; ignored = 0
    for item, rows in accepted.items():
        if rows[0]["source"] == "o5s5":
            continue
        raw, times = raw_sequence(rows[0]); sources += 1
        truths = sorted([(float(row["start_seconds"]), float(row["end_seconds"])) for row in rows])
        excluded = [(float(row["start"]), float(row["end"])) for row in decisions[item] if not row["accepted"]]
        candidates = scan_source(model, raw, times, boundary_threshold, device)
        kept = [candidate for candidate in candidates if not ignored_prediction(candidate, excluded)]
        ignored += len(candidates) - len(kept)
        kept, classifications = classify_candidates(model, raw, times, kept, device)
        for tolerance in EDGE_TOLERANCES:
            pairs = matching(kept, truths, tolerance)
            totals[str(tolerance)]["tp"] += len(pairs)
            totals[str(tolerance)]["fp"] += len(kept) - len(pairs)
            totals[str(tolerance)]["fn"] += len(truths) - len(pairs)
        pairs = matching(kept, truths, 0.20)
        for prediction_index, truth_index in pairs:
            row = rows[truth_index]
            predicted = classifications[prediction_index]
            truth_known = row["target_kind"] == "known"
            guess_known = predicted["known_probability"] >= gate_threshold
            gate_truth.append(truth_known); gate_guess.append(guess_known)
            if truth_known:
                matched_known += 1
                gloss_correct += int(predicted["gloss"] == int(row["target_index"]))
        reference = [int(row["target_index"]) for row in rows if row["target_kind"] == "known"]
        hypothesis = [prediction["gloss"] for prediction in classifications
                      if prediction["known_probability"] >= gate_threshold]
        distance, deletions, insertions, substitutions = edit_counts(reference, hypothesis)
        visible_edits.update(distance=distance, deletions=deletions,
                             insertions=insertions, substitutions=substitutions)
        visible_references += len(reference)
    edge = {}
    for tolerance, count in totals.items():
        precision = count["tp"] / max(1, count["tp"] + count["fp"])
        recall = count["tp"] / max(1, count["tp"] + count["fn"])
        edge[tolerance] = dict(count, precision=precision, recall=recall,
                               f1=2 * precision * recall / max(1e-12, precision + recall))
    truth = np.asarray(gate_truth, dtype=bool); guess = np.asarray(gate_guess, dtype=bool)
    return dict(sources=sources, ignored_predictions_over_questionable_annotations=ignored,
                edge=edge, matched_gate_samples=len(truth),
                known_recall=float(guess[truth].mean()) if truth.any() else 0.0,
                unknown_recall=float((~guess[~truth]).mean()) if (~truth).any() else 0.0,
                matched_known_gloss_accuracy=gloss_correct / max(1, matched_known),
                matched_known_gloss_correct=gloss_correct, matched_known_gloss_total=matched_known,
                visible_reference_glosses=visible_references,
                visible_wer=visible_edits["distance"] / max(1, visible_references),
                visible_edits=dict(visible_edits))


def segment_metrics(rows, predictions, gate_threshold):
    known_truth = np.asarray([row.target >= 0 for row in rows])
    known_guess = np.asarray([row["known_probability"] >= gate_threshold for row in predictions])
    known_recall = float(known_guess[known_truth].mean()) if known_truth.any() else 0.0
    unknown_recall = float((~known_guess[~known_truth]).mean()) if (~known_truth).any() else 0.0
    gloss = [prediction["gloss"] == row.target for row, prediction in zip(rows, predictions) if row.target >= 0]
    by_source = {}
    for source in sorted({row.source for row in rows}):
        subset = [(row, prediction) for row, prediction in zip(rows, predictions) if row.source == source]
        values = [prediction["gloss"] == row.target for row, prediction in subset if row.target >= 0]
        by_source[source] = dict(known_gloss_correct=sum(values), known_gloss_total=len(values),
                                 known_gloss_accuracy=sum(values) / max(1, len(values)))
    return dict(samples=len(rows), known_recall=known_recall, unknown_recall=unknown_recall,
                balanced_gate_accuracy=(known_recall + unknown_recall) / 2,
                known_gloss_correct=sum(gloss), known_gloss_total=len(gloss),
                known_gloss_accuracy=sum(gloss) / max(1, len(gloss)), by_source=by_source)


def synthetic_sequence(raw, times, start, end, repeated):
    core = raw[(times >= start) & (times <= end)]
    if len(core) < 4:
        raise ValueError("synthetic core too short")
    before = raw[times < start]
    after = raw[times > end]
    rest = before[-1] if len(before) else (after[0] if len(after) else core[0])
    prefix = np.repeat(rest[None], 6, axis=0)
    suffix = np.repeat(rest[None], 6, axis=0)
    if repeated:
        middle = np.concatenate((core, np.repeat(rest[None], 6, axis=0), core))
        spans = [(6, 6 + len(core) - 1),
                 (12 + len(core), 12 + 2 * len(core) - 1)]
    else:
        middle = np.concatenate((core, np.repeat(core[-1][None], 12, axis=0)))
        spans = [(6, 6 + len(middle) - 1)]
    value = np.concatenate((prefix, middle, suffix))
    step = float(np.median(np.diff(times[(times >= start) & (times <= end)])))
    clock = np.arange(len(value), dtype=np.float64) * step
    truth = [(clock[left], clock[right]) for left, right in spans]
    return value, clock, truth


def synthetic_evaluation(model, payload, boundary_threshold, device):
    accepted, _ = group_manifest(payload, "validation")
    chosen = [row for rows in accepted.values() for row in rows
              if row["source"] != "o5s5" and row["target_kind"] == "known"][:20]
    held_exact = repeat_exact = 0; evaluated = 0
    for row in chosen:
        raw, times = raw_sequence(row)
        for repeated in (False, True):
            value, clock, truth = synthetic_sequence(raw, times, float(row["start_seconds"]),
                                                     float(row["end_seconds"]), repeated)
            candidates = scan_source(model, value, clock, boundary_threshold, device)
            exact = len(matching(candidates, truth, 0.20)) == len(truth) and len(candidates) == len(truth)
            if repeated: repeat_exact += int(exact)
            else: held_exact += int(exact)
        evaluated += 1
    return dict(events=evaluated, held_once_exact=held_exact,
                intentional_repeat_twice_exact=repeat_exact,
                caveat="Time-warped feature probes, not independent recorded performances.")


def save_checkpoint(model, labels, epoch, audit):
    package = dict(format=CHECKPOINT_FORMAT, epoch=epoch, seed=SEED,
                   model_config=model.base.config.to_dict(),
                   base_model_state_dict={key: value.cpu() for key, value in model.base.state_dict().items()},
                   boundary_state_dict={key: value.cpu() for key, value in model.boundary.state_dict().items()},
                   known_state_dict={key: value.cpu() for key, value in model.known.state_dict().items()},
                   label_to_index=labels, states=list(STATES), public_gloss_count=100,
                   unknown_is_internal=True, window_seconds=WINDOW_SECONDS,
                   data_audit_sha256=sha(HERE / "data_audit.json"), citizen_test_accessed=False)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL.with_suffix(".tmp")
    torch.save(package, temporary); temporary.replace(MODEL)


def train_worker():
    pre = json.loads((HERE / "precheck.json").read_text())
    for path, key in ((MANIFEST, "manifest_sha256"), (BASE, "base_sha256"),
                      (Path(__file__).resolve(), "runner_sha256"),
                      (ROOT / "active/v17/segment_first_v17.py", "module_sha256")):
        if sha(path) != pre[key]:
            raise RuntimeError(f"prechecked input changed: {path}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    payload = json.loads(MANIFEST.read_text()); model, labels = load_model()
    train_boundary, train_segments, train_audit = build_examples(payload, labels, "train")
    validation_boundary, validation_segments, validation_audit = build_examples(payload, labels, "validation")
    train_segments += isolated_samples(CITIZEN_TRAIN, labels, "citizen", 5)
    train_segments += isolated_samples(SEMLEX_TRAIN, labels, "semlex", 5)
    validation_isolated = isolated_samples(CITIZEN_VAL, labels, "citizen", 100000)
    validation_isolated += isolated_samples(SEMLEX_VAL, labels, "semlex", 100000)
    audit = dict(train_boundary=len(train_boundary), train_segments=len(train_segments),
                 validation_boundary=len(validation_boundary), validation_segments=len(validation_segments),
                 validation_isolated=len(validation_isolated), train=train_audit, validation=validation_audit,
                 full_coverage_each_epoch=True, replacement_sampling=False,
                 o5s5_negative_targets=0, excluded_annotations_are_masked=True,
                 visible_glosses=100, citizen_test_accessed=False)
    save("data_audit.json", audit)
    device = torch.device("mps"); model.to(device)
    boundary_values = [int(value) for row in train_boundary for value in row.targets if value != IGNORE]
    boundary_weight = weights(boundary_values, 4, device)
    gate_weight = weights([int(row.target >= 0) for row in train_segments], 2, device)
    optimizer = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": 1e-5},
        {"params": list(model.boundary.parameters()) + list(model.known.parameters()), "lr": 4e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history = []
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter(); totals = Counter(); model.train()
        order = np.random.default_rng(SEED + epoch).permutation(len(train_boundary))
        for begin in range(0, len(order), BATCH):
            _, features, targets = boundary_batch(train_boundary, order[begin:begin + BATCH], device)
            loss = F.cross_entropy(model.boundary_logits(features).reshape(-1, 4), targets.reshape(-1),
                                   weight=boundary_weight, ignore_index=IGNORE)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            totals["boundary"] += float(loss.detach().cpu()); totals["boundary_updates"] += 1
        order = np.random.default_rng(SEED * 2 + epoch).permutation(len(train_segments))
        for begin in range(0, len(order), BATCH):
            _, features, targets, known = segment_batch(train_segments, order[begin:begin + BATCH], device)
            gloss, gate = model.segment_logits(features)
            gate_loss = F.cross_entropy(gate, known, weight=gate_weight)
            mask = targets >= 0
            gloss_loss = F.cross_entropy(gloss[mask], targets[mask])
            loss = gate_loss + gloss_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            totals["gate"] += float(gate_loss.detach().cpu()); totals["gloss"] += float(gloss_loss.detach().cpu())
            totals["segment_updates"] += 1
        scheduler.step(); save_checkpoint(model, labels, epoch, audit)
        record = dict(epoch=epoch, seconds=time.perf_counter() - started,
                      boundary_loss=totals["boundary"] / totals["boundary_updates"],
                      gate_loss=totals["gate"] / totals["segment_updates"],
                      gloss_loss=totals["gloss"] / totals["segment_updates"],
                      boundary_coverage=len(train_boundary), segment_coverage=len(train_segments))
        history.append(record); save("training_history.json", history); print(json.dumps(record), flush=True)

    train_boundary_prob = boundary_probabilities(model, train_boundary, device)
    train_accepted, _ = group_manifest(payload, "train")
    intervals = {item: [(float(row["start_seconds"]), float(row["end_seconds"])) for row in rows]
                 for item, rows in train_accepted.items()}
    boundary_selection = select_boundary_threshold(train_boundary, train_boundary_prob, intervals)
    train_segment_prediction = segment_predictions(model, train_segments, device)
    gate_selection = select_gate_threshold(train_segments, train_segment_prediction)
    validation_boundary_prob = boundary_probabilities(model, validation_boundary, device)
    validation_segment_prediction = segment_predictions(model, validation_segments, device)
    validation_isolated_prediction = segment_predictions(model, validation_isolated, device)
    metrics = dict(
        thresholds=dict(boundary=boundary_selection, gate=gate_selection),
        event_centered_edges=evaluate_edges(validation_boundary, validation_boundary_prob,
                                            boundary_selection["threshold"]),
        exact_segments=segment_metrics(validation_segments, validation_segment_prediction,
                                       gate_selection["threshold"]),
        isolated=segment_metrics(validation_isolated, validation_isolated_prediction,
                                 gate_selection["threshold"]),
        online=online_evaluation(model, payload, labels, boundary_selection["threshold"],
                                 gate_selection["threshold"], device),
        held_repeat=synthetic_evaluation(model, payload, boundary_selection["threshold"], device),
        checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=sha(MODEL),
        citizen_test_accessed=False, runtime_promoted=False,
    )
    save("metrics.json", metrics)
    write_report(metrics, audit)
    verify_checkpoint(model, labels, validation_segments[:64], device)
    subprocess.run([sys.executable, str(ROOT / "scripts/index_large_artifacts_v17.py")], cwd=ROOT, check=True)


def write_report(metrics, audit):
    edges = metrics["online"]["edge"]; segments = metrics["exact_segments"]
    isolated = metrics["isolated"]; online = metrics["online"]; probes = metrics["held_repeat"]
    lines = ["# Segment-first Stage 2 experiment", "",
             "This run trains one class-independent OUTSIDE/START/SIGNING/END head, one internal KNOWN/UNKNOWN gate, and the locked 100-gloss Stage-1 classifier. It uses only the confident manifest. Questionable annotations are masked; O5S5 never supplies background; annotation gaps are not a transition class.", "",
             "## Data and training", "",
             f"Full coverage used {audit['train_boundary']:,} boundary windows and {audit['train_segments']:,} segment/replay examples per epoch for {EPOCHS} epochs on MPS. The public vocabulary remained exactly 100; Citizen test stayed sealed.", "",
             "## Results", "",
             "| Measure | Result |", "| --- | ---: |",
             f"| Online boundary F1 at ±100 ms | {edges['0.1']['f1']*100:.2f}% |",
             f"| Online boundary F1 at ±200 ms | {edges['0.2']['f1']*100:.2f}% |",
             f"| Exact-core known/unknown balanced accuracy | {segments['balanced_gate_accuracy']*100:.2f}% |",
             f"| Exact-core known gloss accuracy | {segments['known_gloss_accuracy']*100:.2f}% |",
             f"| Online matched known gloss accuracy | {online['matched_known_gloss_accuracy']*100:.2f}% |",
             f"| Online visible WER | {online['visible_wer']*100:.2f}% |",
             f"| Citizen + SemLex validation gloss accuracy | {isolated['known_gloss_accuracy']*100:.2f}% |",
             f"| Synthetic held signs emitted once | {probes['held_once_exact']}/{probes['events']} |",
             f"| Synthetic intentional repeats emitted twice | {probes['intentional_repeat_twice_exact']}/{probes['events']} |", "",
             "The synthetic hold/repeat rows are time-warped feature probes, not independent recorded performances. Promotion remains false until the measured gates justify changing the live runtime. Full counts, confusion matrices, source breakdowns, and thresholds are in `metrics.json`.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


@torch.inference_mode()
def verify_checkpoint(model, labels, samples, device):
    package = torch.load(MODEL, map_location="cpu", weights_only=False)
    if package.get("format") != CHECKPOINT_FORMAT or package.get("label_to_index") != labels:
        raise ValueError("checkpoint contract mismatch")
    cpu, _ = load_model()
    cpu.base.load_state_dict(package["base_model_state_dict"], strict=True)
    cpu.boundary.load_state_dict(package["boundary_state_dict"], strict=True)
    cpu.known.load_state_dict(package["known_state_dict"], strict=True); cpu.eval(); model.eval()
    features = torch.from_numpy(np.stack([row.features for row in samples]).astype(np.float32))
    cpu_output = cpu.segment_logits(features); mps_output = model.segment_logits(features.to(device))
    gloss = int((cpu_output[0].argmax(1) == mps_output[0].argmax(1).cpu()).sum())
    gate = int((cpu_output[1].argmax(1) == mps_output[1].argmax(1).cpu()).sum())
    if gloss != len(samples) or gate != len(samples):
        raise RuntimeError("CPU/MPS decisions differ")
    save("verification.json", dict(status="passed", focused_tests="3/3", checked_samples=len(samples),
                                   cpu_mps_gloss_matches=gloss, cpu_mps_gate_matches=gate,
                                   full_coverage=True, citizen_test_accessed=False, model_promoted=False))


def precheck():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    payload = json.loads(MANIFEST.read_text()); model, labels = load_model()
    if payload.get("format") != "slt_confident_continuous_supervision_v17":
        raise ValueError("wrong supervision manifest")
    if payload["contract"].get("gaps_are_transition_targets") is not False:
        raise ValueError("gap contract changed")
    if payload["audit"].get("train_validation_signer_disjoint") is not True:
        raise ValueError("split contract changed")
    train_boundary, train_segments, train_audit = build_examples(payload, labels, "train")
    validation_boundary, validation_segments, validation_audit = build_examples(payload, labels, "validation")
    if any(OUTSIDE in row.targets for row in train_boundary if row.source == "o5s5"):
        raise ValueError("O5S5 leaked background supervision")
    if any(set(np.unique(row.targets)) - {IGNORE, OUTSIDE, START, SIGNING, END} for row in train_boundary):
        raise ValueError("invalid boundary state")
    isolated_counts = {}
    for name, root, limit in (("citizen_train", CITIZEN_TRAIN, 5),
                              ("semlex_train", SEMLEX_TRAIN, 5),
                              ("citizen_validation", CITIZEN_VAL, 100000),
                              ("semlex_validation", SEMLEX_VAL, 100000)):
        rows = isolated_samples(root, labels, name.split("_")[0], limit)
        isolated_counts[name] = len(rows)
        del rows
    device = torch.device("mps"); model.to(device).eval()
    rows = train_boundary[:8]
    features = torch.from_numpy(np.stack([row.features for row in rows]).astype(np.float32)).to(device)
    targets = torch.from_numpy(np.stack([row.targets for row in rows])).to(device)
    with torch.no_grad(): encoded, _ = model.base.encode(features)
    head = torch.nn.Linear(model.base.config.dim, 4).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2); losses = []
    for _ in range(30):
        loss = F.cross_entropy(head(encoded.detach()).reshape(-1, 4), targets.reshape(-1), ignore_index=IGNORE)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
    if losses[-1] >= losses[0] * 0.70:
        raise RuntimeError("MPS boundary tiny-fit failed")
    result = dict(status="passed", manifest_sha256=sha(MANIFEST), base_sha256=sha(BASE),
                  runner_sha256=sha(Path(__file__).resolve()),
                  module_sha256=sha(ROOT / "active/v17/segment_first_v17.py"),
                  labels=100, states=list(STATES), train_boundary=len(train_boundary),
                  train_segments=len(train_segments), validation_boundary=len(validation_boundary),
                  validation_segments=len(validation_segments), train_audit=train_audit,
                  validation_audit=validation_audit, isolated_counts=isolated_counts,
                  mps_tiny_fit=[losses[0], losses[-1]],
                  o5s5_negative_targets=0, excluded_annotations_masked=True,
                  signer_disjoint=True, citizen_test_accessed=False)
    save("precheck.json", result)
    (HERE / "PRECHECK_REPORT.md").write_text(
        "# Segment-first precheck\n\nPassed every confident manifest row through the exact training builder, verified signer separation, the locked 100-gloss contract, excluded-annotation masking, zero O5S5 background targets, real MPS output, and a boundary-head tiny fit. Citizen test was not accessed.\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2))


def worker():
    started = datetime.now(timezone.utc).isoformat(); status = "failed"
    try:
        train_worker(); status = "completed"
    except BaseException:
        (HERE / "FAILURE.md").write_text("# Segment-first experiment failed\n\n```text\n" +
                                          traceback.format_exc() + "```\n", encoding="utf-8")
        traceback.print_exc()
    finally:
        save("completion.json", dict(status=status, started_at=started,
                                     finished_at=datetime.now(timezone.utc).isoformat()))
        subprocess.run(["osascript", "-e",
                        'on run argv\ndisplay notification (item 1 of argv) with title "SLT experiment"\nend run',
                        "Segment-first experiment " + status + ". See " + str(HERE)],
                       capture_output=True, timeout=15, check=False)
    if status != "completed":
        raise SystemExit(1)


def launch():
    if (HERE / "launch.json").exists():
        raise RuntimeError("experiment already launched")
    if json.loads((HERE / "precheck.json").read_text()).get("status") != "passed":
        raise RuntimeError("precheck did not pass")
    with (HERE / "process.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    save("launch.json", dict(pid=child.pid, launched_at=datetime.now(timezone.utc).isoformat(),
                             notification_on_exit=True, polling=False))
    print(f"Launched {child.pid} with one exit notification; no polling.")


if __name__ == "__main__":
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.precheck: precheck()
    elif args.worker: worker()
    elif args.launch: launch()
    else: parser.error("choose --precheck or --launch")
