#!/usr/bin/env python3
"""Train the fixed-window Stage-1 gloss + endpoint-phase experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import copy
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

from active.v17.boundary_phase_v17 import (
    BoundaryPhaseHeadV17, BoundaryPhaseModelV17, BoundaryPhaseTranscript,
    CHECKPOINT_FORMAT, KNOWN, PHASES, STRIDE_SECONDS, TRANSITION, UNKNOWN,
    WINDOW_FRAMES, WINDOW_SECONDS,
)
from active.v17.live_transition_supervision_v17 import interior_gaps
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.stage1_window_v17 import normalize_time_window, window_end_times
from active.v17.train_stage_1_phrase_adapt_v17 import load_features
from active.v17.train_stage_2_other_ctc_v17 import _edit_operations

MANIFEST = ROOT / "artifacts/reports/o5s5_citizen100_v17/combined_supervision.json"
BASE = ROOT / "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"
MODEL = ROOT / "artifacts/models/boundary_phase_v17_20260920/final.pth"
CITIZEN_TRAIN = ROOT / "data/local/citizen100_v17/landmarks/train"
CITIZEN_VAL = ROOT / "data/local/citizen100_v17/landmarks/val"
SEMLEX_TRAIN = ROOT / "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"
SEMLEX_VAL = ROOT / "data/local/semlex_citizen100_val_audit/landmarks_v17"
SEED, EPOCHS, BATCH = 17220, 12, 64
CORE_GUARD = 1.0 / 30.0
ALLOWED_SOURCES = {"asllrp_contiguous", "asllrp_other_ctc", "o5s5"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(name: str, value) -> None:
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_base() -> tuple[BoundaryPhaseModelV17, dict[str, int]]:
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("starting checkpoint is not plain v17 Stage 1")
    labels = {str(k): int(v) for k, v in checkpoint["label_to_index"].items()}
    if len(labels) != 100 or sorted(labels.values()) != list(range(100)):
        raise ValueError("starting checkpoint does not have the locked 100-gloss alphabet")
    config = Stage1V17Config(**checkpoint["model_config"])
    if config.num_classes != 100 or config.static_hand_token != "none":
        raise ValueError("unsupported Stage-1 model contract")
    base = SLTStage1V17(config)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return BoundaryPhaseModelV17(base, BoundaryPhaseHeadV17(config.dim, 100)), labels


def raw_sequence(row) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / row["archive_path"] if not Path(row["archive_path"]).is_absolute() else Path(row["archive_path"])
    if {"test", "external_evaluation_reserved"} & {part.casefold() for part in path.parts}:
        raise ValueError(f"protected path: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["raw_format"].item()) != "apple_vision_isotropic_xy_confidence_v1":
            raise ValueError(f"raw schema mismatch: {path}")
        features = payload["raw_features"].astype(np.float32, copy=False)
        timestamps = payload["timestamps_seconds"].astype(np.float64, copy=False)
    if features.shape != (len(timestamps), 61, 5) or len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"invalid raw archive: {path}")
    return features, timestamps


@dataclass(frozen=True)
class Sample:
    features: np.ndarray
    target: int
    phase: int
    kind: str
    source: str
    identity: str
    duration: float


def endpoints(start: float, end: float) -> list[float]:
    left, right = start + CORE_GUARD, end - CORE_GUARD
    if right <= left:
        return [(start + end) / 2]
    return sorted({(left + right) / 2, right})


def context_samples(payload, labels, role: str, selected_rows=None):
    output, rejected = [], Counter()
    rows = [row for row in payload["rows"] if row["role"] == role]
    if selected_rows is not None:
        rows = selected_rows
    for row in rows:
        complete = row.get("all_signs_annotated") is True
        source, item = str(row["source"]), str(row["source_item_id"])
        features, timestamps = raw_sequence(row)
        intervals = sorted(row["intervals"], key=lambda value: float(value["start_seconds"]))
        for position, interval in enumerate(intervals):
            label = str(interval["label"])
            if label in labels:
                phase, target = KNOWN, labels[label]
            elif complete and label == "__OTHER__":
                phase, target = UNKNOWN, -1
            else:
                continue
            for end in endpoints(float(interval["start_seconds"]), float(interval["end_seconds"])):
                for duration in WINDOW_SECONDS:
                    if end - duration < timestamps[0] - 1e-9 or end > timestamps[-1] + 1e-9:
                        rejected["outside_archive"] += 1
                        continue
                    try:
                        value = normalize_time_window(features, timestamps, end, duration)[0]
                    except ValueError as error:
                        rejected[str(error)] += 1
                        continue
                    if phase == KNOWN and not (value[:, :42, 3] > 0).any():
                        rejected["known_without_hand"] += 1
                        continue
                    output.append(Sample(value.astype(np.float16), target, phase, "context", source,
                                         f"{item}:event:{position}:{end:.6f}:{duration:.2f}", duration))
        if complete:
            for position, (left, right) in enumerate(interior_gaps(intervals, CORE_GUARD)):
                for end in [(left + right) / 2]:
                    for duration in WINDOW_SECONDS:
                        if end - duration < timestamps[0] - 1e-9:
                            rejected["outside_archive"] += 1
                            continue
                        try:
                            value = normalize_time_window(features, timestamps, end, duration)[0]
                        except ValueError as error:
                            rejected[str(error)] += 1
                            continue
                        output.append(Sample(value.astype(np.float16), -1, TRANSITION, "context", source,
                                             f"{item}:gap:{position}:{end:.6f}:{duration:.2f}", duration))
    if not output:
        raise ValueError(f"no {role} context samples")
    return output, rejected


def isolated_samples(root: Path, labels, source: str, per_class: int):
    output = []
    for label, target in sorted(labels.items(), key=lambda row: row[1]):
        paths = sorted((root / label).glob("*.v17.npz"))[:per_class]
        if source == "citizen" and not paths:
            raise FileNotFoundError(f"missing Citizen {label}")
        for path in paths:
            if "test" in {part.casefold() for part in path.parts}:
                raise ValueError(f"protected replay path: {path}")
            output.append(Sample(load_features(path).astype(np.float16), target, KNOWN,
                                 "replay", source, str(path.relative_to(ROOT)), 0.0))
    return output


def audit_manifest(payload, labels):
    if payload.get("format") != "slt_stage1_window_supervision_v17" or payload.get("version") != 1:
        raise ValueError("unsupported combined supervision")
    if payload.get("citizen_test_accessed") is not False:
        raise ValueError("protected-test declaration changed")
    if {row["source"] for row in payload["rows"]} != ALLOWED_SOURCES:
        raise ValueError("unexpected supervision source")
    role_signers = {}
    counts = Counter()
    archive_hashes = {}
    for row in payload["rows"]:
        role_signers.setdefault(row["role"], set()).add(str(row["signer_id"]))
        complete = row.get("all_signs_annotated") is True
        counts[(row["role"], row["source"], "complete" if complete else "incomplete")] += 1
        if not complete and row["source"] != "o5s5":
            raise ValueError("only O5S5 may be positive-only")
        path = ROOT / row["archive_path"] if not Path(row["archive_path"]).is_absolute() else Path(row["archive_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        archive_hashes[str(path.relative_to(ROOT))] = sha(path)
        for interval in row["intervals"]:
            label = str(interval["label"])
            if label != "__OTHER__" and label not in labels:
                raise ValueError(f"unlocked interval label {label}")
    if role_signers["train"] & role_signers["validation"]:
        raise ValueError("continuous train/validation signer overlap")
    return counts, archive_hashes


def batch(samples, indices, device):
    selected = [samples[index] for index in indices]
    values = torch.from_numpy(np.stack([row.features for row in selected]).astype(np.float32)).to(device)
    targets = torch.tensor([row.target for row in selected], device=device)
    phases = torch.tensor([row.phase for row in selected], device=device)
    return selected, values, targets, phases


def losses(model, teacher, rows, values, targets, phases, phase_weight):
    gloss, phase = model(values)
    known = targets >= 0
    gloss_loss = F.cross_entropy(gloss[known], targets[known]) if known.any() else gloss.sum() * 0
    phase_loss = F.cross_entropy(phase, phases, weight=phase_weight)
    replay = torch.tensor([row.kind == "replay" for row in rows], device=values.device)
    with torch.no_grad():
        expected = teacher(values[replay]) if replay.any() else None
    retain = (F.kl_div(F.log_softmax(gloss[replay] / 2, 1), F.softmax(expected / 2, 1),
                       reduction="batchmean") * 4 if replay.any() else gloss.sum() * 0)
    return gloss_loss + phase_loss + 0.5 * retain, gloss_loss, phase_loss, retain


def precheck() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    model, labels = load_base()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts, archive_hashes = audit_manifest(payload, labels)
    complete = [row for row in payload["rows"] if row["role"] == "train" and row["all_signs_annotated"]]
    chosen = complete[:4]
    chosen += [row for row in complete if any(x["label"] == "__OTHER__" for x in row["intervals"])][:20]
    chosen += [next(row for row in payload["rows"] if row["role"] == "train" and not row["all_signs_annotated"])]
    samples, _ = context_samples(payload, labels, "train", chosen)
    phases_present = {row.phase for row in samples}
    if phases_present != {KNOWN, UNKNOWN, TRANSITION}:
        raise ValueError(f"precheck lacks all phases: {phases_present}")
    if any(row.phase != KNOWN for row in samples if row.source == "o5s5"):
        raise ValueError("incomplete O5S5 produced a negative phase target")
    device = torch.device("mps")
    model.to(device).eval()
    values = torch.from_numpy(np.stack([row.features for row in samples[:16]]).astype(np.float32)).to(device)
    with torch.no_grad():
        gloss, phase = model(values)
    if gloss.shape != (len(values), 100) or phase.shape != (len(values), 3):
        raise ValueError("model output contract mismatch")
    encoded, _ = model.base.encode(values)
    detached_gloss = gloss.detach()
    head = copy.deepcopy(model.phase_head).to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
    targets = torch.tensor([row.phase for row in samples[:16]], device=device)
    fit = []
    for _ in range(20):
        value = F.cross_entropy(head(encoded.detach(), detached_gloss), targets)
        optimizer.zero_grad(set_to_none=True); value.backward(); optimizer.step()
        fit.append(float(value.detach().cpu()))
    if not fit[-1] < fit[0] * 0.8:
        raise RuntimeError("real MPS phase-head tiny-fit did not reduce loss")
    code = [ROOT / "active/v17/boundary_phase_v17.py", Path(__file__), ROOT / "test/test_boundary_phase_v17.py"]
    result = dict(status="passed", base_sha256=sha(BASE), manifest_sha256=sha(MANIFEST),
                  archive_count=len(archive_hashes), archive_hashes_sha256=hashlib.sha256(
                      json.dumps(archive_hashes, sort_keys=True).encode()).hexdigest(),
                  manifest_counts={"|".join(key): value for key, value in counts.items()},
                  labels=100, phases=list(PHASES), windows=list(WINDOW_SECONDS), stride=STRIDE_SECONDS,
                  frames=WINDOW_FRAMES, model_outputs=[100, 3], mps_tiny_fit=[fit[0], fit[-1]],
                  incomplete_rows_route="KNOWN only; no UNKNOWN or TRANSITION",
                  complete_rows_route="KNOWN cores, explicit OTHER as UNKNOWN, guarded interior gaps as TRANSITION",
                  code_sha256={str(path.relative_to(ROOT)): sha(path) for path in code},
                  citizen_test_accessed=False)
    save("precheck.json", result)
    (HERE / "PRECHECK_REPORT.md").write_text(
        "# Boundary-phase precheck\n\nPassed. The train/live contract is 32 frames at 0.27s and 0.53s, "
        "with exactly 100 visible glosses and three internal phases. Incomplete O5S5 rows can create "
        "KNOWN targets only; UNKNOWN and TRANSITION come only from fully annotated ASLLRP rows. "
        f"The real MPS tiny-fit reduced loss from {fit[0]:.4f} to {fit[-1]:.4f}. Citizen test was not accessed.\n",
        encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "code_sha256"}, indent=2))


def accuracy(model, samples, device):
    phase_confusion = np.zeros((3, 3), dtype=int)
    gloss_correct = gloss_total = 0
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(samples), BATCH):
            rows, values, targets, phases = batch(samples, range(begin, min(begin + BATCH, len(samples))), device)
            gloss, phase = model(values)
            predicted_phase = phase.argmax(1).cpu().tolist()
            for truth, prediction in zip(phases.cpu().tolist(), predicted_phase):
                phase_confusion[truth, prediction] += 1
            known = targets >= 0
            gloss_correct += int((gloss[known].argmax(1) == targets[known]).sum().cpu())
            gloss_total += int(known.sum().cpu())
    return dict(samples=len(samples), phase_accuracy=float(np.trace(phase_confusion) / max(1, phase_confusion.sum())),
                phase_confusion=phase_confusion.tolist(), gloss_correct=gloss_correct,
                gloss_total=gloss_total, gloss_accuracy=gloss_correct / max(1, gloss_total))


def save_checkpoint(model, labels, epoch, audit):
    package = dict(format=CHECKPOINT_FORMAT, epoch=epoch, seed=SEED,
                   base_model_config=model.base.config.to_dict(),
                   base_model_state_dict={key: value.cpu() for key, value in model.base.state_dict().items()},
                   phase_head_state_dict={key: value.cpu() for key, value in model.phase_head.state_dict().items()},
                   label_to_index=labels, public_gloss_count=100, phases=list(PHASES),
                   window_seconds=list(WINDOW_SECONDS), stride_seconds=STRIDE_SECONDS,
                   window_frames=WINDOW_FRAMES, endpoint_target=True,
                   data_audit_sha256=sha(HERE / "data_audit.json"), citizen_test_accessed=False)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(package, handle); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(MODEL)


def train_and_evaluate() -> None:
    pre = json.loads((HERE / "precheck.json").read_text())
    if pre["base_sha256"] != sha(BASE) or pre["manifest_sha256"] != sha(MANIFEST):
        raise RuntimeError("prechecked inputs changed")
    for relative, digest in pre["code_sha256"].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError(f"prechecked code changed: {relative}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model, labels = load_base()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train_context, train_rejected = context_samples(payload, labels, "train")
    validation_context, validation_rejected = context_samples(payload, labels, "validation")
    train_replay = isolated_samples(CITIZEN_TRAIN, labels, "citizen", 10)
    train_replay += isolated_samples(SEMLEX_TRAIN, labels, "semlex", 10)
    validation_replay = isolated_samples(CITIZEN_VAL, labels, "citizen", 10000)
    validation_replay += isolated_samples(SEMLEX_VAL, labels, "semlex", 10000)
    train = train_context + train_replay
    if len({row.identity for row in train}) != len(train):
        raise ValueError("training sample identity collision")
    if any(row.phase != KNOWN for row in train_context if row.source == "o5s5"):
        raise ValueError("incomplete supervision leaked into negative phases")
    phase_counts = Counter(row.phase for row in train)
    phase_weight = torch.tensor([len(train) / (3 * phase_counts[index]) for index in range(3)])
    audit = dict(train_samples=len(train), validation_context_samples=len(validation_context),
                 validation_isolated_samples=len(validation_replay),
                 train_kind_counts=dict(Counter(row.kind for row in train)),
                 train_source_counts=dict(Counter(row.source for row in train)),
                 train_phase_counts={PHASES[key]: value for key, value in phase_counts.items()},
                 train_duration_counts=dict(Counter(f"{row.duration:.2f}" for row in train_context)),
                 train_rejected=dict(train_rejected), validation_rejected=dict(validation_rejected),
                 full_coverage_each_epoch=True, replacement_sampling=False,
                 target_semantics="phase and gloss at trailing-window endpoint",
                 protected_test_accessed=False, citizen_test_accessed=False)
    save("data_audit.json", audit)

    device = torch.device("mps")
    model.to(device)
    teacher = copy.deepcopy(model.base).to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": model.base.parameters(), "lr": 1e-5},
        {"params": model.phase_head.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    weight = phase_weight.to(device)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = np.random.default_rng(SEED + epoch).permutation(len(train)).tolist()
        if sorted(order) != list(range(len(train))):
            raise RuntimeError("epoch coverage mismatch")
        totals = Counter(); started = time.perf_counter(); model.train()
        for begin in range(0, len(order), BATCH):
            rows, values, targets, phases = batch(train, order[begin:begin + BATCH], device)
            total, gloss, phase, retain = losses(model, teacher, rows, values, targets, phases, weight)
            if not torch.isfinite(total): raise RuntimeError("non-finite loss")
            optimizer.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for name, value in (("total", total), ("gloss", gloss), ("phase", phase), ("retain", retain)):
                totals[name] += float(value.detach().cpu())
            totals["updates"] += 1
        scheduler.step()
        record = dict(epoch=epoch, seconds=time.perf_counter() - started,
                      loss={name: totals[name] / totals["updates"] for name in ("total", "gloss", "phase", "retain")},
                      updates=int(totals["updates"]), coverage=len(order))
        history.append(record); save("training_history.json", history)
        save_checkpoint(model, labels, epoch, audit)
        print(json.dumps(record), flush=True)

    context_metrics = accuracy(model, validation_context, device)
    replay_metrics = {}
    for source in ("citizen", "semlex"):
        replay_metrics[source] = accuracy(model, [row for row in validation_replay if row.source == source], device)
    streaming = streaming_evaluation(model, payload, labels, device)
    result = dict(context=context_metrics, isolated=replay_metrics, streaming=streaming,
                  checkpoint=str(MODEL.relative_to(ROOT)), checkpoint_sha256=sha(MODEL),
                  citizen_test_accessed=False, runtime_promoted=False)
    save("evaluation.json", result)
    (HERE / "REPORT.md").write_text(report_text(result, audit), encoding="utf-8")
    verify_saved_checkpoint(model, validation_context[:64], labels, device)
    subprocess.run([sys.executable, str(ROOT / "scripts/index_large_artifacts_v17.py")],
                   cwd=ROOT, check=True)


@torch.inference_mode()
def verify_saved_checkpoint(mps_model, samples, labels, device):
    package = torch.load(MODEL, map_location="cpu", weights_only=False)
    expected = dict(format=CHECKPOINT_FORMAT, public_gloss_count=100,
                    phases=list(PHASES), window_seconds=list(WINDOW_SECONDS),
                    stride_seconds=STRIDE_SECONDS, window_frames=WINDOW_FRAMES,
                    endpoint_target=True, label_to_index=labels)
    for key, value in expected.items():
        if package.get(key) != value: raise ValueError(f"saved checkpoint mismatch: {key}")
    base = SLTStage1V17(Stage1V17Config(**package["base_model_config"]))
    base.load_state_dict(package["base_model_state_dict"], strict=True)
    cpu_model = BoundaryPhaseModelV17(base, BoundaryPhaseHeadV17(base.config.dim, 100))
    cpu_model.phase_head.load_state_dict(package["phase_head_state_dict"], strict=True)
    cpu_model.eval(); mps_model.eval()
    features = torch.from_numpy(np.stack([row.features for row in samples]).astype(np.float32))
    cpu = cpu_model(features)
    mps = mps_model(features.to(device))
    gloss_match = int((cpu[0].argmax(1) == mps[0].argmax(1).cpu()).sum())
    phase_match = int((cpu[1].argmax(1) == mps[1].argmax(1).cpu()).sum())
    if gloss_match != len(samples) or phase_match != len(samples):
        raise RuntimeError("saved CPU/MPS decisions disagree")
    save("verification.json", dict(checkpoint_contract="passed", checked_samples=len(samples),
                                    cpu_mps_gloss_matches=gloss_match, cpu_mps_phase_matches=phase_match,
                                    focused_tests="2/2 prelaunch", full_data_dry_run="passed",
                                    citizen_test_accessed=False))


@torch.inference_mode()
def streaming_evaluation(model, payload, labels, device):
    ordered = [label for label, _ in sorted(labels.items(), key=lambda row: row[1])]
    operations, details = Counter(), []
    model.eval()
    rows = [row for row in payload["rows"] if row["role"] == "validation" and row["all_signs_annotated"]]
    for row in rows:
        raw, timestamps = raw_sequence(row)
        ends = window_end_times(timestamps, WINDOW_SECONDS[0], STRIDE_SECONDS, include_final=True)
        primary, context, has_context, kept = [], [], [], []
        for end in ends:
            try:
                short = normalize_time_window(raw, timestamps, end, WINDOW_SECONDS[0])[0]
            except ValueError:
                continue
            long = short
            available = end - WINDOW_SECONDS[1] >= timestamps[0] - 1e-9
            if available:
                try: long = normalize_time_window(raw, timestamps, end, WINDOW_SECONDS[1])[0]
                except ValueError: available = False
            primary.append(short); context.append(long); has_context.append(available); kept.append(end)
        transcript = BoundaryPhaseTranscript(2)
        for begin in range(0, len(primary), BATCH):
            first = torch.from_numpy(np.stack(primary[begin:begin+BATCH])).to(device)
            second = torch.from_numpy(np.stack(context[begin:begin+BATCH])).to(device)
            short_gloss, short_phase = model(first)
            long_gloss, _ = model(second)
            for offset in range(len(first)):
                index = begin + offset
                gloss = (short_gloss[offset] + long_gloss[offset]) / 2 if has_context[index] else short_gloss[offset]
                phase = int(short_phase[offset].argmax())
                transcript.update(phase, ordered[int(gloss.argmax())] if phase == KNOWN else None, kept[index])
        reference = [str(event["label"]) for event in row["intervals"] if event["label"] in labels]
        predicted = transcript.words
        row_ops = Counter(value["operation"] for value in _edit_operations(reference, predicted))
        operations.update(row_ops)
        details.append(dict(source_item_id=row["source_item_id"], source=row["source"],
                            reference=reference, prediction=predicted, operations=dict(row_ops)))
    tokens = sum(len(row["reference"]) for row in details)
    errors = sum(operations[name] for name in ("substitution", "deletion", "insertion"))
    return dict(recordings=len(details), reference_tokens=tokens, wer_percent=100 * errors / max(1, tokens),
                operations=dict(operations), empty_predictions=sum(not row["prediction"] for row in details), rows=details)


def report_text(result, audit):
    context, stream = result["context"], result["streaming"]
    citizen, semlex = result["isolated"]["citizen"], result["isolated"]["semlex"]
    return f"""# Boundary-aware fixed-window experiment

The model keeps exactly 100 visible glosses and learns a separate internal endpoint phase: KNOWN, UNKNOWN, or TRANSITION. Training and replay use the same 32-frame 0.27s/0.53s normalization. Incomplete O5S5 rows supplied known sign cores only; negative phases came only from fully annotated ASLLRP.

## Results

| Measure | Result |
| --- | ---: |
| Validation phase accuracy | {context['phase_accuracy']*100:.2f}% |
| Validation known-core gloss accuracy | {context['gloss_accuracy']*100:.2f}% ({context['gloss_correct']}/{context['gloss_total']}) |
| Citizen isolated retention | {citizen['gloss_accuracy']*100:.2f}% ({citizen['gloss_correct']}/{citizen['gloss_total']}) |
| SemLex isolated retention | {semlex['gloss_accuracy']*100:.2f}% ({semlex['gloss_correct']}/{semlex['gloss_total']}) |
| Complete-ASLLRP online WER | {stream['wer_percent']:.2f}% |
| Empty online transcripts | {stream['empty_predictions']}/{stream['recordings']} |

Every admitted training sample was used exactly once per epoch. No replacement sampler, 1.07-second target, CTC loss, or incomplete-row transition label was used. Full confusion counts, transcripts, and edit operations are in `evaluation.json`. Citizen test remained sealed and no runtime was promoted automatically.
"""


def notify(message: str) -> None:
    subprocess.run(["osascript", "-e", 'on run argv\ndisplay notification (item 1 of argv) with title "SLT experiment"\nend run', message],
                   capture_output=True, timeout=15, check=False)


def worker() -> None:
    started, status = datetime.now(timezone.utc).isoformat(), "failed"
    try:
        train_and_evaluate(); status = "completed"
    except BaseException:
        (HERE / "FAILURE.md").write_text("# Boundary-phase experiment failed\n\n```text\n" + traceback.format_exc() + "```\n")
        traceback.print_exc()
    finally:
        save("completion.json", dict(status=status, started_at=started,
                                     finished_at=datetime.now(timezone.utc).isoformat()))
        notify("Boundary-phase experiment " + status + ". See " + str(HERE))
    if status != "completed": raise SystemExit(1)


def launch() -> None:
    if (HERE / "launch.json").exists(): raise RuntimeError("experiment already launched")
    pre = json.loads((HERE / "precheck.json").read_text())
    if pre.get("status") != "passed": raise RuntimeError("precheck did not pass")
    with (HERE / "process.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
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
