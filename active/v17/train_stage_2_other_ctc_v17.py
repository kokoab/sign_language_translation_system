#!/usr/bin/env python3
"""Adapt the selected v17 CTC head to full ASLLRP spans with OTHER."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from active.v17.model_stage2_v17 import (
    Stage2TemporalHeadV17,
    Stage2V17Config,
    load_stage2_general_ctc_selector,
    make_stage2_checkpoint,
    warm_start_stage2_with_other,
)
from active.v17.train_stage_2_accuracy_repair_v17 import IsolatedPoolDataset
from active.v17.train_stage_2_v17 import (
    RealPhraseDataset,
    SyntheticCompositionDataset,
    collate,
    edit_distance,
)


LOG = logging.getLogger("train_stage_2_other_ctc_v17")
OTHER_CLASS_INDEX = 100
OTHER_CTC_INDEX = 101
EXPECTED_ENCODER_SHA256 = "1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b"
EXPECTED_EXPERIMENT_MANIFEST_SHA256 = "d120d9747ed01bc7d7cc0d68d2609a25114c75826484536ee942c278820d4af0"
EXPECTED_WARM_START_SHA256 = "6a72ca836247fe8717e0b4a7f930b11b291aa2889b49cc839a9fbc6d86d6cf8e"
EXPECTED_SELECTOR_SHA256 = "0782d052f0500164a2433ebfee86dcce7413c6bcffca03fae379871ece86dc3d"
BASELINE_EXPECTATIONS = {
    "target_edits": 603, "target_tokens": 284,
    "local_edits": 6, "local_tokens": 259,
    "exact_edits": 9, "exact_tokens": 24,
    "contextual_edits": 43, "contextual_tokens": 254,
    "citizen_correct": 331, "citizen_samples": 378,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(sha256(path).encode() + b"\n")
    return digest.hexdigest()


class CombinedDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.offsets = []
        total = 0
        for dataset in datasets:
            self.offsets.append(total)
            total += len(dataset)
        self.length = total

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        for dataset, offset in reversed(list(zip(self.datasets, self.offsets))):
            if index >= offset:
                return dataset[index - offset]
        raise IndexError(index)


def collapse_ctc(sequence: np.ndarray) -> list[int]:
    output = []
    previous = None
    for value in sequence.tolist():
        token = int(value)
        if token != previous and token != 0:
            output.append(token - 1)
        previous = token
    return output


def metric_accumulator():
    return {"edits": 0, "tokens": 0, "exact": 0, "samples": 0}


def add_metric(stats, reference, hypothesis):
    stats["edits"] += edit_distance(reference, hypothesis)
    stats["tokens"] += len(reference)
    stats["exact"] += int(reference == hypothesis)
    stats["samples"] += 1


def finish_metric(stats):
    return {
        **stats,
        "wer": stats["edits"] / max(1, stats["tokens"]),
        "sequence_accuracy": stats["exact"] / max(1, stats["samples"]),
    }


def evaluate(model, loader, device):
    model.eval()
    full = defaultdict(metric_accumulator)
    target_only = defaultdict(metric_accumulator)
    with torch.inference_mode():
        for batch in loader:
            logits, lengths = model(
                batch["features"].to(device), batch["window_mask"].to(device)
            )
            predictions = logits.argmax(dim=-1).cpu().numpy()
            flat_targets = batch["targets"].tolist()
            offset = 0
            for index, (source, length, target_length) in enumerate(zip(
                batch["sources"], lengths.cpu().tolist(), batch["target_lengths"].tolist()
            )):
                reference = [
                    int(value) - 1
                    for value in flat_targets[offset:offset + target_length]
                ]
                offset += target_length
                hypothesis = collapse_ctc(predictions[index, :length])
                add_metric(full[source], reference, hypothesis)
                clean_reference = [value for value in reference if value != OTHER_CLASS_INDEX]
                clean_hypothesis = [value for value in hypothesis if value != OTHER_CLASS_INDEX]
                add_metric(target_only[source], clean_reference, clean_hypothesis)
    return {
        "full": {key: finish_metric(value) for key, value in sorted(full.items())},
        "target_only": {
            key: finish_metric(value) for key, value in sorted(target_only.items())
        },
    }


def selection_key(metrics):
    target = metrics["target_only"]
    local = target["local_phrases"]
    sparse = target["asllrp_contiguous"]
    natural = target["asllrp_other_ctc"]
    natural_full = metrics["full"]["asllrp_other_ctc"]
    old_guard = int(local["edits"] <= 7 and sparse["edits"] <= 11)
    mean_wer = np.mean([
        local["wer"], sparse["wer"], natural["wer"], natural_full["wer"],
    ])
    return (
        old_guard,
        -float(mean_wer),
        -float(natural_full["wer"]),
        -float(sparse["wer"]),
        -float(natural["wer"]),
        -float(local["wer"]),
    )


def sampling_weights(old, natural, synthetic, masses):
    samples = old.samples + natural.samples
    counts = Counter(sample.source for sample in samples)
    counts.update(str(row["source"]) for row in synthetic.rows)
    if set(counts) != set(masses):
        raise ValueError(f"unexpected training sources: {dict(counts)}")
    weights = [masses[sample.source] / counts[sample.source] for sample in samples]
    weights.extend(
        masses[str(row["source"])] / counts[str(row["source"])]
        for row in synthetic.rows
    )
    return torch.as_tensor(weights, dtype=torch.double), counts


def distillation_loss(student_logits, teacher_logits, lengths, replay_mask, temperature):
    if not replay_mask.any():
        return student_logits.sum() * 0.0
    token_mask = torch.arange(student_logits.shape[1], device=lengths.device)[None] < lengths[:, None]
    mask = token_mask & replay_mask[:, None]
    student = student_logits / temperature
    teacher = teacher_logits / temperature
    values = F.kl_div(
        student.log_softmax(dim=-1)[..., :teacher.shape[-1]],
        teacher.softmax(dim=-1), reduction="none",
    ).sum(dim=-1)
    return values[mask].mean() * temperature * temperature


def replay_distillation_mask(sources) -> torch.Tensor:
    return torch.as_tensor([
        source not in {"asllrp_other_ctc", "asl_stem_wiki_verified_interval"}
        for source in sources
    ], dtype=torch.bool)


def transition_masses(arm: str) -> dict[str, float]:
    masses = {
        "asllrp_other_ctc": .40,
        "local_phrases": .20,
        "asllrp_contiguous": .10,
        "isolated_citizen_train": .20 if arm == "with_stem" else .30,
    }
    if arm == "with_stem":
        masses["asl_stem_wiki_verified_interval"] = .10
    elif arm != "no_stem":
        raise ValueError(f"unknown transition arm: {arm}")
    return masses


def _sample_metadata(dataset) -> list[tuple[str, int | tuple[int, ...], str | None]]:
    if isinstance(dataset, CombinedDataset):
        return [row for child in dataset.datasets for row in _sample_metadata(child)]
    if isinstance(dataset, IsolatedPoolDataset):
        return [
            (dataset.source, int(target), None)
            for target in dataset.targets.tolist()
        ]
    rows = []
    for sample in dataset.samples:
        participant = None
        if sample.source == "asl_stem_wiki_verified_interval":
            parts = sample.item_id.split(":")
            participant = parts[1] if len(parts) >= 3 else None
        target = tuple(int(value) for value in sample.targets.tolist())
        rows.append((
            sample.source,
            target[0] if sample.source == "asl_stem_wiki_verified_interval" else "__all__",
            participant,
        ))
    return rows


def transition_sampling_weights(dataset, masses):
    metadata = _sample_metadata(dataset)
    counts = Counter(source for source, _, _ in metadata)
    if set(counts) != set(masses) or not np.isclose(sum(masses.values()), 1.0):
        raise ValueError(f"transition source/mass mismatch: {dict(counts)} {masses}")
    grouped = Counter(metadata)
    classes: dict[str, set] = defaultdict(set)
    participants: dict[tuple[str, object], set] = defaultdict(set)
    for source, target, participant in metadata:
        classes[source].add(target)
        participants[(source, target)].add(participant)
    weights = [
        masses[source]
        / len(classes[source])
        / len(participants[(source, target)])
        / grouped[(source, target, participant)]
        for source, target, participant in metadata
    ]
    return torch.as_tensor(weights, dtype=torch.double), counts


def ctc_lengths_are_feasible(targets, target_lengths, input_lengths) -> bool:
    offset = 0
    for target_length, input_length in zip(target_lengths.tolist(), input_lengths.tolist()):
        sequence = targets[offset:offset + int(target_length)].tolist()
        offset += int(target_length)
        required = len(sequence) + sum(a == b for a, b in zip(sequence, sequence[1:]))
        if required > int(input_length):
            return False
    return offset == len(targets)


def configure_trainable(model: Stage2TemporalHeadV17, epoch: int) -> None:
    projection_only = epoch <= 2
    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            not projection_only
            or name.startswith("input_projection.")
            or name.startswith("ctc_head.")
        )


def transition_optimizer(model, projection_lr, backbone_lr, weight_decay):
    projection = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith("input_projection.") or name.startswith("ctc_head.")
    ]
    projection_ids = {id(parameter) for parameter in projection}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in projection_ids]
    return torch.optim.AdamW([
        {"params": projection, "lr": projection_lr},
        {"params": backbone, "lr": backbone_lr},
    ], weight_decay=weight_decay)


def build_transition_student(initialization) -> Stage2TemporalHeadV17:
    source_config = Stage2V17Config(**initialization["model_config"])
    if source_config.num_classes != 100 or source_config.blank_index != 0:
        raise ValueError("transition warm-start must be the locked 100-class CTC head")
    student = Stage2TemporalHeadV17(Stage2V17Config(
        **{**source_config.to_dict(), "num_classes": 101}
    ))
    warm_start_stage2_with_other(student, initialization)
    return student


def validate_transition_manifest(path: Path, *, required_roles=None, expected_sha256=None):
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("transition experiment manifest hash changed")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "slt_v17_stage2_transition_adapt" or manifest.get("version") != 1:
        raise ValueError("invalid transition experiment manifest format")
    vocabulary = manifest.get("vocabulary", {})
    if (
        vocabulary.get("blank_index") != 0
        or vocabulary.get("locked_gloss_indices") != "1-100"
        or vocabulary.get("other_index") != 101
    ):
        raise ValueError("transition label indices changed")
    if manifest.get("encoder", {}).get("sha256") != EXPECTED_ENCODER_SHA256:
        raise ValueError("transition encoder hash changed")
    roles = Counter()
    for item in manifest.get("inputs", []):
        input_path = Path(item["path"])
        actual = directory_sha256(input_path) if input_path.is_dir() else sha256(input_path)
        if actual != item.get("sha256"):
            raise ValueError(f"transition input hash changed: {input_path}")
        roles[str(item.get("source_role"))] += 1
    required_roles = set(required_roles or {
        "frozen encoder", "verified STEM supervision", "locked vocabulary",
        "ASLLRP OTHER train/validation replay", "exact/local phrase frozen replay",
        "Citizen official-training replay", "Citizen official-validation retention",
    })
    if not required_roles.issubset(roles):
        raise ValueError(f"transition manifest source roles missing: {sorted(required_roles - roles.keys())}")
    train = set(manifest.get("training_participants", []))
    validation = set(manifest.get("validation_participants", []))
    if not train or not validation or train & validation:
        raise ValueError("transition STEM split roles are invalid")
    return manifest


def validate_cached_dataset(dataset, encoder_sha256: str, allowed_sources: set[str]) -> None:
    if isinstance(dataset, IsolatedPoolDataset):
        if dataset.stage1_checkpoint_sha256 != encoder_sha256:
            raise ValueError("isolated replay encoder hash changed")
        if not set(dataset.targets.tolist()).issubset(set(range(100))):
            raise ValueError("isolated replay label index changed")
        return
    sources = {sample.source for sample in dataset.samples}
    if not sources.issubset(allowed_sources):
        raise ValueError(f"unexpected cached source roles: {sorted(sources)}")


def _require_manifest_input(manifest, path: Path, role: str) -> None:
    matches = [item for item in manifest["inputs"] if item.get("source_role") == role]
    if len(matches) != 1 or Path(matches[0]["path"]) != path:
        raise ValueError(f"transition {role} path is not the frozen manifest input")


def _validate_pool_split(path: Path, expected: str) -> None:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
    actual = metadata.get("source_split", metadata.get("role"))
    if actual != expected:
        raise ValueError(f"{path}: isolated replay split changed: {actual}")


def _validate_stem_manifest(root: Path, manifest) -> None:
    vocabulary_path = Path(str(manifest.get("vocabulary", {}).get("path", "")))
    if not vocabulary_path.is_file():
        raise ValueError("STEM semantic vocabulary is absent")
    vocabulary_rows = json.loads(vocabulary_path.read_text()).get("classes", [])
    labels = [str(row.get("canonical_label", "")).upper() for row in vocabulary_rows]
    if len(labels) != 100 or len(set(labels)) != 100:
        raise ValueError("STEM semantic vocabulary must contain 100 unique labels")
    label_indices = {label: index for index, label in enumerate(labels)}
    for row in manifest["rows"]:
        sequence = [str(label).upper() for label in row["target_sequence"]]
        if row["target_indices"] != [label_indices.get(label) for label in sequence]:
            raise ValueError(f"STEM semantic target mismatch: {row['source_item_id']}")
    expected = {
        row["source_item_id"]: (
            row["role"], tuple(row["target_sequence"]), tuple(row["target_indices"]),
            row["participant"], row["signer_id"],
        )
        for row in manifest["rows"]
    }
    observed = {}
    for path in root.rglob("*.stage2_frozen_v17.npz"):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            target_indices = tuple(int(value) for value in payload["target_indices"].tolist())
        item_id = str(metadata.get("source_item_id"))
        parts = item_id.split(":")
        if len(parts) != 3 or parts[0] != "stem" or not parts[1] or not parts[2].isdigit():
            raise ValueError(f"malformed STEM frozen item identity: {item_id}")
        participant = parts[1]
        role = str(metadata.get("role"))
        role_participants = set(manifest[
            "training_participants" if role == "train" else "validation_participants"
        ]) if role in {"train", "validation"} else set()
        if participant not in role_participants:
            raise ValueError(f"STEM participant outside {role} split: {participant}")
        if metadata.get("participant", participant) != participant:
            raise ValueError(f"STEM participant metadata mismatch: {item_id}")
        if metadata.get("signer_id", participant) != participant:
            raise ValueError(f"STEM signer metadata mismatch: {item_id}")
        if item_id in observed:
            raise ValueError(f"duplicate STEM frozen item: {item_id}")
        observed[item_id] = (
            role, tuple(metadata.get("target_sequence", [])), target_indices,
            participant, participant,
        )
    if observed != expected:
        raise ValueError("STEM frozen archive identities/splits/targets changed")


def validate_frozen_inputs(
    path: Path, experiment_manifest: Path, stem_root: Path, context_root: Path
):
    payload = json.loads(path.read_text())
    if (
        payload.get("format") != "slt_stage2_transition_frozen_inputs_v17"
        or payload.get("version") != 1
        or payload.get("encoder_sha256") != EXPECTED_ENCODER_SHA256
    ):
        raise ValueError("invalid transition frozen-input sidecar")
    if (
        Path(payload.get("experiment_manifest", "")).resolve()
        != experiment_manifest.resolve()
        or payload.get("experiment_manifest_sha256") != sha256(experiment_manifest)
    ):
        raise ValueError("frozen inputs do not pin the exact experiment manifest")
    expected = {
        "reviewed STEM train/validation frozen features": stem_root.resolve(),
        "ASLLRP contextual-sign validation frozen features": context_root.resolve(),
    }
    inputs = payload.get("inputs", [])
    if len(inputs) != len(expected):
        raise ValueError("frozen inputs must contain exactly two semantic entries")
    for role, root in expected.items():
        matches = [row for row in inputs if row.get("source_role") == role]
        if len(matches) != 1 or Path(matches[0].get("path", "")).resolve() != root:
            raise ValueError(f"frozen input semantic entry changed: {role}")
        row = matches[0]
        archive_count = sum(1 for _ in root.rglob("*.stage2_frozen_v17.npz"))
        if row.get("sha256") != directory_sha256(root) or row.get("archives") != archive_count:
            raise ValueError(f"frozen input hash/count changed: {root}")
    return payload


def transition_defaults(args):
    fixed = {
        "epochs": 20, "patience": 5, "samples_per_epoch": 1800,
        "batch_size": 16, "head_epochs": 2, "lr": 1e-4,
        "backbone_lr": 5e-6, "weight_decay": .02,
        "distill_weight": 1.0, "temperature": 2.0,
        "mps_memory_fraction": .12,
    }
    for name, value in fixed.items():
        setattr(args, name, value)
    if args.arm == "matched":
        args.seeds = (1701, 1702)
    if not args.seeds or any(seed not in (1701, 1702) for seed in args.seeds):
        raise ValueError("transition seeds are locked to 1701 and 1702")
    return args


def _edit_operations(reference: list[int], hypothesis: list[int]):
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    dp = [[0] * columns for _ in range(rows)]
    back = [[None] * columns for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0], back[i][0] = i, "deletion"
    for j in range(1, columns):
        dp[0][j], back[0][j] = j, "insertion"
    for i in range(1, rows):
        for j in range(1, columns):
            choices = [
                (dp[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]),
                 "match" if reference[i - 1] == hypothesis[j - 1] else "substitution"),
                (dp[i - 1][j] + 1, "deletion"),
                (dp[i][j - 1] + 1, "insertion"),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda value: value[0])
    operations = []
    i, j = len(reference), len(hypothesis)
    while i or j:
        operation = back[i][j]
        operations.append({"operation": operation, "reference_position": i - 1 if i else None})
        if operation in {"match", "substitution"}: i, j = i - 1, j - 1
        elif operation == "deletion": i -= 1
        else: j -= 1
    return list(reversed(operations))


def _metric_summary(examples):
    edits = substitutions = deletions = insertions = repeats = exact = tokens = 0
    position = Counter()
    duration = Counter()
    for row in examples:
        reference, hypothesis = row["reference"], row["prediction"]
        operations = _edit_operations(reference, hypothesis)
        substitutions += sum(op["operation"] == "substitution" for op in operations)
        deletions += sum(op["operation"] == "deletion" for op in operations)
        insertions += sum(op["operation"] == "insertion" for op in operations)
        edits += sum(op["operation"] != "match" for op in operations)
        exact += reference == hypothesis
        tokens += len(reference)
        repeats += abs(
            sum(a == b for a, b in zip(reference, reference[1:]))
            - sum(a == b for a, b in zip(hypothesis, hypothesis[1:]))
        )
        duration[f"{row['windows']}_windows"] += int(reference != hypothesis)
        for op in operations:
            if op["operation"] == "match" or op["reference_position"] is None:
                continue
            where = op["reference_position"]
            position["first" if where == 0 else "last" if where == len(reference) - 1 else "middle"] += 1
    return {
        "edits": edits, "tokens": tokens, "wer": edits / max(1, tokens),
        "exact": exact, "samples": len(examples),
        "sequence_accuracy": exact / max(1, len(examples)),
        "substitutions": substitutions, "deletions": deletions,
        "insertions": insertions, "repeated_sign_errors": repeats,
        "position_error_buckets": dict(position),
        "duration_error_buckets": dict(duration),
    }


def evaluate_transition(model, loader, device):
    model.eval()
    examples = defaultdict(list)
    with torch.inference_mode():
        for batch in loader:
            logits, lengths = model(batch["features"].to(device), batch["window_mask"].to(device))
            predictions = logits.argmax(-1).cpu().numpy()
            offset = 0
            for index, (source, length, target_length) in enumerate(zip(
                batch["sources"], lengths.cpu().tolist(), batch["target_lengths"].tolist()
            )):
                reference = [int(value) - 1 for value in batch["targets"][offset:offset + target_length].tolist()]
                offset += target_length
                hypothesis = collapse_ctc(predictions[index, :length])
                name = {
                    "asllrp_other_ctc": "target",
                    "local_phrases": "local",
                    "asllrp_contiguous": "exact",
                    "asllrp_segmented_validation": "contextual",
                    "isolated_citizen_validation": "citizen",
                    "asl_stem_wiki_verified_interval": "stem",
                }[source]
                row = {
                    "item_id": batch["item_ids"][index], "source": source,
                    "reference": reference, "prediction": hypothesis,
                    "windows": int(batch["window_mask"][index].sum()),
                }
                if name == "target":
                    examples["target_full"].append(row)
                    row = dict(row,
                        reference=[value for value in reference if value != OTHER_CLASS_INDEX],
                        prediction=[value for value in hypothesis if value != OTHER_CLASS_INDEX],
                    )
                examples[name].append(row)
    metrics = {name: _metric_summary(rows) for name, rows in examples.items()}
    flat = {
        "target_edits": metrics["target"]["edits"], "target_tokens": metrics["target"]["tokens"],
        "local_edits": metrics["local"]["edits"], "local_tokens": metrics["local"]["tokens"],
        "exact_edits": metrics["exact"]["edits"], "exact_tokens": metrics["exact"]["tokens"],
        "contextual_edits": metrics["contextual"]["edits"], "contextual_tokens": metrics["contextual"]["tokens"],
        "citizen_correct": metrics["citizen"]["exact"], "citizen_samples": metrics["citizen"]["samples"],
        "stem_correct": metrics.get("stem", {}).get("exact", 0),
        "stem_samples": metrics.get("stem", {}).get("samples", 0),
    }
    return {"summary": flat, "domains": metrics, "examples": dict(examples)}


def eligibility(candidate, baseline) -> bool:
    return bool(
        candidate["target_edits"] <= .90 * baseline["target_edits"]
        and candidate["local_edits"] <= baseline["local_edits"]
        and candidate["exact_edits"] <= baseline["exact_edits"]
        and candidate["contextual_edits"] <= baseline["contextual_edits"]
        and candidate["citizen_correct"] >= math.ceil(
            (baseline["citizen_correct"] / baseline["citizen_samples"] - .01)
            * candidate["citizen_samples"]
        )
    )


def candidate_key(candidate, baseline, epoch):
    retention = candidate["local_edits"] + candidate["exact_edits"] + candidate["contextual_edits"]
    return (
        int(eligibility(candidate, baseline)),
        -candidate["target_edits"] / max(1, candidate["target_tokens"]),
        -retention,
        -epoch,
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_initialization(path: Path, temporal_mix: float = 1.0):
    """Return student initialization and an unchanged 100-class distillation teacher."""
    if not 0.0 <= temporal_mix <= 1.0:
        raise ValueError("temporal mix must be in [0,1]")
    initialization = torch.load(path, map_location="cpu", weights_only=False)
    if initialization.get("format") == "slt_stage2_ctc_v17":
        return initialization, initialization, None
    if initialization.get("format") != "slt_stage2_temporal_pretrain_v17":
        raise ValueError("warm-start must be v17 Stage 2 or temporal pretraining")
    if initialization.get("ctc_head_trained") is not False:
        raise ValueError("temporal pretraining must leave the CTC head frozen")
    teacher_path = Path(str(initialization["base_checkpoint"]))
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)
    if teacher.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("temporal pretraining base is not v17 Stage 2")
    if sha256(teacher_path) != initialization.get("base_checkpoint_sha256"):
        raise ValueError("temporal pretraining base checkpoint hash changed")
    if initialization["model_config"] != teacher["model_config"]:
        raise ValueError("temporal initialization and teacher configurations differ")
    mixed_state = {}
    for name, value in initialization["model_state_dict"].items():
        base = teacher["model_state_dict"][name]
        if name.startswith("ctc_head."):
            if not torch.equal(value, base):
                raise ValueError("temporal pretraining changed the frozen CTC head")
            mixed_state[name] = base
        else:
            mixed_state[name] = base + temporal_mix * (value - base)
    student = dict(initialization)
    student["format"] = "slt_stage2_ctc_v17"
    student["model_state_dict"] = mixed_state
    return student, teacher, {
        "checkpoint": path.as_posix(),
        "checkpoint_sha256": sha256(path),
        "source_split": initialization.get("source_split"),
        "temporal_mix": temporal_mix,
        "teacher": teacher_path.as_posix(),
        "teacher_sha256": sha256(teacher_path),
    }


def _validate_feature_root(root: Path, roles: set[str], expected_encoder: str) -> None:
    counts = Counter()
    for path in root.rglob("*.stage2_frozen_v17.npz"):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            targets = payload["target_indices"].astype(np.int64)
        if metadata.get("stage1_checkpoint_sha256") != expected_encoder:
            raise ValueError(f"{path}: frozen encoder hash changed")
        if metadata.get("source") not in roles or metadata.get("role") not in {"train", "validation"}:
            raise ValueError(f"{path}: frozen source/split role changed")
        if any(int(value) not in range(101) for value in targets):
            raise ValueError(f"{path}: frozen target index changed")
        counts[str(metadata["role"])] += 1
    if not counts:
        raise ValueError(f"no frozen features under {root}")


def _validate_measured_baseline(summary) -> None:
    for name, expected in BASELINE_EXPECTATIONS.items():
        if summary.get(name) != expected:
            raise ValueError(
                f"accepted-selector baseline changed for {name}: "
                f"{summary.get(name)} != {expected}"
            )


def _transition_train_seed(
    seed, arm, train_dataset, weights, validation_loader, initialization,
    teacher, teacher_path, baseline, args, device, output_root,
):
    seed_everything(seed)
    model = build_transition_student(initialization).to(device)
    optimizer = transition_optimizer(
        model, args.lr, args.backbone_lr, args.weight_decay
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, collate_fn=collate,
    )
    initialized = evaluate_transition(model, validation_loader, device)
    history = [{"epoch": 0, "initialized_candidate": initialized}]
    best_key = None
    best_state = best_metrics = None
    best_epoch = 0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        configure_trainable(model, epoch)
        model.train()
        totals = Counter()
        for batch in loader:
            features = batch["features"].to(device)
            window_mask = batch["window_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            replay_mask = replay_distillation_mask(batch["sources"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = model(features, window_mask)
            if not ctc_lengths_are_feasible(targets, target_lengths, lengths):
                raise ValueError("CTC target is infeasible, including repeated-target blanks")
            ctc = criterion(
                logits.float().cpu().log_softmax(-1).transpose(0, 1),
                targets.cpu(), lengths.cpu(), target_lengths.cpu(),
            ).to(device)
            if replay_mask.any():
                replay_indices = torch.nonzero(replay_mask, as_tuple=False).flatten()
                with torch.inference_mode():
                    teacher_logits, teacher_lengths = teacher(
                        features.index_select(0, replay_indices),
                        window_mask.index_select(0, replay_indices),
                    )
                replay_lengths = lengths.index_select(0, replay_indices)
                if not torch.equal(replay_lengths, teacher_lengths):
                    raise RuntimeError("teacher/student temporal lengths differ")
                distill = distillation_loss(
                    logits.index_select(0, replay_indices), teacher_logits,
                    replay_lengths, torch.ones_like(replay_indices, dtype=torch.bool),
                    args.temperature,
                )
            else:
                distill = logits.sum() * 0.0
            loss = ctc + args.distill_weight * distill
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite transition loss seed={seed} epoch={epoch}")
            loss.backward()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in trainable
            ):
                raise RuntimeError(f"non-finite transition gradient seed={seed} epoch={epoch}")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            count = len(window_mask)
            totals["samples"] += count
            totals["ctc"] += float(ctc.detach()) * count
            totals["distill"] += float(distill.detach()) * count
        validation = evaluate_transition(model, validation_loader, device)
        key = candidate_key(validation["summary"], baseline["summary"], epoch)
        history.append({
            "epoch": epoch,
            "ctc_loss": totals["ctc"] / totals["samples"],
            "distill_loss": totals["distill"] / totals["samples"],
            "eligible": bool(key[0]), "selection_key": list(key),
            "validation": validation,
        })
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics = key, epoch, validation
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "transition arm=%s seed=%d epoch=%d ctc=%.5f distill=%.5f key=%s patience=%d",
            arm, seed, epoch, totals["ctc"] / totals["samples"],
            totals["distill"] / totals["samples"], best_key, patience,
        )
        if epoch > 2 and patience >= args.patience:
            break
        if device.type == "mps":
            torch.mps.empty_cache()
    seed_root = output_root / arm / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    checkpoint = make_stage2_checkpoint(
        model, best_state, seed=seed, epoch=best_epoch,
        validation_metrics=best_metrics["summary"],
        selection_key=list(best_key), transition_arm=arm,
        transition_eligible=bool(best_key[0]),
        warm_started_from=args.warm_start.as_posix(),
        warm_started_from_sha256=sha256(args.warm_start),
        distilled_from=teacher_path.as_posix(),
        distilled_from_sha256=sha256(teacher_path),
        experiment_manifest=args.experiment_manifest.as_posix(),
        experiment_manifest_sha256=sha256(args.experiment_manifest),
        frozen_inputs=args.frozen_inputs.as_posix(),
        frozen_inputs_sha256=sha256(args.frozen_inputs),
        encoder_sha256=EXPECTED_ENCODER_SHA256,
        other_class_index=OTHER_CLASS_INDEX, other_ctc_index=OTHER_CTC_INDEX,
        projection_epochs=2, projection_lr=args.lr,
        backbone_lr=args.backbone_lr, training_source_sampling_mass=transition_masses(arm),
        citizen_test_accessed=False, semlex_test_accessed=False,
        local_test_accessed=False, rit_external_evaluation_accessed=False,
    )
    checkpoint_path = seed_root / "best_model.pth"
    torch.save(checkpoint, checkpoint_path)
    history_path = seed_root / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    return {
        "arm": arm, "seed": seed, "best_epoch": best_epoch,
        "eligible": bool(best_key[0]), "selection_key": list(best_key),
        "validation": best_metrics,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "history": history_path.as_posix(),
    }


def run_transition(args):
    args = transition_defaults(args)
    manifest = validate_transition_manifest(
        args.experiment_manifest,
        expected_sha256=EXPECTED_EXPERIMENT_MANIFEST_SHA256,
    )
    frozen_inputs = validate_frozen_inputs(
        args.frozen_inputs, args.experiment_manifest,
        args.stem_root, args.context_validation_root,
    )
    if sha256(args.warm_start) != EXPECTED_WARM_START_SHA256:
        raise ValueError("transition warm-start hash changed")
    if sha256(args.teacher) != EXPECTED_SELECTOR_SHA256:
        raise ValueError("accepted selector teacher hash changed")
    initialization, _, provenance = resolve_initialization(args.warm_start, 1.0)
    teacher, teacher_payload = load_stage2_general_ctc_selector(args.teacher)
    if teacher.config.num_classes != 100 or teacher.config.blank_index != 0:
        raise ValueError("accepted selector label contract changed")
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    old_train = RealPhraseDataset(args.replay_root, "train")
    natural_train = RealPhraseDataset(args.other_root, "train")
    citizen_train = IsolatedPoolDataset(
        args.citizen_train, "isolated_citizen_train", augment_boundaries=False
    )
    stem_train = RealPhraseDataset(args.stem_root, "train")
    old_validation = RealPhraseDataset(args.replay_root, "validation")
    natural_validation = RealPhraseDataset(args.other_root, "validation")
    context_validation = RealPhraseDataset(args.context_validation_root, "validation")
    citizen_validation = IsolatedPoolDataset(
        args.citizen_validation, "isolated_citizen_validation", augment_boundaries=False
    )
    stem_validation = RealPhraseDataset(args.stem_root, "validation")
    _validate_feature_root(args.replay_root, {"local_phrases", "asllrp_contiguous"}, EXPECTED_ENCODER_SHA256)
    _validate_feature_root(args.other_root, {"asllrp_other_ctc"}, EXPECTED_ENCODER_SHA256)
    _validate_feature_root(args.context_validation_root, {"asllrp_segmented_validation"}, EXPECTED_ENCODER_SHA256)
    _validate_feature_root(args.stem_root, {"asl_stem_wiki_verified_interval"}, EXPECTED_ENCODER_SHA256)
    _validate_stem_manifest(args.stem_root, manifest)
    _require_manifest_input(manifest, args.other_root, "ASLLRP OTHER train/validation replay")
    _require_manifest_input(manifest, args.replay_root, "exact/local phrase frozen replay")
    _require_manifest_input(manifest, args.citizen_train, "Citizen official-training replay")
    _require_manifest_input(manifest, args.citizen_validation, "Citizen official-validation retention")
    validate_cached_dataset(citizen_train, EXPECTED_ENCODER_SHA256, set())
    validate_cached_dataset(citizen_validation, EXPECTED_ENCODER_SHA256, set())
    _validate_pool_split(args.citizen_train, "citizen_official_train_only")
    _validate_pool_split(args.citizen_validation, "validation")
    validation = CombinedDataset([
        old_validation, natural_validation, context_validation,
        citizen_validation, stem_validation,
    ])
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    baseline = evaluate_transition(teacher, validation_loader, device)
    _validate_measured_baseline(baseline["summary"])
    arms = ("no_stem", "with_stem") if args.arm == "matched" else (args.arm,)
    output_root = (
        Path("artifacts/models/stage2_v17_transition_adapt_v1")
        if args.output == Path("artifacts/models/stage2_v17_asllrp_other_ctc_v1")
        else args.output
    )
    results = []
    counts_by_arm = {}
    for arm in arms:
        datasets = [old_train, natural_train, citizen_train]
        if arm == "with_stem":
            datasets.append(stem_train)
        train_dataset = CombinedDataset(datasets)
        weights, counts = transition_sampling_weights(train_dataset, transition_masses(arm))
        counts_by_arm[arm] = dict(counts)
        for seed in args.seeds:
            results.append(_transition_train_seed(
                seed, arm, train_dataset, weights, validation_loader,
                initialization, teacher, args.teacher, baseline, args, device, output_root,
            ))
    args.transition_report_root.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "slt_stage2_transition_adapt_training_v17",
        "experiment_manifest": args.experiment_manifest.as_posix(),
        "experiment_manifest_sha256": sha256(args.experiment_manifest),
        "frozen_inputs": args.frozen_inputs.as_posix(),
        "frozen_inputs_sha256": sha256(args.frozen_inputs),
        "frozen_inputs_format": frozen_inputs["format"],
        "manifest_format": manifest["format"],
        "encoder_sha256": EXPECTED_ENCODER_SHA256,
        "warm_start": args.warm_start.as_posix(),
        "warm_start_sha256": sha256(args.warm_start),
        "teacher": args.teacher.as_posix(), "teacher_sha256": sha256(args.teacher),
        "teacher_format": teacher_payload["format"],
        "baseline": baseline, "expected_baseline": BASELINE_EXPECTATIONS,
        "training_counts": counts_by_arm, "results": results,
        "fixed_design": {
            "epochs": args.epochs, "patience": args.patience,
            "samples_per_epoch": args.samples_per_epoch, "batch_size": args.batch_size,
            "projection_epochs": args.head_epochs, "projection_lr": args.lr,
            "backbone_lr": args.backbone_lr, "weight_decay": args.weight_decay,
            "gradient_clip": 1.0, "distill_weight": args.distill_weight,
            "temperature": args.temperature,
        },
        "temporal_initialization": provenance,
        "citizen_test_accessed": False, "semlex_test_accessed": False,
        "local_test_accessed": False, "rit_external_evaluation_accessed": False,
        "test_evaluated": False,
    }
    report_name = (
        "training_result.json" if args.arm == "matched"
        else f"{args.arm}_seed_{'_'.join(str(seed) for seed in args.seeds)}.json"
    )
    report_path = args.transition_report_root / report_name
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["report"] = report_path.as_posix()
    return report


def train_one(
    seed, train_dataset, weights, validation_loader,
    initialization, teacher_checkpoint, args, device,
):
    seed_everything(seed)
    source_config = Stage2V17Config(**initialization["model_config"])
    model = Stage2TemporalHeadV17(Stage2V17Config(
        **{**source_config.to_dict(), "num_classes": 101}
    )).to(device)
    warm_start_stage2_with_other(model, initialization)
    teacher = Stage2TemporalHeadV17(source_config).to(device).eval()
    teacher.load_state_dict(teacher_checkpoint["model_state_dict"], strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    head_parameters = list(model.ctc_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    for parameter in backbone_parameters:
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": args.lr},
        {"params": backbone_parameters, "lr": args.backbone_lr},
    ], weight_decay=args.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    sampler = WeightedRandomSampler(
        weights, num_samples=args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=0, collate_fn=collate,
    )
    baseline = evaluate(model, validation_loader, device)
    best_key = selection_key(baseline)
    best_metrics = baseline
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_epoch = 0
    history = [{"epoch": 0, "validation": baseline, "selection_key": list(best_key)}]
    patience = 0
    peak_mps_current = peak_mps_driver = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_epochs + 1:
            for parameter in backbone_parameters:
                parameter.requires_grad = True
        model.train()
        total_ctc = total_distill = seen = 0.0
        for batch in loader:
            features = batch["features"].to(device)
            window_mask = batch["window_mask"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            replay_mask = torch.as_tensor(
                [source != "asllrp_other_ctc" for source in batch["sources"]],
                dtype=torch.bool, device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, lengths = model(features, window_mask)
            ctc = criterion(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets, lengths, target_lengths,
            )
            with torch.inference_mode():
                teacher_logits, teacher_lengths = teacher(features, window_mask)
            if not torch.equal(lengths, teacher_lengths):
                raise RuntimeError("teacher/student temporal lengths differ")
            distill = distillation_loss(
                logits, teacher_logits, lengths, replay_mask, args.temperature
            )
            loss = ctc + args.distill_weight * distill
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(window_mask)
            total_ctc += float(ctc.detach()) * count
            total_distill += float(distill.detach()) * count
            seen += count
        metrics = evaluate(model, validation_loader, device)
        if device.type == "mps":
            torch.mps.synchronize()
            peak_mps_current = max(
                peak_mps_current, int(torch.mps.current_allocated_memory())
            )
            peak_mps_driver = max(
                peak_mps_driver, int(torch.mps.driver_allocated_memory())
            )
            torch.mps.empty_cache()
        key = selection_key(metrics)
        history.append({
            "epoch": epoch,
            "ctc_loss": total_ctc / seen,
            "distill_loss": total_distill / seen,
            "validation": metrics,
            "selection_key": list(key),
        })
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "seed=%d epoch=%d ctc=%.4f distill=%.4f key=%s patience=%d",
            seed, epoch, total_ctc / seen, total_distill / seen,
            tuple(round(value, 5) for value in key), patience,
        )
        if epoch > args.head_epochs and patience >= args.patience:
            break
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "validation_metrics": best_metrics,
        "state_dict": best_state,
        "history": history,
        "model_config": model.config.to_dict(),
        "peak_mps_current_bytes": peak_mps_current,
        "peak_mps_driver_bytes": peak_mps_driver,
    }


def run(args):
    if getattr(args, "experiment_manifest", None) is not None:
        return run_transition(args)
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    device = torch.device(device_name)
    if device.type == "mps":
        if not 0 < args.mps_memory_fraction <= 0.12:
            raise ValueError("MPS memory fraction must be in (0, 0.12]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    initialization, teacher_checkpoint, temporal_provenance = resolve_initialization(
        args.warm_start, args.temporal_mix
    )
    old_train = RealPhraseDataset(args.replay_root, "train")
    natural_train = RealPhraseDataset(args.other_root, "train")
    synthetic_train = SyntheticCompositionDataset(args.synthetic_pool, args.synthetic_plan)
    old_validation = RealPhraseDataset(args.replay_root, "validation")
    natural_validation = RealPhraseDataset(args.other_root, "validation")
    train_dataset = CombinedDataset([old_train, natural_train, synthetic_train])
    validation_dataset = CombinedDataset([old_validation, natural_validation])
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    masses = {
        "local_phrases": args.local_mass,
        "asllrp_contiguous": args.sparse_asllrp_mass,
        "asllrp_other_ctc": args.natural_asllrp_mass,
        "synthetic_citizen_train": args.synthetic_citizen_mass,
        "synthetic_multivoice_train": args.synthetic_multivoice_mass,
    }
    if not np.isclose(sum(masses.values()), 1.0):
        raise ValueError("training source masses must sum to one")
    weights, counts = sampling_weights(old_train, natural_train, synthetic_train, masses)
    started = time.monotonic()
    results = [
        train_one(
            seed, train_dataset, weights, validation_loader,
            initialization, teacher_checkpoint, args, device,
        )
        for seed in args.seeds
    ]
    winner = max(results, key=lambda value: tuple(value["selection_key"]))
    args.output.mkdir(parents=True, exist_ok=True)
    model = Stage2TemporalHeadV17(Stage2V17Config(**winner["model_config"]))
    selected = make_stage2_checkpoint(
        model, winner.pop("state_dict"), seed=winner["seed"], epoch=winner["best_epoch"],
        validation_metrics=winner["validation_metrics"],
        selection_key=winner["selection_key"],
        warm_started_from=args.warm_start.as_posix(),
        warm_started_from_sha256=sha256(args.warm_start),
        temporal_initialization=temporal_provenance,
        replay_cache_root=args.replay_root.as_posix(),
        other_cache_root=args.other_root.as_posix(),
        other_manifest=args.other_manifest.as_posix(),
        other_manifest_sha256=sha256(args.other_manifest),
        synthetic_pool=args.synthetic_pool.as_posix(),
        synthetic_pool_sha256=sha256(args.synthetic_pool),
        synthetic_plan=args.synthetic_plan.as_posix(),
        synthetic_plan_sha256=sha256(args.synthetic_plan),
        other_class_index=OTHER_CLASS_INDEX,
        other_ctc_index=OTHER_CTC_INDEX,
        decoder_policy="collapse CTC then remove class index 100 (OTHER)",
        training_source_sampling_mass=masses,
        distill_weight=args.distill_weight,
        temperature=args.temperature,
        citizen_test_accessed=False, semlex_test_accessed=False,
        local_test_accessed=False, rit_external_evaluation_accessed=False,
    )
    checkpoint_path = args.output / "best_model.pth"
    torch.save(selected, checkpoint_path)
    for result in results:
        result.pop("state_dict", None)
    report = {
        "format": "slt_stage2_other_ctc_training_result_v17",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "selected_seed": winner["seed"],
        "selected_epoch": winner["best_epoch"],
        "selection_key": winner["selection_key"],
        "validation_metrics": winner["validation_metrics"],
        "training_counts": dict(counts),
        "training_source_sampling_mass": masses,
        "temporal_initialization": temporal_provenance,
        "validation_counts": {
            "legacy": len(old_validation), "natural_asllrp": len(natural_validation),
        },
        "seconds": time.monotonic() - started,
        "device": str(device),
        "candidate_results": results,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "rit_external_evaluation_accessed": False,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--experiment-manifest", type=Path)
    value.add_argument(
        "--frozen-inputs", type=Path,
        default=Path("artifacts/reports/stage2_v17_transition_adapt_v1/frozen_inputs.json"),
    )
    value.add_argument("--arm", choices=("no_stem", "with_stem", "matched"), default="matched")
    value.add_argument(
        "--teacher", type=Path,
        default=Path("artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"),
    )
    value.add_argument(
        "--stem-root", type=Path,
        default=Path("data/local/stage2_v17_transition_adapt_v1/frozen_features"),
    )
    value.add_argument(
        "--citizen-train", type=Path,
        default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"),
    )
    value.add_argument(
        "--citizen-validation", type=Path,
        default=Path("data/local/stage2_v17_isolated_replay/citizen_validation.npz"),
    )
    value.add_argument(
        "--context-validation-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"),
    )
    value.add_argument(
        "--transition-report-root", type=Path,
        default=Path("artifacts/reports/stage2_v17_transition_adapt_v1"),
    )
    value.add_argument("--replay-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    value.add_argument("--other-root", type=Path, default=Path("data/local/stage2_v17_asllrp_other_frozen_features"))
    value.add_argument("--other-manifest", type=Path, default=Path("active/v17/stage2_asllrp_other_ctc_manifest_v17.json"))
    value.add_argument("--warm-start", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_adaptation_v3/best_model.pth"))
    value.add_argument("--temporal-mix", type=float, default=1.0)
    value.add_argument("--synthetic-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"))
    value.add_argument("--synthetic-plan", type=Path, default=Path("active/v17/stage2_multivoice_transfer_plan_v17.json"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_asllrp_other_ctc_v1"))
    value.add_argument("--device", default="auto")
    value.add_argument("--mps-memory-fraction", type=float, default=0.12)
    value.add_argument("--seeds", type=lambda text: tuple(int(value) for value in text.split(",")), default=(1701,))
    value.add_argument("--epochs", type=int, default=40)
    value.add_argument("--head-epochs", type=int, default=5)
    value.add_argument("--patience", type=int, default=10)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--samples-per-epoch", type=int, default=1800)
    value.add_argument("--lr", type=float, default=3e-4)
    value.add_argument("--backbone-lr", type=float, default=4e-5)
    value.add_argument("--weight-decay", type=float, default=0.02)
    value.add_argument("--distill-weight", type=float, default=0.35)
    value.add_argument("--temperature", type=float, default=2.0)
    value.add_argument("--local-mass", type=float, default=0.18)
    value.add_argument("--sparse-asllrp-mass", type=float, default=0.12)
    value.add_argument("--natural-asllrp-mass", type=float, default=0.40)
    value.add_argument("--synthetic-citizen-mass", type=float, default=0.12)
    value.add_argument("--synthetic-multivoice-mass", type=float, default=0.18)
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(parser().parse_args()), indent=2))
