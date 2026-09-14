#!/usr/bin/env python3
"""Prepare and train the bounded contextual Stage-1 window candidate.

Preparation accepts only timestamped v17 sequences whose manifest explicitly says
that every sign, including out-of-vocabulary signs, was annotated.  Training is an
additional explicit action so an audit can never start a model run by accident.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.live_transition_supervision_v17 import interior_gaps
from active.v17.model_reel_emission_v17 import (
    ReelEmissionHeadConfig,
    ReelEmissionHeadV17,
    ReelEmissionStage1V17,
    pool_stage1_encoded,
    reel_temporal_summary,
    load_reel_emission_checkpoint,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.stage1_window_v17 import (
    CHECKPOINT_FORMAT,
    RAW_FORMAT,
    STRIDE_SECONDS,
    WINDOW_FRAMES,
    WINDOW_SECONDS,
    normalize_time_window,
    raw_observation_features,
    window_end_times,
    window_sample_times,
)
from active.v17.train_stage_1_phrase_adapt_v17 import IsolatedReplay, load_features


NO_EMIT_INDEX = 100
NO_EMIT = "__NO_EMIT__"
WINDOW_DURATIONS = (0.27, 0.53, 1.07)
TRANSITION_GUARD_SECONDS = 0.10
CATEGORY_SHARES = {"replay": 0.50, "context": 0.30, "background": 0.20}


@dataclass(frozen=True)
class ContextSample:
    features: np.ndarray
    foreground: np.ndarray
    target: int
    category: str
    source: str
    signer: str
    identity: str
    duration_seconds: float
    schedule: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected(path: Path) -> bool:
    return bool({"test", "external_evaluation_reserved"} & {
        part.casefold() for part in path.parts
    })


def _load_sequence(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if _protected(path):
        raise ValueError(f"refusing protected sequence: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if str(payload["raw_format"].item()) != RAW_FORMAT:
            raise ValueError(f"{path}: globally normalized features are not valid raw observations")
        features = payload["raw_features"].astype(np.float32, copy=False)
        timestamps = payload["timestamps_seconds"].astype(np.float64, copy=False)
    if (
        features.ndim != 3 or features.shape[1:] != (61, 5)
        or timestamps.shape != (len(features),)
        or len(timestamps) < 2 or not np.isfinite(features).all()
        or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError(f"{path}: invalid raw observation arrays")
    return features, timestamps


def audit_existing_supervision(
    supervision: dict[str, object], frozen: dict[str, object]
) -> dict[str, object]:
    """Audit the accepted all-sign intervals before raw observations are cached."""
    if (
        supervision.get("format") != "slt_stage2_live_transition_supervision_v17"
        or supervision.get("version") != 1
    ):
        raise ValueError("unsupported all-sign supervision")
    frozen_rows = {str(row["source_item_id"]): row for row in frozen.get("rows", [])}
    result: dict[str, object] = {}
    role_signers: dict[str, set[str]] = {}
    for role in ("train", "validation"):
        items = [
            (item_id, row) for item_id, row in supervision.get("items", {}).items()
            if row.get("role") == role and row.get("annotation_status") == "available"
        ]
        if not items or any(item_id not in frozen_rows for item_id, _ in items):
            raise ValueError(f"{role} lacks frozen rows for checked supervision")
        signers = {str(frozen_rows[item_id]["signer_id"]) for item_id, _ in items}
        labels = {
            str(event["label"]) for _, row in items for event in row["sign_intervals"]
            if event.get("label") != "__OTHER__"
        }
        pairs = {
            (str(left["label"]), str(right["label"]))
            for _, row in items
            for left, right in zip(row["sign_intervals"], row["sign_intervals"][1:])
            if left.get("label") != "__OTHER__" and right.get("label") != "__OTHER__"
        }
        gaps = [
            gap for _, row in items
            for gap in interior_gaps(row["sign_intervals"], TRANSITION_GUARD_SECONDS)
        ]
        backgrounds = sum(
            max(0, int(np.floor(((end - start) - WINDOW_SECONDS) / STRIDE_SECONDS)) + 1)
            for start, end in gaps
        )
        if not backgrounds:
            raise ValueError(f"{role} has no full verified 0.53-second background window")
        result[role] = {
            "annotated_items": len(items), "distinct_signers": len(signers),
            "distinct_known_signs": len(labels), "distinct_known_sign_pairs": len(pairs),
            "guarded_gap_seconds": sum(end - start for start, end in gaps),
            "potential_full_0_53_second_background_windows": backgrounds,
        }
        role_signers[role] = signers
    if role_signers["train"] & role_signers["validation"]:
        raise ValueError("all-sign supervision train/validation signers overlap")
    result["train_validation_signer_disjoint"] = True
    result["training_launched"] = False
    return result


def materialize_raw_supervision(args: argparse.Namespace) -> dict[str, object]:
    """Replay checked source videos through the live observer into raw arrays."""
    from scripts.diagnose_stage1_window_v17 import recording_observations
    from scripts.cache_stage2_live_matched_v17 import input_contract
    from scripts.live_reel_continuous_v17 import parser as live_parser
    from scripts.extract_stage2_multimodal_v17 import safe_name

    supervision = json.loads(args.existing_supervision.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_inputs.read_text(encoding="utf-8"))
    audit_existing_supervision(supervision, frozen)
    checked = supervision["items"]
    live_args = live_parser().parse_args([
        "--no-display", "--no-speech", "--naturalizer", "literal"
    ])
    observer_contract = input_contract(live_args)
    manifest_rows = []
    for ordinal, row in enumerate(frozen["rows"], 1):
        item_id = str(row["source_item_id"])
        annotation = checked.get(item_id, {})
        if annotation.get("annotation_status") != "available":
            continue
        video = Path(row["video_path"])
        if _protected(video) or _sha256(video) != row["video_sha256"]:
            raise ValueError(f"protected or changed source video: {video}")
        output = args.raw_root / row["role"] / row["source"] / f"{safe_name(row)}.stage1_window_raw_v17.npz"
        metadata = {
            "source_item_id": item_id, "video_path": str(video),
            "video_sha256": row["video_sha256"],
            "observer_contract": observer_contract,
        }
        if output.exists():
            with np.load(output, allow_pickle=False) as payload:
                cached_metadata = json.loads(str(payload["metadata_json"].item()))
                if (
                    str(payload["raw_format"].item()) != RAW_FORMAT
                    or cached_metadata != metadata
                ):
                    raise ValueError(f"existing raw archive provenance changed: {output}")
        else:
            observations, _ = recording_observations({"video": str(video)}, live_args)
            raw, timestamps = raw_observation_features(observations)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary, raw_features=raw, timestamps_seconds=timestamps,
                raw_format=np.array(RAW_FORMAT),
                metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            )
            temporary.replace(output)
        manifest_rows.append({
            "role": row["role"], "source": row["source"],
            "signer_id": row["signer_id"], "source_item_id": item_id,
            "archive_path": str(output), "all_signs_annotated": True,
            "intervals": annotation["sign_intervals"],
            "video_sha256": row["video_sha256"],
        })
        if ordinal % 25 == 0:
            progress = {
                "completed_checked_rows": len(manifest_rows),
                "last_source_item_id": item_id,
                "observer_contract": observer_contract,
            }
            (args.supervision.parent / "materialize_progress.json").write_text(
                json.dumps(progress, indent=2) + "\n", encoding="utf-8"
            )
            print(f"materialized {len(manifest_rows)} checked sequences", flush=True)
    payload = {
        "format": "slt_stage1_window_supervision_v17", "version": 1,
        "raw_format": RAW_FORMAT, "rows": manifest_rows,
        "observer_contract": observer_contract,
        "all_signs_source_sha256": _sha256(args.existing_supervision),
        "frozen_inputs_sha256": _sha256(args.frozen_inputs),
        "citizen_test_accessed": False,
    }
    args.supervision.parent.mkdir(parents=True, exist_ok=True)
    if args.supervision.exists() and json.loads(args.supervision.read_text()) != payload:
        raise ValueError("supervision manifest already exists with different content")
    args.supervision.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"materialized_rows": len(manifest_rows), "supervision": str(args.supervision)}


def _overlaps(interval: dict[str, object], start: float, end: float) -> bool:
    return float(interval["start_seconds"]) < end and float(interval["end_seconds"]) > start


def _sample(
    features: np.ndarray, timestamps: np.ndarray, interval: dict[str, object] | None,
    *, end: float, duration: float, target: int, category: str, source: str,
    signer: str, identity: str, schedule: str,
    other_intervals: list[dict[str, object]] | None = None,
) -> ContextSample | None:
    retained = timestamps[
        (timestamps >= end - duration - 1e-9) & (timestamps <= end + 1e-9)
    ]
    samples = window_sample_times(retained, end, duration, WINDOW_FRAMES)
    foreground = np.zeros(WINDOW_FRAMES, dtype=bool) if interval is None else (
        (samples >= float(interval["start_seconds"]))
        & (samples <= float(interval["end_seconds"]))
    )
    for other in other_intervals or []:
        foreground &= ~(
            (samples >= float(other["start_seconds"]))
            & (samples <= float(other["end_seconds"]))
        )
    normalized = normalize_time_window(features, timestamps, end, duration)[0]
    if interval is not None and not (foreground & (normalized[:, :42, 3] > 0).any(axis=1)).any():
        return None
    return ContextSample(
        normalized,
        foreground, target, category, source, signer, identity,
        duration, schedule,
    )


def load_context_samples(
    manifest_path: Path, labels: dict[str, int], role: str,
) -> tuple[list[ContextSample], dict[str, object]]:
    """Create only unambiguous positives and guarded, verified background windows."""
    if role not in {"train", "validation"} or _protected(manifest_path):
        raise ValueError("role must be train/validation and manifest must not be protected")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "slt_stage1_window_supervision_v17" or payload.get("version") != 1:
        raise ValueError("unsupported Stage-1 window supervision manifest")
    rows = [row for row in payload.get("rows", []) if row.get("role") == role]
    if not rows:
        raise ValueError(f"manifest has no {role} rows")
    output: list[ContextSample] = []
    signs: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    signers: set[str] = set()
    annotation_seconds = timeline_seconds = 0.0
    duration_counts: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    positive_only_rows = 0
    for row in rows:
        complete = row.get("all_signs_annotated") is True
        positive_only_rows += not complete
        source, signer, item = str(row["source"]), str(row["signer_id"]), str(row["source_item_id"])
        features, timestamps = _load_sequence(Path(row["archive_path"]))
        intervals = sorted(row.get("intervals", []), key=lambda value: float(value["start_seconds"]))
        if not intervals:
            raise ValueError(f"{item}: no verified sign intervals")
        for interval in intervals:
            start, end = float(interval["start_seconds"]), float(interval["end_seconds"])
            if not np.isfinite((start, end)).all() or end <= start:
                raise ValueError(f"{item}: invalid sign interval")
        known = [interval for interval in intervals if interval.get("label") in labels]
        signs.update(str(interval["label"]) for interval in known)
        pairs.update(
            (str(left["label"]), str(right["label"]))
            for left, right in zip(intervals, intervals[1:])
            if left.get("label") in labels and right.get("label") in labels
        )
        signers.add(signer)
        annotation_seconds += sum(float(x["end_seconds"]) - float(x["start_seconds"]) for x in intervals)
        timeline_seconds += float(timestamps[-1] - timestamps[0])
        for position, interval in enumerate(known):
            center = (float(interval["start_seconds"]) + float(interval["end_seconds"])) / 2
            others = [other for other in intervals if other is not interval]
            if any(
                float(other["start_seconds"]) <= center <= float(other["end_seconds"])
                for other in others
            ):
                rejections["ambiguous_center_annotation"] += len(WINDOW_DURATIONS)
                continue
            for duration in WINDOW_DURATIONS:
                start, end = center - duration / 2, center + duration / 2
                if start < timestamps[0] or end > timestamps[-1]:
                    continue
                try:
                    sample = _sample(
                        features, timestamps, interval, end=end, duration=duration,
                        target=labels[str(interval["label"])], category="context",
                        source=source, signer=signer, identity=f"{item}:center:{position}:{duration:.2f}",
                        schedule="centered", other_intervals=others,
                    )
                except ValueError as error:
                    rejections[str(error)] += 1
                    continue
                if sample is not None:
                    output.append(sample)
                    duration_counts[f"{duration:.2f}"] += 1
                else:
                    rejections["no_foreground_hand_observation"] += 1
        for end in window_end_times(timestamps, WINDOW_SECONDS, STRIDE_SECONDS):
            start = end - WINDOW_SECONDS
            center = (start + end) / 2
            candidates = [
                x for x in intervals
                if float(x["start_seconds"]) <= center <= float(x["end_seconds"])
            ]
            if len(candidates) == 1 and candidates[0].get("label") in labels:
                interval = candidates[0]
                try:
                    sample = _sample(
                        features, timestamps, interval, end=end, duration=WINDOW_SECONDS,
                        target=labels[str(interval["label"])], category="context",
                        source=source, signer=signer, identity=f"{item}:schedule:{end:.6f}",
                        schedule="trailing-0.53/0.13",
                        other_intervals=[other for other in intervals if other is not interval],
                    )
                except ValueError as error:
                    rejections[str(error)] += 1
                    continue
                if sample is not None:
                    output.append(sample)
                    duration_counts["0.53_schedule"] += 1
                else:
                    rejections["no_foreground_hand_observation"] += 1
        for gap_start, gap_end in interior_gaps(intervals, TRANSITION_GUARD_SECONDS) if complete else []:
            end = gap_start + WINDOW_SECONDS
            ordinal = 0
            while end <= gap_end + 1e-9:
                try:
                    sample = _sample(
                        features, timestamps, None, end=end, duration=WINDOW_SECONDS,
                        target=NO_EMIT_INDEX, category="background", source=source,
                        signer=signer, identity=f"{item}:background:{ordinal}",
                        schedule="trailing-0.53/0.13",
                    )
                except ValueError as error:
                    rejections[str(error)] += 1
                else:
                    assert sample is not None
                    output.append(sample)
                ordinal += 1
                end += STRIDE_SECONDS
    counts = Counter(sample.category for sample in output)
    if not counts["context"]:
        raise ValueError(f"{role} has no verified contextual positives")
    if not counts["background"]:
        raise ValueError(f"{role} has no verified background windows after 0.10s guards")
    audit = {
        "role": role, "rows": len(rows), "context_windows": counts["context"],
        "background_windows": counts["background"], "distinct_signs": len(signs),
        "distinct_sign_pairs": len(pairs), "distinct_signers": len(signers),
        "duration_windows": dict(duration_counts),
        "annotation_coverage": annotation_seconds / timeline_seconds if timeline_seconds else 0.0,
        "positive_only_rows": positive_only_rows,
        "guard_seconds": TRANSITION_GUARD_SECONDS,
        "equal_partition_ground_truth_used": False,
        "rejected_windows": dict(rejections),
    }
    return output, audit


def sample_weights(rows) -> torch.Tensor:
    """Balance 50/30/20 categories, then sources and classes within each."""
    keys = [(str(row[0]), str(row[1]), int(row[2])) for row in rows]
    categories = set(category for category, _, _ in keys)
    if categories != set(CATEGORY_SHARES):
        raise ValueError("replay, context, and verified background are all required")
    sources = Counter(category for category, _ in set((c, s) for c, s, _ in keys))
    source_classes = Counter((category, source) for category, source, _ in set(keys))
    counts = Counter(keys)
    return torch.tensor([
        CATEGORY_SHARES[category] / sources[category] / source_classes[(category, source)]
        / counts[(category, source, target)]
        for category, source, target in keys
    ], dtype=torch.double)


def window_objective(
    window_logits: torch.Tensor, foreground_logits: torch.Tensor,
    teacher_logits: torch.Tensor, targets: torch.Tensor, categories,
) -> dict[str, torch.Tensor]:
    window_loss = F.cross_entropy(window_logits, targets)
    context = torch.tensor([value == "context" for value in categories], device=targets.device)
    replay = torch.tensor([value == "replay" for value in categories], device=targets.device)
    zero = window_logits.sum() * 0.0
    foreground_loss = (
        F.cross_entropy(foreground_logits[context], targets[context])
        if context.any() else zero
    )
    replay_loss = (
        F.kl_div(
            F.log_softmax(window_logits[replay, :100] / 2.0, dim=1),
            F.softmax(teacher_logits[replay] / 2.0, dim=1),
            reduction="batchmean",
        ) * 4.0
        if replay.any() else zero
    )
    return {
        "window": window_loss, "foreground": foreground_loss,
        "replay": replay_loss,
        "total": window_loss + 0.5 * foreground_loss + replay_loss,
    }


def epoch_indices(rows, generator):
    weights = sample_weights(rows)
    chosen = []
    for category, share in CATEGORY_SHARES.items():
        indices = torch.tensor([i for i, row in enumerate(rows) if row[0] == category])
        draw = torch.multinomial(weights[indices], round(3000 * share), replacement=True, generator=generator)
        chosen.extend(indices[draw].tolist())
    return [chosen[i] for i in torch.randperm(len(chosen), generator=generator).tolist()]


class _Rows(Dataset):
    def __init__(self, replay: list[tuple[Path, int, str]], context: list[ContextSample]):
        self.replay, self.context = replay, context
        self.keys = [("replay", source, target) for _, target, source in replay] + [
            (row.category, row.source, row.target) for row in context
        ]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if index < len(self.replay):
            path, target, source = self.replay[index]
            features = load_features(path)
            return torch.from_numpy(features), torch.ones(32, dtype=torch.bool), target, "replay", source
        row = self.context[index - len(self.replay)]
        return torch.from_numpy(row.features.copy()), torch.from_numpy(row.foreground.copy()), row.target, row.category, row.source


def _replay_rows(root: Path, labels: dict[str, int], source: str, per_class: int) -> list[tuple[Path, int, str]]:
    dataset = IsolatedReplay(root, labels, 0, per_class, require_all_classes=(source == "citizen"))
    return [(path, target, source) for path, target, _ in dataset.rows]


def _load_start(checkpoint: dict[str, object]) -> ReelEmissionStage1V17:
    if checkpoint.get("format") == "slt_stage1_reel_emission_v17":
        return load_reel_emission_checkpoint(checkpoint)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("base must be a v17 Stage-1 or Reel emission checkpoint")
    config = Stage1V17Config(**checkpoint["model_config"])
    if config.num_classes != 100:
        raise ValueError("plain Stage-1 starting checkpoint must have 100 glosses")
    base = SLTStage1V17(config)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    head = ReelEmissionHeadV17(ReelEmissionHeadConfig(
        stage1_dim=config.dim, num_glosses=config.num_classes
    ))
    return ReelEmissionStage1V17(base, head).eval()


def _validate_seed(seed: int, eligibility: Path | None) -> None:
    if seed == 17112 and (
        eligibility is None
        or json.loads(eligibility.read_text(encoding="utf-8")).get("eligible") is not True
    ):
        raise ValueError("seed 17112 requires an eligible complete-streaming result for seed 17111")


def verify_development_freeze(path, base, supervision):
    frozen = json.loads(path.read_text())
    if frozen.get('status') != 'baselines_frozen':
        raise ValueError('complete matched baselines must be frozen before training')
    if frozen.get('base_sha256') != _sha256(base) or frozen.get('supervision_sha256') != _sha256(supervision):
        raise ValueError('candidate starting model or supervision changed after freeze')
    for name in ('artifacts', 'raw_inputs'):
        if not frozen.get(name):
            raise ValueError('missing frozen ' + name)
        for filename, digest in frozen[name].items():
            if _sha256(Path(filename)) != digest:
                raise ValueError('frozen input changed: ' + filename)
    return _sha256(path)


@torch.inference_mode()
def _accuracy(model, dataset: Dataset, device: torch.device, batch_size: int) -> float:
    correct = total = 0
    model.eval()
    for features, _, targets, _, _ in DataLoader(dataset, batch_size=batch_size):
        predictions = model(features.to(device)).argmax(1).cpu()
        mask = targets < 100
        correct += int((predictions[mask] == targets[mask]).sum())
        total += int(mask.sum())
    return correct / max(total, 1)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.materialize:
        return materialize_raw_supervision(args)
    if args.audit_existing:
        supervision = json.loads(args.existing_supervision.read_text(encoding="utf-8"))
        frozen = json.loads(args.frozen_inputs.read_text(encoding="utf-8"))
        audit = {
            "format": "slt_stage1_window_data_audit_v17", "version": 1,
            **audit_existing_supervision(supervision, frozen),
            "blocker": "timestamped raw live-observation cache has not been materialized",
            "citizen_test_accessed": False,
        }
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        return audit
    if args.train:
        _validate_seed(args.seed, args.seed_17111_eligibility)
        development_sha256 = verify_development_freeze(args.development_freeze, args.base, args.supervision)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    model = _load_start(checkpoint)
    labels = {str(k): int(v) for k, v in checkpoint["label_to_index"].items() if int(v) < 100}
    train_context, train_audit = load_context_samples(args.supervision, labels, "train")
    validation_context, validation_audit = load_context_samples(args.supervision, labels, "validation")
    payload = json.loads(args.supervision.read_text(encoding="utf-8"))
    train_signers = {str(r["signer_id"]) for r in payload["rows"] if r["role"] == "train"}
    validation_signers = {str(r["signer_id"]) for r in payload["rows"] if r["role"] == "validation"}
    if train_signers & validation_signers:
        raise ValueError("contextual train/validation signers overlap")
    audit = {
        "format": "slt_stage1_window_data_audit_v17", "version": 1,
        "supervision": str(args.supervision), "supervision_sha256": _sha256(args.supervision),
        "train": train_audit, "validation": validation_audit,
        "train_validation_signer_disjoint": True,
        "schedule": {"window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS,
                     "training_window_seconds": list(WINDOW_DURATIONS), "frames": WINDOW_FRAMES},
        "category_shares": CATEGORY_SHARES, "citizen_test_accessed": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not args.train:
        return audit
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device)
    train_replay = _replay_rows(args.citizen_train, labels, "citizen", args.replay_per_class)
    train_replay += _replay_rows(args.semlex_train, labels, "semlex", args.replay_per_class)
    validation_replay = _replay_rows(args.citizen_validation, labels, "citizen", 10000)
    validation_replay += _replay_rows(args.semlex_validation, labels, "semlex", 10000)
    train_data = _Rows(train_replay, train_context)
    isolated_validation = {
        source: _Rows([row for row in validation_replay if row[2] == source], [])
        for source in ("citizen", "semlex")
    }
    sampler_generator = torch.Generator().manual_seed(args.seed)
    model.to(device)
    teacher = copy.deepcopy(model.base).eval()
    for parameter in teacher.parameters(): parameter.requires_grad = False
    teacher.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline = {
        source: _accuracy(model.base, dataset, device, 64)
        for source, dataset in isolated_validation.items()
    }
    history = []
    started = time.perf_counter()
    for epoch in range(1, 13):
        loader = DataLoader(train_data, batch_size=64,
                            sampler=epoch_indices(train_data.keys, sampler_generator), num_workers=0)
        model.train(); totals = Counter()
        for features, foreground, targets, categories, _ in loader:
            features, foreground, targets = features.to(device), foreground.to(device), targets.to(device)
            encoded, active = model.base.encode(features)
            pooled = pool_stage1_encoded(model.base, encoded, active)
            class_logits = model.base.classifier(pooled)
            window_logits = torch.cat((class_logits, model.emission_head(reel_temporal_summary(encoded, class_logits))), dim=1)
            foreground_logits = model.base.classifier(pool_stage1_encoded(model.base, encoded, foreground & active))
            replay = torch.tensor([c == 'replay' for c in categories], device=device)
            with torch.no_grad():
                teacher_logits = torch.zeros((len(features), 100), device=device)
                if replay.any():
                    teacher_logits[replay] = teacher(features[replay])
            losses = window_objective(window_logits, foreground_logits, teacher_logits, targets, categories)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite Stage-1 window training loss")
            optimizer.zero_grad(set_to_none=True); losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            for key, value in losses.items(): totals[key] += float(value.detach().cpu())
            totals["updates"] += 1
        scheduler.step()
        validation_accuracy = {
            source: _accuracy(model, dataset, device, 64)
            for source, dataset in isolated_validation.items()
        }
        row = {"epoch": epoch, "loss": totals["total"] / len(loader),
               "isolated_validation_accuracy": validation_accuracy}
        history.append(row)
        print(json.dumps(row), flush=True)
        selected = {
            "format": CHECKPOINT_FORMAT, "epoch": epoch,
            "base_model_config": model.base.config.to_dict(),
            "base_model_state_dict": {k: v.cpu() for k, v in model.base.state_dict().items()},
            "emission_head_config": model.emission_head.config.to_dict(),
            "emission_head_state_dict": {k: v.cpu() for k, v in model.emission_head.state_dict().items()},
            "label_to_index": {**labels, NO_EMIT: NO_EMIT_INDEX},
            "stage1_window": {
                "base_checkpoint": str(args.base), "base_sha256": _sha256(args.base),
                "development_freeze_sha256": development_sha256,
                "window_seconds": WINDOW_SECONDS, "stride_seconds": STRIDE_SECONDS,
                "training_window_seconds": list(WINDOW_DURATIONS), "frames": WINDOW_FRAMES,
                "no_emit_index": NO_EMIT_INDEX, "data_audit": audit,
                "isolated_validation_baseline": baseline,
                "isolated_validation_current": validation_accuracy,
                "selection_status": "unselected_pending_complete_streaming_evaluation",
                "recipe": {"seed": args.seed, "epochs": 12, "samples_per_epoch": 3000,
                           "batch_size": 64, "optimizer": "AdamW", "learning_rate": 2e-5,
                           "weight_decay": 1e-4, "scheduler": "cosine", "gradient_clip": 1.0,
                           "loss_weights": {"window": 1.0, "foreground": 0.5, "replay": 1.0},
                           "category_shares": CATEGORY_SHARES},
                "epoch_optimizer_updates": int(totals["updates"]),
                "completed_optimizer_updates": epoch * len(loader),
            },
        }
        torch.save(selected, args.output_dir / f"epoch_{epoch:02d}.pth")
    result = {"format": "slt_stage1_window_training_v17", "history": history,
              "elapsed_seconds": time.perf_counter() - started,
              "selection": None, "all_epochs_saved": True, "citizen_test_accessed": False}
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base", type=Path, default=Path("artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"))
    value.add_argument("--supervision", type=Path, default=Path("artifacts/reports/stage1_window_v17/supervision_manifest.json"))
    value.add_argument("--existing-supervision", type=Path, default=Path("artifacts/reports/stage2_v17_revisable_v1/supervision.json"))
    value.add_argument("--frozen-inputs", type=Path, default=Path("artifacts/reports/stage2_v17_live_matched_v1/frozen_inputs.json"))
    value.add_argument("--audit", type=Path, default=Path("artifacts/reports/stage1_window_v17/data_audit.json"))
    value.add_argument("--raw-root", type=Path, default=Path("data/local/stage1_window_v17/raw_observations"))
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/models/stage1_window_v17_seed17111"))
    value.add_argument("--citizen-train", type=Path, default=Path("data/local/citizen100_v17/landmarks/train"))
    value.add_argument("--citizen-validation", type=Path, default=Path("data/local/citizen100_v17/landmarks/val"))
    value.add_argument("--semlex-train", type=Path, default=Path("data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"))
    value.add_argument("--semlex-validation", type=Path, default=Path("data/local/semlex_citizen100_val_audit/landmarks_v17"))
    value.add_argument("--replay-per-class", type=int, default=10)
    value.add_argument("--seed", type=int, choices=(17111, 17112), default=17111)
    value.add_argument("--seed-17111-eligibility", type=Path)
    value.add_argument("--development-freeze", type=Path, default=Path("artifacts/reports/stage1_window_v17/development_freeze.json"))
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--train", action="store_true", help="run the frozen 12-epoch recipe after audit")
    value.add_argument("--audit-existing", action="store_true", help="audit accepted all-sign intervals without extracting or training")
    value.add_argument("--materialize", action="store_true", help="cache checked raw live observations and write the training manifest")
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
