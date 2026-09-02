#!/usr/bin/env python3
"""Train a separate Stage-1 ``NO_EMIT`` output for reel-style live inference.

The accepted 100-gloss landmark model is frozen.  Only one additional linear row is
learned from complete signs versus incomplete prefixes and between-sign transitions.
Consequently the ranking of the original 100 glosses cannot change.  Citizen and
SemLex isolated clips provide vocabulary-wide replay; genuine signer-disjoint ASLLRP
phrases and the local phrase cache provide conversational timing.  Test and externally
reserved data are rejected.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.geometry_v17 import resample_features
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.model_reel_emission_v17 import (
    ReelEmissionHeadConfig,
    ReelEmissionHeadV17,
    ReelEmissionStage1V17,
    pool_stage1_encoded,
    reel_temporal_summary,
)


NO_EMIT = "__NO_EMIT__"
DEFAULT_BASE = Path("artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth")


@dataclass(frozen=True)
class BoundarySample:
    features: np.ndarray
    target: int
    domain: str
    kind: str
    identity: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def refuse_protected(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered = {part.casefold() for part in path.parts}
        if "test" in lowered or "external_evaluation_reserved" in lowered:
            raise ValueError(f"reel emission training refuses protected path: {path}")


def load_v17(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        value = payload["features"].astype(np.float32, copy=False)
    if value.shape != (32, 61, 5) or not np.isfinite(value).all():
        raise ValueError(f"invalid v17 landmark archive: {path}")
    return value


def crop_resample(value: np.ndarray, start: float, stop: float) -> np.ndarray:
    first = max(0, min(len(value) - 1, int(round(start))))
    last = max(first + 1, min(len(value), int(round(stop))))
    if last - first < 2:
        first = max(0, last - 2)
    return resample_features(value[first:last], 32).astype(np.float32)


def restore_source_frames(
    windows: np.ndarray, source_ranges: np.ndarray
) -> np.ndarray:
    """Approximately invert per-window resampling using recorded source ranges."""
    restored = []
    for window, (start, stop) in zip(windows, source_ranges):
        count = int(stop) - int(start)
        if count > 0:
            restored.append(resample_features(window.astype(np.float32), count))
    if not restored:
        raise ValueError("phrase cache contains no source frames")
    return np.concatenate(restored).astype(np.float32)


def isolated_samples(
    root: Path,
    labels: dict[str, int],
    no_emit_index: int,
    *,
    domain: str,
    per_class: int,
    prefix_fractions: tuple[float, ...],
) -> Iterator[BoundarySample]:
    for label, target in sorted(labels.items(), key=lambda row: row[1]):
        paths = sorted((root / label).glob("*.v17.npz"))[:per_class]
        if not paths:
            continue
        for path in paths:
            value = load_v17(path)
            yield BoundarySample(value.copy(), target, domain, "complete", path.name)
            for fraction in prefix_fractions:
                yield BoundarySample(
                    crop_resample(value, 0, len(value) * fraction),
                    no_emit_index,
                    domain,
                    "prefix",
                    f"{path.name}:prefix:{fraction:.2f}",
                )


def phrase_payload(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        frames = restore_source_frames(
            payload["landmarks"], payload["window_source_ranges"]
        )
    return frames, metadata


def local_phrase_samples(
    root: Path,
    labels: dict[str, int],
    no_emit_index: int,
    *,
    prefix_fraction: float,
) -> Iterator[BoundarySample]:
    for path in sorted(root.glob("*.npz")):
        frames, metadata = phrase_payload(path)
        sequence = [str(value) for value in metadata["target_sequence"]]
        if not sequence or any(value not in labels for value in sequence):
            continue
        width = len(frames) / len(sequence)
        for position, label in enumerate(sequence):
            left, right = position * width, (position + 1) * width
            context = 0.06 * width
            yield BoundarySample(
                crop_resample(frames, left - context, right + context),
                labels[label], "local_phrase", "complete", f"{path.name}:{position}",
            )
            yield BoundarySample(
                crop_resample(frames, left - context, left + prefix_fraction * width),
                no_emit_index, "local_phrase", "prefix",
                f"{path.name}:{position}:prefix",
            )
        for position in range(len(sequence) - 1):
            boundary = (position + 1) * width
            yield BoundarySample(
                crop_resample(frames, boundary - 0.30 * width, boundary + 0.30 * width),
                no_emit_index, "local_phrase", "transition",
                f"{path.name}:{position}:transition",
            )


def asllrp_annotations(
    span_manifest: Path, segmented_manifest: Path
) -> dict[str, tuple[dict[str, object], list[dict[str, object]]]]:
    spans = json.loads(span_manifest.read_text(encoding="utf-8"))["spans"]
    signs = json.loads(segmented_manifest.read_text(encoding="utf-8"))["videos"]
    by_parent: dict[str, list[dict[str, object]]] = {}
    for sign in signs:
        if sign.get("source") != "asllrp" or sign.get("split_role") != "train_candidate":
            continue
        by_parent.setdefault(str(sign["utterance_video_filename"]), []).append(sign)
    output = {}
    for span in spans:
        if span.get("source") != "asllrp" or span.get("split_role") != "train_candidate":
            continue
        parent = str(span["utterance_video_filename"])
        first, last = int(span["span_start_frame_global"]), int(span["span_end_frame_global"])
        selected = sorted(
            (
                sign for sign in by_parent.get(parent, [])
                if first <= int(sign["sign_start_frame"])
                <= int(sign["sign_end_frame"]) <= last
            ),
            key=lambda row: int(row["sign_start_frame"]),
        )
        item_id = f"asllrp:{parent}:span{int(span['span_index_in_utterance']):02d}"
        output[item_id] = (span, selected)
    return output


def asllrp_phrase_samples(
    root: Path,
    annotations: dict[str, tuple[dict[str, object], list[dict[str, object]]]],
    labels: dict[str, int],
    no_emit_index: int,
    *,
    prefix_fraction: float,
) -> Iterator[BoundarySample]:
    for path in sorted(root.glob("*.npz")):
        frames, metadata = phrase_payload(path)
        item_id = str(metadata["source_item_id"])
        if item_id not in annotations:
            raise ValueError(f"missing manual alignment for {item_id}")
        span, signs = annotations[item_id]
        sequence = [str(value) for value in metadata["target_sequence"]]
        aligned_labels = [str(value["canonical_label"]) for value in signs]
        if sequence != aligned_labels:
            raise ValueError(f"{item_id}: phrase/manual labels disagree")
        crop_global_start = int(span["utterance_start_frame_global"]) + int(
            span["crop_start_frame_local"]
        )
        intervals = [
            (
                int(sign["sign_start_frame"]) - crop_global_start,
                int(sign["sign_end_frame"]) - crop_global_start + 1,
            )
            for sign in signs
        ]
        for position, (label, (left, right)) in enumerate(zip(sequence, intervals)):
            width = max(2, right - left)
            context = 0.08 * width
            yield BoundarySample(
                crop_resample(frames, left - context, right + context),
                labels[label], "asllrp_phrase", "complete", f"{item_id}:{position}",
            )
            yield BoundarySample(
                crop_resample(frames, left - context, left + prefix_fraction * width),
                no_emit_index, "asllrp_phrase", "prefix",
                f"{item_id}:{position}:prefix",
            )
        for position, ((_, left_stop), (right_start, right_stop)) in enumerate(
            zip(intervals, intervals[1:])
        ):
            left_start = intervals[position][0]
            left_width = max(2, left_stop - left_start)
            right_width = max(2, right_stop - right_start)
            yield BoundarySample(
                crop_resample(
                    frames,
                    left_stop - 0.35 * left_width,
                    right_start + 0.35 * right_width,
                ),
                no_emit_index, "asllrp_phrase", "transition",
                f"{item_id}:{position}:transition",
            )


def collect_samples(
    *,
    split: str,
    labels: dict[str, int],
    no_emit_index: int,
    citizen_root: Path,
    semlex_root: Path,
    phrase_root: Path,
    annotations: dict[str, tuple[dict[str, object], list[dict[str, object]]]],
    per_class: int,
) -> list[BoundarySample]:
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    citizen_split = "train" if split == "train" else "val"
    semlex_split = (
        semlex_root / "full_clean_landmarks_v17"
        if split == "train" else semlex_root / "landmarks_v17"
    )
    samples = list(isolated_samples(
        citizen_root / citizen_split, labels, no_emit_index,
        domain="citizen", per_class=per_class,
        prefix_fractions=(0.50, 0.72) if split == "train" else (0.61,),
    ))
    samples.extend(isolated_samples(
        semlex_split, labels, no_emit_index,
        domain="semlex", per_class=per_class,
        prefix_fractions=(0.50, 0.72) if split == "train" else (0.61,),
    ))
    cache_split = "train" if split == "train" else "validation"
    samples.extend(local_phrase_samples(
        phrase_root / cache_split / "local_phrases", labels, no_emit_index,
        prefix_fraction=0.60 if split == "train" else 0.65,
    ))
    samples.extend(asllrp_phrase_samples(
        phrase_root / cache_split / "asllrp_contiguous", annotations,
        labels, no_emit_index,
        prefix_fraction=0.60 if split == "train" else 0.65,
    ))
    if not samples:
        raise ValueError(f"no {split} boundary samples")
    return samples


@torch.no_grad()
def encode_samples(
    model: SLTStage1V17,
    samples: list[BoundarySample],
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.to(device).eval()
    summary_rows, logit_rows = [], []
    for start in range(0, len(samples), batch_size):
        batch = torch.from_numpy(np.stack([
            value.features for value in samples[start : start + batch_size]
        ])).to(device)
        encoded, active = model.encode(batch)
        pooled = pool_stage1_encoded(model, encoded, active)
        logits = model.classifier(pooled)
        summary_rows.append(reel_temporal_summary(encoded, logits).float().cpu())
        logit_rows.append(logits.float().cpu())
    model.to("cpu")
    return {
        "summary": torch.cat(summary_rows),
        "class_logits": torch.cat(logit_rows),
        "targets": torch.tensor([value.target for value in samples]),
        "domains": [value.domain for value in samples],
        "kinds": [value.kind for value in samples],
        "identities": [value.identity for value in samples],
    }


def wait_probability(
    head: nn.Module, summary: torch.Tensor, class_logits: torch.Tensor
) -> torch.Tensor:
    wait_logit = head(summary).squeeze(1)
    return torch.sigmoid(wait_logit - torch.logsumexp(class_logits, dim=1))


def threshold_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    expected_wait = targets.astype(bool)
    predicted_wait = probabilities >= threshold
    wait_total = int(expected_wait.sum())
    complete_total = int((~expected_wait).sum())
    wait_correct = int((predicted_wait & expected_wait).sum())
    complete_correct = int((~predicted_wait & ~expected_wait).sum())
    wait_recall = wait_correct / max(wait_total, 1)
    complete_accept = complete_correct / max(complete_total, 1)
    return {
        "threshold": float(threshold),
        "samples": len(targets),
        "wait_samples": wait_total,
        "complete_samples": complete_total,
        "wait_recall": wait_recall,
        "complete_accept_rate": complete_accept,
        "balanced_accuracy": 0.5 * (wait_recall + complete_accept),
    }


def select_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    domains: np.ndarray | None = None,
    kinds: np.ndarray | None = None,
) -> dict[str, float | int]:
    thresholds = np.concatenate((
        np.linspace(0.01, 0.99, 99),
        np.asarray((0.995, 0.999, 0.9995, 0.9999, 0.99999)),
    ))
    rows = [
        threshold_metrics(probabilities, targets, float(value))
        for value in thresholds
    ]
    if domains is not None and kinds is not None:
        asllrp_complete = (domains == "asllrp_phrase") & (kinds == "complete")
        asllrp_wait = (domains == "asllrp_phrase") & (kinds != "complete")
        local_complete = (domains == "local_phrase") & (kinds == "complete")
        for value in rows:
            threshold = float(value["threshold"])
            value["asllrp_complete_accept_rate"] = float(
                (probabilities[asllrp_complete] < threshold).mean()
            )
            value["asllrp_wait_recall"] = float(
                (probabilities[asllrp_wait] >= threshold).mean()
            )
            value["asllrp_balanced_accuracy"] = 0.5 * (
                value["asllrp_complete_accept_rate"] + value["asllrp_wait_recall"]
            )
            value["local_complete_accept_rate"] = float(
                (probabilities[local_complete] < threshold).mean()
            )
    eligible = [
        value for value in rows
        if value["complete_accept_rate"] >= 0.95
        and value.get("asllrp_complete_accept_rate", 1.0) >= 0.75
        and value.get("local_complete_accept_rate", 1.0) >= 0.90
    ]
    pool = eligible or rows
    return max(
        pool,
        key=lambda value: (
            value.get("asllrp_balanced_accuracy", 0.0),
            value["balanced_accuracy"], value["wait_recall"],
            value["complete_accept_rate"], -value["threshold"],
        ),
    )


def evaluate_encoded(
    head: nn.Module,
    encoded: dict[str, object],
    no_emit_index: int,
    threshold: float,
) -> dict[str, object]:
    summary = encoded["summary"]
    class_logits = encoded["class_logits"]
    targets = encoded["targets"]
    probabilities = wait_probability(head.cpu(), summary, class_logits).detach().numpy()
    target_array = targets.numpy()
    waits = target_array == no_emit_index
    output: dict[str, object] = threshold_metrics(probabilities, waits, threshold)
    complete = ~waits
    predicted = class_logits.argmax(1).numpy()
    output["complete_gloss_accuracy"] = float(
        (predicted[complete] == target_array[complete]).mean()
    ) if complete.any() else 0.0
    output["accepted_correct_gloss_rate"] = float(
        ((predicted == target_array) & complete & (probabilities < threshold)).sum()
        / max(int(complete.sum()), 1)
    )
    groups: dict[str, object] = {}
    domains = np.asarray(encoded["domains"])
    kinds = np.asarray(encoded["kinds"])
    for domain in sorted(set(domains.tolist())):
        for kind_group, mask in (
            ("complete", (domains == domain) & (kinds == "complete")),
            ("no_emit", (domains == domain) & (kinds != "complete")),
        ):
            if mask.any():
                groups[f"{domain}_{kind_group}"] = threshold_metrics(
                    probabilities[mask], waits[mask], threshold
                )
    output["groups"] = groups
    output["probability_quantiles"] = {
        "complete_p50": float(np.quantile(probabilities[complete], 0.50)),
        "complete_p95": float(np.quantile(probabilities[complete], 0.95)),
        "no_emit_p05": float(np.quantile(probabilities[waits], 0.05)),
        "no_emit_p50": float(np.quantile(probabilities[waits], 0.50)),
    }
    return output

def run(args: argparse.Namespace) -> dict[str, object]:
    protected = (
        args.citizen_root, args.semlex_train_root, args.semlex_val_root,
        args.phrase_root, args.span_manifest, args.segmented_manifest,
    )
    refuse_protected(protected)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("base must be a v17 landmark Stage-1 checkpoint")
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    if sorted(labels.values()) != list(range(100)) or NO_EMIT in labels:
        raise ValueError("expected the frozen 100-gloss label map")
    base = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    annotations = asllrp_annotations(args.span_manifest, args.segmented_manifest)
    train_samples = collect_samples(
        split="train", labels=labels, no_emit_index=100,
        citizen_root=args.citizen_root,
        semlex_root=args.semlex_train_root,
        phrase_root=args.phrase_root, annotations=annotations,
        per_class=args.train_per_class,
    )
    validation_samples = collect_samples(
        split="validation", labels=labels, no_emit_index=100,
        citizen_root=args.citizen_root,
        semlex_root=args.semlex_val_root,
        phrase_root=args.phrase_root, annotations=annotations,
        per_class=args.validation_per_class,
    )
    started = time.perf_counter()
    train = encode_samples(base, train_samples, device, args.embedding_batch_size)
    validation = encode_samples(base, validation_samples, device, args.embedding_batch_size)

    head_config = ReelEmissionHeadConfig(
        stage1_dim=base.config.dim,
        num_glosses=base.config.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    head = ReelEmissionHeadV17(head_config).to(device)
    summary = train["summary"].to(device)
    class_logits = train["class_logits"].to(device)
    wait_targets = (train["targets"] == 100).float().to(device)
    group_names = [
        f"{domain}:{'complete' if kind == 'complete' else 'no_emit'}"
        for domain, kind in zip(train["domains"], train["kinds"])
    ]
    group_counts = Counter(group_names)
    sample_weights = torch.tensor(
        [1.0 / group_counts[name] for name in group_names], dtype=torch.float32,
        device=device,
    )
    sample_weights /= sample_weights.mean()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    best = None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(summary), generator=generator)
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size].to(device)
            probabilities_logit = head(summary[indices]).squeeze(1) - torch.logsumexp(
                class_logits[indices], dim=1
            )
            losses = F.binary_cross_entropy_with_logits(
                probabilities_logit, wait_targets[indices], reduction="none"
            )
            loss = (losses * sample_weights[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
        head.eval().cpu()
        probabilities = wait_probability(
            head, validation["summary"], validation["class_logits"]
        ).detach().numpy()
        threshold = select_threshold(
            probabilities,
            validation["targets"].numpy() == 100,
            np.asarray(validation["domains"]),
            np.asarray(validation["kinds"]),
        )
        row = {"epoch": epoch, "loss": total / len(order), **threshold}
        history.append(row)
        key = (
            row.get("asllrp_balanced_accuracy", 0.0),
            row["balanced_accuracy"], row["wait_recall"],
        )
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "state": {key: value.detach().clone() for key, value in head.state_dict().items()},
                "threshold": float(row["threshold"]),
            }
        head.to(device)
    if best is None:
        raise RuntimeError("training produced no checkpoint")
    head.load_state_dict(best["state"])
    head.cpu().eval()
    threshold = float(best["threshold"])
    train_metrics = evaluate_encoded(head, train, 100, threshold)
    validation_metrics = evaluate_encoded(head, validation, 100, threshold)
    reel_model = ReelEmissionStage1V17(base, head).eval()
    augmented_labels = dict(labels)
    augmented_labels[NO_EMIT] = 100
    with torch.inference_mode():
        probe = torch.from_numpy(validation_samples[0].features)[None]
        old_logits = base(probe)
        new_logits = reel_model(probe)[:, :100]
    preserved_max_abs = float((old_logits - new_logits).abs().max())
    if preserved_max_abs != 0.0:
        raise RuntimeError(f"original gloss logits changed: {preserved_max_abs}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "best_model.pth"
    selected = {
        "format": "slt_stage1_reel_emission_v17",
        "epoch": int(best["epoch"]),
        "base_model_config": base.config.to_dict(),
        "base_model_state_dict": base.state_dict(),
        "emission_head_config": head.config.to_dict(),
        "emission_head_state_dict": head.state_dict(),
        "label_to_index": augmented_labels,
        "validation_metrics": validation_metrics,
        "reel_emission": {
            "base_checkpoint": str(args.base),
            "base_sha256": sha256(args.base),
            "no_emit_label": NO_EMIT,
            "no_emit_index": 100,
            "runtime_no_emit_probability_threshold": threshold,
            "original_100_gloss_logits_preserved_max_abs": preserved_max_abs,
            "training_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "training_groups": dict(Counter(
                f"{value.domain}:{value.kind}" for value in train_samples
            )),
            "validation_groups": dict(Counter(
                f"{value.domain}:{value.kind}" for value in validation_samples
            )),
            "asllrp_validation_signer_disjoint": True,
            "test_accessed": False,
            "external_evaluation_reserved_accessed": False,
        },
    }
    torch.save(selected, output)
    result = {
        "format": "slt_stage1_reel_emission_adaptation_v17",
        "output": str(output),
        "base": str(args.base),
        "base_sha256": sha256(args.base),
        "device": str(device),
        "selected_epoch": int(best["epoch"]),
        "runtime_no_emit_probability_threshold": threshold,
        "original_100_gloss_logits_preserved_max_abs": preserved_max_abs,
        "train": train_metrics,
        "validation": validation_metrics,
        "history": history,
        "training_groups": selected["reel_emission"]["training_groups"],
        "validation_groups": selected["reel_emission"]["validation_groups"],
        "elapsed_seconds": time.perf_counter() - started,
        "test_accessed": False,
        "external_evaluation_reserved_accessed": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base", type=Path, default=DEFAULT_BASE)
    value.add_argument("--citizen-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    value.add_argument("--semlex-train-root", type=Path, default=Path("data/local/semlex_citizen100_train_audit"))
    value.add_argument("--semlex-val-root", type=Path, default=Path("data/local/semlex_citizen100_val_audit"))
    value.add_argument("--phrase-root", type=Path, default=Path("data/local/stage2_v17_multimodal"))
    value.add_argument("--span-manifest", type=Path, default=Path("data/local/asllrp_contiguous_phrases_v17/manifest.json"))
    value.add_argument("--segmented-manifest", type=Path, default=Path("data/local/asllrp_segmented_citizen100_v17/manifest.json"))
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/models/stage1_v17_reel_emission_v3"))
    value.add_argument("--epochs", type=int, default=60)
    value.add_argument("--batch-size", type=int, default=512)
    value.add_argument("--embedding-batch-size", type=int, default=128)
    value.add_argument("--learning-rate", type=float, default=2e-3)
    value.add_argument("--hidden-dim", type=int, default=64)
    value.add_argument("--dropout", type=float, default=0.25)
    value.add_argument("--weight-decay", type=float, default=1e-3)
    value.add_argument("--train-per-class", type=int, default=10)
    value.add_argument("--validation-per-class", type=int, default=10000)
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--seed", type=int, default=17031)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
