#!/usr/bin/env python3
"""Cache, train, and select the frozen-encoder unified v17 multimodal student."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_rgb_semlex_val_v17 import validation_items
from active.v17.extract_hand_rgb_supplement_v17 import selection_items
from active.v17.model_hand_mobileclip2_v17 import (
    HandMobileCLIP2Stage1Config,
    HandMobileCLIP2Stage1V17,
)
from active.v17.model_unified_multimodal_v17 import (
    UnifiedFusionHeadV17,
    UnifiedMultimodalV17Config,
    per_sample_zscore,
)
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.model_visual_speech_v17 import (
    VisualSpeechTeacherV17,
    VisualSpeechTeacherV17Config,
)
from active.v17.schema_hand_mobileclip2_v17 import (
    HandMobileCLIP2V17Config,
    schema_fingerprint as hand_schema_fingerprint,
)
from active.v17.schema_v17 import V17Config, schema_fingerprint
from active.v17.train_stage_1_v17 import load_v17_archive, mask_mouth_nodes_v17
from active.v17.train_stage_1_visual_speech_features_v17 import FrozenVisualFeatureDataset


LOG = logging.getLogger("unified_multimodal_student_v17")
SEEDS = (1701, 3407, 5101)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PairRecord:
    source: str
    split: str
    item_id: str
    label: str
    target: int
    landmark_path: Path
    hand_path: Path
    mask_mouth: bool = False


def label_map(manifest_path: Path) -> dict[str, int]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {
        str(row["canonical_label"]): int(row["class_index"])
        for row in payload["classes"]
    }
    if sorted(result.values()) != list(range(len(result))):
        raise ValueError("Citizen label map is not contiguous")
    return result


def citizen_records(
    landmark_root: Path, hand_root: Path, split: str, labels: dict[str, int],
    rejections_path: Path,
) -> list[PairRecord]:
    if split not in ("train", "val"):
        raise ValueError("Citizen test is sealed")
    with rejections_path.open(encoding="utf-8", newline="") as stream:
        rejected = {
            (str(row["split"]), str(row["canonical_label"]), str(row["video"]))
            for row in csv.DictReader(stream)
        }
    output = []
    for label, target in labels.items():
        for landmark in sorted((landmark_root / split / label).glob("*.v17.npz")):
            stem = landmark.name.removesuffix(".v17.npz")
            if (split, label, f"{stem}.mp4") in rejected:
                continue
            hand = hand_root / split / label / f"{stem}.hand_mobileclip2_v17.npz"
            if not hand.is_file():
                raise FileNotFoundError(hand)
            output.append(PairRecord("citizen", split, f"{label}/{stem}", label, target, landmark, hand))
    return output


def supplement_records(
    manifest: Path,
    source: str,
    hand_root: Path,
    labels: dict[str, int],
) -> list[PairRecord]:
    items, _ = selection_items(manifest, source)
    output = []
    for item in items:
        if item.label not in labels:
            continue
        hand_folder = "val" if source == "local_deep_clean_val" else source
        hand = hand_root / hand_folder / item.label / f"{item.item_id}.hand_mobileclip2_v17.npz"
        if not hand.is_file():
            raise FileNotFoundError(hand)
        output.append(PairRecord(
            source="local" if source.startswith("local_deep_clean") else source,
            split="val" if source.endswith("_val") else "train",
            item_id=f"{item.label}/{item.item_id}",
            label=item.label,
            target=labels[item.label],
            landmark_path=item.landmark_path,
            hand_path=hand,
            mask_mouth=source.startswith("local_deep_clean"),
        ))
    return output


def semlex_validation_records(
    manifest: Path, hand_root: Path, labels: dict[str, int]
) -> list[PairRecord]:
    items, _ = validation_items(manifest)
    output = []
    for item in items:
        hand = hand_root / item.label / f"{item.item_id}.hand_mobileclip2_v17.npz"
        if not hand.is_file():
            raise FileNotFoundError(hand)
        output.append(PairRecord(
            "semlex", "val", f"{item.label}/{item.item_id}", item.label,
            labels[item.label], item.landmark_path, hand,
        ))
    return output


def load_hand(path: Path, expected_schema: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        embeddings = payload["embeddings"].astype(np.float32)
        valid = payload["valid"].astype(np.bool_)
        boxes = payload["boxes_normalized"].astype(np.float32)
        metadata = json.loads(str(payload["metadata_json"]))
    if (
        embeddings.shape != (16, 3, 512)
        or valid.shape != (16, 3)
        or boxes.shape != (16, 3, 4)
        or metadata.get("schema_fingerprint") != expected_schema
        or metadata.get("test_accessed") not in (None, False)
    ):
        raise ValueError(f"invalid hand archive: {path}")
    if not np.isfinite(embeddings).all() or not np.isfinite(boxes).all():
        raise ValueError(f"non-finite hand archive: {path}")
    if not np.all(embeddings[~valid] == 0):
        raise ValueError(f"invalid views are nonzero: {path}")
    return tuple(torch.from_numpy(value.copy()) for value in (embeddings, valid, boxes))


def load_landmark_model(path: Path) -> tuple[SLTStage1V17, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError(f"not a v17 landmark checkpoint: {path}")
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint


def load_hand_model(path: Path) -> tuple[HandMobileCLIP2Stage1V17, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_hand_mobileclip2_v17":
        raise ValueError(f"not a v17 hand checkpoint: {path}")
    model = HandMobileCLIP2Stage1V17(
        HandMobileCLIP2Stage1Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint


@torch.inference_mode()
def encode_records(
    records: list[PairRecord],
    landmark_model: SLTStage1V17,
    hand_model: HandMobileCLIP2Stage1V17,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    landmark_model.to(device).eval()
    hand_model.to(device).eval()
    landmark_features = []
    hand_features = []
    landmark_logits = []
    hand_logits = []
    expected_landmark = schema_fingerprint(V17Config())
    expected_hand = hand_schema_fingerprint(HandMobileCLIP2V17Config())
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        landmarks = []
        hands = []
        valids = []
        boxes = []
        for row in rows:
            landmark = load_v17_archive(row.landmark_path, expected_landmark)
            if row.mask_mouth:
                landmark = mask_mouth_nodes_v17(landmark)
            hand, valid, box = load_hand(row.hand_path, expected_hand)
            landmarks.append(landmark)
            hands.append(hand)
            valids.append(valid)
            boxes.append(box)
        landmark_batch = torch.stack(landmarks).to(device)
        hand_batch = torch.stack(hands).to(device)
        valid_batch = torch.stack(valids).to(device)
        box_batch = torch.stack(boxes).to(device)
        l_logits, l_features = landmark_model(landmark_batch, return_embeddings=True)
        h_features = hand_model.forward_features(hand_batch, valid_batch, box_batch)
        h_logits = hand_model.classifier(h_features)
        landmark_features.append(l_features.float().cpu().numpy())
        hand_features.append(h_features.float().cpu().numpy())
        landmark_logits.append(l_logits.float().cpu().numpy())
        hand_logits.append(h_logits.float().cpu().numpy())
        if device.type == "mps":
            torch.mps.synchronize()
    return {
        "landmark_features": np.concatenate(landmark_features).astype(np.float32),
        "hand_features": np.concatenate(hand_features).astype(np.float32),
        "landmark_logits": np.concatenate(landmark_logits).astype(np.float32),
        "hand_logits": np.concatenate(hand_logits).astype(np.float32),
        "targets": np.asarray([row.target for row in records], dtype=np.int64),
        "item_ids": np.asarray([row.item_id for row in records]),
    }


@torch.inference_mode()
def landmark_logits_for_records(
    records: list[PairRecord], model: SLTStage1V17, device: torch.device, batch_size: int
) -> np.ndarray:
    model.to(device).eval()
    expected = schema_fingerprint(V17Config())
    output = []
    for start in range(0, len(records), batch_size):
        values = torch.stack([
            load_v17_archive(row.landmark_path, expected)
            for row in records[start : start + batch_size]
        ]).to(device)
        output.append(model(values).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


@torch.inference_mode()
def visual_logits(
    cache_path: Path, checkpoint_path: Path, expected_ids: np.ndarray,
    device: torch.device, batch_size: int,
) -> np.ndarray:
    dataset = FrozenVisualFeatureDataset(cache_path, "train")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VisualSpeechTeacherV17(
        VisualSpeechTeacherV17Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    logits = []
    for features, valid, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits.append(model.forward_features(features.to(device), valid.to(device)).float().cpu().numpy())
    raw_ids = [str(value).replace("\\", "/") for value in dataset.item_ids]
    lookup = {}
    for index, value in enumerate(raw_ids):
        value = value.removeprefix("train/")
        for suffix in (".visual_speech_v17.npz", ".mp4"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
        lookup[value] = index
    if set(lookup) != set(map(str, expected_ids)):
        missing = sorted(set(map(str, expected_ids)) - set(lookup))[:3]
        extra = sorted(set(lookup) - set(map(str, expected_ids)))[:3]
        raise ValueError(f"visual cache IDs differ: missing={missing} extra={extra}")
    order = np.asarray([lookup[str(value)] for value in expected_ids])
    return np.concatenate(logits).astype(np.float32)[order]


def zscore_np(value: np.ndarray) -> np.ndarray:
    centered = value - value.mean(axis=1, keepdims=True)
    return centered / np.maximum(value.std(axis=1, keepdims=True), 1e-6)


def save_cache(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def build_caches(args: argparse.Namespace, device: torch.device) -> dict[str, Path]:
    labels = label_map(args.manifest)
    records = {
        "citizen_train": citizen_records(args.citizen_landmarks, args.citizen_hand, "train", labels, args.citizen_rejections),
        "citizen_val": citizen_records(args.citizen_landmarks, args.citizen_hand, "val", labels, args.citizen_rejections),
        "semlex_train": supplement_records(args.semlex_train_manifest, "semlex", args.supplement_hand, labels),
        "semlex_val": semlex_validation_records(args.semlex_val_manifest, args.semlex_val_hand, labels),
        "local_train": supplement_records(args.local_train_manifest, "local_deep_clean", args.local_hand, labels),
        "local_val": supplement_records(args.local_val_manifest, "local_deep_clean_val", args.local_hand, labels),
    }
    expected_counts = {
        "citizen_train": 1475, "citizen_val": 378, "semlex_train": 1388,
        "semlex_val": 978, "local_train": 13381, "local_val": 2896,
    }
    actual = {name: len(value) for name, value in records.items()}
    if actual != expected_counts:
        raise ValueError(f"unified record counts changed: {actual}")
    landmark_model, _ = load_landmark_model(args.landmark_checkpoint)
    hand_model, _ = load_hand_model(args.hand_checkpoint)
    paths = {}
    for name, rows in records.items():
        path = args.cache_dir / f"{name}.npz"
        paths[name] = path
        if path.is_file() and not args.rebuild_cache:
            continue
        started = time.monotonic()
        arrays = encode_records(rows, landmark_model, hand_model, device, args.encode_batch_size)
        if name == "citizen_train":
            teacher_landmark, _ = load_landmark_model(args.teacher_landmark_checkpoint)
            teacher_landmark_logits = landmark_logits_for_records(
                rows, teacher_landmark, device, args.encode_batch_size
            )
            teacher_landmark.to("cpu")
            mouth = visual_logits(
                args.mouth_train_cache, args.mouth_checkpoint, arrays["item_ids"],
                device, args.encode_batch_size,
            )
            lower = visual_logits(
                args.lower_train_cache, args.lower_checkpoint, arrays["item_ids"],
                device, args.encode_batch_size,
            )
            arrays["teacher_scores"] = (
                0.30 * zscore_np(teacher_landmark_logits)
                + 0.15 * zscore_np(mouth)
                + 0.35 * zscore_np(lower)
                + 0.20 * zscore_np(arrays["hand_logits"])
            ).astype(np.float32)
        else:
            arrays["teacher_scores"] = np.full(
                (len(rows), len(labels)), np.nan, dtype=np.float32
            )
        metadata = {
            "format": "slt_v17_unified_student_feature_cache",
            "domain_split": name,
            "samples": len(rows),
            "landmark_checkpoint_sha256": sha256_file(args.landmark_checkpoint),
            "hand_checkpoint_sha256": sha256_file(args.hand_checkpoint),
            "teacher_available": name == "citizen_train",
            "local_mouth_policy": "four_lip_points_zero" if name.startswith("local_") else "full_face",
            "citizen_test_accessed": False,
            "semlex_test_accessed": False,
            "local_test_accessed": False,
            "seconds": time.monotonic() - started,
        }
        save_cache(path, arrays, metadata)
        LOG.info("cached %s samples=%d seconds=%.1f", name, len(rows), metadata["seconds"])
    return paths


def load_cache(path: Path) -> dict[str, torch.Tensor | np.ndarray | dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        result = {key: payload[key].copy() for key in payload.files if key != "metadata_json"}
    if metadata.get("format") != "slt_v17_unified_student_feature_cache":
        raise ValueError(f"invalid unified cache: {path}")
    result["metadata"] = metadata
    return result


def metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    predicted = logits.argmax(1)
    top5 = logits.topk(5, dim=1).indices
    classes = logits.shape[1]
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets.numpy(), predicted.numpy()), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(0), 1)
    recall = true_positive / np.maximum(confusion.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "top1": 100.0 * float((predicted == targets).float().mean()),
        "top1_correct": int((predicted == targets).sum()),
        "top5": 100.0 * float((top5 == targets[:, None]).any(1).float().mean()),
        "macro_f1": 100.0 * float(f1.mean()),
        "samples": len(targets),
    }


def tensors(cache: dict[str, object]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(cache[key]) for key in (
        "landmark_features", "hand_features", "landmark_logits", "hand_logits",
        "targets", "teacher_scores",
    ))


@torch.inference_mode()
def evaluate_head(head: UnifiedFusionHeadV17, cache: dict[str, object]) -> dict[str, float | int]:
    head.eval()
    landmark, hand, landmark_logits, hand_logits, targets, _ = tensors(cache)
    return metrics(head(landmark, hand, landmark_logits, hand_logits), targets)


def selection_key(domain_metrics: dict[str, dict[str, float | int]]) -> tuple[float, int, int, int]:
    mean = sum(float(domain_metrics[name]["top1"]) for name in ("citizen", "semlex", "local")) / 3.0
    return (
        mean,
        int(domain_metrics["citizen"]["top1_correct"]),
        int(domain_metrics["semlex"]["top1_correct"]),
        int(domain_metrics["local"]["top1_correct"]),
    )


def train_seed(
    seed: int,
    train_caches: dict[str, dict[str, object]],
    val_caches: dict[str, dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = UnifiedMultimodalV17Config(
        num_classes=100, feature_dim=256, hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    head = UnifiedFusionHeadV17(config)
    source_order = ("citizen", "semlex", "local")
    joined = []
    source_index = []
    for index, source in enumerate(source_order):
        current = tensors(train_caches[source])
        joined.append(current)
        source_index.extend([index] * len(current[4]))
    combined = tuple(torch.cat([row[index] for row in joined]) for index in range(6))
    source_index_tensor = torch.tensor(source_index, dtype=torch.long)
    targets = combined[4]
    weights = torch.empty(len(targets), dtype=torch.double)
    for source_id in range(3):
        source_mask = source_index_tensor == source_id
        counts = torch.bincount(targets[source_mask], minlength=100).clamp_min(1)
        weights[source_mask] = (1.0 / 3.0) / counts[targets[source_mask]].double()
    sampler = WeightedRandomSampler(
        weights, len(targets), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    dataset = TensorDataset(*combined, source_index_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.02
    )
    output = args.output / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_key: tuple[float, int, int, int] | None = None
    stale = 0

    def consider(epoch: int) -> bool:
        nonlocal best_key, stale
        domain = {name: evaluate_head(head, val_caches[name]) for name in source_order}
        key = selection_key(domain)
        eligible = int(domain["citizen"]["top1_correct"]) >= 361
        row = {"epoch": epoch, "eligible": eligible, "selection_key": list(key), "domains": domain}
        history.append(row)
        if eligible and (best_key is None or key > best_key):
            best_key = key
            stale = 0
            torch.save({
                "format": "slt_v17_unified_fusion_head",
                "seed": seed,
                "epoch": epoch,
                "head_config": config.to_dict(),
                "head_state_dict": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
                "domain_metrics": domain,
                "selection_key": list(key),
                "citizen_floor_correct": 361,
                "test_evaluated": False,
            }, output / "best_head.pth")
            return True
        stale += int(epoch > 0)
        return False

    consider(0)
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = seen = 0
        for landmark, hand, landmark_logits, hand_logits, target, teacher, source in loader:
            optimizer.zero_grad(set_to_none=True)
            output_logits, residual = head(
                landmark, hand, landmark_logits, hand_logits, return_residual=True
            )
            hard = F.cross_entropy(output_logits, target, label_smoothing=args.label_smoothing)
            teacher_mask = source == 0
            if teacher_mask.any():
                temperature = args.temperature
                soft = F.kl_div(
                    F.log_softmax(output_logits[teacher_mask] / temperature, dim=1),
                    F.softmax(teacher[teacher_mask] / temperature, dim=1),
                    reduction="batchmean",
                ) * temperature**2
            else:
                soft = output_logits.sum() * 0.0
            residual_penalty = residual.square().mean()
            loss = hard + args.distillation_weight * soft + args.residual_weight * residual_penalty
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        scheduler.step()
        improved = consider(epoch)
        history[-1].update({"train_loss": total / seen, "lr": scheduler.get_last_lr()[0]})
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if epoch == 1 or epoch % 10 == 0 or improved:
            LOG.info("seed=%d epoch=%d loss=%.4f key=%s", seed, epoch, total / seen, history[-1]["selection_key"])
        if stale >= args.patience:
            break
    checkpoint = torch.load(output / "best_head.pth", map_location="cpu", weights_only=False)
    result = {
        "seed": seed,
        "epochs_completed": len(history) - 1,
        "selected_epoch": checkpoint["epoch"],
        "domain_metrics": checkpoint["domain_metrics"],
        "selection_key": checkpoint["selection_key"],
        "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "test_evaluated": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def package_winner(
    results: list[dict[str, object]], args: argparse.Namespace, cache_paths: dict[str, Path]
) -> dict[str, object]:
    winner = max(results, key=lambda row: tuple(row["selection_key"]))
    head_checkpoint = torch.load(
        args.output / f"seed_{winner['seed']}" / "best_head.pth",
        map_location="cpu", weights_only=False,
    )
    landmark_checkpoint = torch.load(args.landmark_checkpoint, map_location="cpu", weights_only=False)
    hand_checkpoint = torch.load(args.hand_checkpoint, map_location="cpu", weights_only=False)
    provenance = {
        "selection_protocol": "equal_domain_mean_then_citizen_semlex_local_correct_tiebreakers",
        "candidate_seeds": list(SEEDS),
        "source_replay": {"citizen": 0.34, "semlex": 0.33, "local": 0.33},
        "landmark_encoder_frozen": True,
        "hand_encoder_frozen": True,
        "local_mouth_policy": "zero_only_four_lip_points",
        "citizen_teacher_distillation": "fixed_four_stream_30_15_35_20",
        "semlex_teacher_distillation": False,
        "local_teacher_distillation": False,
        "cache_sha256": {name: sha256_file(path) for name, path in cache_paths.items()},
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    package = {
        "format": "slt_stage1_unified_multimodal_v17",
        "format_version": 1,
        "epoch": head_checkpoint["epoch"],
        "seed": head_checkpoint["seed"],
        "landmark_model_config": landmark_checkpoint["model_config"],
        "landmark_model_state_dict": landmark_checkpoint["model_state_dict"],
        "hand_model_config": hand_checkpoint["model_config"],
        "hand_model_state_dict": hand_checkpoint["model_state_dict"],
        "head_config": head_checkpoint["head_config"],
        "head_state_dict": head_checkpoint["head_state_dict"],
        "label_to_index": landmark_checkpoint["label_to_index"],
        "manifest_sha256": landmark_checkpoint["manifest_sha256"],
        "landmark_schema_fingerprint": landmark_checkpoint["schema_fingerprint"],
        "hand_schema_fingerprint": hand_checkpoint["schema_fingerprint"],
        "source_checkpoints": {
            "landmark": {"path": str(args.landmark_checkpoint), "sha256": sha256_file(args.landmark_checkpoint)},
            "hand": {"path": str(args.hand_checkpoint), "sha256": sha256_file(args.hand_checkpoint)},
            "teacher_landmark": {"path": str(args.teacher_landmark_checkpoint), "sha256": sha256_file(args.teacher_landmark_checkpoint)},
            "teacher_mouth": {"path": str(args.mouth_checkpoint), "sha256": sha256_file(args.mouth_checkpoint)},
            "teacher_lower": {"path": str(args.lower_checkpoint), "sha256": sha256_file(args.lower_checkpoint)},
        },
        "validation_metrics": head_checkpoint["domain_metrics"],
        "selection_key": head_checkpoint["selection_key"],
        "training_data_provenance": provenance,
        "test_evaluated": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    temporary = args.output / "best_model.pth.tmp"
    torch.save(package, temporary)
    temporary.replace(args.output / "best_model.pth")
    result = {
        "selected_seed": winner["seed"],
        "selected_epoch": winner["selected_epoch"],
        "validation_metrics": winner["domain_metrics"],
        "selection_key": winner["selection_key"],
        "checkpoint": str(args.output / "best_model.pth"),
        "checkpoint_sha256": sha256_file(args.output / "best_model.pth"),
        "candidate_results": results,
        "test_evaluated": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device)
    if device.type == "mps":
        if not 0 < args.mps_memory_fraction <= 0.25:
            raise ValueError("MPS memory fraction must be in (0, 0.25]")
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    cache_paths = build_caches(args, device)
    train_caches = {name: load_cache(cache_paths[f"{name}_train"]) for name in ("citizen", "semlex", "local")}
    val_caches = {name: load_cache(cache_paths[f"{name}_val"]) for name in ("citizen", "semlex", "local")}
    results = [train_seed(seed, train_caches, val_caches, args) for seed in SEEDS]
    return package_winner(results, args, cache_paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--citizen-landmarks", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    parser.add_argument("--citizen-hand", type=Path, default=Path("data/local/citizen100_v17/hand_mobileclip2_s0"))
    parser.add_argument("--citizen-rejections", type=Path, default=Path("data/local/citizen100_v17/rejections.csv"))
    parser.add_argument("--semlex-train-manifest", type=Path, default=Path("data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json"))
    parser.add_argument("--semlex-val-manifest", type=Path, default=Path("data/local/semlex_citizen100_val_audit/selection_plan.json"))
    parser.add_argument("--semlex-val-hand", type=Path, default=Path("data/local/semlex_citizen100_val_audit/hand_mobileclip2_s0"))
    parser.add_argument("--supplement-hand", type=Path, default=Path("data/local/hand_mobileclip2_supplements_v17"))
    parser.add_argument("--local-train-manifest", type=Path, default=Path("data/local/local_deep_clean_v17/train_final_manifest.json"))
    parser.add_argument("--local-val-manifest", type=Path, default=Path("data/local/local_deep_clean_v17/val_final_manifest.json"))
    parser.add_argument("--local-hand", type=Path, default=Path("data/local/local_deep_clean_v17/hand_mobileclip2_s0"))
    parser.add_argument("--landmark-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_local_deep_clean_mouth_masked_replay_ft_v1/best_promotion_gate_model.pth"))
    parser.add_argument("--hand-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_hand_mobileclip2_local_deep_clean_replay_ft_v1/best_model.pth"))
    parser.add_argument("--teacher-landmark-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_citizen_semlex_full_clean_balanced/best_model.pth"))
    parser.add_argument("--mouth-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_visual_speech_auto_avsr_mouth_frozen/best_model.pth"))
    parser.add_argument("--lower-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_visual_speech_auto_avsr_lower_face_frozen/best_model.pth"))
    parser.add_argument("--mouth-train-cache", type=Path, default=Path("data/local/citizen100_v17/visual_speech_auto_avsr/mouth_train.npz"))
    parser.add_argument("--lower-train-cache", type=Path, default=Path("data/local/citizen100_v17/visual_speech_auto_avsr/lower_face_train.npz"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/generated/unified_multimodal_student_v17"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--distillation-weight", type=float, default=0.50)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
