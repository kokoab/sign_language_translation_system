#!/usr/bin/env python3
"""Adapt a separate unified Stage-1 head to genuine local phrase segments.

The landmark and hand encoders remain frozen.  Existing Citizen, SemLex, and local
isolated feature caches are replayed with base-model distillation, while only the
new fusion head is optimized.  Validation/test paths containing ``test`` are refused.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.geometry_v17 import resample_features
from active.v17.train_unified_multimodal_student_v17 import load_cache, metrics


DEFAULT_BASE = Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth")
DEFAULT_CACHE = Path("artifacts/generated/unified_multimodal_student_v17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_hand_path(rgb_path: Path, hand_root: Path) -> Path:
    name = rgb_path.name.replace(
        ".stage2_rgb_v17.npz", ".stage2_hand_mobileclip2_v17.npz"
    )
    return hand_root / name


def hand_segment(
    embeddings: np.ndarray, valid: np.ndarray, boxes: np.ndarray,
    start: int, end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = max(0, int(np.floor(start / 2)))
    last = min(len(embeddings), max(first + 1, int(np.ceil(end / 2))))
    positions = np.rint(np.linspace(first, last - 1, 16)).astype(int)
    return embeddings[positions], valid[positions], boxes[positions]


@torch.inference_mode()
def phrase_cache(
    model, rgb_root: Path, hand_root: Path, labels: dict[str, int],
    device: torch.device, *, variants: int, jitter: float, seed: int,
    activity_crops: bool = False,
) -> dict[str, torch.Tensor]:
    generator = random.Random(seed)
    landmarks: list[np.ndarray] = []
    hands: list[np.ndarray] = []
    valids: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    targets: list[int] = []
    for rgb_path in sorted(rgb_root.glob("*.npz")):
        with np.load(rgb_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            landmark = np.concatenate(
                payload["landmarks"].astype(np.float32, copy=False), axis=0
            )
        with np.load(paired_hand_path(rgb_path, hand_root), allow_pickle=False) as payload:
            embeddings = np.concatenate(
                payload["embeddings"].astype(np.float32, copy=False), axis=0
            )
            valid = np.concatenate(payload["valid"].astype(np.bool_), axis=0)
            hand_boxes = np.concatenate(
                payload["boxes_normalized"].astype(np.float32, copy=False), axis=0
            )
        sequence = [str(item) for item in metadata["target_sequence"]]
        if any(label not in labels for label in sequence):
            continue
        width = len(landmark) / len(sequence)
        for position, label in enumerate(sequence):
            for variant in range(variants):
                radius = width * jitter
                left = position * width
                right = (position + 1) * width
                if activity_crops and variant:
                    crop = (
                        (0.15, 0.00), (0.30, 0.05), (0.40, 0.15)
                    )[min(variant - 1, 2)]
                    left += width * crop[0]
                    right -= width * crop[1]
                    left += generator.uniform(-radius, radius)
                    right += generator.uniform(-radius, radius)
                elif variant:
                    left += generator.uniform(-radius, radius)
                    right += generator.uniform(-radius, radius)
                context = width * 0.06
                start = max(0, int(round(left - context)))
                end = min(len(landmark), int(round(right + context)))
                if end - start < 4:
                    continue
                hand, hand_valid, box = hand_segment(
                    embeddings, valid, hand_boxes, start, end
                )
                landmarks.append(
                    resample_features(landmark[start:end], 32).astype(np.float32)
                )
                hands.append(hand)
                valids.append(hand_valid)
                boxes.append(box)
                targets.append(labels[label])
    if not targets:
        raise ValueError(f"no compatible phrase segments under {rgb_root}")

    landmark_features = []
    hand_features = []
    landmark_logits = []
    hand_logits = []
    model.to(device).eval()
    for start in range(0, len(targets), 64):
        landmark_batch = torch.from_numpy(
            np.stack(landmarks[start : start + 64])
        ).to(device)
        hand_batch = torch.from_numpy(np.stack(hands[start : start + 64])).to(device)
        valid_batch = torch.from_numpy(np.stack(valids[start : start + 64])).to(device)
        box_batch = torch.from_numpy(np.stack(boxes[start : start + 64])).to(device)
        l_logits, l_features = model.landmark_model(
            landmark_batch, return_embeddings=True
        )
        h_features = model.hand_model.forward_features(
            hand_batch, valid_batch, box_batch
        )
        h_logits = model.hand_model.classifier(h_features)
        landmark_features.append(l_features.cpu())
        hand_features.append(h_features.cpu())
        landmark_logits.append(l_logits.cpu())
        hand_logits.append(h_logits.cpu())
    model.to("cpu")
    return {
        "landmark_features": torch.cat(landmark_features),
        "hand_features": torch.cat(hand_features),
        "landmark_logits": torch.cat(landmark_logits),
        "hand_logits": torch.cat(hand_logits),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


def cache_tensors(cache: dict[str, object]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(cache[key]) for key in (
        "landmark_features", "hand_features", "landmark_logits", "hand_logits",
        "targets",
    ))


@torch.inference_mode()
def evaluate(head, values: tuple[torch.Tensor, ...]) -> dict[str, float | int]:
    head.eval()
    logits = head(*values[:4])
    return metrics(logits.cpu(), values[4].cpu())


@torch.inference_mode()
def evaluate_pair(
    head, values: tuple[torch.Tensor, ...], pair: tuple[int, int],
) -> dict[str, float | int]:
    head.eval()
    target = values[4]
    usable = (target == pair[0]) | (target == pair[1])
    if not usable.any():
        return {"top1": 0.0, "top1_correct": 0, "samples": 0}
    logits = head(*(value[usable] for value in values[:4]))[:, list(pair)]
    expected = (target[usable] == pair[1]).long()
    correct = int((logits.argmax(1).cpu() == expected.cpu()).sum())
    return {
        "top1": 100.0 * correct / len(expected),
        "top1_correct": correct,
        "samples": int(len(expected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--isolated-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--phrase-rgb", type=Path,
        default=Path("data/local/stage2_v17_multimodal"),
    )
    parser.add_argument(
        "--phrase-hand", type=Path,
        default=Path("data/local/stage2_v17_hand_mobileclip2"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "artifacts/models/stage1_v17_unified_phrase_pair_adapt_reel_v3"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--parameter-anchor", type=float, default=0.01)
    parser.add_argument("--pair-loss-weight", type=float, default=2.0)
    parser.add_argument("--pair-sampling-multiplier", type=float, default=4.0)
    parser.add_argument("--train-variants", type=int, default=4)
    parser.add_argument("--boundary-jitter", type=float, default=0.10)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=27117)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if any("test" in {part.lower() for part in path.parts} for path in (
        args.phrase_rgb, args.phrase_hand,
    )):
        raise ValueError("phrase adaptation refuses test paths")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model, checkpoint = load_model(args.base)
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    pair = (labels["GOOD"], labels["THANKYOU"])
    phrase_train = phrase_cache(
        model,
        args.phrase_rgb / "train/local_phrases",
        args.phrase_hand / "train/local_phrases",
        labels, device,
        variants=args.train_variants, jitter=args.boundary_jitter, seed=args.seed,
        activity_crops=True,
    )
    phrase_val = phrase_cache(
        model,
        args.phrase_rgb / "validation/local_phrases",
        args.phrase_hand / "validation/local_phrases",
        labels, device,
        variants=1, jitter=0.0, seed=args.seed,
    )
    phrase_val_activity = phrase_cache(
        model,
        args.phrase_rgb / "validation/local_phrases",
        args.phrase_hand / "validation/local_phrases",
        labels, device,
        variants=4, jitter=0.02, seed=args.seed + 1,
        activity_crops=True,
    )
    caches = {
        source: {
            split: cache_tensors(load_cache(args.isolated_cache / f"{source}_{split}.npz"))
            for split in ("train", "val")
        }
        for source in ("citizen", "semlex", "local")
    }
    train_values = [caches[source]["train"] for source in ("citizen", "semlex", "local")]
    train_values.append(tuple(phrase_train[key] for key in (
        "landmark_features", "hand_features", "landmark_logits", "hand_logits", "targets"
    )))
    combined = tuple(torch.cat([value[index] for value in train_values]) for index in range(5))
    source_ids = torch.cat([
        torch.full((len(value[4]),), index, dtype=torch.long)
        for index, value in enumerate(train_values)
    ])
    shares = (0.30, 0.20, 0.25, 0.25)
    pair_counts = Counter(zip(source_ids.tolist(), combined[4].tolist()))
    class_counts = Counter(source for source, _ in pair_counts)
    weights = torch.tensor([
        shares[source] / class_counts[source] / pair_counts[(source, int(target))]
        for source, target in zip(source_ids.tolist(), combined[4].tolist())
    ], dtype=torch.double)
    pair_mask = (combined[4] == pair[0]) | (combined[4] == pair[1])
    weights[pair_mask] *= args.pair_sampling_multiplier
    sampler = WeightedRandomSampler(
        weights, args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(*combined, source_ids), batch_size=args.batch_size,
        sampler=sampler, num_workers=0,
    )

    head = copy.deepcopy(model.fusion_head).to(device)
    teacher = copy.deepcopy(model.fusion_head).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    anchors = {name: value.detach().clone().to(device) for name, value in head.named_parameters()}
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    validation = {
        "citizen": caches["citizen"]["val"],
        "semlex": caches["semlex"]["val"],
        "local": caches["local"]["val"],
        "phrase": tuple(phrase_val[key] for key in (
            "landmark_features", "hand_features", "landmark_logits", "hand_logits", "targets"
        )),
        "phrase_activity": tuple(phrase_val_activity[key] for key in (
            "landmark_features", "hand_features", "landmark_logits", "hand_logits",
            "targets",
        )),
    }
    baseline = {name: evaluate(head.cpu(), value) for name, value in validation.items()}
    baseline_pair = {
        name: evaluate_pair(head, value, pair) for name, value in validation.items()
    }
    head.to(device)
    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    started = time.perf_counter()
    print(json.dumps({
        "baseline": baseline, "baseline_pair": baseline_pair, "device": str(device)
    }, indent=2))
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        seen = 0
        for landmark, hand, l_logits, h_logits, target, source in loader:
            inputs = tuple(value.to(device) for value in (landmark, hand, l_logits, h_logits))
            target = target.to(device)
            source = source.to(device)
            output = head(*inputs)
            hard = F.cross_entropy(output, target)
            pair_rows = (target == pair[0]) | (target == pair[1])
            if pair_rows.any():
                pair_target = (target[pair_rows] == pair[1]).long()
                pair_loss = F.cross_entropy(
                    output[pair_rows][:, list(pair)], pair_target
                )
            else:
                pair_loss = output.sum() * 0.0
            replay = source != 3
            if replay.any():
                with torch.no_grad():
                    teacher_output = teacher(*(value[replay] for value in inputs))
                distill = F.kl_div(
                    F.log_softmax(output[replay] / 2.0, dim=1),
                    F.softmax(teacher_output / 2.0, dim=1),
                    reduction="batchmean",
                ) * 4.0
            else:
                distill = output.sum() * 0.0
            anchor = sum(
                (parameter - anchors[name]).square().mean()
                for name, parameter in head.named_parameters()
            )
            loss = (
                hard + args.pair_loss_weight * pair_loss
                + args.distill_weight * distill + args.parameter_anchor * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(target)
            seen += len(target)
        scheduler.step()
        head.to("cpu")
        domain = {name: evaluate(head, value) for name, value in validation.items()}
        pair_domain = {
            name: evaluate_pair(head, value, pair) for name, value in validation.items()
        }
        head.to(device)
        eligible = all(
            domain[name]["top1"] >= baseline[name]["top1"] - 1.0
            for name in ("citizen", "semlex", "local")
        )
        row = {
            "epoch": epoch, "loss": total / seen, "eligible": eligible,
            "domains": domain, "learning_rate": optimizer.param_groups[0]["lr"],
            "pair_domains": pair_domain,
        }
        history.append(row)
        print(json.dumps(row))
        key = (
            float(pair_domain["phrase_activity"]["top1"]),
            float(domain["phrase_activity"]["top1"]),
            float(domain["phrase"]["top1"]),
            sum(float(domain[name]["top1"]) for name in ("citizen", "semlex", "local")),
        )
        if eligible and (best is None or key > tuple(best["selection_key"])):
            best = {
                **row, "selection_key": list(key),
                "head_state_dict": copy.deepcopy(head.state_dict()),
            }
    if best is None:
        raise RuntimeError("no unified adaptation passed all isolated-domain gates")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected = copy.deepcopy(checkpoint)
    selected["head_state_dict"] = best.pop("head_state_dict")
    selected["epoch"] = int(best["epoch"])
    selected["validation_metrics"] = best["domains"]
    selected["phrase_adaptation"] = {
        "base": str(args.base), "base_sha256": sha256(args.base),
        "phrase_train_segments": len(phrase_train["targets"]),
        "phrase_validation_segments": len(phrase_val["targets"]),
        "phrase_activity_validation_segments": len(phrase_val_activity["targets"]),
        "encoders_frozen": True, "selected": best,
        "pair_labels": ["GOOD", "THANKYOU"],
        "pair_loss_weight": args.pair_loss_weight,
        "pair_sampling_multiplier": args.pair_sampling_multiplier,
        "citizen_test_accessed": False, "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    output = args.output_dir / "best_model.pth"
    torch.save(selected, output)
    result = {
        "format": "slt_stage1_unified_phrase_adaptation_v17",
        "base": str(args.base), "base_sha256": sha256(args.base),
        "output": str(output), "output_sha256": sha256(output),
        "baseline": baseline, "baseline_pair": baseline_pair,
        "selected": best, "history": history,
        "elapsed_seconds": time.perf_counter() - started, "device": str(device),
        "citizen_test_accessed": False, "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
