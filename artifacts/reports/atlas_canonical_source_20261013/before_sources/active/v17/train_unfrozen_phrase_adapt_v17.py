#!/usr/bin/env python3
"""Does unfreezing the unified Stage-1 encoders on phrase segments raise the oracle ceiling?

The shipped verifier (stage1_v17_unified_phrase_activity_adapt_reel_v2) adapted only the
fusion head; its landmark and hand encoders were trained on isolated clips. With perfect
held-out ASLLRP intervals it reaches 66.7%. This trainer runs two arms from the same
pre-phrase base (stage1_v17_unified_multimodal_student_v1) on the same data and seed:

  frozen   - fusion head only (control: separates "new split" from "unfreezing")
  unfrozen - landmark encoder + hand temporal encoder + head, encoders at a lower LR

Phrase supervision is the approved v2 manifest's TRAIN membership only (resolved by video
hash onto the multimodal archives that carry hand embeddings). The shipped head was fitted
on an older split that contains 158 approved-validation clips, so it is not used as the
initialisation. Isolated Citizen/SemLex/local train clips are replayed with distillation to
the frozen base. Checkpoints are eligible only if every isolated validation domain holds the
recipe's floors, which are set against the shipped model. The held-out ASLLRP oracle is
reported every epoch and never used for selection. No test split is read.
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from active.v17.approved_phrase_data_v17 import verify_manifest
from active.v17.export_unified_multimodal_coreml_v17 import load_model
from active.v17.geometry_v17 import resample_features
from active.v17.schema_hand_mobileclip2_v17 import (
    HandMobileCLIP2V17Config, schema_fingerprint as hand_schema_fingerprint,
)
from active.v17.schema_v17 import V17Config, schema_fingerprint
from active.v17.train_stage_1_v17 import load_v17_archive, mask_mouth_nodes_v17
from active.v17.train_unified_multimodal_student_v17 import (
    build_parser as student_parser, citizen_records, label_map, load_hand, metrics,
    semlex_validation_records, supplement_records,
)
from active.v17.train_unified_phrase_adapt_v17 import hand_segment, paired_hand_path

RECIPE = Path("active/v17/unfrozen_phrase_adapt_manifest_20260925.json")
SOURCES = ("citizen", "semlex", "local")
PHRASE_SOURCES = ("local_phrases", "asllrp_contiguous")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def isolated_raw(cache_dir: Path, extractor: str = "apple",
                 mediapipe_root: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Raw encoder inputs for the exact record lists behind the shipped feature caches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    student = student_parser().parse_args([])
    labels = label_map(student.manifest)
    builders = {
        "citizen_train": lambda: citizen_records(student.citizen_landmarks, student.citizen_hand, "train", labels, student.citizen_rejections),
        "citizen_val": lambda: citizen_records(student.citizen_landmarks, student.citizen_hand, "val", labels, student.citizen_rejections),
        "semlex_train": lambda: supplement_records(student.semlex_train_manifest, "semlex", student.supplement_hand, labels),
        "semlex_val": lambda: semlex_validation_records(student.semlex_val_manifest, student.semlex_val_hand, labels),
        "local_train": lambda: supplement_records(student.local_train_manifest, "local_deep_clean", student.local_hand, labels),
        "local_val": lambda: supplement_records(student.local_val_manifest, "local_deep_clean_val", student.local_hand, labels),
    }
    expected_landmark = schema_fingerprint(V17Config())
    expected_hand = hand_schema_fingerprint(HandMobileCLIP2V17Config())
    if extractor == "mediapipe_full":
        # Android family: the same Apple record lists swapped to their MediaPipe mirrors.
        from active.v17 import train_unified_multimodal_student_v17 as unified
        unified._EXTRACTOR = "mediapipe_full"
        expected_landmark, expected_hand = unified.expected_schemas()
        builders = {name: (lambda build=build: unified.mediapipe_records(build(), mediapipe_root)[0])
                    for name, build in builders.items()}
    output = {}
    for name, build in builders.items():
        path = cache_dir / f"{name}.npz"
        if not path.is_file():
            started = time.monotonic()
            rows = build()
            landmarks, hands, valids, boxes = [], [], [], []
            for row in rows:
                landmark = load_v17_archive(row.landmark_path, expected_landmark)
                if row.mask_mouth:
                    landmark = mask_mouth_nodes_v17(landmark)
                hand, valid, box = load_hand(row.hand_path, expected_hand)
                landmarks.append(landmark.numpy())
                hands.append(hand.numpy().astype(np.float16))
                valids.append(valid.numpy())
                boxes.append(box.numpy())
            np.savez(
                path, landmarks=np.stack(landmarks), hand_embeddings=np.stack(hands),
                hand_valid=np.stack(valids), hand_boxes=np.stack(boxes),
                targets=np.asarray([row.target for row in rows], np.int64),
                item_ids=np.asarray([row.item_id for row in rows]),
            )
            print(f"cached raw {name}: {len(rows)} in {time.monotonic() - started:.0f}s", flush=True)
        with np.load(path, allow_pickle=False) as payload:
            output[name] = {key: payload[key] for key in payload.files}
    return output


def approved_membership(manifest: Path) -> dict[str, str]:
    """video_sha256 -> approved role, for the phrase sources this recipe admits."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    roles = {}
    for row in payload["admitted"]:
        if row.get("source") in PHRASE_SOURCES:
            roles[str(row["video_sha256"])] = str(row["role"])
    return roles


def phrase_raw(
    rgb_root: Path, hand_root: Path, roles: dict[str, str], want_role: str,
    labels: dict[str, int], *, variants: int, jitter: float, seed: int,
    activity_crops: bool, sources: tuple[str, ...] = PHRASE_SOURCES,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Equal-width per-sign segments (the shipped recipe), as raw encoder inputs."""
    generator = random.Random(seed)
    keep = {key: [] for key in ("landmarks", "hand_embeddings", "hand_valid", "hand_boxes", "targets")}
    used = []
    for split in ("train", "validation"):
        for source in sources:
            for rgb_path in sorted((rgb_root / split / source).glob("*.npz")):
                with np.load(rgb_path, allow_pickle=False) as payload:
                    metadata = json.loads(str(payload["metadata_json"].item()))
                    if roles.get(str(metadata.get("video_sha256"))) != want_role:
                        continue
                    landmark = np.concatenate(payload["landmarks"].astype(np.float32), axis=0)
                with np.load(paired_hand_path(rgb_path, hand_root / split / source), allow_pickle=False) as payload:
                    embeddings = np.concatenate(payload["embeddings"].astype(np.float32), axis=0)
                    valid = np.concatenate(payload["valid"].astype(np.bool_), axis=0)
                    hand_boxes = np.concatenate(payload["boxes_normalized"].astype(np.float32), axis=0)
                sequence = [str(item) for item in metadata["target_sequence"]]
                if any(label not in labels for label in sequence):
                    continue
                used.append(str(metadata["video_sha256"]))
                width = len(landmark) / len(sequence)
                for position, label in enumerate(sequence):
                    for variant in range(variants):
                        radius = width * jitter
                        left, right = position * width, (position + 1) * width
                        if activity_crops and variant:
                            crop = ((0.15, 0.00), (0.30, 0.05), (0.40, 0.15))[min(variant - 1, 2)]
                            left += width * crop[0]
                            right -= width * crop[1]
                        if variant:
                            left += generator.uniform(-radius, radius)
                            right += generator.uniform(-radius, radius)
                        context = width * 0.06
                        start = max(0, int(round(left - context)))
                        end = min(len(landmark), int(round(right + context)))
                        if end - start < 4:
                            continue
                        hand, hand_valid, box = hand_segment(embeddings, valid, hand_boxes, start, end)
                        keep["landmarks"].append(resample_features(landmark[start:end], 32).astype(np.float32))
                        keep["hand_embeddings"].append(hand.astype(np.float16))
                        keep["hand_valid"].append(hand_valid)
                        keep["hand_boxes"].append(box)
                        keep["targets"].append(labels[label])
    return {key: np.asarray(value) if key == "targets" else np.stack(value) for key, value in keep.items()}, used


def tensors(values: dict[str, np.ndarray]) -> tuple[torch.Tensor, ...]:
    return (
        torch.from_numpy(values["landmarks"].astype(np.float32)),
        torch.from_numpy(values["hand_embeddings"]),
        torch.from_numpy(values["hand_valid"].astype(np.bool_)),
        torch.from_numpy(values["hand_boxes"].astype(np.float32)),
        torch.from_numpy(values["targets"].astype(np.int64)),
    )


@torch.inference_mode()
def logits_for(model, values: tuple[torch.Tensor, ...], device: torch.device, batch: int = 256) -> torch.Tensor:
    model.eval()
    output = []
    for start in range(0, len(values[0]), batch):
        inputs = [value[start:start + batch].to(device) for value in values[:4]]
        inputs[1] = inputs[1].float()
        output.append(model(*inputs).float().cpu())
    return torch.cat(output)


def evaluate(model, values, device) -> dict[str, float | int]:
    return metrics(logits_for(model, values, device), values[4])


def oracle_score(model, oracle, device) -> list[dict[str, object]]:
    """Top-1 per pad on held-out ASLLRP intervals; unusable windows count as misses."""
    if oracle is None:
        return []
    arrays, rows, index = oracle
    logits = logits_for(model, arrays, device)
    predicted = logits.argmax(1).tolist()
    out = []
    for pad in sorted({row["pad"] for row in rows}):
        chosen = [row for row in rows if row["pad"] == pad]
        hits = sum(
            1 for row in chosen
            if row["usable"] and predicted[row["array_index"]] == index[row["gloss"]]
        )
        out.append({"pad": pad, "n": len(chosen), "correct": hits, "top1": 100.0 * hits / len(chosen)})
    return out


def load_oracle(path: Path, labels: dict[str, int]):
    if not path.with_suffix(".npz").is_file():
        return None
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as payload:
        arrays = (
            torch.from_numpy(payload["landmarks"].astype(np.float32)),
            torch.from_numpy(payload["hand_embeddings"].astype(np.float16)),
            torch.from_numpy(payload["hand_valid"].astype(np.bool_)),
            torch.from_numpy(payload["hand_boxes"].astype(np.float32)),
        )
    rows = json.loads(path.with_suffix(".json").read_text())["rows"]
    position = 0
    for row in rows:
        row["array_index"] = position if row["usable"] else None
        position += bool(row["usable"])
    return arrays, rows, labels


def eligible(domains: dict, floors: dict[str, float]) -> bool:
    return all(domains[name]["top1"] >= floors[name] - 1e-9 for name in SOURCES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("frozen", "unfrozen"), required=True)
    parser.add_argument("--recipe", type=Path, default=RECIPE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--micro-batch", type=int, default=0,
                        help="memory-only: split each recipe batch; 0 = whole batch")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if not recipe.get("training_ready"):
        raise RuntimeError("recipe is not training-ready")
    for key in ("base", "reference", "approved_manifest"):
        if sha256(Path(recipe[key])) != recipe[f"{key}_sha256"]:
            raise ValueError(f"{key} hash changed: {recipe[key]}")
    verify_manifest(Path(recipe["approved_manifest"]))
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    epochs = args.epochs or int(recipe["epochs"])
    micro = args.micro_batch or int(recipe["batch_size"])
    seed = int(recipe["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )

    model, checkpoint = load_model(Path(recipe["base"]))
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    pair = (labels["GOOD"], labels["THANKYOU"])
    reference = torch.load(recipe["reference"], map_location="cpu", weights_only=False)
    floors = {
        name: max(float(reference["validation_metrics"][name]["top1"]) - float(recipe["isolated_tolerance"][name]), 0.0)
        for name in SOURCES
    }

    raw = isolated_raw(Path(recipe["raw_cache_dir"]))
    roles = approved_membership(Path(recipe["approved_manifest"]))
    rgb_root, hand_root = Path(recipe["phrase_rgb"]), Path(recipe["phrase_hand"])
    phrase_train, train_ids = phrase_raw(rgb_root, hand_root, roles, "train", labels,
                                         variants=4, jitter=0.10, seed=seed, activity_crops=True)
    # Selection sees local phrase validation only: the 12 approved ASLLRP validation videos
    # are the held-out oracle and must not steer checkpoint choice.
    phrase_val, val_ids = phrase_raw(rgb_root, hand_root, roles, "validation", labels,
                                     variants=1, jitter=0.0, seed=seed, activity_crops=False,
                                     sources=("local_phrases",))
    phrase_activity, _ = phrase_raw(rgb_root, hand_root, roles, "validation", labels,
                                    variants=4, jitter=0.02, seed=seed + 1, activity_crops=True,
                                    sources=("local_phrases",))
    if set(train_ids) & set(val_ids):
        raise ValueError("approved phrase train/validation overlap")
    print(json.dumps({"phrase_train_clips": len(train_ids), "phrase_train_segments": len(phrase_train["targets"]),
                      "phrase_val_clips": len(val_ids), "phrase_val_segments": len(phrase_val["targets"]),
                      "floors": floors, "device": str(device)}), flush=True)

    train_parts = [tensors(raw[f"{source}_train"]) for source in SOURCES] + [tensors(phrase_train)]
    combined = tuple(torch.cat([part[index] for part in train_parts]) for index in range(5))
    source_ids = torch.cat([torch.full((len(part[4]),), index, dtype=torch.long) for index, part in enumerate(train_parts)])
    teacher = copy.deepcopy(model).to(device)
    teacher_logits = logits_for(teacher, combined, device)
    teacher.to("cpu")
    del teacher

    shares = recipe["source_shares"]
    pair_counts = Counter(zip(source_ids.tolist(), combined[4].tolist()))
    class_counts = Counter(source for source, _ in pair_counts)
    weights = torch.tensor([
        shares[source] / class_counts[source] / pair_counts[(source, int(target))]
        for source, target in zip(source_ids.tolist(), combined[4].tolist())
    ], dtype=torch.double)
    weights[(combined[4] == pair[0]) | (combined[4] == pair[1])] *= float(recipe["pair_sampling_multiplier"])
    sampler = WeightedRandomSampler(weights, int(recipe["samples_per_epoch"]), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(*combined, teacher_logits, source_ids),
                        batch_size=int(recipe["batch_size"]), sampler=sampler, num_workers=0)

    encoders = list(model.landmark_model.parameters()) + list(model.hand_model.parameters())
    if args.arm == "frozen":
        for parameter in encoders:
            parameter.requires_grad = False
        groups = [{"params": list(model.fusion_head.parameters()), "lr": float(recipe["head_lr"])}]
    else:
        groups = [
            {"params": list(model.fusion_head.parameters()), "lr": float(recipe["head_lr"])},
            {"params": encoders, "lr": float(recipe["encoder_lr"])},
        ]
    trainable = [p for group in groups for p in group["params"]]
    anchors = [p.detach().clone().to(device) for p in trainable]
    model.to(device)
    optimizer = torch.optim.AdamW(groups, weight_decay=float(recipe["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    validation = {name: tensors(raw[f"{name}_val"]) for name in SOURCES}
    validation["phrase"] = tensors(phrase_val)
    validation["phrase_activity"] = tensors(phrase_activity)
    oracle = load_oracle(Path(recipe["oracle_inputs"]), labels)

    def snapshot(epoch: int, loss: float) -> dict[str, object]:
        domains = {name: evaluate(model, value, device) for name, value in validation.items()}
        return {"epoch": epoch, "loss": loss, "eligible": eligible(domains, floors),
                "domains": domains, "oracle_heldout_asllrp": oracle_score(model, oracle, device)}

    history = [snapshot(0, float("nan"))]
    print(json.dumps(history[0]), flush=True)
    best, started = None, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if args.arm == "frozen":
            model.landmark_model.eval()
            model.hand_model.eval()
        total = seen = 0.0
        for batch in loader:
            # Exact gradient accumulation: every term is a per-row sum divided by the FULL
            # batch's row count, so micro-batching only changes memory, not the objective.
            target_all, source_all = batch[4], batch[6]
            pair_all = (target_all == pair[0]) | (target_all == pair[1])
            n_rows, n_pair, n_replay = len(target_all), int(pair_all.sum()), int((source_all != 3).sum())
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for lo in range(0, n_rows, micro):
                landmark, hand, valid, box, target, t_logits, source = (
                    value[lo:lo + micro].to(device) for value in batch)
                output = model(landmark, hand.float(), valid, box)
                hard = F.cross_entropy(output, target, reduction="sum") / n_rows
                pair_rows = (target == pair[0]) | (target == pair[1])
                pair_loss = (F.cross_entropy(output[pair_rows][:, list(pair)], (target[pair_rows] == pair[1]).long(),
                                             reduction="sum") / n_pair if pair_rows.any() else output.sum() * 0.0)
                replay = source != 3
                distill = (F.kl_div(F.log_softmax(output[replay] / 2.0, 1), F.softmax(t_logits[replay] / 2.0, 1),
                                    reduction="sum") / n_replay * 4.0 if replay.any() else output.sum() * 0.0)
                loss = (hard + float(recipe["pair_loss_weight"]) * pair_loss
                        + float(recipe["distill_weight"]) * distill)
                loss.backward()
                step_loss += float(loss.detach().cpu())
            anchor = float(recipe["parameter_anchor"]) * sum(
                (p - a).square().mean() for p, a in zip(trainable, anchors))
            anchor.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += (step_loss + float(anchor.detach().cpu())) * n_rows
            seen += n_rows
        scheduler.step()
        row = snapshot(epoch, total / seen)
        row["minutes"] = (time.perf_counter() - started) / 60
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (row["domains"]["phrase_activity"]["top1"], row["domains"]["phrase"]["top1"],
               sum(row["domains"][name]["top1"] for name in SOURCES))
        if row["eligible"] and (best is None or key > tuple(best["selection_key"])):
            best = {**row, "selection_key": list(key), "state": copy.deepcopy(model.state_dict())}

    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {
        "arm": args.arm, "recipe": str(args.recipe), "recipe_sha256": sha256(args.recipe),
        "trainer_sha256": sha256(Path(__file__)), "epochs": epochs, "micro_batch": micro, "seed": seed,
        "phrase_train_video_sha256": sorted(train_ids), "phrase_val_video_sha256": sorted(val_ids),
        "floors": floors, "encoders_frozen": args.arm == "frozen",
        "oracle_used_for_selection": False,
        "citizen_test_accessed": False, "semlex_test_accessed": False, "local_test_accessed": False,
    }
    (args.output_dir / "history.json").write_text(json.dumps({"provenance": provenance, "history": history}, indent=1) + "\n")
    if best is None:
        print("NO ELIGIBLE EPOCH: every epoch broke an isolated floor; nothing saved as a model", flush=True)
        return
    state = best.pop("state")
    model.load_state_dict(state)
    selected = copy.deepcopy(checkpoint)
    selected["landmark_model_state_dict"] = model.landmark_model.state_dict()
    selected["hand_model_state_dict"] = model.hand_model.state_dict()
    selected["head_state_dict"] = model.fusion_head.state_dict()
    selected["epoch"] = int(best["epoch"])
    selected["validation_metrics"] = best["domains"]
    selected["phrase_adaptation"] = {**provenance, "selected": best}
    selected["test_evaluated"] = False
    torch.save(selected, args.output_dir / "best_model.pth")
    print(json.dumps({"selected_epoch": best["epoch"], "domains": best["domains"],
                      "oracle": best["oracle_heldout_asllrp"]}), flush=True)


if __name__ == "__main__":
    main()
