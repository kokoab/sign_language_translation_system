#!/usr/bin/env python3
"""Adapt a separate v17 Stage-1 checkpoint on manually aligned ASLLRP sign cores."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.geometry_v17 import resample_features
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_phrase_adapt_v17 import (
    IsolatedReplay,
    PhraseSegments,
)
from active.v17.train_stage_1_reel_emission_v17 import asllrp_annotations
from active.v17.train_stage_1_v17 import augment_v17
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ASLLRPCores(Dataset):
    """Exact sign cores from manual ASLLRP intervals, with train-only jitter."""

    def __init__(
        self, root: Path, annotations: dict[str, tuple[dict, list[dict]]],
        labels: dict[str, int], *, training: bool,
    ) -> None:
        self.training = training
        self.phrases: list[np.ndarray] = []
        self.rows: list[tuple[int, int, int, int]] = []
        self.identities: list[str] = []
        for path in sorted(root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"].item()))
                frames = restore_source_frames(
                    payload["landmarks"], payload["window_source_ranges"]
                )
            item_id = str(metadata["source_item_id"])
            span, signs = annotations[item_id]
            sequence = [str(value) for value in metadata["target_sequence"]]
            aligned = [str(value["canonical_label"]) for value in signs]
            if sequence != aligned or any(value not in labels for value in sequence):
                raise ValueError(f"manual labels disagree for {item_id}")
            crop_start = int(span["utterance_start_frame_global"]) + int(
                span["crop_start_frame_local"]
            )
            phrase_index = len(self.phrases)
            self.phrases.append(frames.astype(np.float16))
            self.identities.append(item_id)
            for sign in signs:
                start = int(sign["sign_start_frame"]) - crop_start
                stop = int(sign["sign_end_frame"]) - crop_start + 1
                self.rows.append(
                    (phrase_index, start, stop, labels[str(sign["canonical_label"])])
                )
        if not self.rows:
            raise ValueError(f"no ASLLRP sign cores under {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        phrase_index, start, stop, target = self.rows[index]
        frames = self.phrases[phrase_index].astype(np.float32)
        width = max(2, stop - start)
        if self.training:
            jitter = max(1.0, width * 0.10)
            start += random.uniform(-jitter, jitter)
            stop += random.uniform(-jitter, jitter)
        context = max(1.0, width * 0.10)
        first = max(0, int(round(start - context)))
        last = min(len(frames), int(round(stop + context)))
        if last - first < 2:
            first, last = max(0, first - 1), min(len(frames), last + 1)
        value = resample_features(frames[first:last], 32).astype(np.float32)
        return torch.from_numpy(value.copy()), target, 2


class LocalPhraseReplay(PhraseSegments):
    def __getitem__(self, index: int):
        features, target, _ = super().__getitem__(index)
        return features, target, 3


def balanced_weights(dataset: ConcatDataset, shares: dict[int, float]) -> torch.Tensor:
    rows: list[tuple[int, int]] = []
    for child in dataset.datasets:
        if isinstance(child, IsolatedReplay):
            rows.extend((source, target) for _, target, source in child.rows)
        elif isinstance(child, LocalPhraseReplay):
            rows.extend((3, row[3]) for row in child.rows)
        else:
            rows.extend((2, row[3]) for row in child.rows)
    pair_counts = Counter(rows)
    source_classes = Counter(source for source, _ in pair_counts)
    return torch.tensor([
        shares[source] / source_classes[source] / pair_counts[(source, target)]
        for source, target in rows
    ], dtype=torch.double)


@torch.inference_mode()
def metrics(
    model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int
) -> dict[str, float]:
    model.eval()
    correct1 = correct5 = total = 0
    for features, targets, _ in DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    ):
        logits = model(features.to(device)).cpu()
        order = logits.topk(min(5, logits.shape[1]), dim=1).indices
        correct1 += int((order[:, 0] == targets).sum())
        correct5 += int((order == targets[:, None]).any(dim=1).sum())
        total += len(targets)
    return {"samples": total, "top1": correct1 / total, "top5": correct5 / total}


def run(args: argparse.Namespace) -> dict[str, object]:
    for path in (
        args.citizen_train, args.citizen_validation, args.semlex_train,
        args.semlex_validation, args.phrase_root,
    ):
        if "test" in {part.casefold() for part in path.parts}:
            raise ValueError(f"refusing protected path: {path}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("base must be a v17 Stage-1 checkpoint")
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    annotations = asllrp_annotations(args.span_manifest, args.segmented_manifest)
    citizen_train = IsolatedReplay(
        args.citizen_train, labels, 0, args.replay_per_class
    )
    semlex_train = IsolatedReplay(
        args.semlex_train, labels, 1, args.replay_per_class,
        require_all_classes=False,
    )
    asllrp_train = ASLLRPCores(
        args.phrase_root / "train/asllrp_contiguous", annotations, labels,
        training=True,
    )
    local_train = LocalPhraseReplay(
        args.phrase_root / "train/local_phrases", labels,
        training=True, boundary_jitter=0.10,
    )
    citizen_validation = IsolatedReplay(
        args.citizen_validation, labels, 0, 10000
    )
    semlex_validation = IsolatedReplay(
        args.semlex_validation, labels, 1, 10000, require_all_classes=False
    )
    local_validation = PhraseSegments(
        args.phrase_root / "validation/local_phrases", labels,
        training=False, boundary_jitter=0.0,
    )
    asllrp_validation = ASLLRPCores(
        args.phrase_root / "validation/asllrp_contiguous", annotations, labels,
        training=False,
    )
    train = ConcatDataset((citizen_train, semlex_train, asllrp_train, local_train))
    shares = {
        0: args.citizen_share, 1: args.semlex_share,
        2: args.asllrp_share, 3: args.local_share,
    }
    if not np.isclose(sum(shares.values()), 1.0) or min(shares.values()) <= 0:
        raise ValueError("source shares must be positive and sum to one")
    sampler = WeightedRandomSampler(
        balanced_weights(train, shares), args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        train, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    validations = {
        "citizen": citizen_validation, "semlex": semlex_validation,
        "local_segments": local_validation, "asllrp_cores": asllrp_validation,
    }
    baseline = {
        name: metrics(model, dataset, device, args.batch_size)
        for name, dataset in validations.items()
    }
    print(json.dumps({"baseline": baseline, "device": str(device)}, indent=2))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    best = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        for features, targets, sources in loader:
            features, targets, sources = (
                features.to(device), targets.to(device), sources.to(device)
            )
            augmented = augment_v17(features)
            logits = model(augmented)
            supervised = F.cross_entropy(logits, targets)
            replay = sources != 2
            distill = torch.zeros((), device=device)
            if replay.any():
                with torch.no_grad():
                    teacher_logits = teacher(augmented[replay])
                distill = F.kl_div(
                    F.log_softmax(logits[replay] / 2.0, dim=1),
                    F.softmax(teacher_logits / 2.0, dim=1),
                    reduction="batchmean",
                ) * 4.0
            loss = supervised + args.distill_weight * distill
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu())
        scheduler.step()
        current = {
            name: metrics(model, dataset, device, args.batch_size)
            for name, dataset in validations.items()
        }
        passes = (
            current["citizen"]["top1"] >= baseline["citizen"]["top1"] - args.replay_tolerance
            and current["semlex"]["top1"] >= baseline["semlex"]["top1"] - args.replay_tolerance
            and current["local_segments"]["top1"] >= baseline["local_segments"]["top1"] - args.local_tolerance
        )
        row = {
            "epoch": epoch, "loss": loss_total / len(loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "passes_retention_gates": passes, **current,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        key = (
            current["asllrp_cores"]["top1"], current["asllrp_cores"]["top5"],
            current["citizen"]["top1"] + current["semlex"]["top1"],
        )
        if passes and (best is None or key > best["key"]):
            best = {
                "key": key, "epoch": epoch, "metrics": current,
                "state": copy.deepcopy(model.state_dict()),
            }
    if best is None:
        raise RuntimeError("no epoch passed the retention gates")
    model.load_state_dict(best["state"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "best_model.pth"
    selected = copy.deepcopy(checkpoint)
    selected["model_state_dict"] = {key: value.cpu() for key, value in model.state_dict().items()}
    selected["epoch"] = int(best["epoch"])
    selected["asllrp_core_adaptation"] = {
        "base_checkpoint": str(args.base), "base_sha256": sha256(args.base),
        "train_cores": len(asllrp_train), "validation_cores": len(asllrp_validation),
        "local_replay_segments": len(local_train),
        "manual_boundaries": True, "baseline": baseline,
        "selected": best["metrics"], "source_shares": shares,
        "test_accessed": False, "external_evaluation_reserved_accessed": False,
    }
    torch.save(selected, output)
    result = {
        "format": "slt_stage1_asllrp_core_adaptation_v17",
        "output": str(output), "device": str(device), "selected_epoch": best["epoch"],
        "baseline": baseline, "selected": best["metrics"], "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "test_accessed": False, "external_evaluation_reserved_accessed": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base", type=Path, default=Path(
        "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"
    ))
    value.add_argument("--phrase-root", type=Path, default=Path(
        "data/local/stage2_v17_multimodal"
    ))
    value.add_argument("--span-manifest", type=Path, default=Path(
        "data/local/asllrp_contiguous_phrases_v17/manifest.json"
    ))
    value.add_argument("--segmented-manifest", type=Path, default=Path(
        "data/local/asllrp_segmented_citizen100_v17/manifest.json"
    ))
    value.add_argument("--citizen-train", type=Path, default=Path(
        "data/local/citizen100_v17/landmarks/train"
    ))
    value.add_argument("--citizen-validation", type=Path, default=Path(
        "data/local/citizen100_v17/landmarks/val"
    ))
    value.add_argument("--semlex-train", type=Path, default=Path(
        "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"
    ))
    value.add_argument("--semlex-validation", type=Path, default=Path(
        "data/local/semlex_citizen100_val_audit/landmarks_v17"
    ))
    value.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/models/stage1_v17_asllrp_core_adapt_v1"
    ))
    value.add_argument("--epochs", type=int, default=12)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--samples-per-epoch", type=int, default=3000)
    value.add_argument("--replay-per-class", type=int, default=10)
    value.add_argument("--citizen-share", type=float, default=0.30)
    value.add_argument("--semlex-share", type=float, default=0.25)
    value.add_argument("--asllrp-share", type=float, default=0.25)
    value.add_argument("--local-share", type=float, default=0.20)
    value.add_argument("--learning-rate", type=float, default=2e-5)
    value.add_argument("--weight-decay", type=float, default=1e-4)
    value.add_argument("--distill-weight", type=float, default=1.0)
    value.add_argument("--replay-tolerance", type=float, default=0.015)
    value.add_argument("--local-tolerance", type=float, default=0.03)
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--seed", type=int, default=17061)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
