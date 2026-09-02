#!/usr/bin/env python3
"""Fine-tune a separate Stage-1 landmark model on genuine phrase segments.

The accepted isolated checkpoint is never modified.  Phrase supervision comes only
from the existing train/validation local phrase caches; Citizen and SemLex isolated
clips are replayed to limit forgetting.  No test or sealed split is accepted.
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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

from active.v17.geometry_v17 import resample_features
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_v17 import augment_v17


DEFAULT_BASE = Path(
    "artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/"
    "stage1_v17_orientation_robust_v1/best_model.pth"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        value = payload["features"].astype(np.float32, copy=False)
    if value.shape != (32, 61, 5) or not np.isfinite(value).all():
        raise ValueError(f"invalid v17 archive: {path}")
    return value


class IsolatedReplay(Dataset):
    def __init__(
        self, root: Path, labels: dict[str, int], source: int, per_class: int,
        *, require_all_classes: bool = True,
    ) -> None:
        self.rows: list[tuple[Path, int, int]] = []
        for label, target in sorted(labels.items(), key=lambda row: row[1]):
            paths = sorted((root / label).glob("*.v17.npz"))[:per_class]
            if not paths and require_all_classes:
                raise FileNotFoundError(f"no replay clips for {label} under {root}")
            self.rows.extend((path, target, source) for path in paths)
        if not self.rows:
            raise ValueError(f"no replay clips found under {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        path, target, source = self.rows[index]
        return torch.from_numpy(load_features(path).copy()), target, source


class PhraseSegments(Dataset):
    """Target-ordered phrase partitions with boundary jitter on train only."""

    def __init__(
        self, root: Path, labels: dict[str, int], *, training: bool,
        boundary_jitter: float,
    ) -> None:
        self.training = training
        self.boundary_jitter = boundary_jitter
        self.phrases: list[np.ndarray] = []
        self.rows: list[tuple[int, int, int, int]] = []
        self.identities: list[str] = []
        for path in sorted(root.glob("*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"].item()))
                value = np.concatenate(
                    payload["landmarks"].astype(np.float32, copy=False), axis=0
                )
            sequence = [str(item) for item in metadata["target_sequence"]]
            if not sequence or any(label not in labels for label in sequence):
                continue
            phrase_index = len(self.phrases)
            self.phrases.append(value.astype(np.float16))
            self.identities.append(path.name)
            for position, label in enumerate(sequence):
                self.rows.append((phrase_index, position, len(sequence), labels[label]))
        if not self.rows:
            raise ValueError(f"no compatible phrase segments under {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        phrase_index, position, count, target = self.rows[index]
        phrase = self.phrases[phrase_index].astype(np.float32)
        width = len(phrase) / count
        start = position * width
        end = (position + 1) * width
        if self.training and self.boundary_jitter:
            radius = width * self.boundary_jitter
            start += random.uniform(-radius, radius)
            end += random.uniform(-radius, radius)
        # A little genuine neighboring motion teaches the classifier not to require
        # a neutral pose, without turning the sample into two complete signs.
        context = width * 0.06
        start = max(0, int(round(start - context)))
        end = min(len(phrase), int(round(end + context)))
        if end - start < 4:
            end = min(len(phrase), start + 4)
            start = max(0, end - 4)
        value = resample_features(phrase[start:end], 32).astype(np.float32)
        return torch.from_numpy(value.copy()), target, 2


def source_balanced_weights(
    dataset: ConcatDataset, source_share: dict[int, float]
) -> torch.Tensor:
    sources: list[int] = []
    targets: list[int] = []
    for child in dataset.datasets:
        if isinstance(child, IsolatedReplay):
            sources.extend(row[2] for row in child.rows)
            targets.extend(row[1] for row in child.rows)
        else:
            sources.extend([2] * len(child.rows))
            targets.extend(row[3] for row in child.rows)
    pair_counts = Counter(zip(sources, targets))
    class_counts = Counter(source for source, _ in pair_counts)
    weights = [
        source_share[source]
        / class_counts[source]
        / pair_counts[(source, target)]
        for source, target in zip(sources, targets)
    ]
    return torch.tensor(weights, dtype=torch.double)


def accuracy(model: nn.Module, dataset: Dataset, device: torch.device, batch: int) -> float:
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0)
    correct = total = 0
    model.eval()
    with torch.inference_mode():
        for features, targets, _ in loader:
            prediction = model(features.to(device)).argmax(1).cpu()
            correct += int((prediction == targets).sum())
            total += len(targets)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--citizen-train", type=Path,
        default=Path("data/local/citizen100_v17/landmarks/train"),
    )
    parser.add_argument(
        "--citizen-val", type=Path,
        default=Path("data/local/citizen100_v17/landmarks/val"),
    )
    parser.add_argument(
        "--semlex-train", type=Path,
        default=Path(
            "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"
        ),
    )
    parser.add_argument(
        "--semlex-val", type=Path,
        default=Path("data/local/semlex_citizen100_val_audit/landmarks_v17"),
    )
    parser.add_argument(
        "--phrase-root", type=Path,
        default=Path("data/local/stage2_v17_multimodal"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/models/stage1_v17_phrase_adapt_reel_v1"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--distill-weight", type=float, default=0.20)
    parser.add_argument("--citizen-share", type=float, default=0.50)
    parser.add_argument("--semlex-share", type=float, default=0.25)
    parser.add_argument("--phrase-share", type=float, default=0.25)
    parser.add_argument("--boundary-jitter", type=float, default=0.10)
    parser.add_argument("--replay-per-class", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=17021)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if min(args.epochs, args.batch_size, args.samples_per_epoch, args.replay_per_class) <= 0:
        raise ValueError("training sizes must be positive")
    if "test" in args.citizen_val.parts or "test" in args.semlex_val.parts:
        raise ValueError("this adaptation never accepts test roots")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("base is not a v17 Stage-1 checkpoint")
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    if sorted(labels.values()) != list(range(100)):
        raise ValueError("expected the frozen 100-gloss label map")

    citizen_train = IsolatedReplay(
        args.citizen_train, labels, 0, args.replay_per_class
    )
    semlex_train = IsolatedReplay(
        args.semlex_train, labels, 1, args.replay_per_class,
        require_all_classes=False,
    )
    phrase_train = PhraseSegments(
        args.phrase_root / "train/local_phrases", labels,
        training=True, boundary_jitter=args.boundary_jitter,
    )
    citizen_val = IsolatedReplay(args.citizen_val, labels, 0, 10000)
    semlex_val = IsolatedReplay(
        args.semlex_val, labels, 1, 10000, require_all_classes=False
    )
    phrase_val = PhraseSegments(
        args.phrase_root / "validation/local_phrases", labels,
        training=False, boundary_jitter=0.0,
    )
    train = ConcatDataset((citizen_train, semlex_train, phrase_train))
    shares = {
        0: args.citizen_share, 1: args.semlex_share, 2: args.phrase_share
    }
    if any(value <= 0 for value in shares.values()) or not np.isclose(
        sum(shares.values()), 1.0
    ):
        raise ValueError("Citizen, SemLex, and phrase shares must be positive and sum to one")
    weights = source_balanced_weights(train, shares)
    sampler = WeightedRandomSampler(
        weights, args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        train, batch_size=args.batch_size, sampler=sampler, num_workers=0,
    )

    config = Stage1V17Config(**checkpoint["model_config"])
    model = SLTStage1V17(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    baseline = {
        "citizen_val": accuracy(model, citizen_val, device, args.batch_size),
        "semlex_val": accuracy(model, semlex_val, device, args.batch_size),
        "phrase_segment_val": accuracy(model, phrase_val, device, args.batch_size),
    }
    print(json.dumps({"baseline": baseline, "device": str(device)}, indent=2))
    best: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, targets, sources in loader:
            features = features.to(device)
            targets = targets.to(device)
            sources = sources.to(device)
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
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite phrase adaptation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        scheduler.step()
        metrics = {
            "epoch": epoch,
            "loss": total_loss / max(batches, 1),
            "citizen_val": accuracy(model, citizen_val, device, args.batch_size),
            "semlex_val": accuracy(model, semlex_val, device, args.batch_size),
            "phrase_segment_val": accuracy(model, phrase_val, device, args.batch_size),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        metrics["passes_replay_gate"] = (
            metrics["citizen_val"] >= baseline["citizen_val"] - 0.01
            and metrics["semlex_val"] >= baseline["semlex_val"] - 0.01
        )
        history.append(metrics)
        print(json.dumps(metrics))
        if metrics["passes_replay_gate"] and (
            best is None
            or (metrics["phrase_segment_val"], metrics["citizen_val"] + metrics["semlex_val"])
            > (best["phrase_segment_val"], best["citizen_val"] + best["semlex_val"])
        ):
            best = {**metrics, "state": copy.deepcopy(model.state_dict())}

    if best is None:
        raise RuntimeError("no epoch passed the one-point isolated replay gates")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected = copy.deepcopy(checkpoint)
    selected["model_state_dict"] = best.pop("state")
    selected["epoch"] = int(best["epoch"])
    selected["phrase_adaptation"] = {
        "base_checkpoint": str(args.base),
        "base_sha256": sha256(args.base),
        "local_phrase_train_segments": len(phrase_train),
        "local_phrase_validation_segments": len(phrase_val),
        "citizen_replay": len(citizen_train),
        "semlex_replay": len(semlex_train),
        "boundary_method": "equal ordered partitions plus train-only jitter/context",
        "selection": best,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    output = args.output_dir / "best_model.pth"
    torch.save(selected, output)
    result = {
        "format": "slt_stage1_phrase_adaptation_v17",
        "base": str(args.base),
        "base_sha256": sha256(args.base),
        "output": str(output),
        "baseline": baseline,
        "selected": best,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
        "source_shares": shares,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
