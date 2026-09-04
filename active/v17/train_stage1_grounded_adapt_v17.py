#!/usr/bin/env python3
"""Adapt Stage 1 on signer-disjoint local and precisely timed continuous sign cores."""

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
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.geometry_v17 import resample_features
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_phrase_adapt_v17 import IsolatedReplay, PhraseSegments
from active.v17.train_stage_1_reel_emission_v17 import asllrp_annotations
from active.v17.train_stage1_asllrp_core_adapt_v17 import ASLLRPCores, metrics
from active.v17.train_stage_1_v17 import augment_v17
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SourcedPhraseSegments(PhraseSegments):
    def __init__(self, *args, source: int, **kwargs):
        self.source = source
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        features, target, _ = super().__getitem__(index)
        return features, target, self.source


class SourcedASLLRPCores(ASLLRPCores):
    def __init__(self, *args, source: int, **kwargs):
        self.source = source
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        features, target, _ = super().__getitem__(index)
        return features, target, self.source


class NCSLGRCores(Dataset):
    """Strict exact-vocabulary cores from SignStream frame intervals."""

    def __init__(
        self, manifest: Path, labels: dict[str, int], role: str,
        *, training: bool, source: int,
    ) -> None:
        self.training, self.source = training, source
        self.phrases: list[np.ndarray] = []
        self.rows: list[tuple[int, int, int, int]] = []
        payload = json.loads(manifest.read_text())
        for item in payload["rows"]:
            if item["role"] != role or not item["has_strict_target"]:
                continue
            with np.load(item["archive_path"], allow_pickle=False) as archive:
                frames = restore_source_frames(
                    archive["landmarks"], archive["window_source_ranges"]
                )
            phrase_index = len(self.phrases)
            self.phrases.append(frames.astype(np.float16))
            for event in item["events"]:
                label = event["canonical_label"]
                if label not in labels:
                    continue
                self.rows.append((
                    phrase_index,
                    int(event["source_start_frame"]),
                    int(event["source_end_frame_exclusive"]),
                    labels[label],
                ))
        if not self.rows:
            raise ValueError(f"no strict NCSLGR {role} cores")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        phrase, start, stop, target = self.rows[index]
        frames = self.phrases[phrase].astype(np.float32)
        width = max(2, stop - start)
        if self.training:
            start += random.uniform(-0.08, 0.08) * width
            stop += random.uniform(-0.08, 0.08) * width
        context = 0.08 * width
        first = max(0, int(round(start - context)))
        last = min(len(frames), int(round(stop + context)))
        if last - first < 2:
            first, last = max(0, first - 1), min(len(frames), last + 1)
        value = resample_features(frames[first:last], 32).astype(np.float32)
        return torch.from_numpy(value.copy()), target, self.source


def weights(dataset: ConcatDataset, shares: dict[int, float]) -> torch.Tensor:
    pairs = []
    for child in dataset.datasets:
        source = child.source if hasattr(child, "source") else child.rows[0][2]
        if isinstance(child, IsolatedReplay):
            pairs.extend((row[2], row[1]) for row in child.rows)
        else:
            pairs.extend((source, row[3]) for row in child.rows)
    counts = Counter(pairs)
    classes = Counter(source for source, _ in counts)
    return torch.tensor([
        shares[source] / classes[source] / counts[(source, target)]
        for source, target in pairs
    ], dtype=torch.double)


def controlled_augment(features: torch.Tensor) -> torch.Tensor:
    value = augment_v17(
        features, full_roll_probability=0.02,
        maximum_roll_degrees=18.0, mild_roll_degrees=8.0,
    )
    # Short detector gaps are represented by carrying the preceding observation;
    # never fabricate coordinates through a missing interval.
    if torch.rand((), device=value.device) < 0.40:
        for sample in range(len(value)):
            drop = torch.rand(value.shape[1], device=value.device) < 0.08
            drop[0] = False
            indices = torch.arange(value.shape[1], device=value.device)
            indices[drop] -= 1
            value[sample] = value[sample].index_select(0, indices)
    return value


def run(args: argparse.Namespace) -> dict[str, object]:
    protected = (
        args.citizen_train, args.citizen_validation,
        args.semlex_train, args.semlex_validation, args.phrase_root,
    )
    if any("test" in {part.casefold() for part in path.parts} for path in protected):
        raise ValueError("grounded adaptation refuses test paths")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.base, map_location="cpu", weights_only=False)
    labels = {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}
    annotations = asllrp_annotations(args.span_manifest, args.segmented_manifest)

    train = (
        IsolatedReplay(args.citizen_train, labels, 0, args.replay_per_class),
        IsolatedReplay(args.semlex_train, labels, 1, args.replay_per_class, require_all_classes=False),
        SourcedPhraseSegments(args.phrase_root / "train/local_phrases", labels,
            training=True, boundary_jitter=0.08, source=2),
        SourcedASLLRPCores(args.phrase_root / "train/asllrp_contiguous", annotations,
            labels, training=True, source=3),
        NCSLGRCores(args.ncslgr_manifest, labels, "train", training=True, source=4),
    )
    validation = {
        "citizen": IsolatedReplay(args.citizen_validation, labels, 0, 10000),
        "semlex": IsolatedReplay(args.semlex_validation, labels, 1, 10000, require_all_classes=False),
        "local_signer": SourcedPhraseSegments(args.phrase_root / "validation/local_phrases", labels,
            training=False, boundary_jitter=0.0, source=2),
        "asllrp_cores": SourcedASLLRPCores(args.phrase_root / "validation/asllrp_contiguous",
            annotations, labels, training=False, source=3),
        "ncslgr_cores": NCSLGRCores(args.ncslgr_manifest, labels, "validation",
            training=False, source=4),
    }
    combined = ConcatDataset(train)
    shares = {0: 0.28, 1: 0.22, 2: 0.16, 3: 0.14, 4: 0.20}
    sampler = WeightedRandomSampler(
        weights(combined, shares), args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.to(device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters(): parameter.requires_grad = False
    baseline = {name: metrics(model, data, device, args.batch_size) for name, data in validation.items()}
    print(json.dumps({"baseline": baseline, "device": str(device)}, indent=2), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best, history, started = None, [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for features, targets, sources in loader:
            features, targets, sources = features.to(device), targets.to(device), sources.to(device)
            augmented = controlled_augment(features)
            logits = model(augmented)
            supervised = F.cross_entropy(logits, targets)
            replay = sources < 2
            distill = torch.zeros((), device=device)
            if replay.any():
                with torch.no_grad(): teacher_logits = teacher(augmented[replay])
                distill = F.kl_div(
                    F.log_softmax(logits[replay] / 2.0, dim=1),
                    F.softmax(teacher_logits / 2.0, dim=1), reduction="batchmean",
                ) * 4.0
            loss = supervised + args.distill_weight * distill
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss.detach().cpu())
        scheduler.step()
        current = {name: metrics(model, data, device, args.batch_size) for name, data in validation.items()}
        passes = (
            current["citizen"]["top1"] >= baseline["citizen"]["top1"] - 0.015
            and current["semlex"]["top1"] >= baseline["semlex"]["top1"] - 0.015
            and current["local_signer"]["top1"] >= baseline["local_signer"]["top1"] - 0.03
        )
        row = {"epoch": epoch, "loss": total / len(loader), "passes_retention": passes, **current}
        history.append(row); print(json.dumps(row), flush=True)
        key = (
            current["ncslgr_cores"]["top1"], current["asllrp_cores"]["top1"],
            current["local_signer"]["top1"],
        )
        if passes and (best is None or key > best["key"]):
            best = {"key": key, "epoch": epoch, "metrics": current,
                    "state": copy.deepcopy(model.state_dict())}
    if best is None:
        raise RuntimeError("no epoch passed retention gates")
    model.load_state_dict(best["state"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "best_model.pth"
    selected = copy.deepcopy(checkpoint)
    selected["model_state_dict"] = {key: value.cpu() for key, value in model.state_dict().items()}
    selected["epoch"] = best["epoch"]
    selected["grounded_adaptation"] = {
        "base_checkpoint": str(args.base), "base_sha256": sha256(args.base),
        "baseline": baseline, "selected": best["metrics"], "source_shares": shares,
        "controlled_augmentation": "mild geometry/noise/warp plus 8% carried-frame gaps",
        "test_accessed": False,
    }
    torch.save(selected, output)
    result = {
        "format": "slt_stage1_grounded_adaptation_v17", "output": str(output),
        "selected_epoch": best["epoch"], "baseline": baseline,
        "selected": best["metrics"], "history": history,
        "elapsed_seconds": time.perf_counter() - started, "device": str(device),
        "test_accessed": False,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base", type=Path, default=Path(
        "artifacts/models/stage1_v17_asllrp_core_adapt_v1/best_model.pth"))
    value.add_argument("--phrase-root", type=Path, default=Path(
        "data/local/stage2_v17_grounded_signer_split"))
    value.add_argument("--ncslgr-manifest", type=Path, default=Path(
        "active/v17/ncslgr_supervised_manifest_v17.json"))
    value.add_argument("--span-manifest", type=Path, default=Path(
        "data/local/asllrp_contiguous_phrases_v17/manifest.json"))
    value.add_argument("--segmented-manifest", type=Path, default=Path(
        "data/local/asllrp_segmented_citizen100_v17/manifest.json"))
    value.add_argument("--citizen-train", type=Path, default=Path(
        "data/local/citizen100_v17/landmarks/train"))
    value.add_argument("--citizen-validation", type=Path, default=Path(
        "data/local/citizen100_v17/landmarks/val"))
    value.add_argument("--semlex-train", type=Path, default=Path(
        "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"))
    value.add_argument("--semlex-validation", type=Path, default=Path(
        "data/local/semlex_citizen100_val_audit/landmarks_v17"))
    value.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/models/stage1_v17_grounded_adapt_v1"))
    value.add_argument("--epochs", type=int, default=14)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--samples-per-epoch", type=int, default=3500)
    value.add_argument("--replay-per-class", type=int, default=10)
    value.add_argument("--learning-rate", type=float, default=1.5e-5)
    value.add_argument("--distill-weight", type=float, default=0.8)
    value.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    value.add_argument("--seed", type=int, default=17071)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
