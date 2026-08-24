#!/usr/bin/env python3
"""Self-supervise the v17 hand-geometry MLP with bone-orientation triplets.

Only Citizen train and the explicitly approved SemLex train supplement are loaded.
The classifier, validation split, and both test splits are outside this pretraining job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v17.model_v17 import (
        ArticulatedPoseEmbeddingV17,
        articulated_bone_distance,
        masked_hand_bone_geometry,
    )
    from active.v17.schema_v17 import V17Config, schema_fingerprint
    from active.v17.train_stage_1_v17 import (
        Citizen100V17Dataset,
        SemLexSupplementV17Dataset,
        sha256_file,
    )
else:
    from .model_v17 import (
        ArticulatedPoseEmbeddingV17,
        articulated_bone_distance,
        masked_hand_bone_geometry,
    )
    from .schema_v17 import V17Config, schema_fingerprint
    from .train_stage_1_v17 import (
        Citizen100V17Dataset,
        SemLexSupplementV17Dataset,
        sha256_file,
    )


def collect_source_frames(
    dataset: torch.utils.data.Dataset,
    *,
    maximum_frames: int,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    """Collect deterministic frames with at least one sufficiently observed hand."""
    retained: list[torch.Tensor] = []
    total = 0
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    for features, _ in loader:
        frames = features.float().flatten(0, 1)
        left = (frames[:, :21, 3] > 0.5).sum(dim=1)
        right = (frames[:, 21:42, 3] > 0.5).sum(dim=1)
        frames = frames[(left >= 12) | (right >= 12)]
        if not len(frames):
            continue
        remaining = maximum_frames - total
        retained.append(frames[:remaining])
        total += min(len(frames), remaining)
        if total >= maximum_frames:
            break
    if not retained:
        raise RuntimeError("no quality-gated hand frames were found")
    return torch.cat(retained, dim=0)


def make_triplet_indices(
    geometry: torch.Tensor,
    batch_size: int,
    candidate_pool: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mine a local positive and a separated negative from random candidates."""
    samples = len(geometry)
    anchors = torch.randint(
        samples, (batch_size,), device=geometry.device, generator=generator
    )
    candidates = torch.randint(
        samples,
        (batch_size, candidate_pool),
        device=geometry.device,
        generator=generator,
    )
    anchor_geometry = geometry.index_select(0, anchors).unsqueeze(1)
    candidate_geometry = geometry[candidates]
    distances = articulated_bone_distance(anchor_geometry, candidate_geometry)
    common = (
        (anchor_geometry[..., 4] > 0.5)
        & (candidate_geometry[..., 4] > 0.5)
    ).sum(dim=-1)
    usable = (common >= 8) & (candidates != anchors.unsqueeze(1))
    distances = distances.masked_fill(~usable, torch.inf)
    order = distances.argsort(dim=1)
    ordered_distance = distances.gather(1, order)
    valid_count = torch.isfinite(ordered_distance).sum(dim=1)
    keep = valid_count >= 16
    if not keep.any():
        raise RuntimeError("triplet mining found no anchors with sufficient overlap")
    anchors = anchors[keep]
    candidates = candidates[keep]
    order = order[keep]
    valid_count = valid_count[keep]

    # Randomize within the closest eight candidates and around the 75th percentile.
    positive_rank = torch.randint(
        8, (len(anchors),), device=geometry.device, generator=generator
    )
    negative_rank = torch.floor(0.75 * (valid_count - 1).float()).long()
    row = torch.arange(len(anchors), device=geometry.device)
    positives = candidates[row, order[row, positive_rank]]
    negatives = candidates[row, order[row, negative_rank]]
    return anchors, positives, negatives


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs < 1 or args.triplets < 1 or args.batch_size < 2:
        raise ValueError("epochs/triplets must be positive and batch size at least two")
    if args.candidate_pool < 16:
        raise ValueError("candidate pool must contain at least 16 frames")
    if args.maximum_frames < 64:
        raise ValueError("maximum frames must be at least 64")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    expected_schema = schema_fingerprint(V17Config())
    citizen = Citizen100V17Dataset(
        args.data_root,
        "train",
        args.manifest,
        args.rejections,
        cache=False,
        expected_schema=expected_schema,
    )
    semlex = SemLexSupplementV17Dataset(
        args.supplement_root,
        args.supplement_manifest,
        citizen.label_to_index,
        cache=False,
        expected_schema=expected_schema,
    )
    per_source = args.maximum_frames // 2
    citizen_frames = collect_source_frames(
        citizen,
        maximum_frames=per_source,
        batch_size=args.load_batch_size,
        workers=args.workers,
    )
    semlex_frames = collect_source_frames(
        semlex,
        maximum_frames=per_source,
        batch_size=args.load_batch_size,
        workers=args.workers,
    )
    frames = torch.cat((citizen_frames, semlex_frames), dim=0)
    geometry = masked_hand_bone_geometry(frames.unsqueeze(1)).squeeze(1)
    frames = frames.to(device)
    geometry = geometry.to(device)

    model = ArticulatedPoseEmbeddingV17().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = math.ceil(args.triplets / args.batch_size)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.02
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    history: list[dict[str, float]] = []
    processed = 0
    steps_per_epoch = math.ceil(total_steps / args.epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_triplets = 0
        for _ in range(steps_per_epoch):
            if processed >= total_steps:
                break
            anchor, positive, negative = make_triplet_indices(
                geometry,
                args.batch_size,
                args.candidate_pool,
                generator,
            )
            anchor_embedding = model(frames.index_select(0, anchor).unsqueeze(1)).squeeze(1)
            positive_embedding = model(
                frames.index_select(0, positive).unsqueeze(1)
            ).squeeze(1)
            negative_embedding = model(
                frames.index_select(0, negative).unsqueeze(1)
            ).squeeze(1)
            loss = F.triplet_margin_loss(
                anchor_embedding,
                positive_embedding,
                negative_embedding,
                margin=args.margin,
                p=2,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            count = len(anchor)
            epoch_loss += float(loss.detach()) * count
            epoch_triplets += count
            processed += 1
        history.append(
            {
                "epoch": float(epoch),
                "loss": epoch_loss / max(epoch_triplets, 1),
                "triplets": float(epoch_triplets),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if processed >= total_steps:
            break

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    triplets_seen = int(sum(row["triplets"] for row in history))
    checkpoint = {
        "format": "slt_v17_articulated_pose_pretrain",
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "manifest_sha256": sha256_file(Path(args.manifest)),
        "supplement_manifest_sha256": sha256_file(Path(args.supplement_manifest)),
        "schema_fingerprint": expected_schema,
        "epochs": len(history),
        "triplets": triplets_seen,
        "objective": (
            "triplet_margin_0.2_missing_aware_length_weighted_hand_bone_orientation"
        ),
        "citizen_train_clips": len(citizen),
        "semlex_train_clips": len(semlex),
        "citizen_frames": len(citizen_frames),
        "semlex_frames": len(semlex_frames),
        "candidate_pool": args.candidate_pool,
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
    }
    torch.save(checkpoint, output / "articulated_pose_pretrained.pth")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    result["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    result["final_loss"] = history[-1]["loss"]
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("data/local/citizen100_v17/landmarks"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--rejections", type=Path,
        default=Path("data/local/citizen100_v17/rejections.csv"),
    )
    parser.add_argument(
        "--supplement-root", type=Path,
        default=Path(
            "data/local/semlex_citizen100_train_audit/full_clean_landmarks_v17"
        ),
    )
    parser.add_argument(
        "--supplement-manifest", type=Path,
        default=Path(
            "data/local/semlex_citizen100_train_audit/full_clean_train_candidates.json"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--triplets", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--maximum-frames", type=int, default=60_000)
    parser.add_argument("--load-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
