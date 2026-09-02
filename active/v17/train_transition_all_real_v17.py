#!/usr/bin/env python3
"""Source-balanced adaptation of the transition model on all genuine corpora."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.06")

import argparse
import gc
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
)
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    evaluate,
    loss_terms,
    sha256,
)


LOG = logging.getLogger("train_transition_all_real_v17")
NCSLGR_TRAIN = {"ncslgr:BENJAMIN_JAMES_BAHAN"}
NCSLGR_VALIDATION = {"ncslgr:NORMA_BOWERS_TOURANGEAU"}


def dataset(
    root: Path, *, seed: int, fixed: bool, roles: set[str],
    sources: set[str] | None = None, signers: set[str] | None = None,
) -> TransitionWindowDataset:
    return TransitionWindowDataset(
        root, signers, seed=seed, fixed_masks=fixed, all_archives=True,
        roles=roles, sources=sources,
    )


def balanced_sampler(families: list[torch.utils.data.Dataset], samples: int, seed: int):
    weights = torch.cat([
        torch.full((len(value),), 1.0 / len(value)) for value in families
    ]).double()
    return WeightedRandomSampler(
        weights, num_samples=samples, replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def validation_metrics(model, loaders, device):
    return {name: evaluate(model, loader, device) for name, loader in loaders.items()}


def selection_score(metrics: dict[str, dict[str, float]]) -> float:
    return float(np.mean([
        metrics[name]["relative_score_improvement"]
        for name in ("local", "asllrp_other", "ncslgr")
    ]))


def eligible(
    metrics: dict[str, dict[str, float]],
    initial: dict[str, dict[str, float]], tolerance: float,
) -> bool:
    return all(
        metrics[name]["relative_score_improvement"]
        >= initial[name]["relative_score_improvement"] - tolerance
        for name in ("local", "asllrp_other", "ncslgr", "how2sign_guard", "web_guard")
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)

    local_train = dataset(args.local_root, seed=1701, fixed=False, roles={"train"})
    asllrp_old = dataset(args.asllrp_old_root, seed=1801, fixed=False, roles={"train"})
    asllrp_other = dataset(args.asllrp_other_train_root, seed=1901, fixed=False, roles={"train"})
    flores = dataset(args.flores_root, seed=2001, fixed=False, roles={"train"})
    how2sign = dataset(
        args.how2sign_ncslgr_root, seed=2101, fixed=False, roles={"train"},
        sources={"how2sign_unlabeled_continuous"},
    )
    ncslgr = dataset(
        args.how2sign_ncslgr_root, seed=2201, fixed=False, roles={"train"},
        sources={"ncslgr_public_continuous"}, signers=NCSLGR_TRAIN,
    )
    youtube = dataset(
        args.youtube_root, seed=2301, fixed=False, roles={"train"},
        sources={"youtube_asl"},
    )
    openasl = dataset(args.openasl_root, seed=2401, fixed=False, roles={"train"})
    families = [
        local_train,
        ConcatDataset((asllrp_old, asllrp_other)),
        flores,
        how2sign,
        ncslgr,
        ConcatDataset((youtube, openasl)),
    ]
    combined = ConcatDataset(families)
    sampler = balanced_sampler(
        families, args.samples_per_epoch or len(combined), args.seed
    )
    loader = DataLoader(
        combined, batch_size=args.batch_size, sampler=sampler, num_workers=0
    )

    validation_sets = {
        "local": dataset(args.local_root, seed=4701, fixed=True, roles={"validation"}),
        "asllrp_other": dataset(args.asllrp_other_validation_root, seed=4801, fixed=True, roles={"validation"}),
        "ncslgr": dataset(
            args.how2sign_ncslgr_root, seed=4901, fixed=True, roles={"train"},
            sources={"ncslgr_public_continuous"}, signers=NCSLGR_VALIDATION,
        ),
        "how2sign_guard": dataset(
            args.how2sign_ncslgr_root, seed=5001, fixed=True, roles={"train"},
            sources={"how2sign_unlabeled_continuous"},
        ),
        "web_guard": dataset(
            args.youtube_root, seed=5101, fixed=True, roles={"train"},
            sources={"youtube_asl"},
        ),
        "openasl_reference": dataset(args.openasl_root, seed=5201, fixed=True, roles={"validation"}),
    }
    validation_loaders = {
        name: DataLoader(value, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for name, value in validation_sets.items()
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    initial_checkpoint = torch.load(
        args.initial_checkpoint, map_location="cpu", weights_only=False
    )
    if initial_checkpoint.get("format") != "slt_transition_inpainter_v17":
        raise ValueError("unexpected initial transition checkpoint")
    config = TransitionInpainterV17Config(**initial_checkpoint["model_config"])
    model = TransitionInpainterV17(config).to(device)
    model.load_state_dict(initial_checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    initial = validation_metrics(model, validation_loaders, device)
    best_metrics = initial
    best_score = selection_score(initial)
    best_epoch = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    history = [{
        "epoch": 0, "validation": initial,
        "selection_score": best_score, "eligible": True,
    }]
    patience = 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = seen = 0
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, mask)
            terms = loss_terms(prediction, features, mask)
            loss = (
                terms["spatial"]
                + args.auxiliary_weight * terms["auxiliary"]
                + args.velocity_weight * terms["velocity"]
                + args.acceleration_weight * terms["acceleration"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite all-real transition loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        metrics = validation_metrics(model, validation_loaders, device)
        score = selection_score(metrics)
        passes = eligible(metrics, initial, args.regression_tolerance)
        history.append({
            "epoch": epoch, "train_loss": total / seen,
            "validation": metrics, "selection_score": score, "eligible": passes,
        })
        if passes and score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        LOG.info(
            "epoch=%d loss=%.6f score=%.4f eligible=%s best=%d local=%.4f",
            epoch, total / seen, score, passes, best_epoch,
            metrics["local"]["relative_score_improvement"],
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if patience >= args.patience:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_transition_inpainter_v17",
        "version": 1,
        "model_config": config.to_dict(),
        "model_state_dict": best_state,
        "seed": args.seed,
        "epoch": best_epoch,
        "held_out_signer": None,
        "initial_checkpoint": args.initial_checkpoint.as_posix(),
        "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
        "training_family_windows": {
            "local": len(local_train),
            "asllrp": len(asllrp_old) + len(asllrp_other),
            "two_m_flores": len(flores),
            "how2sign": len(how2sign),
            "ncslgr": len(ncslgr),
            "web": len(youtube) + len(openasl),
        },
        "selection_domains": ["local", "asllrp_other", "ncslgr"],
        "guard_domains": ["how2sign_guard", "web_guard"],
        "best_validation": best_metrics,
        "source_balancing": "equal probability for six corpus families",
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    checkpoint_path = args.output / "model.pth"
    torch.save(checkpoint, checkpoint_path)
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    result = {
        "format": "slt_transition_all_real_result_v17",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "initial_checkpoint_sha256": checkpoint["initial_checkpoint_sha256"],
        "best_epoch": best_epoch,
        "initial_validation": initial,
        "best_validation": best_metrics,
        "training_family_windows": checkpoint["training_family_windows"],
        "seconds": time.monotonic() - started,
        "claim_boundary": "masked genuine-motion reconstruction only; not semantic phrase generation or human naturalness",
        "test_evaluated": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data/local/local_phrase_motion_landmarks_v17"))
    parser.add_argument("--asllrp-old-root", type=Path, default=Path("data/local/stage2_v17_multimodal/train/asllrp_contiguous"))
    parser.add_argument("--asllrp-other-train-root", type=Path, default=Path("data/local/stage2_v17_asllrp_other_multimodal/train/asllrp_other_ctc"))
    parser.add_argument("--asllrp-other-validation-root", type=Path, default=Path("data/local/stage2_v17_asllrp_other_multimodal/validation/asllrp_other_ctc"))
    parser.add_argument("--flores-root", type=Path, default=Path("data/local/stage2_v17_2m_flores_multimodal/train/two_m_flores_asl"))
    parser.add_argument("--how2sign-ncslgr-root", type=Path, default=Path("data/local/how2sign_transition_landmarks_v17"))
    parser.add_argument("--youtube-root", type=Path, default=Path("data/local/youtube_asl_transition_landmarks_v17"))
    parser.add_argument("--openasl-root", type=Path, default=Path("data/local/openasl_transition_landmarks_v17"))
    parser.add_argument("--initial-checkpoint", type=Path, default=Path("artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/model.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/transition_all_real_v17_v1"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--auxiliary-weight", type=float, default=0.10)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--acceleration-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--regression-tolerance", type=float, default=0.005)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print(json.dumps(run(build_parser().parse_args()), indent=2))
