#!/usr/bin/env python3
"""Reload and evaluate a selected Stage-2 v17 checkpoint on one frozen role."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import load_stage2_model_v17
from active.v17.train_stage_2_v17 import RealPhraseDataset, collate, evaluate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace):
    model, checkpoint = load_stage2_model_v17(args.checkpoint)
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device
    )
    model.to(device).eval()
    dataset = RealPhraseDataset(args.cache_root, args.role)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    metrics = evaluate(model, loader, device)
    report = {
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "role": args.role,
        "samples": len(dataset),
        "metrics": metrics,
        "checkpoint_test_evaluated_before_run": checkpoint.get("test_evaluated"),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/models/stage2_v17_unified_ctc_v1/best_model.pth"),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument("--role", default="validation")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_evaluation/validation_reload.json"),
    )
    return parser


def main():
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
