#!/usr/bin/env python3
"""Audit one transition model on signer-held-out studio and channel-held-out web data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
    interpolate_masked_context,
)
from active.v17.train_transition_inpainter_multicorpus_v17 import manifest_signers
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    discover_signers,
    sha256,
)
from scripts.evaluate_transition_inpainter_naturalness_v17 import (
    bootstrap_improvement,
    paired_discrimination,
    per_window_score,
)


def audit_dataset(
    model: TransitionInpainterV17,
    dataset: TransitionWindowDataset,
    batch_size: int,
    folds: int,
    bootstrap_iterations: int,
) -> dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    genuine = []
    learned = []
    linear = []
    masks = []
    groups = []
    for batch in loader:
        features = batch["features"]
        mask = batch["mask"]
        with torch.inference_mode():
            prediction = model(features, mask)
            interpolation = interpolate_masked_context(features, mask)
        genuine.extend(features.numpy())
        learned.extend(prediction.numpy())
        linear.extend(interpolation.numpy())
        masks.extend(mask.numpy())
        groups.extend(str(item).rsplit(":", 1)[0] for item in batch["item"])
    target = torch.from_numpy(np.stack(genuine))
    mask_tensor = torch.from_numpy(np.stack(masks))
    learned_score = per_window_score(
        torch.from_numpy(np.stack(learned)), target, mask_tensor
    )
    linear_score = per_window_score(
        torch.from_numpy(np.stack(linear)), target, mask_tensor
    )
    group_array = np.asarray(groups)
    return {
        "paired_windows": len(dataset),
        "source_clips": len(set(groups)),
        "reconstruction_vs_linear": bootstrap_improvement(
            learned_score, linear_score, bootstrap_iterations
        ),
        "genuine_vs_learned_discriminator": paired_discrimination(
            genuine, learned, masks, group_array, folds
        ),
        "genuine_vs_linear_discriminator": paired_discrimination(
            genuine, linear, masks, group_array, folds
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_transition_inpainter_v17":
        raise ValueError("unexpected transition checkpoint")
    model = TransitionInpainterV17(
        TransitionInpainterV17Config(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, manifest_web_validation = manifest_signers(args.web_manifest)
    available_web = discover_signers(args.web_root)
    web_validation = manifest_web_validation & available_web
    missing_web_validation = manifest_web_validation - available_web
    if not web_validation:
        raise ValueError("no usable channel-held-out web voices")
    web_dataset = TransitionWindowDataset(
        args.web_root, web_validation, seed=5701, fixed_masks=True
    )
    domains = {
        "youtube_asl_channel_heldout": audit_dataset(
            model,
            web_dataset,
            args.batch_size,
            args.folds,
            args.bootstrap_iterations,
        )
    }
    how2sign_signer = args.how2sign_signer or checkpoint.get("held_out_signer")
    if how2sign_signer:
        how2sign_dataset = TransitionWindowDataset(
            args.how2sign_root, {str(how2sign_signer)}, seed=4701, fixed_masks=True
        )
        domains["how2sign_signer_heldout"] = audit_dataset(
            model,
            how2sign_dataset,
            args.batch_size,
            args.folds,
            args.bootstrap_iterations,
        )
    report = {
        "format": "transition_inpainter_multicorpus_naturalness_audit_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "how2sign_signer": how2sign_signer,
        "youtube_asl_validation_voices": len(web_validation),
        "youtube_asl_missing_validation_voices": len(missing_web_validation),
        "domains": domains,
        "interpretation": (
            "Both domains are unseen voice groups. Lower discriminator performance "
            "than linear is evidence of a closer landmark distribution, not human naturalness."
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--how2sign-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument("--how2sign-signer")
    parser.add_argument(
        "--web-root", type=Path,
        default=Path("data/local/youtube_asl_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--web-manifest", type=Path,
        default=Path("active/v17/youtube_asl_transition_manifest_v17.json"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
