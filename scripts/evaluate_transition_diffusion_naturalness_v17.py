#!/usr/bin/env python3
"""Evaluate stochastic transition samples on a genuine held-out signer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_diffusion_v17 import (
    TransitionResidualDiffusionV17,
    TransitionResidualDiffusionV17Config,
)
from active.v17.model_transition_inpainter_v17 import interpolate_masked_context
from active.v17.train_transition_diffusion_v17 import load_mean_model
from active.v17.train_transition_inpainter_v17 import TransitionWindowDataset, sha256
from scripts.evaluate_transition_inpainter_naturalness_v17 import (
    bootstrap_improvement,
    paired_discrimination,
    per_window_score,
)


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_transition_residual_diffusion_v17":
        raise ValueError("unexpected transition diffusion checkpoint")
    diffusion = TransitionResidualDiffusionV17(
        TransitionResidualDiffusionV17Config(**checkpoint["model_config"])
    )
    diffusion.load_state_dict(checkpoint["model_state_dict"])
    diffusion.eval().to(device)
    mean_model = load_mean_model(Path(checkpoint["mean_checkpoint"]), device)
    scale = checkpoint["residual_scale"].to(device)
    held_out = str(checkpoint["held_out_signer"])
    complete = TransitionWindowDataset(
        args.landmark_root, {held_out}, seed=4701, fixed_masks=True
    )
    indices = list(range(len(complete)))
    if args.max_windows:
        indices = indices[:args.max_windows]
    dataset = Subset(complete, indices)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    genuine = []
    masks = []
    groups = []
    mean_rows = []
    linear_rows = []
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        with torch.inference_mode():
            mean = mean_model(features, mask)
            linear = interpolate_masked_context(features, mask)
        genuine.extend(features.cpu().numpy())
        masks.extend(mask.cpu().numpy())
        mean_rows.extend(mean.cpu().numpy())
        linear_rows.extend(linear.cpu().numpy())
        groups.extend(str(item).rsplit(":", 1)[0] for item in batch["item"])
    target_tensor = torch.from_numpy(np.stack(genuine))
    mask_tensor = torch.from_numpy(np.stack(masks))
    mean_tensor = torch.from_numpy(np.stack(mean_rows))
    linear_tensor = torch.from_numpy(np.stack(linear_rows))
    mean_scores = per_window_score(mean_tensor, target_tensor, mask_tensor)
    linear_scores = per_window_score(linear_tensor, target_tensor, mask_tensor)
    group_array = np.asarray(groups)

    temperatures = {}
    for temperature in args.temperatures:
        torch.manual_seed(args.sample_seed)
        generated = []
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            with torch.inference_mode():
                mean = mean_model(features, mask)
                normalized = diffusion.sample_normalized_residual(
                    mean, mask, temperature=temperature,
                    sampling_steps=args.sampling_steps,
                )
                sample = mean.clone()
                spatial = mean[..., :3] + normalized * scale[None, None]
                sample[..., :3] = torch.where(
                    mask[:, :, None, None], spatial, mean[..., :3]
                )
            generated.extend(sample.cpu().numpy())
        generated_tensor = torch.from_numpy(np.stack(generated))
        generated_scores = per_window_score(
            generated_tensor, target_tensor, mask_tensor
        )
        temperatures[str(temperature)] = {
            "reconstruction_vs_deterministic_mean": bootstrap_improvement(
                generated_scores, mean_scores, args.bootstrap_iterations
            ),
            "reconstruction_vs_linear": bootstrap_improvement(
                generated_scores, linear_scores, args.bootstrap_iterations
            ),
            "genuine_vs_generated_discriminator": paired_discrimination(
                genuine, generated, masks, group_array, args.folds
            ),
        }
    report = {
        "format": "transition_residual_diffusion_naturalness_audit_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "mean_checkpoint": checkpoint["mean_checkpoint"],
        "mean_checkpoint_sha256": checkpoint["mean_checkpoint_sha256"],
        "held_out_signer": held_out,
        "paired_windows": len(dataset),
        "held_out_source_clips": len(set(groups)),
        "sampling_steps": args.sampling_steps,
        "sample_seed": args.sample_seed,
        "temperatures": temperatures,
        "deterministic_mean_discriminator": paired_discrimination(
            genuine, mean_rows, masks, group_array, args.folds
        ),
        "linear_discriminator": paired_discrimination(
            genuine, linear_rows, masks, group_array, args.folds
        ),
        "claim_boundary": (
            "machine feature distribution is not rendered-video or human-perceptual naturalness"
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
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("artifacts/models/transition_residual_diffusion_v17_h8_smoke/best_model.pth"),
    )
    parser.add_argument(
        "--landmark-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/transition_residual_diffusion_v17_h8_smoke.json"),
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=9701)
    parser.add_argument("--max-windows", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.10)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
