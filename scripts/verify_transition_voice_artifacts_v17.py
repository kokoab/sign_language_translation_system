#!/usr/bin/env python3
"""Cold-reload and structurally verify the final v17 transition voice artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_diffusion_v17 import (
    TransitionResidualDiffusionV17,
    TransitionResidualDiffusionV17Config,
)
from active.v17.model_transition_span_v17 import (
    TransitionSpanPredictorV17,
    TransitionSpanV17Config,
)
from active.v17.train_transition_diffusion_v17 import load_mean_model
from active.v17.train_transition_inpainter_v17 import (
    TransitionWindowDataset,
    sha256,
)
from active.v17.train_transition_span_multicorpus_v17 import TransitionSpanDataset


def run(args: argparse.Namespace) -> dict[str, object]:
    mean_hash = sha256(args.mean_checkpoint)
    diffusion_hash = sha256(args.diffusion_checkpoint)
    diffusion_checkpoint = torch.load(
        args.diffusion_checkpoint, map_location="cpu", weights_only=False
    )
    if diffusion_checkpoint.get("format") != "slt_transition_residual_diffusion_v17":
        raise ValueError("unexpected diffusion checkpoint format")
    if diffusion_checkpoint.get("mean_checkpoint_sha256") != mean_hash:
        raise ValueError("diffusion artifact does not pin the supplied mean artifact")

    mean_model = load_mean_model(args.mean_checkpoint, torch.device("cpu"))
    diffusion = TransitionResidualDiffusionV17(
        TransitionResidualDiffusionV17Config(
            **diffusion_checkpoint["model_config"]
        )
    )
    diffusion.load_state_dict(diffusion_checkpoint["model_state_dict"])
    diffusion.eval()
    scale = diffusion_checkpoint["residual_scale"].float()

    signer_rows = diffusion_checkpoint.get(
        "how2sign_train_signers", diffusion_checkpoint.get("train_signers")
    )
    if not signer_rows:
        raise ValueError("diffusion checkpoint has no training voice provenance")
    signer = sorted(signer_rows)[0]
    dataset = TransitionWindowDataset(
        args.landmark_root, {signer}, seed=4701, fixed_masks=True
    )
    row = dataset[0]
    features = row["features"].unsqueeze(0)
    mask = row["mask"].unsqueeze(0)
    visible = ~mask[:, :, None, None].expand_as(features)
    with torch.inference_mode():
        mean = mean_model(features, mask)
    if not torch.isfinite(mean).all() or not torch.equal(mean[visible], features[visible]):
        raise RuntimeError("mean artifact failed finite/visible preservation checks")

    temperature_reports = {}
    for temperature in args.temperatures:
        with torch.inference_mode():
            normalized = diffusion.sample_normalized_residual(
                mean,
                mask,
                temperature=temperature,
                generator=torch.Generator().manual_seed(args.sample_seed),
                sampling_steps=args.sampling_steps,
            )
            generated = mean.clone()
            spatial = mean[..., :3] + normalized * scale[None, None]
            generated[..., :3] = torch.where(
                mask[:, :, None, None], spatial, mean[..., :3]
            )
        delta = generated[..., :3] - mean[..., :3]
        residual = delta[mask[:, :, None, None].expand_as(delta)]
        bound = float(6.0 * scale.max()) + 1e-6
        maximum = float(residual.abs().max())
        if not torch.isfinite(generated).all():
            raise RuntimeError("stochastic artifact produced non-finite output")
        if not torch.equal(generated[visible], features[visible]):
            raise RuntimeError("stochastic artifact changed visible context")
        if maximum == 0.0 or maximum > bound:
            raise RuntimeError("stochastic residual is zero or outside its hard bound")
        temperature_reports[str(temperature)] = {
            "masked_mean_absolute_delta_from_mean": float(residual.abs().mean()),
            "masked_max_absolute_delta_from_mean": maximum,
            "hard_maximum_delta_bound": bound,
            "finite": True,
            "visible_context_exactly_preserved": True,
        }

    timing_report = None
    timing_hash = None
    if args.timing_checkpoint is not None:
        timing_hash = sha256(args.timing_checkpoint)
        timing_checkpoint = torch.load(
            args.timing_checkpoint, map_location="cpu", weights_only=False
        )
        if timing_checkpoint.get("format") != "slt_transition_span_predictor_v17":
            raise ValueError("unexpected timing checkpoint format")
        timing = TransitionSpanPredictorV17(
            TransitionSpanV17Config(**timing_checkpoint["model_config"])
        )
        timing.load_state_dict(timing_checkpoint["model_state_dict"])
        timing.eval()
        timing_row = TransitionSpanDataset(args.landmark_root, {signer})[0]
        with torch.inference_mode():
            timing_logits = timing(timing_row["context"].unsqueeze(0))
        predicted_span = int(timing_logits.argmax(dim=1).item()) + 4
        if timing_logits.shape != (1, 9) or not torch.isfinite(timing_logits).all():
            raise RuntimeError("timing artifact failed shape/finite checks")
        if not 4 <= predicted_span <= 12:
            raise RuntimeError("timing artifact predicted an invalid span")
        timing_report = {
            "checkpoint": args.timing_checkpoint.as_posix(),
            "checkpoint_sha256": timing_hash,
            "logits_shape": list(timing_logits.shape),
            "finite": True,
            "predicted_span": predicted_span,
            "valid_span": True,
        }

    report = {
        "format": "transition_voice_artifact_cold_reload_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mean_checkpoint": args.mean_checkpoint.as_posix(),
        "mean_checkpoint_sha256": mean_hash,
        "diffusion_checkpoint": args.diffusion_checkpoint.as_posix(),
        "diffusion_checkpoint_sha256": diffusion_hash,
        "timing": timing_report,
        "verified_signer": signer,
        "verified_item": row["item"],
        "sampling_steps": args.sampling_steps,
        "sample_seed": args.sample_seed,
        "temperatures": temperature_reports,
        "passed": True,
        "claim_boundary": (
            "structural cold-reload verification is not human-perceptual naturalness"
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
        "--mean-checkpoint",
        type=Path,
        default=Path("artifacts/models/transition_inpainter_v17_all6_final/model.pth"),
    )
    parser.add_argument("--timing-checkpoint", type=Path)
    parser.add_argument(
        "--diffusion-checkpoint",
        type=Path,
        default=Path("artifacts/models/transition_residual_diffusion_v17_all6_final/model.pth"),
    )
    parser.add_argument(
        "--landmark-root",
        type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/transition_voice_artifact_cold_reload_v17.json"),
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=(0.10, 0.20))
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=1701)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
