#!/usr/bin/env python3
"""Build a hash-pinned evidence manifest for the v17 multi-voice transition stack."""

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

from active.v17.train_transition_inpainter_v17 import sha256


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verified_result(result_path: Path) -> tuple[dict[str, object], Path, str]:
    result = read_json(result_path)
    checkpoint = Path(str(result["checkpoint"]))
    digest = sha256(checkpoint)
    if result.get("checkpoint_sha256") != digest:
        raise ValueError(f"checkpoint digest mismatch: {checkpoint}")
    return result, checkpoint, digest


def run(args: argparse.Namespace) -> dict[str, object]:
    mean_result, mean_path, mean_hash = verified_result(args.mean_result)
    diffusion_result, diffusion_path, diffusion_hash = verified_result(
        args.diffusion_result
    )
    timing_result, timing_path, timing_hash = verified_result(args.timing_result)
    mean = torch.load(mean_path, map_location="cpu", weights_only=False)
    diffusion = torch.load(diffusion_path, map_location="cpu", weights_only=False)
    timing = torch.load(timing_path, map_location="cpu", weights_only=False)
    expected_formats = (
        (mean, "slt_transition_inpainter_v17"),
        (diffusion, "slt_transition_residual_diffusion_v17"),
        (timing, "slt_transition_span_predictor_v17"),
    )
    for checkpoint, expected in expected_formats:
        if checkpoint.get("format") != expected:
            raise ValueError(f"expected checkpoint format {expected}")
    if diffusion.get("mean_checkpoint_sha256") != mean_hash:
        raise ValueError("diffusion checkpoint is not pinned to the final mean")

    h2s = set(mean["how2sign_train_signers"])
    web = set(mean["youtube_asl_train_voice_proxies"])
    for checkpoint in (diffusion, timing):
        if set(checkpoint["how2sign_train_signers"]) != h2s:
            raise ValueError("How2Sign voice coverage differs between components")
        if set(checkpoint["youtube_asl_train_voice_proxies"]) != web:
            raise ValueError("public voice coverage differs between components")
    if len(h2s) != 6 or len(web) < 96:
        raise ValueError("multi-voice breadth floor failed")

    motion_loso = read_json(args.motion_loso)
    timing_loso = read_json(args.timing_loso)
    cold_reload = read_json(args.cold_reload)
    if not cold_reload.get("passed"):
        raise ValueError("cold reload did not pass")
    if cold_reload.get("mean_checkpoint_sha256") != mean_hash:
        raise ValueError("cold reload mean digest mismatch")
    if cold_reload.get("diffusion_checkpoint_sha256") != diffusion_hash:
        raise ValueError("cold reload diffusion digest mismatch")
    if (cold_reload.get("timing") or {}).get("checkpoint_sha256") != timing_hash:
        raise ValueError("cold reload timing digest mismatch")

    report = {
        "format": "transition_multivoice_package_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "generalizable landmark-space transition voice: deterministic mean, "
            "bounded stochastic variation, and context-conditioned elapsed span"
        ),
        "voice_coverage": {
            "controlled_how2sign_signers": len(h2s),
            "youtube_asl_channel_voice_proxies": len(web),
            "total_sources_or_proxies": len(h2s) + len(web),
            "important_boundary": (
                "YouTube channels are acquisition-level voice proxies, not "
                "identity-verified unique signers"
            ),
        },
        "components": {
            "deterministic_mean": {
                "checkpoint": mean_path.as_posix(),
                "sha256": mean_hash,
                "parameters": sum(v.numel() for v in mean["model_state_dict"].values()),
                "result": args.mean_result.as_posix(),
            },
            "stochastic_residual": {
                "checkpoint": diffusion_path.as_posix(),
                "sha256": diffusion_hash,
                "parameters": sum(
                    v.numel() for v in diffusion["model_state_dict"].values()
                ),
                "mean_checkpoint_sha256": diffusion["mean_checkpoint_sha256"],
                "recommended_temperatures": diffusion["recommended_temperatures"],
                "result": args.diffusion_result.as_posix(),
            },
            "timing": {
                "checkpoint": timing_path.as_posix(),
                "sha256": timing_hash,
                "parameters": sum(v.numel() for v in timing["model_state_dict"].values()),
                "span_frames": [4, 12],
                "result": args.timing_result.as_posix(),
            },
        },
        "held_out_evidence": {
            "motion_report": args.motion_loso.as_posix(),
            "motion_report_sha256": sha256(args.motion_loso),
            "motion": motion_loso["aggregate"],
            "timing_report": args.timing_loso.as_posix(),
            "timing_report_sha256": sha256(args.timing_loso),
            "timing": timing_loso["aggregate"],
        },
        "cold_reload": {
            "report": args.cold_reload.as_posix(),
            "sha256": sha256(args.cold_reload),
            "passed": True,
        },
        "research_basis": [
            {
                "title": "Sign-D2C: Discrete to Continuous transition generation",
                "url": "https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Discrete_to_Continuous_Generating_Smooth_Transition_Poses_from_Sign_Language_CVPR_2025_paper.pdf",
            },
            {
                "title": "Signing at Scale: learned co-articulation and signer evaluation",
                "url": "https://openaccess.thecvf.com/content/CVPR2022/html/Saunders_Signing_at_Scale_Learning_to_Co-Articulate_Signs_for_Large-Scale_Photo-Realistic_CVPR_2022_paper.html",
            },
        ],
        "claim_boundary": (
            "This is held-out reconstruction/discriminator/timing evidence in "
            "landmark space. It does not yet prove semantic prosody, RGB realism, "
            "or Deaf-signer-rated human naturalness."
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mean-result", type=Path,
        default=Path("artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/result.json"),
    )
    parser.add_argument(
        "--diffusion-result", type=Path,
        default=Path("artifacts/models/transition_residual_diffusion_multicorpus_v17_allvoices_final/result.json"),
    )
    parser.add_argument(
        "--timing-result", type=Path,
        default=Path("artifacts/models/transition_span_multicorpus_v17_allvoices_final/result.json"),
    )
    parser.add_argument(
        "--motion-loso", type=Path,
        default=Path("artifacts/reports/transition_inpainter_multicorpus_loso_summary_v17.json"),
    )
    parser.add_argument(
        "--timing-loso", type=Path,
        default=Path("artifacts/reports/transition_span_loso_summary_v17.json"),
    )
    parser.add_argument(
        "--cold-reload", type=Path,
        default=Path("artifacts/reports/transition_multivoice_package_cold_reload_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/reports/transition_multivoice_package_v17.json"),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
