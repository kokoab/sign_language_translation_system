#!/usr/bin/env python3
"""Build the frozen train-all 16-D content-gated signing-voice profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_signing_voice_profile_v17 import (
    estimate_voice_profile,
    fit_profile_latent,
)
from active.v17.train_signing_voice_v17 import build_prototypes, load_content_model, sha256


def run(args):
    with np.load(args.pool, allow_pickle=False) as payload:
        landmarks = payload["landmarks"].astype(np.float32)
        targets = payload["target_indices"].astype(np.int64)
        signers = payload["signer_ids"].astype(str)
        observed_frames = payload["observed_frames"].astype(np.int16)
    voices = sorted({
        signer for signer in set(signers.tolist())
        if len(set(targets[signers == signer].tolist())) >= 2
    })
    indices = np.arange(len(landmarks))[np.isin(signers, voices)]
    prototypes_tensor, medoids = build_prototypes(landmarks, targets, indices)
    prototypes = prototypes_tensor.numpy().astype(np.float32)
    profiles = [
        estimate_voice_profile(
            landmarks, targets, indices[signers[indices] == voice], prototypes
        )
        for voice in voices
    ]
    latent_mean, latent_components, latents = fit_profile_latent(
        profiles, args.latent_dim
    )
    class_medians = np.asarray([
        np.median(observed_frames[indices[targets[indices] == target]])
        for target in range(100)
    ], dtype=np.float32)
    duration_ratios = []
    for voice in voices:
        rows = indices[signers[indices] == voice]
        duration_ratios.append(float(np.median(
            observed_frames[rows] / class_medians[targets[rows]].clip(min=1)
        )))
    _, label_to_index = load_content_model(args.content_checkpoint, torch.device("cpu"))
    summary = json.loads(args.summary.read_text())
    if summary.get("format") != "slt_signing_voice_profile_signer_disjoint_summary_v17":
        raise ValueError("unexpected signer-disjoint evidence summary")
    if summary["frozen_design"] != {
        "latent_dim": 16,
        "profile": "per-node median XYZ signing-space offset",
        "curve_strength": 0.0,
        "content_gate_strengths": [1.0, 0.75, 0.5, 0.4, 0.25, 0.0],
        "content_gate": "select strongest profile retaining requested frozen Stage-1 label",
    }:
        raise ValueError("signer-disjoint evidence does not pin the frozen final design")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "slt_signing_voice_profile_v17",
        "version": 1,
        "latent_dim": args.latent_dim,
        "latent_mean": torch.from_numpy(latent_mean),
        "latent_components": torch.from_numpy(latent_components),
        "train_voice_style_latents": torch.from_numpy(latents),
        "train_voice_profiles": torch.from_numpy(np.stack([value.vector() for value in profiles])),
        "content_prototypes": prototypes_tensor,
        "content_prototype_pool_indices": medoids,
        "curve_strength": 0.0,
        "adaptive_content_gate": True,
        "content_gate_strengths": [1.0, 0.75, 0.50, 0.40, 0.25, 0.0],
        "label_to_index": label_to_index,
        "train_voices": voices,
        "validation_voices": [],
        "train_voice_duration_ratios": torch.tensor(duration_ratios),
        "class_median_observed_frames": torch.from_numpy(class_medians),
        "fold": None,
        "selection_performed": False,
        "signer_disjoint_summary": args.summary.as_posix(),
        "signer_disjoint_summary_sha256": sha256(args.summary),
        "signer_disjoint_aggregate": summary["aggregate"],
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "content_checkpoint": args.content_checkpoint.as_posix(),
        "content_checkpoint_sha256": sha256(args.content_checkpoint),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    checkpoint_path = args.output / "model.pth"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "latent_dim": args.latent_dim,
        "train_voices": len(voices),
        "train_examples": len(indices),
        "selection_performed": False,
        "signer_disjoint_summary": args.summary.as_posix(),
        "signer_disjoint_summary_sha256": sha256(args.summary),
        "inherited_evidence": summary["aggregate"],
        "claim_boundary": "train-all profile inherits held-out evidence; it has no independent naturalness score",
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/local/signing_voice_v17/train_only_landmark_pool.npz"))
    parser.add_argument("--content-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--summary", type=Path, default=Path("artifacts/reports/signing_voice_profile_signer_disjoint_summary_v17.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/signing_voice_profile_v17_allvoices_final"))
    parser.add_argument("--latent-dim", type=int, default=16)
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
