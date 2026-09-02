#!/usr/bin/env python3
"""Build separate recognizer observations and render-only animation rigs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from active.v17.landmark_anatomy_v17 import (
    IMPUTED_CONFIDENCE,
    anatomy_coverage,
    build_anatomy_template,
    complete_landmark_anatomy,
)
from active.v17.train_signing_voice_v17 import load_content_model, sha256


@torch.inference_mode()
def predict(model, values: np.ndarray) -> np.ndarray:
    rows = []
    for start in range(0, len(values), 64):
        rows.append(model(torch.from_numpy(values[start:start + 64])).argmax(1).numpy())
    return np.concatenate(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    profile = torch.load(args.profile_checkpoint, map_location="cpu", weights_only=False)
    if profile.get("format") != "slt_signing_voice_profile_v17":
        raise ValueError("unexpected signing-voice profile checkpoint")
    if sha256(args.pool) != profile.get("pool_sha256"):
        raise ValueError("profile checkpoint and anatomy pool differ")
    if sha256(args.content_checkpoint) != profile.get("content_checkpoint_sha256"):
        raise ValueError("profile checkpoint and content classifier differ")
    with np.load(args.pool, allow_pickle=False) as payload:
        landmarks = payload["landmarks"].astype(np.float32)
        targets = payload["target_indices"].astype(np.int64)
        item_ids = payload["item_ids"].astype(str)
        signers = payload["signer_ids"].astype(str)
        source_codes = payload["source_codes"].astype(np.uint8)
    label_to_index = {str(key): int(value) for key, value in profile["label_to_index"].items()}
    if set(targets.tolist()) != set(range(100)):
        raise ValueError("train-only anatomy pool does not cover all 100 classes")
    model, content_labels = load_content_model(args.content_checkpoint, torch.device("cpu"))
    if content_labels != label_to_index:
        raise ValueError("content classifier and profile class maps differ")

    template = build_anatomy_template(landmarks)
    selected_rigs = []
    selected_rows = []
    selected_coverages = []
    candidate_counts = []
    correct_candidate_counts = []
    for target in range(100):
        rows = np.flatnonzero(targets == target)
        rigs = np.stack([
            complete_landmark_anatomy(landmarks[row], template)[0] for row in rows
        ])
        rig_predictions = predict(model, rigs)
        observation_predictions = predict(model, landmarks[rows])
        correct = np.flatnonzero(
            (rig_predictions == target) & (observation_predictions == target)
        )
        if not len(correct):
            raise RuntimeError(
                f"no prototype preserves class {target} in both observation and rig form"
            )
        coverages = np.asarray([anatomy_coverage(landmarks[row]) for row in rows])
        correct_coverages = coverages[correct]
        high_coverage = correct[
            correct_coverages >= np.quantile(correct_coverages, 0.75)
        ]
        raw = landmarks[rows]
        present = raw[..., 3:4] > 0
        mean_xyz = (raw[..., :3] * present).sum(axis=0) / present.sum(axis=0).clip(min=1)
        errors = []
        for candidate in high_coverage:
            source = landmarks[rows[candidate]]
            valid = source[..., 3] > 0
            errors.append(float(np.square(source[..., :3] - mean_xyz)[valid].mean()))
        choice = int(high_coverage[int(np.argmin(errors))])
        selected_rigs.append(rigs[choice])
        selected_rows.append(int(rows[choice]))
        selected_coverages.append(float(coverages[choice]))
        candidate_counts.append(len(rows))
        correct_candidate_counts.append(len(correct))

    recognition_prototypes = landmarks[selected_rows].astype(np.float32)
    animation_rig_prototypes = np.stack(selected_rigs).astype(np.float32)
    expected = np.arange(100)
    if not np.array_equal(predict(model, recognition_prototypes), expected):
        raise RuntimeError("selected observation prototypes fail the 100-class content gate")
    if not np.array_equal(predict(model, animation_rig_prototypes), expected):
        raise RuntimeError("selected animation rigs fail the 100-class content gate")
    source_observed = landmarks[selected_rows, ..., 3] > 0
    source_hand_activity = source_observed[..., :42].any(axis=2)
    source_hand_participation = np.stack((
        source_observed[..., :21].any(axis=(1, 2)),
        source_observed[..., 21:42].any(axis=(1, 2)),
    ), axis=1)
    rig_hand_participation = np.stack((
        (animation_rig_prototypes[..., :21, 3] > 0).any(axis=(1, 2)),
        (animation_rig_prototypes[..., 21:42, 3] > 0).any(axis=(1, 2)),
    ), axis=1)
    if (
        not source_hand_activity.any(axis=1).all()
        or not np.array_equal(source_hand_participation, rig_hand_participation)
        or not (animation_rig_prototypes[..., 42:, 3] == 1).all()
        or np.array_equal(source_observed, np.ones_like(source_observed))
    ):
        raise RuntimeError("selected animation prototypes violate anatomy/activity contract")

    args.output.mkdir(parents=True, exist_ok=True)
    package = args.output / "anatomy.npz"
    metadata = {
        "format": "slt_signing_landmark_anatomy_v17",
        "version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": "citizen_semlex_asllrp_train_only",
        "classes": 100,
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "profile_checkpoint": args.profile_checkpoint.as_posix(),
        "profile_checkpoint_sha256": sha256(args.profile_checkpoint),
        "content_checkpoint": args.content_checkpoint.as_posix(),
        "content_checkpoint_sha256": sha256(args.content_checkpoint),
        "label_to_index": label_to_index,
        "imputed_confidence": IMPUTED_CONFIDENCE,
        "prototype_selection": (
            "content-correct candidates only; top coverage quartile; minimum "
            "observed-node error to the class mean"
        ),
        "data_contract": (
            "recognition_prototypes retain the detector observation mask; "
            "animation_rig_prototypes are render-only and must never be supplied "
            "to recognizer training, validation, or testing; a never-observed hand "
            "remains absent in the animation rig"
        ),
        "animation_completion_contract": (
            "preserve every observed XYZ; temporally fill intermittent nodes; "
            "complete face/body from train-only medians; complete only participating "
            "hands and bridge at most three missing detector frames"
        ),
        "selected_source_item_ids": item_ids[selected_rows].tolist(),
        "selected_source_signers": signers[selected_rows].tolist(),
        "selected_source_codes": source_codes[selected_rows].astype(int).tolist(),
        "selected_source_coverages": selected_coverages,
        "candidate_counts": candidate_counts,
        "content_correct_candidate_counts": correct_candidate_counts,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    temporary = package.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        recognition_prototypes=recognition_prototypes.astype(np.float16),
        animation_rig_prototypes=animation_rig_prototypes.astype(np.float16),
        source_observed_masks=source_observed,
        source_hand_activity=source_hand_activity,
        source_hand_participation=source_hand_participation,
        prototype_pool_indices=np.asarray(selected_rows, dtype=np.int64),
        canonical_absolute_xyz=template["absolute_xyz"],
        canonical_hand_shapes=template["hand_shapes"],
        canonical_wrist_from_elbow=template["wrist_from_elbow"],
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(package)
    report = {
        "format": "slt_signing_landmark_anatomy_result_v17",
        "package": package.as_posix(),
        "package_sha256": sha256(package),
        "classes": 100,
        "recognition_content_accuracy": 1.0,
        "animation_rig_content_accuracy": 1.0,
        "observation_presence_fraction": float(source_observed.mean()),
        "animation_rig_presence_fraction": float(
            (animation_rig_prototypes[..., 3] > 0).mean()
        ),
        "one_handed_classes": int((source_hand_participation.sum(axis=1) == 1).sum()),
        "two_handed_classes": int((source_hand_participation.sum(axis=1) == 2).sum()),
        "hand_participation_preserved": True,
        "selected_coverage_minimum": min(selected_coverages),
        "selected_coverage_median": float(np.median(selected_coverages)),
        "selected_coverage_maximum": max(selected_coverages),
        "all_classes_have_content_safe_candidate": min(correct_candidate_counts) > 0,
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
        "claim_boundary": (
            "the participation-preserving rig is render-only; neither content gates nor anatomy "
            "completion establish native-sign linguistic or perceptual naturalness"
        ),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--pool", type=Path, default=Path("data/local/signing_voice_v17/train_only_landmark_pool.npz"))
    value.add_argument("--profile-checkpoint", type=Path, default=Path("artifacts/models/signing_voice_profile_v17_allvoices_final/model.pth"))
    value.add_argument("--content-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    value.add_argument("--output", type=Path, default=Path("artifacts/models/signing_landmark_anatomy_v17_v3"))
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
