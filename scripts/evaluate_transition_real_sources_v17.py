#!/usr/bin/env python3
"""Evaluate one frozen transition model on every compatible genuine source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Iterator

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_transition_inpainter_v17 import (
    TransitionInpainterV17,
    TransitionInpainterV17Config,
    interpolate_masked_context,
)
from scripts.evaluate_transition_inpainter_naturalness_v17 import per_window_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_mask(index: int, seed: int = 1701) -> np.ndarray:
    rng = random.Random(seed + index * 104729)
    length = rng.randint(4, 12)
    start = rng.randint(3, 32 - length - 3)
    result = np.zeros(32, dtype=np.bool_)
    result[start:start + length] = True
    return result


def windows(root: Path, source: str | None) -> Iterator[tuple[str, np.ndarray]]:
    for path in sorted(root.glob("**/*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            if "landmarks" not in payload.files or "metadata_json" not in payload.files:
                continue
            meta = json.loads(str(payload["metadata_json"].item()))
            if str(meta.get("role", "train")) != "train":
                continue
            if source is not None and str(meta.get("source")) != source:
                continue
            values = payload["landmarks"].astype(np.float32)
            if "window_valid" in payload.files:
                valid = payload["window_valid"].astype(np.bool_)
            elif "landmark_window_valid" in payload.files:
                valid = payload["landmark_window_valid"].astype(np.bool_)
            else:
                valid = np.ones(len(values), dtype=np.bool_)
            for index in np.flatnonzero(valid):
                yield path.as_posix(), values[int(index)]


def evaluate(
    model: TransitionInpainterV17,
    rows: Iterator[tuple[str, np.ndarray]],
    batch_size: int,
) -> dict[str, float | int]:
    learned_scores: list[np.ndarray] = []
    linear_scores: list[np.ndarray] = []
    clips: set[str] = set()
    batch: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    seen = 0

    def flush() -> None:
        nonlocal batch, masks
        if not batch:
            return
        target = torch.from_numpy(np.stack(batch))
        mask = torch.from_numpy(np.stack(masks))
        with torch.inference_mode():
            predicted = model(target, mask)
            interpolated = interpolate_masked_context(target, mask)
        learned_scores.append(per_window_score(predicted, target, mask))
        linear_scores.append(per_window_score(interpolated, target, mask))
        batch = []
        masks = []

    for clip, value in rows:
        clips.add(clip)
        batch.append(value)
        masks.append(fixed_mask(seen))
        seen += 1
        if len(batch) >= batch_size:
            flush()
    flush()
    if not learned_scores:
        raise ValueError("no compatible train-side windows")
    learned = np.concatenate(learned_scores)
    linear = np.concatenate(linear_scores)
    return {
        "source_clips": len(clips),
        "windows": int(len(learned)),
        "learned_score": float(learned.mean()),
        "linear_score": float(linear.mean()),
        "relative_improvement_vs_linear": float(
            (linear.mean() - learned.mean()) / max(linear.mean(), 1e-12)
        ),
        "windows_improved_fraction": float(np.mean(learned < linear)),
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
    specs = (
        ("local_phrases", args.local, None),
        ("asllrp_contiguous", args.asllrp_contiguous, None),
        ("asllrp_other_ctc", args.asllrp_other, None),
        ("two_m_flores_asl", args.flores, None),
        ("how2sign", args.how2sign_ncslgr, "how2sign_unlabeled_continuous"),
        ("ncslgr", args.how2sign_ncslgr, "ncslgr_public_continuous"),
        ("youtube_asl_train", args.youtube, "youtube_asl"),
        ("openasl_train", args.openasl, "openasl_unlabeled_continuous"),
    )
    domains = {
        name: evaluate(model, windows(root, source), args.batch_size)
        for name, root, source in specs
    }
    report = {
        "format": "slt_transition_real_source_transfer_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "question": "Does the frozen transition model reconstruct masked genuine intervals better than endpoint interpolation across every compatible downloaded source?",
        "domains": domains,
        "interpretation": "Positive improvement supports cross-domain masked reconstruction only. It does not establish semantic coarticulation, phrase generation, or human naturalness.",
        "openasl_status": "all three retained train-split derivatives are v17-compatible; only the one train-role channel is scored here",
        "local_phrase_status": "all nine phrase families are v17-compatible motion-only data; folder phrases are provenance, not CTC targets",
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
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/model.pth"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_real_motion_reference_audit/reconstruction_transfer.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local", type=Path, default=Path("data/local/stage2_v17_multimodal/train/local_phrases"))
    parser.add_argument("--asllrp-contiguous", type=Path, default=Path("data/local/stage2_v17_multimodal/train/asllrp_contiguous"))
    parser.add_argument("--asllrp-other", type=Path, default=Path("data/local/stage2_v17_asllrp_other_multimodal/train/asllrp_other_ctc"))
    parser.add_argument("--flores", type=Path, default=Path("data/local/stage2_v17_2m_flores_multimodal/train/two_m_flores_asl"))
    parser.add_argument("--how2sign-ncslgr", type=Path, default=Path("data/local/how2sign_transition_landmarks_v17"))
    parser.add_argument("--youtube", type=Path, default=Path("data/local/youtube_asl_transition_landmarks_v17"))
    parser.add_argument("--openasl", type=Path, default=Path("data/local/openasl_transition_landmarks_v17"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
