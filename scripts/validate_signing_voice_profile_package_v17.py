#!/usr/bin/env python3
"""Cold-reload and integrity-check the final v17 signing-voice profile package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_signing_voice_profile_v17 import apply_voice_profile, decode_profile
from active.v17.signing_voice_phrase_v17 import (
    NovelVoiceRecipe,
    compose_phrase,
    load_transition_voice,
    voice_duration_ratio,
)
from active.v17.train_signing_voice_v17 import load_content_model, sha256


def decode_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("final signing-voice video cannot be opened")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (height, width):
            raise RuntimeError("video frame geometry changed during decode")
        frames += 1
    capture.release()
    if not frames:
        raise RuntimeError("final signing-voice video has no decodable frames")
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def run(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_signing_voice_profile_v17":
        raise ValueError("unexpected signing-voice profile checkpoint")
    report = json.loads(args.demo_report.read_text())
    if report.get("format") != "slt_signing_voice_phrase_demo_v17":
        raise ValueError("unexpected signing-voice visual report")
    if report["signing_voice_checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("visual report does not pin the final voice checkpoint")
    for path_key, hash_key in (("video", "video_sha256"), ("preview", "preview_sha256")):
        path = Path(report[path_key])
        if sha256(path) != report[hash_key]:
            raise ValueError(f"{path_key} hash mismatch")
    summary = json.loads(Path(checkpoint["signer_disjoint_summary"]).read_text())
    if sha256(Path(checkpoint["signer_disjoint_summary"])) != checkpoint["signer_disjoint_summary_sha256"]:
        raise ValueError("signer-disjoint evidence hash mismatch")
    if summary["aggregate"] != checkpoint["signer_disjoint_aggregate"]:
        raise ValueError("train-all checkpoint changed inherited fold evidence")

    mean, timing = load_transition_voice(args.mean_checkpoint, args.timing_checkpoint)
    content_model, labels = load_content_model(args.content_checkpoint, torch.device("cpu"))
    if labels != {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}:
        raise ValueError("cold-reload content map mismatch")
    if report["mean_checkpoint_sha256"] != sha256(args.mean_checkpoint):
        raise ValueError("transition mean hash mismatch")
    if report["timing_checkpoint_sha256"] != sha256(args.timing_checkpoint):
        raise ValueError("transition timing hash mismatch")
    if report["content_checkpoint_sha256"] != sha256(args.content_checkpoint):
        raise ValueError("content checkpoint hash mismatch")

    voice_to_index = {
        voice: index for index, voice in enumerate(checkpoint["train_voices"])
    }
    cold_voices = []
    for voice, raw_row in zip(report["voices"], report["raw_voices"]):
        indices = tuple(voice_to_index[value] for value in voice["source_voice_ids"])
        recipe = NovelVoiceRecipe(voice["name"], indices, tuple(voice["weights"]))
        weights = torch.tensor(recipe.weights, dtype=torch.float32)
        weights /= weights.sum()
        latent = (
            checkpoint["train_voice_style_latents"][list(indices)].float()
            * weights[:, None]
        ).sum(dim=0)
        profile = decode_profile(
            latent.numpy(), checkpoint["latent_mean"].numpy(),
            checkpoint["latent_components"].numpy(),
        )
        targets = [labels[value] for value in report["requested_glosses"]]
        isolated = []
        predictions = []
        for target, strength in zip(targets, voice["selected_profile_strengths"]):
            prototype = checkpoint["content_prototypes"][target].numpy().astype(np.float32)
            generated = apply_voice_profile(
                prototype, profile, profile_strength=float(strength),
                curve_strength=float(checkpoint["curve_strength"]) * float(strength),
            )
            isolated.append(generated)
            with torch.inference_mode():
                predictions.append(int(content_model(torch.from_numpy(generated)[None]).argmax(1)))
        if predictions != targets:
            raise ValueError(f"cold-reload content mismatch for voice {voice['name']}")
        phrase, timeline = compose_phrase(
            isolated, targets, voice_duration_ratio(checkpoint, recipe),
            checkpoint["class_median_observed_frames"], mean, timing,
        )
        raw_path = Path(raw_row["path"])
        if sha256(raw_path) != raw_row["sha256"]:
            raise ValueError("raw generated voice hash mismatch")
        with np.load(raw_path, allow_pickle=False) as payload:
            stored = payload["landmarks"].astype(np.float32)
            metadata = json.loads(str(payload["metadata_json"]))
        if stored.shape != phrase.shape or not np.allclose(stored, phrase, atol=2e-3):
            raise ValueError(f"cold-reload phrase mismatch for voice {voice['name']}")
        if metadata["timeline"] != timeline:
            raise ValueError(f"cold-reload timeline mismatch for voice {voice['name']}")
        cold_voices.append({
            "name": voice["name"], "frames": len(phrase),
            "stage1_predictions": [report["requested_glosses"][i] for i in range(len(targets))],
            "finite": bool(np.isfinite(phrase).all()),
        })

    style_matrix = np.asarray(report["voice_style_cosine_matrix"])
    off_diagonal = style_matrix[~np.eye(len(style_matrix), dtype=np.bool_)]
    if float(off_diagonal.max()) >= 0.95:
        raise ValueError("novel generated voices are not mutually distinct")
    if any(not row["all_stage1_predictions_correct"] for row in report["voices"]):
        raise ValueError("visual report contains a content-gate failure")
    video = decode_video(Path(report["video"]))
    result = {
        "format": "slt_signing_voice_profile_package_cold_reload_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint.as_posix(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "signer_disjoint_summary_sha256": checkpoint["signer_disjoint_summary_sha256"],
        "demo_report": args.demo_report.as_posix(),
        "demo_report_sha256": sha256(args.demo_report),
        "video": video,
        "maximum_pairwise_voice_cosine": float(off_diagonal.max()),
        "voices": cold_voices,
        "all_content_predictions_correct": True,
        "all_raw_trajectories_reproduced": True,
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/signing_voice_profile_v17_allvoices_final/model.pth"))
    parser.add_argument("--demo-report", type=Path, default=Path("artifacts/reports/signing_voice_phrase_v17.json"))
    parser.add_argument("--mean-checkpoint", type=Path, default=Path("artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/model.pth"))
    parser.add_argument("--timing-checkpoint", type=Path, default=Path("artifacts/models/transition_span_multicorpus_v17_allvoices_final/model.pth"))
    parser.add_argument("--content-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/signing_voice_profile_package_cold_reload_v17.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
