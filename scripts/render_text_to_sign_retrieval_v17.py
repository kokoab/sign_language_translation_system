#!/usr/bin/env python3
"""Render exact known text/gloss phrases from genuine full-motion landmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.train_signing_voice_v17 import sha256
from scripts.build_generated_phrase_review_v17 import load_separated_landmark_artifact
from scripts.render_genuine_local_phrase_reference_v17 import coordinate_bounds
from scripts.render_signing_voice_phrase_v17 import avatar_panel, put_text


def normalize_phrase(text: str) -> str:
    tokens = re.findall(r"[A-Z0-9]+", text.upper())
    if tokens[:2] == ["THANK", "YOU"]:
        tokens[:2] = ["THANKYOU"]
    if not tokens:
        raise ValueError("text/gloss phrase is empty")
    return "_".join(tokens)


def resolve_item(report: dict[str, object], text: str) -> dict[str, object]:
    key = normalize_phrase(text)
    items = {str(row["phrase"]): row for row in report["items"]}
    if key not in items:
        raise ValueError(
            f"unsupported exact phrase {key!r}; available: {', '.join(sorted(items))}"
        )
    return items[key]


def render_frame(
    rig: np.ndarray,
    frame: int,
    requested: str,
    phrase: str,
    bounds: tuple[np.ndarray, float],
) -> np.ndarray:
    canvas = np.full((720, 720, 3), 8, np.uint8)
    put_text(canvas, "TEXT TO SIGN - GENUINE RETRIEVAL", (22, 40), 0.72, (244, 247, 251), 2)
    put_text(canvas, f"Input: {requested}", (22, 72), 0.54, (160, 181, 210), 1)
    put_text(canvas, f"Gloss: {phrase.replace('_', ' ')}", (22, 100), 0.54, (160, 181, 210), 1)
    panel = avatar_panel(
        rig, frame, (680, 550), *bounds, (56, 186, 255), filled_avatar=False
    )
    canvas[125:675, 20:700] = panel
    put_text(canvas, "Real full-phrase motion; render completion only", (22, 707), 0.45, (185, 196, 214), 1)
    return canvas


def run(args: argparse.Namespace) -> dict[str, object]:
    catalog = json.loads(args.catalog.read_text())
    item = resolve_item(catalog, args.text)
    artifact = Path(item["artifact"])
    rig, _, metadata, diagnostics = load_separated_landmark_artifact(artifact)
    if not metadata.get("genuine_motion") or metadata.get("generated_motion"):
        raise ValueError("retrieval catalog does not point to genuine motion")
    bounds = coordinate_bounds(rig)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (720, 720)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the text-to-sign retrieval video")
    repeats = max(1, round(args.output_fps / float(item["fps"])))
    first = render_frame(rig, 0, args.text, str(item["phrase"]), bounds)
    for _ in range(args.output_fps):
        writer.write(first)
    for frame in range(len(rig)):
        image = render_frame(rig, frame, args.text, str(item["phrase"]), bounds)
        for _ in range(repeats):
            writer.write(image)
    writer.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ], check=True)
    temporary.unlink()
    report = {
        "format": "slt_text_to_sign_genuine_retrieval_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_text": args.text,
        "resolved_phrase": item["phrase"],
        "mode": "exact_genuine_phrase_retrieval",
        "generated_motion": False,
        "source_item_id": item["source_item_id"],
        "source_artifact": artifact.as_posix(),
        "source_artifact_sha256": sha256(artifact),
        "observation_presence_fraction": diagnostics["observation_presence_fraction"],
        "video": args.output.as_posix(),
        "video_sha256": sha256(args.output),
        "claim_boundary": (
            "This is a safe exact-phrase text-to-sign baseline using genuine motion. "
            "It does not generate unseen gloss combinations."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--text", required=True)
    value.add_argument("--catalog", type=Path, default=Path("artifacts/reports/stage2_v17_genuine_local_phrase_reference_v1/report.json"))
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_text_to_sign_retrieval_v1/phrase.mp4"))
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_text_to_sign_retrieval_v1/report.json"))
    value.add_argument("--output-fps", type=int, default=30)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
