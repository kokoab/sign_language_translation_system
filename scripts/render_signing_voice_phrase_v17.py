#!/usr/bin/env python3
"""Generate and render one complete phrase in three novel landmark signing voices."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.signing_voice_phrase_v17 import (
    compose_phrase,
    load_transition_voice,
    NovelVoiceRecipe,
    voice_duration_ratio,
)
from active.v17.model_signing_voice_profile_v17 import (
    apply_voice_profile,
    decode_profile,
)
from active.v17.train_signing_voice_v17 import load_content_model, sha256


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
FACE_EDGES = (
    (42, 44), (44, 45), (45, 43), (42, 46), (46, 47), (47, 43),
    (42, 56), (43, 56), (56, 48), (49, 51), (51, 50),
    (50, 52), (52, 49), (53, 54), (54, 55),
)
VOICE_COLORS = ((56, 186, 255), (255, 170, 74), (161, 102, 255))


def build_profile_voice_recipes(checkpoint, seed=1701, candidates=30_000):
    """Select mixtures by decoded profile novelty, not latent-space direction."""
    latents = checkpoint["train_voice_style_latents"].float()
    train_profiles = F.normalize(checkpoint["train_voice_profiles"].float(), dim=1)
    mean = checkpoint["latent_mean"].float()
    components = checkpoint["latent_components"].float()
    rng = np.random.default_rng(seed)
    styles, novelty, ingredients = [], [], []
    for _ in range(candidates):
        indices = tuple(int(value) for value in rng.choice(len(latents), 3, replace=False))
        weights = rng.dirichlet(np.full(3, 2.0))
        if weights.min() < 0.10 or weights.max() > 0.60:
            continue
        weight = torch.tensor(weights, dtype=latents.dtype)
        latent = (latents[list(indices)] * weight[:, None]).sum(dim=0)
        profile = F.normalize(mean + latent @ components, dim=0)
        styles.append(profile)
        novelty.append(float((train_profiles @ profile).max()))
        ingredients.append((indices, tuple(float(value) for value in weights)))
    style_matrix = torch.stack(styles)
    novelty_tensor = torch.tensor(novelty)
    selected = [int(torch.argmin(novelty_tensor))]
    while len(selected) < 3:
        score = torch.maximum(
            (style_matrix @ style_matrix[selected].T).max(dim=1).values,
            novelty_tensor,
        )
        score[selected] = 2.0
        selected.append(int(torch.argmin(score)))
    return [
        NovelVoiceRecipe(name, ingredients[index][0], ingredients[index][1])
        for name, index in zip(("Aster", "Cobalt", "Juniper"), selected)
    ]


def coordinate_bounds(voices: list[dict[str, object]]):
    points = []
    for voice in voices:
        features = voice["phrase"]
        present = features[..., 3] > 0
        points.append(features[..., :2][present])
    values = np.concatenate(points)
    low, high = np.percentile(values, (1, 99), axis=0)
    center = (low + high) / 2
    extent = max(float((high - low).max()) * 0.62, 1.0)
    return center, extent


def point_xy(point, center, extent, size):
    width, height = size
    scale = min(width, height) * 0.82 / (2 * extent)
    return (
        int(width / 2 + (point[0] - center[0]) * scale),
        int(height / 2 + (point[1] - center[1]) * scale),
    )


def line_if_present(canvas, frame, first, second, color, center, extent, width):
    if frame[first, 3] <= 0 or frame[second, 3] <= 0:
        return
    size = (canvas.shape[1], canvas.shape[0])
    cv2.line(
        canvas, point_xy(frame[first], center, extent, size),
        point_xy(frame[second], center, extent, size), color, width, cv2.LINE_AA,
    )


def avatar_panel(features, frame_index, size, center, extent, accent):
    width, height = size
    canvas = np.full((height, width, 3), (13, 16, 23), np.uint8)
    frame = features[frame_index]
    # Filled head and torso make this readable as an articulated avatar rather than
    # a diagnostic landmark plot. Fine facial and hand motion remains data-driven.
    face_nodes = frame[42:57]
    face_present = face_nodes[:, 3] > 0
    if face_present.any():
        points = [point_xy(value, center, extent, size) for value in face_nodes[face_present]]
        array = np.asarray(points)
        face_center = tuple(np.mean(array, axis=0).round().astype(int))
        radius_x = max(22, int(np.ptp(array[:, 0]) * 0.72))
        radius_y = max(30, int(np.ptp(array[:, 1]) * 0.72))
        cv2.ellipse(canvas, face_center, (radius_x, radius_y), 0, 0, 360, (69, 76, 91), -1, cv2.LINE_AA)
    if all(frame[index, 3] > 0 for index in (57, 58)):
        left = point_xy(frame[57], center, extent, size)
        right = point_xy(frame[58], center, extent, size)
        shoulder_y = (left[1] + right[1]) // 2
        bottom_y = min(height - 20, shoulder_y + max(100, abs(right[0] - left[0])))
        polygon = np.asarray([left, right, (right[0] + 35, bottom_y), (left[0] - 35, bottom_y)])
        cv2.fillConvexPoly(canvas, polygon, (38, 48, 67), cv2.LINE_AA)
    for first, second in ((57, 58), (57, 59), (59, 0), (58, 60), (60, 21)):
        line_if_present(canvas, frame, first, second, (112, 124, 146), center, extent, 12)
    for edge in FACE_EDGES:
        line_if_present(canvas, frame, edge[0], edge[1], (218, 221, 231), center, extent, 2)
    for edge in HAND_EDGES:
        line_if_present(canvas, frame, edge[0], edge[1], accent, center, extent, 4)
        line_if_present(canvas, frame, edge[0] + 21, edge[1] + 21, accent, center, extent, 4)
    for start, stop in ((0, 21), (21, 42)):
        for index in range(start, stop):
            if frame[index, 3] > 0:
                cv2.circle(canvas, point_xy(frame[index], center, extent, size), 3, accent, -1, cv2.LINE_AA)
    # Wrist trails expose timing and path differences between synthesized voices.
    for wrist in (0, 21):
        trail = []
        for index in range(max(0, frame_index - 5), frame_index + 1):
            if features[index, wrist, 3] > 0:
                trail.append(point_xy(features[index, wrist], center, extent, size))
        for first, second in zip(trail, trail[1:]):
            cv2.line(canvas, first, second, tuple(int(v * 0.55) for v in accent), 2, cv2.LINE_AA)
    return canvas


def put_text(image, text, origin, scale=0.62, color=(238, 241, 246), thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def current_segment(voice, frame_index):
    for row in voice["timeline"]:
        if int(row["start"]) <= frame_index < int(row["stop"]):
            if row["kind"] == "transition":
                return "learned coarticulation"
            return voice["index_to_label"][int(row["target"])]
    return "hold"


def draw_timeline(canvas, voice, x, y, width, frame_index):
    total = len(voice["phrase"])
    for row in voice["timeline"]:
        x0 = x + round(width * int(row["start"]) / total)
        x1 = x + round(width * int(row["stop"]) / total)
        color = (55, 170, 240) if row["kind"] == "gloss" else (60, 215, 250)
        cv2.rectangle(canvas, (x0, y), (max(x0 + 1, x1), y + 10), color, -1)
    cursor = x + round(width * min(frame_index, total - 1) / max(1, total - 1))
    cv2.line(canvas, (cursor, y - 5), (cursor, y + 16), (255, 255, 255), 2)


def render_frame(voices, logical_frame, bounds, size=(1920, 900)):
    width, height = size
    canvas = np.full((height, width, 3), 8, np.uint8)
    phrase_text = " ".join(voices[0]["glosses"])
    put_text(canvas, "ONE GENERATED PHRASE, THREE NOVEL SIGNING VOICES", (36, 44), 0.90, (248, 248, 250), 2)
    put_text(canvas, f"Requested gloss sequence: {phrase_text}", (36, 78), 0.64, (170, 184, 205), 1)
    panel_width, panel_height = 616, 650
    center, extent = bounds
    for index, (voice, accent) in enumerate(zip(voices, VOICE_COLORS)):
        x = 16 + index * 634
        frame_index = min(logical_frame, len(voice["phrase"]) - 1)
        panel = avatar_panel(voice["phrase"], frame_index, (panel_width, panel_height), center, extent, accent)
        cv2.rectangle(panel, (1, 1), (panel_width - 2, panel_height - 2), accent, 3)
        canvas[112:112 + panel_height, x:x + panel_width] = panel
        put_text(canvas, f"Voice {voice['name']}", (x + 14, 105), 0.72, accent, 2)
        put_text(canvas, current_segment(voice, frame_index), (x + 14, 795), 0.62, accent, 2)
        put_text(canvas, f"{len(voice['phrase'])} generated frames", (x + 14, 824), 0.49, (155, 170, 193), 1)
        draw_timeline(canvas, voice, x + 14, 844, panel_width - 28, frame_index)
    put_text(canvas, "Blue = generated gloss motion | Yellow = learned transition | no human trajectory is being replayed", (34, 886), 0.53, (178, 190, 208), 1)
    return canvas


def slate(writer, lines, frames, size=(1920, 900)):
    canvas = np.full((size[1], size[0], 3), 8, np.uint8)
    y = 320
    for index, line in enumerate(lines):
        scale = 1.05 if index == 0 else 0.67
        color = (68, 194, 255) if index == 0 else (220, 226, 236)
        measured = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        put_text(canvas, line, ((size[0] - measured[0]) // 2, y), scale, color, 2)
        y += 68 if index == 0 else 48
    for _ in range(frames):
        writer.write(canvas)


def run(args):
    device = torch.device("cpu")
    checkpoint = torch.load(
        args.signing_voice_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != "slt_signing_voice_profile_v17":
        raise ValueError("the final renderer requires a signing-voice profile checkpoint")
    mean, timing = load_transition_voice(args.mean_checkpoint, args.timing_checkpoint, device)
    content_model, content_labels = load_content_model(args.content_checkpoint, device)
    if content_labels != {str(key): int(value) for key, value in checkpoint["label_to_index"].items()}:
        raise ValueError("content evaluator and signing voice use different class maps")
    index_to_label = {value: key for key, value in content_labels.items()}
    centroids = checkpoint["train_voice_style_latents"].float()
    recipes = build_profile_voice_recipes(checkpoint)
    voices = []
    raw_isolated = []
    for recipe in recipes:
        weights = torch.tensor(recipe.weights, dtype=centroids.dtype)
        weights /= weights.sum()
        raw_style = (
            centroids[list(recipe.source_voice_indices)] * weights[:, None]
        ).sum(dim=0)
        profile = decode_profile(
            raw_style.numpy(), checkpoint["latent_mean"].numpy(),
            checkpoint["latent_components"].numpy(),
        )
        style = F.normalize(torch.from_numpy(profile.vector()), dim=0)
        targets = [int(checkpoint["label_to_index"][gloss]) for gloss in args.glosses]
        isolated = []
        predictions = []
        selected_strengths = []
        for target in targets:
            prototype = checkpoint["content_prototypes"][target].numpy().astype(np.float32)
            candidates = [
                apply_voice_profile(
                    prototype, profile, profile_strength=float(strength),
                    curve_strength=float(checkpoint["curve_strength"]) * float(strength),
                )
                for strength in checkpoint["content_gate_strengths"]
            ]
            with torch.inference_mode():
                candidate_predictions = content_model(
                    torch.from_numpy(np.stack(candidates))
                ).argmax(dim=1).tolist()
            choice = len(candidates) - 1
            for index, prediction in enumerate(candidate_predictions):
                if prediction == target:
                    choice = index
                    break
            isolated.append(candidates[choice])
            predictions.append(candidate_predictions[choice])
            selected_strengths.append(float(checkpoint["content_gate_strengths"][choice]))
        ratio = voice_duration_ratio(checkpoint, recipe)
        phrase, timeline = compose_phrase(
            isolated, targets, ratio, checkpoint["class_median_observed_frames"],
            mean, timing, device,
        )
        voice = {
            "name": recipe.name,
            "recipe": recipe,
            "style": style,
            "raw_style": raw_style,
            "targets": targets,
            "predictions": predictions,
            "isolated": isolated,
            "phrase": phrase,
            "timeline": timeline,
            "duration_ratio": ratio,
            "selected_profile_strengths": selected_strengths,
            "glosses": args.glosses,
            "index_to_label": index_to_label,
        }
        voices.append(voice)
        raw_isolated.append(np.stack(isolated))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.video, args.preview, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    raw_reports = []
    for voice in voices:
        path = args.output_dir / f"voice_{voice['name'].lower()}.npz"
        metadata = {
            "name": voice["name"], "glosses": args.glosses,
            "targets": voice["targets"], "predictions": voice["predictions"],
            "timeline": voice["timeline"], "duration_ratio": voice["duration_ratio"],
            "selected_profile_strengths": voice["selected_profile_strengths"],
        }
        np.savez_compressed(
            path, landmarks=voice["phrase"].astype(np.float16),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        raw_reports.append({"path": path.as_posix(), "sha256": sha256(path)})

    bounds = coordinate_bounds(voices)
    temporary = args.video.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 900))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the signing-voice video")
    slate(writer, [
        "V17 AI SIGNING VOICES",
        "Content prototypes + novel style latents + generated whole signs + learned transitions",
        "Abstract articulated avatars; this is not photorealistic RGB generation",
    ], args.fps * 2)
    repeats = max(1, args.fps // args.logical_fps)
    for frame in range(max(len(voice["phrase"]) for voice in voices)):
        image = render_frame(voices, frame, bounds)
        for _ in range(repeats):
            writer.write(image)
    for _ in range(args.fps):
        writer.write(render_frame(voices, 10_000, bounds))
    writer.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.video),
    ], check=True)
    temporary.unlink()
    preview_frame = max(len(voice["phrase"]) for voice in voices) // 2
    cv2.imwrite(str(args.preview), render_frame(voices, preview_frame, bounds))

    styles = torch.stack([voice["style"] for voice in voices])
    style_similarity = (styles @ styles.T).numpy()
    isolated_values = np.stack(raw_isolated)
    pairwise_motion = {}
    for first in range(len(voices)):
        for second in range(first + 1, len(voices)):
            pairwise_motion[f"{voices[first]['name']}_{voices[second]['name']}"] = float(
                np.mean(np.abs(isolated_values[first, ..., :3] - isolated_values[second, ..., :3]))
            )
    train_similarity = styles @ F.normalize(
        checkpoint["train_voice_profiles"].float(), dim=1
    ).T
    report_voices = []
    for index, voice in enumerate(voices):
        recipe = voice["recipe"]
        report_voices.append({
            "name": voice["name"],
            "source_voice_ids": [checkpoint["train_voices"][value] for value in recipe.source_voice_indices],
            "weights": list(recipe.weights),
            "maximum_cosine_to_any_training_voice": float(train_similarity[index].max()),
            "duration_ratio": voice["duration_ratio"],
            "selected_profile_strengths": voice["selected_profile_strengths"],
            "phrase_frames": len(voice["phrase"]),
            "targets": [index_to_label[value] for value in voice["targets"]],
            "stage1_predictions": [index_to_label[value] for value in voice["predictions"]],
            "all_stage1_predictions_correct": voice["targets"] == voice["predictions"],
            "transition_spans": [
                int(row["stop"]) - int(row["start"])
                for row in voice["timeline"] if row["kind"] == "transition"
            ],
        })
    report = {
        "format": "slt_signing_voice_phrase_demo_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_glosses": args.glosses,
        "video": args.video.as_posix(), "video_sha256": sha256(args.video),
        "preview": args.preview.as_posix(), "preview_sha256": sha256(args.preview),
        "raw_voices": raw_reports,
        "signing_voice_checkpoint": args.signing_voice_checkpoint.as_posix(),
        "signing_voice_checkpoint_sha256": sha256(args.signing_voice_checkpoint),
        "mean_checkpoint_sha256": sha256(args.mean_checkpoint),
        "timing_checkpoint_sha256": sha256(args.timing_checkpoint),
        "content_checkpoint_sha256": sha256(args.content_checkpoint),
        "voice_style_cosine_matrix": style_similarity.tolist(),
        "pairwise_mean_absolute_xyz_difference": pairwise_motion,
        "voices": report_voices,
        "claim_boundary": (
            "This is a first content-conditioned, novel-latent, complete landmark signing-voice system. "
            "It is an abstract avatar and has not been rated as linguistically natural by fluent Deaf signers."
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signing-voice-checkpoint", type=Path, required=True)
    parser.add_argument("--mean-checkpoint", type=Path, default=Path("artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/model.pth"))
    parser.add_argument("--timing-checkpoint", type=Path, default=Path("artifacts/models/transition_span_multicorpus_v17_allvoices_final/model.pth"))
    parser.add_argument("--content-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--glosses", nargs="+", default=["HELLO", "HOW", "YOU"])
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/signing_voice_phrase_v17"))
    parser.add_argument("--video", type=Path, default=Path("artifacts/reports/signing_voice_phrase_v17.mp4"))
    parser.add_argument("--preview", type=Path, default=Path("artifacts/reports/signing_voice_phrase_v17_preview.png"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/signing_voice_phrase_v17.json"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--logical-fps", type=int, default=15)
    return parser


def main():
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
