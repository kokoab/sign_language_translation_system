#!/usr/bin/env python3
"""Render a labeled RGB/landmark comparison of final v17 transition synthesis."""

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
    interpolate_masked_context,
    sha256,
)


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
FACE_EDGES = (
    (42, 44), (44, 45), (45, 43),
    (42, 46), (46, 47), (47, 43),
    (42, 56), (43, 56), (56, 48),
    (49, 51), (51, 50), (50, 52), (52, 49),
    (53, 54), (54, 55), (48, 49), (48, 50),
)
BODY_EDGES = ((57, 58), (57, 59), (58, 60))
COLORS = {
    "left": (255, 210, 50),
    "right": (230, 80, 235),
    "face": (50, 205, 255),
    "body": (235, 235, 235),
}


def load_checkpoint_models(args: argparse.Namespace):
    device = torch.device("cpu")
    mean = load_mean_model(args.mean_checkpoint, device)
    diffusion_row = torch.load(
        args.diffusion_checkpoint, map_location="cpu", weights_only=False
    )
    if diffusion_row.get("mean_checkpoint_sha256") != sha256(args.mean_checkpoint):
        raise ValueError("diffusion checkpoint does not pin the supplied mean")
    diffusion = TransitionResidualDiffusionV17(
        TransitionResidualDiffusionV17Config(**diffusion_row["model_config"])
    )
    diffusion.load_state_dict(diffusion_row["model_state_dict"])
    diffusion.eval()
    timing_row = torch.load(
        args.timing_checkpoint, map_location="cpu", weights_only=False
    )
    timing = TransitionSpanPredictorV17(
        TransitionSpanV17Config(**timing_row["model_config"])
    )
    timing.load_state_dict(timing_row["model_state_dict"])
    timing.eval()
    return mean, diffusion, diffusion_row["residual_scale"].float(), timing


def motion_score(features: np.ndarray) -> float:
    hand = features[:, :42]
    present = (hand[1:, :, 3] > 0) & (hand[:-1, :, 3] > 0)
    speed = np.linalg.norm(hand[1:, :, :2] - hand[:-1, :, :2], axis=-1)
    if not present.any():
        return 0.0
    both = min(
        float((features[:, :21, 3] > 0).mean()),
        float((features[:, 21:42, 3] > 0).mean()),
    )
    return float(speed[present].mean()) * (0.5 + both)


def timing_context(features: torch.Tensor, span: int) -> torch.Tensor:
    start = (32 - span) // 2
    stop = start + span
    return torch.cat((features[start - 8:start], features[stop:stop + 8]), dim=0)


def candidate_rows(root: Path, timing: TransitionSpanPredictorV17):
    rough = []
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            ranges = payload["window_source_ranges"]
            for index, (features, valid) in enumerate(
                zip(payload["landmarks"], payload["window_valid"])
            ):
                if not valid or int(ranges[index, 1] - ranges[index, 0]) != 32:
                    continue
                rough.append((motion_score(features.astype(np.float32)), path, index, metadata))
    rough.sort(reverse=True, key=lambda value: value[0])
    ranked = []
    for base_score, path, index, metadata in rough[:80]:
        with np.load(path, allow_pickle=False) as payload:
            features = torch.from_numpy(payload["landmarks"][index].astype(np.float32))
            source_range = payload["window_source_ranges"][index].astype(int).tolist()
        contexts = torch.stack([timing_context(features, span) for span in range(4, 13)])
        with torch.inference_mode():
            predictions = timing(contexts).argmax(dim=1).cpu().numpy() + 4
        exact = [span for span, prediction in zip(range(4, 13), predictions) if span == prediction]
        span = min(exact, key=lambda value: abs(value - 8)) if exact else 8
        prediction = int(predictions[span - 4])
        center = (32 - span) // 2
        local = motion_score(features[center:center + span].numpy())
        ranked.append({
            "score": base_score + local,
            "path": path,
            "window": index,
            "metadata": metadata,
            "features": features,
            "source_range": source_range,
            "span": span,
            "predicted_span": prediction,
        })
    return sorted(ranked, reverse=True, key=lambda row: row["score"])


def choose_examples(timing: TransitionSpanPredictorV17, args: argparse.Namespace):
    specifications = (
        (args.how2sign_root / "how2sign_3", "How2Sign signer 3"),
        (args.how2sign_root / "how2sign_5", "How2Sign signer 5"),
        (args.web_root, "Public channel voice proxy"),
    )
    rows = []
    used_videos = set()
    for root, label in specifications:
        for row in candidate_rows(root, timing):
            video = row["metadata"]["video_metadata"]["video_path"]
            if video not in used_videos:
                row["label"] = label
                used_videos.add(video)
                rows.append(row)
                break
        else:
            raise RuntimeError(f"no renderable transition candidate under {root}")
    return rows


def synthesize(row, mean, diffusion, scale, seed: int):
    features = row["features"].unsqueeze(0)
    span = row["span"]
    start = (32 - span) // 2
    mask = torch.zeros((1, 32), dtype=torch.bool)
    mask[:, start:start + span] = True
    with torch.inference_mode():
        learned = mean(features, mask)
        linear = interpolate_masked_context(features, mask)
    outputs = {
        "Genuine landmarks": features.squeeze(0).numpy(),
        "Linear interpolation": linear.squeeze(0).numpy(),
        "Learned deterministic": learned.squeeze(0).numpy(),
    }
    for offset, temperature in enumerate((0.10, 0.20)):
        with torch.inference_mode():
            normalized = diffusion.sample_normalized_residual(
                learned,
                mask,
                temperature=temperature,
                generator=torch.Generator().manual_seed(seed + offset),
                sampling_steps=10,
            )
            sample = learned.clone()
            spatial = learned[..., :3] + normalized * scale[None, None]
            sample[..., :3] = torch.where(
                mask[:, :, None, None], spatial, learned[..., :3]
            )
        outputs[f"Stochastic temp {temperature:.2f}"] = sample.squeeze(0).numpy()
    row["mask_start"] = start
    row["mask_stop"] = start + span
    row["outputs"] = outputs
    return row


def read_rgb_frames(video_path: Path, source_range: list[int], size: tuple[int, int]):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    start, stop = source_range
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(stop - start):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(fit_image(frame, size))
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    while len(frames) < 32:
        frames.append(frames[-1].copy())
    positions = np.linspace(0, len(frames) - 1, 32).round().astype(int)
    return [frames[position] for position in positions]


def fit_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), 12, np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def coordinate_bounds(outputs: dict[str, np.ndarray]):
    values = []
    for features in outputs.values():
        present = features[..., 3] > 0
        values.append(features[..., :2][present])
    points = np.concatenate(values)
    low = np.percentile(points, 1, axis=0)
    high = np.percentile(points, 99, axis=0)
    center = (low + high) / 2
    extent = max(float((high - low).max()) * 0.65, 1.0)
    return center, extent


def point_xy(point: np.ndarray, center: np.ndarray, extent: float, size):
    width, height = size
    scale = min(width, height) * 0.78 / (2 * extent)
    return (
        int(width / 2 + (point[0] - center[0]) * scale),
        int(height / 2 + (point[1] - center[1]) * scale),
    )


def draw_edge(canvas, frame, edge, color, center, extent, width=2):
    first, second = edge
    if frame[first, 3] <= 0 or frame[second, 3] <= 0:
        return
    cv2.line(
        canvas,
        point_xy(frame[first], center, extent, (canvas.shape[1], canvas.shape[0])),
        point_xy(frame[second], center, extent, (canvas.shape[1], canvas.shape[0])),
        color,
        width,
        cv2.LINE_AA,
    )


def skeleton_panel(features, frame_index, size, center, extent):
    width, height = size
    canvas = np.full((height, width, 3), 12, np.uint8)
    frame = features[frame_index]
    for edge in HAND_EDGES:
        draw_edge(canvas, frame, edge, COLORS["left"], center, extent, 3)
        draw_edge(canvas, frame, (edge[0] + 21, edge[1] + 21), COLORS["right"], center, extent, 3)
    for edge in FACE_EDGES:
        draw_edge(canvas, frame, edge, COLORS["face"], center, extent, 2)
    for edge in BODY_EDGES:
        draw_edge(canvas, frame, edge, COLORS["body"], center, extent, 3)
    for start, stop, color in (
        (0, 21, COLORS["left"]), (21, 42, COLORS["right"]),
        (42, 57, COLORS["face"]), (57, 61, COLORS["body"]),
    ):
        for node in range(start, stop):
            if frame[node, 3] > 0:
                cv2.circle(
                    canvas,
                    point_xy(frame[node], center, extent, size),
                    3 if node < 42 else 2,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
    # A short wrist trail makes differences in timing and curvature easier to see.
    for wrist, color in ((0, COLORS["left"]), (21, COLORS["right"])):
        trail = []
        for index in range(max(0, frame_index - 5), frame_index + 1):
            if features[index, wrist, 3] > 0:
                trail.append(point_xy(features[index, wrist], center, extent, size))
        for first, second in zip(trail, trail[1:]):
            cv2.line(canvas, first, second, tuple(int(v * 0.55) for v in color), 2, cv2.LINE_AA)
    return canvas


def put_text(image, text, origin, scale=0.65, color=(245, 245, 245), thickness=1):
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
        thickness, cv2.LINE_AA,
    )


def labeled_panel(panel, label, synthetic, active):
    color = (60, 210, 255) if synthetic else (80, 220, 100)
    border = (40, 70, 90)
    if active:
        border = color
    cv2.rectangle(panel, (1, 1), (panel.shape[1] - 2, panel.shape[0] - 2), border, 4)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (8, 8, 8), -1)
    put_text(panel, label, (16, 28), 0.68, color, 2)
    if active:
        tag = "SYNTHETIC INTERVAL" if synthetic else "HUMAN REFERENCE INTERVAL"
        width = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
        put_text(panel, tag, (panel.shape[1] - width - 14, 27), 0.46, color, 1)
    return panel


def chapter_frame(row, logical_frame, frame_size=(1920, 1080)):
    width, height = frame_size
    canvas = np.full((height, width, 3), 8, np.uint8)
    panel_width, panel_height = 620, 420
    panel_size = (panel_width, panel_height)
    x_positions = (15, 650, 1285)
    y_positions = (105, 540)
    active = row["mask_start"] <= logical_frame < row["mask_stop"]
    source = Path(row["metadata"]["video_metadata"]["video_path"])
    header = (
        f"Example {row['chapter']}  |  {row['label']}  |  "
        f"hidden human span {row['span']} frames  |  timing model predicted {row['predicted_span']}"
    )
    put_text(canvas, header, (28, 45), 0.72, (245, 245, 245), 2)
    put_text(
        canvas,
        f"source: {source.name}  |  frame {logical_frame + 1}/32",
        (28, 76), 0.55, (165, 175, 190), 1,
    )

    panels = []
    panels.append(labeled_panel(
        row["rgb_frames"][logical_frame].copy(), "Original human RGB", False, active
    ))
    center, extent = row["bounds"]
    for label in (
        "Genuine landmarks", "Linear interpolation", "Learned deterministic",
        "Stochastic temp 0.10", "Stochastic temp 0.20",
    ):
        rendered = skeleton_panel(
            row["outputs"][label], logical_frame, panel_size, center, extent
        )
        panels.append(labeled_panel(
            rendered,
            label,
            label not in ("Genuine landmarks", "Original human RGB"),
            active,
        ))
    for panel, x, y in zip(
        panels,
        (x_positions[0], x_positions[1], x_positions[2], x_positions[0], x_positions[1], x_positions[2]),
        (y_positions[0], y_positions[0], y_positions[0], y_positions[1], y_positions[1], y_positions[1]),
    ):
        canvas[y:y + panel_height, x:x + panel_width] = panel

    progress_x, progress_y, progress_w = 30, 1010, 1860
    cv2.rectangle(canvas, (progress_x, progress_y), (progress_x + progress_w, progress_y + 10), (45, 45, 45), -1)
    mask_x0 = progress_x + round(progress_w * row["mask_start"] / 32)
    mask_x1 = progress_x + round(progress_w * row["mask_stop"] / 32)
    cv2.rectangle(canvas, (mask_x0, progress_y), (mask_x1, progress_y + 10), (60, 210, 255), -1)
    cursor = progress_x + round(progress_w * logical_frame / 31)
    cv2.line(canvas, (cursor, progress_y - 5), (cursor, progress_y + 16), (255, 255, 255), 2)
    put_text(
        canvas,
        "Yellow timeline section = frames replaced by each synthetic method. All other landmark values are identical.",
        (30, 1055), 0.55, (190, 200, 215), 1,
    )
    return canvas


def slate(lines: list[str], frames: int, writer, size=(1920, 1080)):
    width, height = size
    canvas = np.full((height, width, 3), 8, np.uint8)
    y = 350
    for index, line in enumerate(lines):
        scale = 1.05 if index == 0 else 0.68
        color = (60, 210, 255) if index == 0 else (220, 225, 235)
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        put_text(canvas, line, ((width - text_size[0]) // 2, y), scale, color, 2)
        y += 74 if index == 0 else 50
    for _ in range(frames):
        writer.write(canvas)


def run(args: argparse.Namespace):
    mean, diffusion, scale, timing = load_checkpoint_models(args)
    rows = choose_examples(timing, args)
    for chapter, row in enumerate(rows, start=1):
        row["chapter"] = chapter
        synthesize(row, mean, diffusion, scale, args.sample_seed + chapter * 100)
        row["rgb_frames"] = read_rgb_frames(
            Path(row["metadata"]["video_metadata"]["video_path"]),
            row["source_range"],
            (620, 420),
        )
        row["bounds"] = coordinate_bounds(row["outputs"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".mp4v.mp4")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the demonstration video")
    slate([
        "V17 MULTI-VOICE TRANSITION SYNTHESIS",
        "Original RGB + genuine landmarks + three synthetic methods",
        "Only the marked middle interval is synthesized; this is landmark output, not RGB generation.",
    ], args.fps * 2, writer)
    repeats = max(1, args.fps // args.logical_fps)
    for row in rows:
        slate([
            f"EXAMPLE {row['chapter']}: {row['label']}",
            f"Human interval: {row['span']} frames | Timing prediction: {row['predicted_span']} frames",
        ], args.fps, writer)
        for frame_index in range(32):
            frame = chapter_frame(row, frame_index)
            for _ in range(repeats):
                writer.write(frame)
        for _ in range(args.fps // 2):
            writer.write(chapter_frame(row, 31))
    writer.release()
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ], check=True)
    temporary.unlink()

    preview = chapter_frame(rows[0], rows[0]["mask_start"] + rows[0]["span"] // 2)
    cv2.imwrite(str(args.preview), preview)
    report_rows = []
    for row in rows:
        report_rows.append({
            "chapter": row["chapter"],
            "label": row["label"],
            "archive": row["path"].as_posix(),
            "window": row["window"],
            "source_video": row["metadata"]["video_metadata"]["video_path"],
            "source_video_sha256": row["metadata"]["video_sha256"],
            "human_span": row["span"],
            "predicted_span": row["predicted_span"],
            "mask_range": [row["mask_start"], row["mask_stop"]],
        })
    report = {
        "format": "transition_multivoice_visual_demo_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": args.output.as_posix(),
        "video_sha256": sha256(args.output),
        "preview": args.preview.as_posix(),
        "preview_sha256": sha256(args.preview),
        "mean_checkpoint_sha256": sha256(args.mean_checkpoint),
        "diffusion_checkpoint_sha256": sha256(args.diffusion_checkpoint),
        "timing_checkpoint_sha256": sha256(args.timing_checkpoint),
        "fps": args.fps,
        "logical_fps": args.logical_fps,
        "examples": report_rows,
        "claim_boundary": (
            "This is an abstract landmark rendering. It is not synthesized RGB, "
            "semantic prosody evidence, or a human-naturalness rating."
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mean-checkpoint", type=Path,
        default=Path("artifacts/models/transition_inpainter_multicorpus_v17_allvoices_final/model.pth"),
    )
    parser.add_argument(
        "--diffusion-checkpoint", type=Path,
        default=Path("artifacts/models/transition_residual_diffusion_multicorpus_v17_allvoices_final/model.pth"),
    )
    parser.add_argument(
        "--timing-checkpoint", type=Path,
        default=Path("artifacts/models/transition_span_multicorpus_v17_allvoices_final/model.pth"),
    )
    parser.add_argument(
        "--how2sign-root", type=Path,
        default=Path("data/local/how2sign_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--web-root", type=Path,
        default=Path("data/local/youtube_asl_transition_landmarks_v17"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/reports/transition_multivoice_visual_demo_v17.mp4"),
    )
    parser.add_argument(
        "--preview", type=Path,
        default=Path("artifacts/reports/transition_multivoice_visual_demo_v17_preview.png"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/transition_multivoice_visual_demo_v17.json"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--logical-fps", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=17701)
    return parser


def main():
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
