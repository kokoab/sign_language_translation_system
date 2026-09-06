#!/usr/bin/env python3
"""Compare exact train-source footage, rejected avatar, and source-world retargeting.

World hand geometry is a separate detector estimate used only for animation review.
The source footage contains isolated signs: generated joins have no paired ground
truth video and are explicitly shown without a source panel.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import retarget_avatar, bone_length_metrics, HAND_EDGES, interpolate_world_hand
from active.v17.extract_mediapipe_v17 import MediaPipeHybridDetector, DEFAULT_MODEL_PATH
from active.v17.schema_mediapipe_v17 import MediaPipeV17Config
from active.v17.extract_v17 import AppleVisionDetector, assign_hands
from scripts.render_rigged_avatar_v17 import _load_makehuman, render, sha256, segment_label


def video_frames(path):
    capture = cv2.VideoCapture(str(path)); frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"no video frames: {path}")
    return frames


def fit(frame, width, height):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    canvas = np.full((height, width, 3), 25, np.uint8)
    y, x = (height - len(resized)) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + len(resized), x:x + resized.shape[1]] = resized
    return canvas


def match_world_hands(hands, reference):
    """Attach world estimates to observed wrist tracks, not unstable chirality."""
    slots = [side for side in ("left", "right") if reference[side] is not None]
    result = {"left": None, "right": None}
    if not hands or not slots:
        return result
    pairs = min(len(slots), len(hands))
    choices = [(sum(np.linalg.norm(h.xy[0] - reference[s].xy[0]) for s, h in zip(ss, hh)), ss, hh)
               for ss in permutations(slots, pairs) for hh in permutations(hands, pairs)]
    _, sides, matched = min(choices, key=lambda item: item[0])
    for side, hand in zip(sides, matched):
        if np.linalg.norm(hand.xy[0] - reference[side].xy[0]) < .15:
            result[side] = hand
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-report", type=Path, default=Path("artifacts/reports/stage2_v17_grounded_text_input_demo_v2/report.json"))
    p.add_argument("--phrase", default="HELLO HOW YOU")
    p.add_argument("--previous-video", type=Path, default=Path("artifacts/reports/rigged_avatar_v17_v5/hello_how_you.rigged_avatar.mp4"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--handshape-lexicon", type=Path,
                   help="optional explicit ASL-LEX animation constraints; never treated as observations")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report = json.loads(a.source_report.read_text())
    row = next(r for r in report["items"] if r["phrase"] == a.phrase)
    with np.load(row["artifact"]) as archive:
        xyz = archive["animation_rig_xyz"].astype(np.float32)
        presence = archive["animation_rig_presence"].astype(bool)
        observed = archive["observation_presence"].astype(bool)
        observation_xyz = archive["observation_xyz"].astype(np.float32)
        metadata = json.loads(str(archive["metadata_json"]))
    detector = MediaPipeHybridDetector(DEFAULT_MODEL_PATH, MediaPipeV17Config(include_apple_auxiliary=False))
    apple = AppleVisionDetector()
    world = np.full((len(xyz), 2, 21, 3), np.nan, np.float32)
    world_imputed = np.zeros((len(xyz), 2), bool)
    repeats = np.full(len(xyz), 2, int)
    source_images, references, source_rows = {}, {}, []
    gloss_rows = [r for r in metadata["timeline"] if r["kind"] == "gloss"]
    for path, segment in zip(row["source_landmarks"], gloss_rows):
        if "/train/" not in path:
            raise ValueError("source audit is train-only")
        with np.load(path) as archive:
            source_meta = json.loads(str(archive["metadata_json"]))
            features = archive["features"].astype(np.float32)
        frames = video_frames(source_meta["video_path"])
        detector.reset_sequence(); previous = {"left": None, "right": None}
        estimate = np.full((len(frames), 2, 21, 3), np.nan, np.float32)
        for i, frame in enumerate(frames):
            detection = detector.detect(frame, False, False)
            reference = assign_hands(apple.detect(frame, False, False).hands, previous)
            assigned = match_world_hands(detection.hands, reference)
            for side, name in enumerate(("left", "right")):
                hand = assigned[name]
                if hand is not None:
                    estimate[i, side] = hand.world_xyz
                    previous[name] = hand.xy[0].copy()
        np.savez_compressed(a.output / f"{segment['gloss'].lower()}_source_world.npz", world_xyz=estimate)
        mapping = []
        for frame in range(segment["start"], segment["stop"]):
            valid = observed[frame, :42]
            distances = ((features[:, :42, :2] - observation_xyz[frame, :42, :2]) ** 2)[:, valid].mean((1, 2))
            index = int(distances.argmin())
            source_index = round(source_meta["hand_trim_start_frame"] + index / (len(features) - 1) *
                                 (source_meta["source_frames_processed"] - 1))
            source_images[frame] = frames[source_index]
            references[frame] = dict(gloss=segment["gloss"], video=source_meta["video_path"],
                                     source_frame=source_index, landmark_frame=index)
            world[frame] = estimate[source_index]
            mapping.append(source_index)
        start, stop = segment["start"], segment["stop"]
        # Restore this sign's actual source duration, including differing source
        # frame rates. The three panels share the resulting presentation clock.
        duration = (max(mapping) - min(mapping) + 1) / source_meta["fps"]
        boundaries = np.rint(np.linspace(0, duration * 30, stop - start + 1)).astype(int)
        repeats[start:stop] = np.diff(boundaries)
        for side in range(2):
            known = np.flatnonzero(np.isfinite(world[start:stop, side]).all((1, 2))) + start
            if not len(known):
                continue
            for index in range(start, stop):
                if index in known:
                    continue
                before, after = known[known < index], known[known > index]
                if len(before) and len(after):
                    first, last = before[-1], after[0]
                    world[index, side] = interpolate_world_hand(world[first, side], world[last, side],
                                                               (index - first) / (last - first))
                else:
                    world[index, side] = world[known[np.argmin(abs(known - index))], side]
                world_imputed[index, side] = True
        source_rows.append(dict(gloss=segment["gloss"], video=source_meta["video_path"],
            video_sha256=sha256(Path(source_meta["video_path"])), fps=source_meta["fps"], source_frames=mapping))
    detector.close()
    # Missing per-frame estimates above use this same sign's detected shape and
    # remain explicitly imputed, including the constant extension at its edges.
    for segment in metadata["timeline"]:
        if segment["kind"] != "transition":
            continue
        left, right = segment["start"] - 1, segment["stop"]
        if left < 0 or right >= len(world):
            continue
        for side in range(2):
            if not np.isfinite(world[[left, right], side]).all():
                continue
            for index in range(segment["start"], segment["stop"]):
                alpha = (index - left) / (right - left)
                world[index, side] = interpolate_world_hand(world[left, side], world[right, side], alpha)
                world_imputed[index, side] = True
    annotations = {}
    if a.handshape_lexicon:
        lexicon = json.loads(a.handshape_lexicon.read_text())
        attributes = {r["name"]: r for r in lexicon["attributes"]}
        for item in lexicon["classes"]:
            if item["canonical_label"] not in a.phrase.split():
                continue
            annotations[item["canonical_label"]] = {
                name: attributes[name]["values"][attributes[name]["targets_by_class_index"][item["class_index"]]]
                for name in ("selected_fingers", "flexion")}
    rig = retarget_avatar(xyz, presence, observed, metadata, hand_world_xyz=world, handshape_annotations=annotations)
    asset = _load_makehuman(Path("artifacts/tools/makehuman_cc0"))
    previous = video_frames(a.previous_video)
    writer = cv2.VideoWriter(str(a.output / "comparison_temporary.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30., (1600, 600))
    selected = {segment["start"] + (segment["stop"] - segment["start"]) // 2 for segment in gloss_rows}
    previews = []
    for i in range(len(xyz)):
        reference = source_images.get(i, np.full((480, 640, 3), 25, np.uint8))
        source = fit(reference, 640, 600)
        old = fit(previous[min(i * 2, len(previous) - 1)], 480, 600)
        new = fit(render(rig, i, segment_label(metadata, i), asset), 480, 600)
        for panel, title in ((source, "Actual source" if i in source_images else "Generated join: no source footage"),
                             (old, "Rejected avatar"), (new, "Source-world retargeting audit")):
            panel[:36] = 25
            cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 230, 255), 1, cv2.LINE_AA)
        canvas = np.concatenate((source, old, new), axis=1)
        for _ in range(repeats[i]):
            writer.write(canvas)
        if i in selected:
            destination = a.output / f"frame_{i:03d}.png"
            cv2.imwrite(str(destination), canvas); previews.append(str(destination))
    writer.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(a.output / "comparison_temporary.mp4"),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(a.output / "comparison.mp4")], check=True)
    (a.output / "comparison_temporary.mp4").unlink()
    np.savez_compressed(a.output / "animation_world_audit.npz", hand_world_xyz=world, world_imputed=world_imputed,
                        presentation_repeats=repeats, hands=rig.hands,
                        shoulders=rig.shoulders, elbows=rig.elbows)
    result = dict(format="source_avatar_comparison_v17", sources=source_rows, source_references=references,
        previews=previews, video=str(a.output / "comparison.mp4"),
        world_hand_frames=int(np.isfinite(world).all((2, 3)).sum()), **bone_length_metrics(rig),
        imputed_world_hand_frames=int(world_imputed.sum()), presentation_fps=30,
        lexical_handshape_constraints=annotations,
        handshape_lexicon_sha256=sha256(a.handshape_lexicon) if a.handshape_lexicon else None,
        duration_seconds=float(repeats.sum() / 30),
        training_eligible=False, test_eligible=False,
        limitations=["3D hand positions are detector estimates, not motion capture",
                     "unobserved hands use visibly unverified neutral fallback",
                     "bone-direction transition interpolation remains experimental",
                     "missing world estimates interpolated or extended from same sign, explicitly imputed",
                     "source gloss duration restored; generated transition duration remains unverified",
                     "isolated source clips do not supply true continuous joins"])
    (a.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"sources", "source_references"}}, indent=2))


if __name__ == "__main__":
    main()
