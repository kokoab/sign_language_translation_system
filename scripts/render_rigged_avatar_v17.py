#!/usr/bin/env python3
"""Render a fixed-anatomy triangulated avatar from a grounded v17 phrase NPZ."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

from active.v17.avatar_rig_v17 import (
    HAND_EDGES, bone_length_metrics, estimate_rest_reference, retarget_avatar,
)


SKIN = np.asarray((117, 142, 181), dtype=np.float32)  # OpenCV BGR
CLOTH = np.asarray((103, 75, 42), dtype=np.float32)
HAIR = np.asarray((31, 25, 23), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(points: np.ndarray, width: int, height: int) -> np.ndarray:
    focal = 1750.0
    depth = 3.2 - points[..., 2]
    return np.stack((width / 2 + focal * points[..., 0] / depth,
                     height * .4 - focal * (points[..., 1] - 1.2) / depth), axis=-1)


def _load_makehuman(folder: Path):
    vertices, faces, group = [], [], ""
    for line in (folder / "base.obj").read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split(maxsplit=1)[1]
        elif line.startswith("f ") and group == "body":
            indices = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
            faces.extend((indices[0], indices[index], indices[index + 1]) for index in range(1, len(indices) - 1))
    original = np.asarray(vertices, dtype=np.float32)
    world = original.copy()
    world[:, 0] *= .1
    world[:, 1] = world[:, 1] * .1 + .76042
    world[:, 2] = (world[:, 2] - .14605) * .1
    skeleton = json.loads((folder / "default.mhskel").read_text())
    weights = json.loads((folder / "default_weights.mhw").read_text())["weights"]

    def joint(name):
        return world[np.asarray(skeleton["joints"][name], dtype=int)].mean(axis=0)

    bones = {}
    for name, row in skeleton["bones"].items():
        bones[name] = (joint(row["head"]), joint(row["tail"]))
    influence = {}
    for name, rows in weights.items():
        kept = [(int(index), float(weight)) for index, weight in rows if int(index) < len(world)]
        if kept:
            influence[name] = (np.asarray([row[0] for row in kept]), np.asarray([row[1] for row in kept], dtype=np.float32))
    body_faces = np.asarray(faces, dtype=np.int32)
    return {"vertices": world, "faces": body_faces, "bones": bones, "weights": influence,
            "mesh_sha256": sha256(folder / "base.obj"), "weights_sha256": sha256(folder / "default_weights.mhw")}


def _map_segment(points, old_head, old_tail, new_head, new_tail, *, old_normal=None, new_normal=None):
    old_axis = old_tail - old_head; old_length = float(np.linalg.norm(old_axis)); old_axis /= old_length
    new_axis = new_tail - new_head; new_length = float(np.linalg.norm(new_axis)); new_axis /= new_length
    cross = np.cross(old_axis, new_axis); sine = float(np.linalg.norm(cross)); cosine = float(old_axis @ new_axis)
    if sine < 1e-7:
        if cosine > 0:
            rotation = np.eye(3, dtype=np.float32)
        else:
            # A 180-degree rotation keeps orientation; -I is a reflection.
            axis = np.cross(old_axis, np.eye(3)[int(np.argmin(np.abs(old_axis)))])
            axis /= np.linalg.norm(axis)
            rotation = 2 * np.outer(axis, axis) - np.eye(3)
    else:
        cross /= sine
        skew = np.asarray(((0, -cross[2], cross[1]), (cross[2], 0, -cross[0]), (-cross[1], cross[0], 0)))
        rotation = np.eye(3) + skew * sine + (skew @ skew) * (1 - cosine)
    if old_normal is not None and new_normal is not None:
        def frame(axis, normal):
            perpendicular = normal - np.dot(normal, axis) * axis
            length = np.linalg.norm(perpendicular)
            if length < 1e-6:
                return None
            perpendicular /= length
            return np.stack((axis, perpendicular, np.cross(axis, perpendicular)), axis=1)
        before, after = frame(old_axis, old_normal), frame(new_axis, new_normal)
        if before is not None and after is not None:
            rotation = after @ before.T
    relative = points - old_head
    axial = (relative @ old_axis)[:, None] * old_axis
    adjusted = relative - axial + axial * (new_length / old_length)
    return new_head + adjusted @ rotation.T


def makehuman_targets(asset, rig, frame):
    targets = {}
    for suffix, side in (("L", 0), ("R", 1)):
        shoulder, elbow, hand = rig.shoulders[frame, side], rig.elbows[frame, side], rig.hands[frame, side]
        wrist = hand[0]; upper_mid = (shoulder + elbow) / 2; lower_mid = (elbow + wrist) / 2
        targets.update({
            f"upperarm01.{suffix}": (shoulder, upper_mid), f"upperarm02.{suffix}": (upper_mid, elbow),
            f"lowerarm01.{suffix}": (elbow, lower_mid), f"lowerarm02.{suffix}": (lower_mid, wrist),
        })
        old_wrist = asset["bones"][f"wrist.{suffix}"][0]
        old_middle = asset["bones"][f"finger3-1.{suffix}"][0]
        old_palm_length = np.linalg.norm(old_middle - old_wrist)
        def palm_head(position):
            fraction = np.linalg.norm(position - old_wrist) / max(old_palm_length, 1e-6)
            return wrist + fraction * (hand[9] - wrist)
        targets[f"wrist.{suffix}"] = (wrist, palm_head(asset["bones"][f"wrist.{suffix}"][1]))
        for index, node in enumerate((5, 9, 13, 17), 1):
            name = f"metacarpal{index}.{suffix}"
            targets[name] = (palm_head(asset["bones"][name][0]), hand[node])
        chains = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                  (13, 14, 15, 16), (17, 18, 19, 20))
        for finger, chain in enumerate(chains, 1):
            for bone, (start, stop) in enumerate(zip(chain[:-1], chain[1:]), 1):
                targets[f"finger{finger}-{bone}.{suffix}"] = (hand[start], hand[stop])
    return targets


def pose_makehuman(asset, rig, frame):
    vertices = asset["vertices"]
    output = vertices.copy()
    delta = np.zeros_like(vertices)
    targets = makehuman_targets(asset, rig, frame)
    palm_normals = {}
    finger_hinges = {}
    for suffix, side in (("L", 0), ("R", 1)):
        old_wrist = asset["bones"][f"wrist.{suffix}"][0]
        old_index = asset["bones"][f"finger2-1.{suffix}"][0]
        old_pinky = asset["bones"][f"finger5-1.{suffix}"][0]
        hand = rig.hands[frame, side]
        palm_normals[suffix] = (np.cross(old_index - old_wrist, old_pinky - old_wrist),
                                np.cross(hand[5] - hand[0], hand[17] - hand[0]))
        finger_hinges[suffix] = (old_index - old_pinky, hand[5] - hand[17])
    for name, (new_head, new_tail) in targets.items():
        if name not in asset["weights"] or name not in asset["bones"]:
            continue
        indices, weights = asset["weights"][name]
        normals = {}
        if name.startswith(("wrist.", "metacarpal", "finger")):
            # The transverse hinge stays stable through finger flexion. Projecting
            # the palm normal onto curled phalanges flips roll beyond 90 degrees.
            basis = finger_hinges if name.startswith(("finger2", "finger3", "finger4", "finger5")) else palm_normals
            normals = dict(zip(("old_normal", "new_normal"), basis[name[-1]]))
        transformed = _map_segment(vertices[indices], *asset["bones"][name], new_head, new_tail, **normals)
        delta[indices] += (transformed - vertices[indices]) * weights[:, None]
    return output + delta


def draw_makehuman(canvas, vertices, faces, bind_vertices):
    screen = project(vertices, canvas.shape[1], canvas.shape[0])
    light = np.asarray((-.25, .45, .86), dtype=np.float32); light /= np.linalg.norm(light)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-6)
    # Clothing follows the bind mesh, not the hand's current screen position.
    centers = bind_vertices[faces].mean(axis=1)
    clothed = (centers[:, 1] > .38) & (centers[:, 1] < 1.30) & (np.abs(centers[:, 0]) < .30)
    bases = np.where(clothed[:, None], CLOTH, SKIN)
    shades = .48 + .52 * np.abs(normals @ light)
    colors = np.clip(bases * shades[:, None], 0, 255).astype(np.uint8)
    projected = np.rint(screen[faces]).astype(np.int32)
    for index in np.argsort(triangles[:, :, 2].mean(axis=1)):
        cv2.fillConvexPoly(canvas, projected[index], tuple(map(int, colors[index])), cv2.LINE_AA)


def render(rig, frame: int, label: str, makehuman=None, width=720, height=900):
    image = np.full((height, width, 3), (21, 24, 30), dtype=np.uint8)
    draw_makehuman(image, pose_makehuman(makehuman, rig, frame), makehuman["faces"], makehuman["vertices"])
    cv2.putText(image, label, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, .72, (238, 242, 247), 2, cv2.LINE_AA)
    state = f"left: {rig.hand_states[frame,0]}   right: {rig.hand_states[frame,1]}"
    cv2.putText(image, state, (24, height - 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (185, 199, 216), 1, cv2.LINE_AA)
    return image


def segment_label(metadata, frame):
    for row in metadata.get("timeline", []):
        if int(row["start"]) <= frame < int(row["stop"]):
            return row.get("gloss", "transition")
    return "hold"


def source_hand_scale_ratio(xyz, presence):
    totals = []
    for side, start in enumerate((0, 21)):
        values = []
        for frame in range(len(xyz)):
            if presence[frame, start:start + 21].sum() < 15: continue
            values.append(sum(np.linalg.norm(xyz[frame, start + b, :2] - xyz[frame, start + a, :2]) for a, b in HAND_EDGES))
        if values: totals.append(max(values) / max(min(values), 1e-8))
    return max(totals) if totals else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/rigged_avatar_v17_v2"))
    parser.add_argument("--makehuman-dir", type=Path, default=Path("artifacts/tools/makehuman_cc0"))
    parser.add_argument("--provenance", type=Path, default=Path("data/local/citizen100_v17/provenance.csv"))
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.fps <= 0: parser.error("--fps must be positive")
    with np.load(args.input, allow_pickle=False) as payload:
        xyz = payload["animation_rig_xyz"].astype(np.float32)
        presence = payload["animation_rig_presence"].astype(bool)
        observations = payload["observation_presence"].astype(bool)
        metadata = json.loads(str(payload["metadata_json"]))
    logical_fps = float(metadata.get("logical_fps", 15.))
    if not np.isfinite(logical_fps) or logical_fps <= 0:
        raise ValueError("positive finite animation frame rate required")
    signer = metadata.get("source_signer")
    archive_rows = []
    with args.provenance.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train" or row["participant"] != signer:
                continue
            archive = Path("data/local/citizen100_v17/landmarks/train") / row["canonical_label"] / f"{Path(row['video']).stem}.v17.npz"
            if archive.exists():
                with np.load(archive, allow_pickle=False) as payload:
                    archive_rows.append((str(archive), payload["features"].astype(np.float32)))
    rest_reference = estimate_rest_reference(archive_rows)
    rig = retarget_avatar(xyz, presence, observations, metadata, rest_reference=rest_reference)
    makehuman = _load_makehuman(args.makehuman_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem.split(".")[0]
    temporary = args.output_dir / f"{stem}.temporary.mp4"
    video = args.output_dir / f"{stem}.rigged_avatar.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (720, 900))
    if not writer.isOpened(): raise RuntimeError("OpenCV could not create video")
    rendered = []
    for frame in range(len(xyz)):
        image = render(rig, frame, segment_label(metadata, frame), makehuman)
        rendered.append(image)
        repeats = round((frame + 1) * args.fps / logical_fps) - round(frame * args.fps / logical_fps)
        for _ in range(repeats):
            writer.write(image)
    writer.release()
    subprocess.run(("ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)), check=True)
    temporary.unlink()
    indices = np.linspace(0, len(rendered) - 1, 6).round().astype(int)
    sheet = np.concatenate([cv2.resize(rendered[index], (360, 450)) for index in indices], axis=1)
    contact = args.output_dir / f"{stem}.contact_sheet.png"
    cv2.imwrite(str(contact), sheet)
    metrics = bone_length_metrics(rig)
    unique, counts = np.unique(rig.hand_states, return_counts=True)
    report = {
        "format": "slt_rigged_avatar_review_v17", "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input), "requested_glosses": metadata.get("requested_glosses", []),
        "video": str(video), "video_sha256": sha256(video),
        "contact_sheet": str(contact), "contact_sheet_sha256": sha256(contact),
        "triangle_mesh": True, "metric_fixed_anatomy": True,
        "human_mesh": "MakeHuman base mesh", "human_mesh_license": "CC0-1.0",
        "human_mesh_source": "https://github.com/makehumancommunity/makehuman/blob/master/makehuman/data/3dobjs/base.obj",
        "human_mesh_sha256": makehuman["mesh_sha256"], "rig_weights_sha256": makehuman["weights_sha256"],
        "source_signer": signer, "rest_profile_source_split": "Citizen official train only",
        "logical_fps": logical_fps, "output_fps": args.fps,
        "rest_profile_candidate_counts": list(rest_reference.candidate_counts),
        "rest_profile_source_archives": list(rest_reference.source_paths),
        "hand_state_counts": dict(zip(unique.tolist(), counts.tolist())),
        "source_max_2d_hand_tree_scale_ratio": source_hand_scale_ratio(xyz, presence),
        **metrics,
        "depth_contract": "v17 spatial channel 3 is a log-scale proxy; hands use an explicit avatar signing plane at .235 m plus a bounded .035 m cue, not recovered source depth",
        "license": "MakeHuman bundled base mesh, skeleton, and weights are CC0; project retargeter and renderer are original code",
        "review_status": "synthetic_native_review_only",
        "training_eligible": False, "test_eligible": False,
        "limitations": ["untextured MakeHuman mesh is more anatomical but not photorealistic", "facial grammar is not retargeted", "bounded depth cue is not 3D reconstruction", "ASL-fluent review pending"],
    }
    report_path = args.output_dir / f"{stem}.report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
