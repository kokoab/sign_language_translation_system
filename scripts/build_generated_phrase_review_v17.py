#!/usr/bin/env python3
"""Audit generated signing-voice videos and build a local native-review index."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_separated_landmark_artifact(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, object]]:
    """Load a review artifact that cannot masquerade as recognizer landmarks."""
    with np.load(path, allow_pickle=False) as payload:
        if "landmarks" in payload.files:
            raise ValueError("ambiguous legacy landmark artifact is unsafe for review")
        required = {
            "animation_rig_xyz", "animation_rig_presence",
            "animation_rig_confidence", "observation_xyz",
            "observation_presence", "observation_confidence", "metadata_json",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"separated landmark artifact is missing {sorted(missing)}")
        rig_xyz = payload["animation_rig_xyz"].astype(np.float32)
        rig_presence = payload["animation_rig_presence"].astype(bool)
        rig_confidence = payload["animation_rig_confidence"].astype(np.float32)
        observation_xyz = payload["observation_xyz"].astype(np.float32)
        observation_presence = payload["observation_presence"].astype(bool)
        observation_confidence = payload["observation_confidence"].astype(np.float32)
        metadata = json.loads(str(payload["metadata_json"]))
    if (
        rig_xyz.ndim != 3 or rig_xyz.shape[1:] != (61, 3)
        or observation_xyz.shape != rig_xyz.shape
        or rig_presence.shape != rig_xyz.shape[:2]
        or rig_confidence.shape != rig_xyz.shape[:2]
        or observation_presence.shape != rig_xyz.shape[:2]
        or observation_confidence.shape != rig_xyz.shape[:2]
    ):
        raise ValueError("unexpected separated landmark artifact shapes")
    if metadata.get("artifact_contract_version") != 3:
        raise ValueError("unexpected separated landmark artifact version")
    if any(metadata.get(key) is not False for key in (
        "training_eligible", "validation_eligible", "test_eligible"
    )):
        raise ValueError("synthetic artifact incorrectly claims dataset eligibility")
    rig = np.zeros(rig_xyz.shape[:2] + (5,), dtype=np.float32)
    rig[..., :3] = rig_xyz
    rig[..., 3] = rig_presence
    rig[..., 4] = rig_confidence
    observation = np.zeros(rig_xyz.shape[:2] + (5,), dtype=np.float32)
    observation[..., :3] = observation_xyz
    observation[..., 3] = observation_presence
    observation[..., 4] = observation_confidence
    if not np.isfinite(rig).all() or not np.isfinite(observation).all():
        raise ValueError("landmark artifact contains non-finite values")
    if np.any(observation_xyz[~observation_presence] != 0):
        raise ValueError("absent observations contain nonzero coordinates")
    if np.any(rig_xyz[~rig_presence] != 0):
        raise ValueError("absent rig nodes contain nonzero coordinates")
    observed_error = np.abs(
        rig_xyz[observation_presence] - observation_xyz[observation_presence]
    )
    diagnostics = {
        "rig_finite": bool(np.isfinite(rig_xyz).all()),
        "observation_finite": bool(np.isfinite(observation).all()),
        "observation_presence_fraction": float(observation_presence.mean()),
        "observation_is_fabricated_all_present": bool(observation_presence.all()),
        "rig_preserves_observed_xyz_max_error": (
            float(observed_error.max()) if len(observed_error) else 0.0
        ),
        "rig_hand_participation": [
            bool(rig_presence[:, :21].any()), bool(rig_presence[:, 21:42].any())
        ],
        "observation_hand_participation": [
            bool(observation_presence[:, :21].any()),
            bool(observation_presence[:, 21:42].any()),
        ],
    }
    diagnostics["hand_participation_preserved"] = (
        diagnostics["rig_hand_participation"]
        == diagnostics["observation_hand_participation"]
    )
    return rig, observation, metadata, diagnostics


def boundary_diagnostics(features: np.ndarray, timeline: list[dict]) -> dict[str, float | int]:
    transition_frames = np.zeros(len(features), dtype=bool)
    joins = []
    handless = presence_jumps = transient_nodes = 0
    for row in timeline:
        if row["kind"] != "transition":
            continue
        start, stop = int(row["start"]), int(row["stop"])
        if not 0 < start < stop < len(features):
            raise ValueError("transition is not bounded by two gloss frames")
        transition_frames[start:stop] = True
        joins.extend((start, stop))
        present = features[start:stop, :, 3] > 0
        handless += int(np.count_nonzero(~present[:, :42].any(axis=1)))
        presence_jumps += int(np.count_nonzero(
            (features[start - 1, :, 3] > 0) != (features[start, :, 3] > 0)
        ))
        presence_jumps += int(np.count_nonzero(
            (features[stop - 1, :, 3] > 0) != (features[stop, :, 3] > 0)
        ))
        absent_at_both_ends = (
            (features[start - 1, :, 3] <= 0) & (features[stop, :, 3] <= 0)
        )
        transient_nodes += int(np.count_nonzero(present[:, absent_at_both_ends].any(axis=0)))

    def motion_ratio(order: int) -> float:
        delta = np.linalg.norm(np.diff(features[:, :42, :3], n=order, axis=0), axis=-1)
        valid = np.ones(delta.shape, dtype=bool)
        hand_present = features[:, :42, 3] > 0
        touches_transition = np.zeros(len(delta), dtype=bool)
        for offset in range(order + 1):
            valid &= hand_present[offset:offset + len(delta)]
            touches_transition |= transition_frames[offset:offset + len(delta)]
        gloss = delta[valid & ~touches_transition[:, None]]
        generated = delta[valid & touches_transition[:, None]]
        if not len(gloss) or not len(generated):
            return 0.0
        reference = float(np.percentile(gloss, 95))
        return float(np.percentile(generated, 95) / reference) if reference > 0 else 0.0

    motion_ratios = {
        name: motion_ratio(order)
        for name, order in (("speed", 1), ("acceleration", 2), ("jerk", 3))
    }
    velocity = np.linalg.norm(np.diff(features[:, :42, :3], axis=0), axis=-1)
    velocity_present = (
        (features[1:, :42, 3] > 0) & (features[:-1, :42, 3] > 0)
    )
    gloss_steps = ~(transition_frames[1:] | transition_frames[:-1])
    reference = velocity[velocity_present & gloss_steps[:, None]]
    reference_p95 = float(np.percentile(reference, 95)) if len(reference) else 0.0
    join_values = []
    for join in joins:
        values = velocity[join - 1][velocity_present[join - 1]]
        join_values.extend(values.tolist())
    transition_steps = transition_frames[1:] | transition_frames[:-1]
    transition_values = velocity[velocity_present & transition_steps[:, None]]

    def ratio(values) -> float:
        if not len(values) or reference_p95 <= 0:
            return 0.0
        return float(np.percentile(values, 95) / reference_p95)

    present = features[..., 3] > 0
    return {
        "frames_with_incomplete_left_hand": int(np.count_nonzero(~present[:, :21].all(axis=1))),
        "frames_with_incomplete_right_hand": int(np.count_nonzero(~present[:, 21:42].all(axis=1))),
        "frames_with_incomplete_face": int(np.count_nonzero(~present[:, 42:57].all(axis=1))),
        "frames_with_incomplete_body": int(np.count_nonzero(~present[:, 57:61].all(axis=1))),
        "presence_changes_anywhere": int(np.count_nonzero(present[1:] != present[:-1])),
        "handless_transition_frames": handless,
        "presence_changes_at_transition_joins": presence_jumps,
        "nodes_present_only_inside_transition": transient_nodes,
        "join_velocity_p95_over_gloss_p95": ratio(join_values),
        "transition_velocity_p95_over_gloss_p95": motion_ratios["speed"],
        "transition_motion_p95_over_gloss_p95": motion_ratios,
    }


def video_metadata(path: Path) -> dict[str, int | float | str]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open {path}")
    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    ok, _ = capture.read()
    capture.release()
    if not ok or metadata["frames"] <= 0:
        raise ValueError(f"cannot decode {path}")
    return metadata


def run(args: argparse.Namespace) -> dict:
    plan = json.loads(args.plan.read_text())
    expected = {
        row["phrase_id"]: row["target_sequence"]
        for row in plan["rows"][:args.limit]
    }
    items = []
    review_rows = []
    for phrase_id, glosses in expected.items():
        folder = args.root / phrase_id
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        if report["requested_glosses"] != glosses:
            raise ValueError(f"{phrase_id} does not match the locked plan")
        video = Path(report["video"])
        preview = Path(report["preview"])
        if sha256(video) != report["video_sha256"] or sha256(preview) != report["preview_sha256"]:
            raise ValueError(f"{phrase_id} rendered hashes changed")
        voices = []
        for voice, raw in zip(report["voices"], report["raw_voices"]):
            raw_path = Path(raw["path"])
            if sha256(raw_path) != raw["sha256"]:
                raise ValueError(f"{phrase_id} raw landmark hash changed")
            rig, features, metadata, artifact_diagnostics = (
                load_separated_landmark_artifact(raw_path)
            )
            diagnostics = boundary_diagnostics(features, metadata["timeline"])
            machine_gate = (
                voice["all_stage1_predictions_correct"]
                and artifact_diagnostics["rig_finite"]
                and artifact_diagnostics["observation_finite"]
                and not artifact_diagnostics["observation_is_fabricated_all_present"]
                and artifact_diagnostics["rig_preserves_observed_xyz_max_error"] <= 1e-3
                and artifact_diagnostics["hand_participation_preserved"]
                and rig.shape[:2] == features.shape[:2]
            )
            voices.append({
                "name": voice["name"],
                "source_voice_ids": voice["source_voice_ids"],
                "source_voice_weights": voice["weights"],
                "frames": voice["phrase_frames"],
                "transition_spans": voice["transition_spans"],
                "stage1_predictions": voice["stage1_predictions"],
                "diagnostics": diagnostics,
                "artifact_diagnostics": artifact_diagnostics,
                "machine_gate": bool(machine_gate),
                "raw_landmarks": raw_path.relative_to(args.root).as_posix(),
                "raw_landmarks_sha256": raw["sha256"],
            })
            review_rows.append({
                "phrase_id": phrase_id,
                "voice": voice["name"],
                "gloss_sequence": " ".join(glosses),
                "lexical_sequence_correct_yes_no": "",
                "transition_1_smooth_yes_no": "",
                "transition_2_smooth_yes_no": "",
                "transition_3_smooth_yes_no": "",
                "no_hand_or_body_pop_yes_no": "",
                "overall_accept_for_training_yes_no": "",
                "notes": "",
            })
        item = {
            "phrase_id": phrase_id,
            "target_sequence": glosses,
            "video": video.relative_to(args.root).as_posix(),
            "video_sha256": report["video_sha256"],
            "preview": preview.relative_to(args.root).as_posix(),
            "preview_sha256": report["preview_sha256"],
            "video_metadata": video_metadata(video),
            "report": report_path.relative_to(args.root).as_posix(),
            "report_sha256": sha256(report_path),
            "voices": voices,
            "machine_gate": all(voice["machine_gate"] for voice in voices),
        }
        items.append(item)

    manifest = {
        "format": "slt_generated_phrase_native_review_v17",
        "version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "role": "synthetic_native_review_only",
        "validation_eligible": False,
        "test_eligible": False,
        "plan": args.plan.as_posix(),
        "plan_sha256": sha256(args.plan),
        "items": items,
        "all_machine_gates_passed": all(item["machine_gate"] for item in items),
        "human_review_required": True,
        "claim_boundary": (
            "The artifact gate verifies separation of detector observations from the "
            "render-only rig. It does not approve motion; native signer review is "
            "required for linguistic and perceptual naturalness."
        ),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (args.root / "review.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_rows[0])
        writer.writeheader()
        writer.writerows(review_rows)

    cards = []
    for item in items:
        voice_rows = "".join(
            f"<tr><td>{escape(voice['name'])}</td>"
            f"<td>{', '.join(map(str, voice['transition_spans']))}</td>"
            f"<td>{voice['diagnostics']['join_velocity_p95_over_gloss_p95']:.2f}×</td>"
            f"<td>{voice['diagnostics']['transition_velocity_p95_over_gloss_p95']:.2f}×</td>"
            f"<td>{'pass' if voice['machine_gate'] else 'FAIL'}</td></tr>"
            for voice in item["voices"]
        )
        cards.append(f"""
<section>
  <h2>{escape(item['phrase_id'])}: {escape(' '.join(item['target_sequence']))}</h2>
  <video controls preload="metadata" poster="{escape(item['preview'])}">
    <source src="{escape(item['video'])}" type="video/mp4">
  </video>
  <p><a href="{escape(item['video'])}">Open/download MP4</a> ·
     <a href="{escape(item['report'])}">generation report</a></p>
  <table><thead><tr><th>Voice</th><th>Transition frames</th><th>Join speed / gloss</th>
  <th>Transition speed / gloss</th><th>Artifact-contract gate</th></tr></thead><tbody>{voice_rows}</tbody></table>
</section>""")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Stage 2 generated phrase review</title><style>
body{{font:16px system-ui,sans-serif;background:#0c0f16;color:#eef1f6;max-width:1100px;margin:auto;padding:24px}}
a{{color:#65c9ff}} section{{border-top:1px solid #354052;padding:20px 0}} video{{width:100%;background:#000}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #354052;padding:7px;text-align:left}}
.warning{{background:#362c12;border:1px solid #846d2d;padding:14px}}
</style></head><body>
<h1>Stage 2 generated phrase pilot: native review</h1>
<p class="warning"><strong>Synthetic review material—not ground truth.</strong> The artifact gate only prevents the
render rig from masquerading as observed landmarks. Only a native signer can approve lexical correctness,
coarticulation, rhythm, and naturalness. Record decisions in
<a href="review.csv">review.csv</a>. Do not use unapproved items for training, validation, or testing.</p>
<p><a href="contact_sheet.png">Open the contact sheet</a> · <a href="manifest.json">audited manifest</a></p>
{''.join(cards)}
</body></html>"""
    (args.root / "index.html").write_text(html)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--root", type=Path, required=True)
    value.add_argument("--plan", type=Path, default=Path("active/v17/stage2_generated_phrase_review_plan_v17.json"))
    value.add_argument("--limit", type=int, default=10)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"items": len(result["items"]), "all_machine_gates_passed": result["all_machine_gates_passed"]}, indent=2))
