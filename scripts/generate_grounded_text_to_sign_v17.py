#!/usr/bin/env python3
"""Compose unseen phrases from genuine isolated signs and learned transitions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.geometry_v17 import resample_features
from active.v17.landmark_anatomy_v17 import anatomy_coverage, complete_landmark_anatomy
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.signing_voice_phrase_v17 import (
    load_transition_voice,
    synthesize_join,
    trim_observed_span,
    trim_transition_span,
)
from active.v17.train_full_trajectory_v17 import FullTrajectoryDataset
from active.v17.train_signing_voice_v17 import sha256
from scripts.audit_real_motion_reference_v17 import Summary
from scripts.build_generated_phrase_review_v17 import boundary_diagnostics
from scripts.evaluate_full_trajectory_generator_v17 import real_summary
from scripts.render_genuine_local_phrase_reference_v17 import coordinate_bounds
from scripts.render_signing_voice_phrase_v17 import avatar_panel, put_text
from scripts.render_signing_voice_phrase_v17 import HAND_EDGES


def hand_participation(features: np.ndarray) -> list[bool]:
    present = np.asarray(features)[..., 3] > 0
    return [bool(present[:, :21].any()), bool(present[:, 21:42].any())]


def parse_phrases(values: list[str]) -> list[tuple[str, ...]]:
    phrases = [tuple(part.strip().upper() for part in value.split(",") if part.strip()) for value in values]
    if not phrases or any(not phrase for phrase in phrases):
        raise ValueError("each phrase must contain at least one comma-separated gloss")
    return phrases


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[A-Z0-9]+", value.upper()))


def resolve_text(value: str, catalog: dict[str, object]) -> tuple[str, ...]:
    aliases = {
        normalize_text(alias): tuple(row["target_glosses"])
        for row in catalog["phrases"] for alias in row["text_aliases"]
    }
    key = normalize_text(value)
    if key not in aliases:
        raise ValueError(
            f"unsupported curated text {value!r}; available: {', '.join(sorted(aliases))}"
        )
    return aliases[key]


def provenance_by_signer(
    path: Path, glosses: set[str]
) -> dict[str, dict[str, list[dict[str, str]]]]:
    output: dict[str, dict[str, list[dict[str, str]]]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "train" or row["canonical_label"] not in glosses:
                continue
            output.setdefault(row["participant"], {}).setdefault(
                row["canonical_label"], []
            ).append(row)
    return output


def landmark_path(root: Path, row: dict[str, str]) -> Path:
    return root / "train" / row["canonical_label"] / f"{Path(row['video']).stem}.v17.npz"


def load_isolated(path: Path, logical_fps: float = 15.) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        value = payload["features"].astype(np.float32)
        metadata = json.loads(str(payload["metadata_json"]))
    fps = float(metadata["fps"])
    processed = int(metadata["source_frames_processed"])
    decoded, sampled = int(metadata["decoded_frame_count"]), int(metadata["sampled_frame_count"])
    if min(fps, logical_fps, processed, decoded, sampled) <= 0 or not np.isfinite([fps, logical_fps]).all():
        raise ValueError("source timing must be finite and positive")
    duration = processed * decoded / sampled / fps
    value = resample_features(value, max(4, round(duration * logical_fps)))
    return trim_observed_span(value)


def isolated_candidates(
    root: Path,
    rows: list[dict[str, str]],
    gloss: str,
    stage1,
    labels: dict[int, str],
    logical_fps: float = 15.,
) -> list[tuple[Path, np.ndarray, tuple[str, float]]]:
    choices = []
    for row in rows:
        path = landmark_path(root, row)
        value = load_isolated(path, logical_fps)
        source_prediction = recognize(stage1, labels, value)
        prepared_prediction = recognize(
            stage1, labels, trim_transition_span(value)
        )
        source_correct = source_prediction[0] == gloss
        prepared_correct = prepared_prediction[0] == gloss
        choices.append((
            source_correct and prepared_correct,
            source_correct,
            min(source_prediction[1], prepared_prediction[1]),
            path.as_posix(), path, value, source_prediction,
        ))
    if not choices:
        raise ValueError(f"no isolated candidates for {gloss}")
    choices.sort(key=lambda row: row[:4], reverse=True)
    return [(path, value, prediction) for *_, path, value, prediction in choices]


def select_isolated_candidate(
    root: Path,
    rows: list[dict[str, str]],
    gloss: str,
    stage1,
    labels: dict[int, str],
) -> tuple[Path, np.ndarray, tuple[str, float]]:
    return isolated_candidates(root, rows, gloss, stage1, labels)[0]


def compose(signs, glosses, mean, timing, device):
    signs = [trim_transition_span(sign) for sign in signs]
    stream = signs[0]
    timeline = [{
        "kind": "gloss", "gloss": glosses[0], "start": 0, "stop": len(stream),
        "hand_participation": hand_participation(stream),
    }]
    for sign, gloss in zip(signs[1:], glosses[1:]):
        transition, sign, span, entry_trim = synthesize_join(
            stream, sign, mean, timing, device
        )
        left_sides = hand_participation(stream[timeline[-1]["start"]:timeline[-1]["stop"]])
        right_sides = hand_participation(sign)
        transition_sides = hand_participation(transition)
        if any(value and not (left or right) for value, left, right in zip(
            transition_sides, left_sides, right_sides
        )):
            raise RuntimeError("transition invented a hand absent from both neighboring signs")
        start = len(stream)
        stream = np.concatenate((stream, transition, sign), axis=0)
        timeline.extend((
            {
                "kind": "transition", "start": start, "stop": start + span,
                "right_entry_trim_frames": entry_trim,
            },
            {
                "kind": "gloss", "gloss": gloss, "start": start + span,
                "stop": len(stream), "hand_participation": right_sides,
            },
        ))
    return stream.astype(np.float32), timeline


def load_stage1(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("grounded generation requires a v17 Stage-1 checkpoint")
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), {int(value): key for key, value in checkpoint["label_to_index"].items()}


@torch.inference_mode()
def recognize(model, labels, features):
    probability = model(torch.from_numpy(resample_features(features, 32).astype(np.float32))[None]).softmax(dim=-1)[0]
    confidence, index = probability.max(dim=0)
    return labels[int(index)], float(confidence)


def motion_summary(features):
    value = Summary()
    value.add(resample_features(features, 128))
    return value.result()


def hand_bone_diagnostics(features, timeline):
    transition = np.zeros(len(features), dtype=bool)
    for row in timeline:
        if row["kind"] == "transition":
            transition[int(row["start"]):int(row["stop"])] = True
    values = {}
    for name, frames in (("gloss", ~transition), ("transition", transition)):
        lengths = []
        for start in (0, 21):
            for first, second in HAND_EDGES:
                valid = (
                    frames & (features[:, start + first, 3] > 0)
                    & (features[:, start + second, 3] > 0)
                )
                lengths.extend(np.linalg.norm(
                    features[valid, start + first, :3]
                    - features[valid, start + second, :3], axis=1,
                ).tolist())
        values[name] = {
            key: float(np.quantile(lengths, quantile))
            for key, quantile in (("p05", 0.05), ("p50", 0.50), ("p95", 0.95))
        }
    ratios = {
        key: values["transition"][key] / max(values["gloss"][key], 1e-8)
        for key in ("p05", "p50", "p95")
    }
    return {"bone_lengths": values, "transition_over_gloss": ratios}


def render_frame(rig, frame, requested_text, phrase, segment, bounds):
    canvas = np.full((720, 720, 3), 8, np.uint8)
    put_text(canvas, "GROUNDED TEXT TO SIGN - NATIVE REVIEW", (20, 38), 0.68, (245, 247, 251), 2)
    put_text(canvas, f"Text: {requested_text}", (20, 70), 0.48, (175, 188, 207), 1)
    put_text(canvas, f"Gloss: {phrase}", (20, 99), 0.56, (74, 190, 255), 2)
    put_text(canvas, segment, (20, 128), 0.48, (175, 188, 207), 1)
    panel = avatar_panel(rig, frame, (680, 550), *bounds, (56, 186, 255))
    canvas[145:695, 20:700] = panel
    put_text(canvas, "Exact train-only signs + learned genuine-motion transition", (20, 715), 0.43, (185, 196, 214), 1)
    return canvas


def run(args):
    phrase_values = list(args.phrase or [])
    text_catalog = None
    resolved_texts: dict[str, list[str]] = {}
    if args.text:
        text_catalog = json.loads(args.text_catalog.read_text())
        for value in args.text:
            resolved = resolve_text(value, text_catalog)
            phrase_values.append(",".join(resolved))
            resolved_texts.setdefault(" ".join(resolved), []).append(value)
    phrases = parse_phrases(
        phrase_values or ["GOOD,MORNING", "TOMORROW,SCHOOL,GO"]
    )
    phrases = list(dict.fromkeys(phrases))
    requested = {gloss for phrase in phrases for gloss in phrase}
    rows = provenance_by_signer(args.provenance, requested | {"SORRY"})
    signers = sorted(
        signer for signer, values in rows.items()
        if (
            any(set(phrase) <= set(values) for phrase in phrases)
            if args.allow_different_signers else requested <= set(values)
        )
    )
    if not signers:
        raise ValueError("no train-only Citizen signer covers the requested phrase set")
    device = torch.device(args.device)
    mean, timing = load_transition_voice(args.mean_checkpoint, args.timing_checkpoint, device)
    stage1, stage1_labels = load_stage1(args.stage1_checkpoint)
    manifest = json.loads(args.full_manifest.read_text())
    sources = sorted({str(row["source"]) for row in manifest["rows"]})
    source_to_index = {source: index for index, source in enumerate(sources)}
    local_train = FullTrajectoryDataset(
        manifest, args.full_root, {"train"}, source_to_index,
        sources={"local_phrase_full"},
    )
    reference = real_summary(local_train)

    candidates = []
    for signer in signers:
        phrase_rows = {}
        all_predictions_correct = True
        score = 0.0
        for phrase in phrases:
            if not set(phrase) <= set(rows[signer]):
                continue
            source_options = [
                isolated_candidates(
                    args.isolated_root, rows[signer][gloss], gloss,
                    stage1, stage1_labels,
                    args.logical_fps,
                )
                for gloss in phrase
            ]
            combinations = [[values[0] for values in source_options]]
            for index, values in enumerate(source_options):
                for alternative in values[1:3]:
                    choice = combinations[0].copy()
                    choice[index] = alternative
                    combinations.append(choice)

            phrase_candidates = []
            for selected_sources in combinations:
                paths = [row[0] for row in selected_sources]
                signs = [row[1] for row in selected_sources]
                predictions = [row[2] for row in selected_sources]
                source_predictions_correct = all(
                    prediction[0] == gloss
                    for prediction, gloss in zip(predictions, phrase)
                )
                stream, timeline = compose(signs, phrase, mean, timing, device)
                composed_predictions = [
                    recognize(
                        stage1, stage1_labels,
                        stream[int(part["start"]):int(part["stop"])],
                    )
                    for part in timeline if part["kind"] == "gloss"
                ]
                composed_predictions_correct = all(
                    prediction[0] == gloss
                    for prediction, gloss in zip(composed_predictions, phrase)
                )
                predictions_correct = (
                    source_predictions_correct and composed_predictions_correct
                )
                boundary = boundary_diagnostics(stream, timeline)
                bones = hand_bone_diagnostics(stream, timeline)
                boundary_pass = (
                    boundary["handless_transition_frames"] == 0
                    and boundary["nodes_present_only_inside_transition"] == 0
                    and all(
                        0.25 <= value <= 4.0
                        for value in boundary[
                            "transition_motion_p95_over_gloss_p95"
                        ].values()
                    )
                    and 0.25 <= bones["transition_over_gloss"]["p05"] <= 4.0
                    and 0.5 <= bones["transition_over_gloss"]["p50"] <= 2.0
                    and 0.25 <= bones["transition_over_gloss"]["p95"] <= 4.0
                )
                summary = motion_summary(stream)
                ratios = {
                    name: summary["hand_motion"][name]["p95"]
                    / reference["hand_motion"][name]["p95"]
                    for name in ("speed", "acceleration", "jerk")
                }
                motion_pass = all(0.5 <= value <= 2.0 for value in ratios.values())
                phrase_score = sum(abs(float(np.log(value))) for value in ratios.values())
                phrase_score += 100.0 * (not motion_pass)
                phrase_score += 1_000.0 * (not boundary_pass)
                phrase_score += sum(1.0 - anatomy_coverage(sign) for sign in signs)
                phrase_candidates.append({
                    "stream": stream, "timeline": timeline, "paths": paths,
                    "predictions": predictions,
                    "composed_predictions": composed_predictions,
                    "motion": summary, "motion_ratios": ratios,
                    "motion_pass": motion_pass,
                    "boundary_pass": boundary_pass,
                    "predictions_correct": predictions_correct,
                    "score": phrase_score,
                })
            selected_phrase = min(phrase_candidates, key=lambda row: (
                not row["predictions_correct"], not row["boundary_pass"],
                not row["motion_pass"],
                row["score"], tuple(path.as_posix() for path in row["paths"]),
            ))
            all_predictions_correct &= selected_phrase["predictions_correct"]
            score += selected_phrase["score"]
            phrase_rows[" ".join(phrase)] = selected_phrase
        sorry_audit = None
        if "SORRY" in rows[signer]:
            path, value, prediction = select_isolated_candidate(
                args.isolated_root, rows[signer]["SORRY"], "SORRY",
                stage1, stage1_labels,
            )
            sorry_audit = {
                "path": path, "hand_participation": hand_participation(value),
                "prediction": prediction,
            }
        one_hand_pass = (
            sorry_audit is not None
            and sum(sorry_audit["hand_participation"]) == 1
            and sorry_audit["prediction"][0] == "SORRY"
        )
        gates = {
            "all_exact_components_recognized": all_predictions_correct,
            "all_boundaries_pass_local_motion_and_anatomy": all(
                row["boundary_pass"] for row in phrase_rows.values()
            ),
            "all_motion_orders_within_0_5_to_2x_genuine_train_diagnostic": all(
                row["motion_pass"] for row in phrase_rows.values()
            ),
            "one_hand_audit_does_not_create_second_hand": one_hand_pass,
        }
        required_gates = {
            key: passed for key, passed in gates.items()
            if not key.endswith("_diagnostic")
        }
        candidates.append({
            "signer": signer, "phrases": phrase_rows, "sorry_audit": sorry_audit,
            "score": score, "gates": gates, "passed": all(required_gates.values()),
        })
    if args.allow_different_signers:
        selected_rows = {}
        for phrase in (" ".join(value) for value in phrases):
            eligible = [
                candidate for candidate in candidates
                if candidate["gates"]["one_hand_audit_does_not_create_second_hand"]
                and phrase in candidate["phrases"]
                and candidate["phrases"][phrase]["boundary_pass"]
                and candidate["phrases"][phrase]["predictions_correct"]
            ]
            if not eligible:
                raise RuntimeError(f"no same-signer composition passed for {phrase}")
            choice = min(eligible, key=lambda row: (
                row["phrases"][phrase]["score"], row["signer"]
            ))
            selected_rows[phrase] = (choice, choice["phrases"][phrase])
    else:
        eligible = [row for row in candidates if row["passed"]]
        if not eligible:
            raise RuntimeError("no single signer passed every requested phrase")
        selected = min(eligible, key=lambda row: (row["score"], row["signer"]))
        selected_rows = {
            phrase: (selected, row) for phrase, row in selected["phrases"].items()
        }

    with np.load(args.anatomy_package, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        if metadata.get("version") != 3:
            raise ValueError("grounded rendering requires anatomy contract v3")
        template = {
            "absolute_xyz": payload["canonical_absolute_xyz"].astype(np.float32),
            "hand_shapes": payload["canonical_hand_shapes"].astype(np.float32),
            "wrist_from_elbow": payload["canonical_wrist_from_elbow"].astype(np.float32),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    items = []
    for phrase, (selected, row) in selected_rows.items():
        observation = row["stream"]
        rig, observed = complete_landmark_anatomy(observation, template)
        rig_present = rig[..., 3] > 0
        if not np.array_equal(hand_participation(observation), hand_participation(rig)):
            raise RuntimeError("render rig invented or removed a participating hand")
        if not rig_present[:, 42:].all():
            raise RuntimeError("render rig still contains disappearing face/body nodes")
        for start, stop in ((0, 21), (21, 42)):
            side = rig_present[:, start:stop]
            if np.any(side.any(axis=1) & ~side.all(axis=1)):
                raise RuntimeError("active render hand is not a complete 21-node hand")
        rig_gloss_participation = []
        for part in row["timeline"]:
            if part["kind"] != "gloss":
                continue
            value = hand_participation(rig[int(part["start"]):int(part["stop"])])
            if value != part["hand_participation"]:
                raise RuntimeError(f"render rig changed hand participation for {part['gloss']}")
            rig_gloss_participation.append(value)
        diagnostics = boundary_diagnostics(observation, row["timeline"])
        bone_diagnostics = hand_bone_diagnostics(observation, row["timeline"])
        boundary_pass = (
            diagnostics["handless_transition_frames"] == 0
            and diagnostics["nodes_present_only_inside_transition"] == 0
            and all(
                0.25 <= value <= 4.0
                for value in diagnostics["transition_motion_p95_over_gloss_p95"].values()
            )
            and 0.25 <= bone_diagnostics["transition_over_gloss"]["p05"] <= 4.0
            and 0.5 <= bone_diagnostics["transition_over_gloss"]["p50"] <= 2.0
            and 0.25 <= bone_diagnostics["transition_over_gloss"]["p95"] <= 4.0
        )
        safe = phrase.lower().replace(" ", "_")
        requested_text = " / ".join(resolved_texts.get(phrase, [])) or phrase
        artifact = args.output / f"{safe}.grounded_text_to_sign_v17.npz"
        artifact_metadata = {
            "format": "slt_grounded_text_to_sign_v17", "version": 1,
            "artifact_contract_version": 4,
            "logical_fps": args.logical_fps,
            "source_timing": "restored from source processed frames, sampling fraction and fps before trimming",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "role": "synthetic_native_review_only", "requested_glosses": phrase.split(),
            "source_signer": selected["signer"], "timeline": row["timeline"],
            "genuine_isolated_content": True, "generated_transition_only": True,
            "training_eligible": False, "validation_eligible": False, "test_eligible": False,
        }
        np.savez_compressed(
            artifact,
            animation_rig_xyz=rig[..., :3].astype(np.float16),
            animation_rig_presence=rig_present,
            animation_rig_confidence=rig[..., 4].astype(np.float16),
            observation_xyz=observation[..., :3].astype(np.float16),
            observation_presence=observed,
            observation_confidence=observation[..., 4].astype(np.float16),
            metadata_json=np.asarray(json.dumps(artifact_metadata, sort_keys=True)),
        )
        video = args.output / f"{safe}.mp4"
        temporary = video.with_suffix(".mp4v.mp4")
        writer = cv2.VideoWriter(
            str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (720, 720)
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the review video")
        bounds = coordinate_bounds(rig)
        repeats = max(1, round(args.output_fps / args.logical_fps))
        for frame in range(len(rig)):
            segment = next(
                part for part in row["timeline"]
                if int(part["start"]) <= frame < int(part["stop"])
            )
            label = segment.get("gloss", "learned transition")
            image = render_frame(rig, frame, requested_text, phrase, str(label), bounds)
            for _ in range(repeats):
                writer.write(image)
        writer.release()
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(temporary),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(video),
        ], check=True)
        temporary.unlink()
        preview = args.output / f"{safe}.preview.png"
        preview_frame = len(rig) // 2
        preview_segment = next(
            part for part in row["timeline"]
            if int(part["start"]) <= preview_frame < int(part["stop"])
        )
        cv2.imwrite(str(preview), render_frame(
            rig, preview_frame, requested_text, phrase,
            str(preview_segment.get("gloss", "learned transition")), bounds,
        ))
        items.append({
            "phrase": phrase, "source_signer": selected["signer"],
            "requested_texts": resolved_texts.get(phrase, []),
            "source_landmarks": [path.as_posix() for path in row["paths"]],
            "source_landmark_sha256": [sha256(path) for path in row["paths"]],
            "stage1_predictions": [
                {"label": label, "confidence": confidence}
                for label, confidence in row["predictions"]
            ],
            "composed_stage1_predictions": [
                {"label": label, "confidence": confidence}
                for label, confidence in row["composed_predictions"]
            ],
            "source_gloss_hand_participation": [
                part["hand_participation"] for part in row["timeline"] if part["kind"] == "gloss"
            ],
            "rig_gloss_hand_participation": rig_gloss_participation,
            "motion_p95_over_genuine_local_train": row["motion_ratios"],
            "global_motion_diagnostic_passed": row["motion_pass"],
            "boundary_diagnostics": diagnostics, "boundary_gate_passed": boundary_pass,
            "hand_bone_diagnostics": bone_diagnostics,
            "rig_face_body_complete_every_frame": True,
            "active_rig_hands_have_all_21_nodes": True,
            "artifact": artifact.as_posix(), "artifact_sha256": sha256(artifact),
            "video": video.as_posix(), "video_sha256": sha256(video),
            "preview": preview.as_posix(), "preview_sha256": sha256(preview),
        })
    all_passed = all(item["boundary_gate_passed"] for item in items)
    selected_signers = {phrase: choice[0]["signer"] for phrase, choice in selected_rows.items()}
    unique_signers = sorted(set(selected_signers.values()))
    one_hand_audits = {
        signer: {
            "gloss": "SORRY", "source": candidate["sorry_audit"]["path"].as_posix(),
            "hand_participation": candidate["sorry_audit"]["hand_participation"],
            "stage1_prediction": candidate["sorry_audit"]["prediction"][0],
            "passed": candidate["gates"]["one_hand_audit_does_not_create_second_hand"],
        }
        for signer in unique_signers
        for candidate in candidates if candidate["signer"] == signer
    }
    previews = [cv2.imread(item["preview"]) for item in items]
    if any(value is None for value in previews):
        raise RuntimeError("could not reload a generated preview")
    if len(previews) % 2:
        previews.append(np.full_like(previews[0], 8))
    contact_sheet = args.output / "contact_sheet.png"
    cv2.imwrite(str(contact_sheet), np.vstack([
        np.hstack(previews[index:index + 2]) for index in range(0, len(previews), 2)
    ]))
    cards = "".join(
        f'<section><h2>{escape(item["phrase"])}</h2>'
        f'<p>Text: {escape(" / ".join(item["requested_texts"]) or item["phrase"])}</p>'
        f'<img src="{escape(Path(item["preview"]).name)}" alt="{escape(item["phrase"])} preview">'
        f'<video controls preload="metadata" src="{escape(Path(item["video"]).name)}"></video>'
        f'<p>Signer {escape(item["source_signer"])} · '
        f'<a href="{escape(Path(item["artifact"]).name)}">landmark artifact</a></p></section>'
        for item in items
    )
    index = args.output / "index.html"
    index.write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>Grounded text-to-sign native review</title><style>'
        'body{font:16px system-ui;background:#0c0f16;color:#eef1f6;max-width:900px;margin:auto;padding:24px}'
        'section{border-top:1px solid #354052;padding:18px 0}video,img{width:100%;margin:8px 0}a{color:#65c9ff}'
        '.warning{background:#362c12;border:1px solid #846d2d;padding:14px}</style>'
        '<h1>Grounded text-to-sign: native review</h1><p class="warning">Synthetic review material. '
        'Lexical segments are genuine train-only isolated signs; transitions are generated. '
        'Do not use for training, validation, or testing without a separate approved protocol.</p>'
        '<p>Check exact lexical meaning, hand participation, transition smoothness, and overall naturalness.</p>'
        + cards,
        encoding="utf-8",
    )
    report = {
        "format": "slt_grounded_text_to_sign_report_v17", "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_uses_holdout_phrase_trajectory": False,
        "requested_texts": list(args.text or []),
        "text_catalog": args.text_catalog.as_posix() if text_catalog is not None else None,
        "text_catalog_sha256": sha256(args.text_catalog) if text_catalog is not None else None,
        "selection_rule": (
            "one train-only signer per phrase; exact source and composed Stage-1 "
            "labels; hard local boundary/anatomy gates; closest global genuine-motion "
            "diagnostic"
        ),
        "selected_signer": unique_signers[0] if len(unique_signers) == 1 else None,
        "selected_signers": selected_signers,
        "one_hand_audit": one_hand_audits[unique_signers[0]] if len(unique_signers) == 1 else None,
        "one_hand_audits": one_hand_audits,
        "items": items, "all_machine_gates_passed": all_passed,
        "all_global_motion_diagnostics_passed": all(
            item["global_motion_diagnostic_passed"] for item in items
        ),
        "contact_sheet": contact_sheet.as_posix(),
        "contact_sheet_sha256": sha256(contact_sheet),
        "review_index": index.as_posix(), "review_index_sha256": sha256(index),
        "native_signer_review_required": True,
        "training_eligible": False, "validation_eligible": False, "test_eligible": False,
        "claim_boundary": (
            "Lexical motion comes from exact labeled train-only isolated clips and only "
            "the boundaries are generated. Machine gates permit native review, not use "
            "as recognition ground truth or a claim of fluent natural signing."
        ),
        "citizen_test_accessed": False, "semlex_test_accessed": False,
        "local_test_accessed": False, "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all_passed:
        raise RuntimeError("rendered review artifacts failed boundary gates")
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--phrase", action="append")
    value.add_argument("--text", action="append")
    value.add_argument(
        "--text-catalog", type=Path,
        default=Path("active/v17/text_to_sign_phrase_catalog_v17.json"),
    )
    value.add_argument("--provenance", type=Path, default=Path("data/local/citizen100_v17/provenance.csv"))
    value.add_argument("--isolated-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    value.add_argument("--full-manifest", type=Path, default=Path("active/v17/full_trajectory_generation_manifest_v17.json"))
    value.add_argument("--full-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_baseline/best_model.pth"))
    value.add_argument("--mean-checkpoint", type=Path, default=Path("artifacts/models/transition_all_real_v17_v1/model.pth"))
    value.add_argument("--timing-checkpoint", type=Path, default=Path("artifacts/models/transition_span_multicorpus_v17_allvoices_final/model.pth"))
    value.add_argument("--anatomy-package", type=Path, default=Path("artifacts/models/signing_landmark_anatomy_v17_v3/anatomy.npz"))
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_grounded_text_to_sign_pilot_v1"))
    value.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    value.add_argument("--output-fps", type=int, default=30)
    value.add_argument("--logical-fps", type=int, default=15)
    value.add_argument("--allow-different-signers", action="store_true")
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"selected_signer": result["selected_signer"], "items": len(result["items"]), "passed": result["all_machine_gates_passed"]}, indent=2))
