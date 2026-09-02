#!/usr/bin/env python3
"""Stress-test grounded transitions across hand-participation pattern changes."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.geometry_v17 import resample_features
from active.v17.signing_voice_phrase_v17 import (
    load_transition_voice,
    synthesize_join,
    trim_transition_span,
    trim_observed_span,
)
from active.v17.train_full_trajectory_v17 import FullTrajectoryDataset
from active.v17.train_signing_voice_v17 import sha256
from scripts.audit_real_motion_reference_v17 import Summary
from scripts.build_generated_phrase_review_v17 import boundary_diagnostics
from scripts.evaluate_full_trajectory_generator_v17 import real_summary
from scripts.generate_grounded_text_to_sign_v17 import (
    hand_bone_diagnostics,
    hand_participation,
    landmark_path,
    load_isolated,
    load_stage1,
    provenance_by_signer,
    recognize,
)


def sample_rows(rows, maximum, rng):
    if len(rows) <= maximum:
        return rows
    indices = np.sort(rng.choice(len(rows), maximum, replace=False))
    return [rows[index] for index in indices]


def run(args):
    stage1, labels = load_stage1(args.stage1_checkpoint)
    source_rows = defaultdict(list)
    with args.provenance.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == "train" and row["participant"] == args.signer:
                source_rows[row["canonical_label"]].append(row)
    if not source_rows:
        raise ValueError(f"unknown train signer {args.signer}")
    signs = {}
    predictions = {}
    for gloss, candidates in sorted(source_rows.items()):
        choices = []
        for row in candidates:
            path = landmark_path(args.isolated_root, row)
            value = load_isolated(path)
            prediction = recognize(stage1, labels, value)
            if prediction[0] == gloss:
                choices.append((prediction[1], path.as_posix(), value))
        if choices:
            confidence, path, value = max(choices, key=lambda row: (row[0], row[1]))
            signs[gloss] = value
            predictions[gloss] = {"confidence": confidence, "path": path}
    if len(signs) != len(labels):
        raise ValueError(
            f"{args.signer} has {len(signs)}/{len(labels)} correctly recognized glosses"
        )

    grouped = defaultdict(list)
    for first in sorted(signs):
        for second in sorted(signs):
            if first == second:
                continue
            pattern = f"{hand_participation(signs[first])}->{hand_participation(signs[second])}"
            grouped[pattern].append((first, second))
    rng = np.random.default_rng(args.seed)
    selected = [
        pair for pattern in sorted(grouped)
        for pair in sample_rows(grouped[pattern], args.samples_per_pattern, rng)
    ]

    device = torch.device(args.device)
    mean, timing = load_transition_voice(args.mean_checkpoint, args.timing_checkpoint, device)
    manifest = json.loads(args.full_manifest.read_text())
    sources = sorted({str(row["source"]) for row in manifest["rows"]})
    source_to_index = {source: index for index, source in enumerate(sources)}
    local_train = FullTrajectoryDataset(
        manifest, args.full_root, {"train"}, source_to_index,
        sources={"local_phrase_full"},
    )
    motion_reference = real_summary(local_train)

    rows = []
    for index, (first, second) in enumerate(selected, start=1):
        left = trim_transition_span(signs[first])
        right = trim_transition_span(signs[second])
        transition, right, span, entry_trim = synthesize_join(
            left, right, mean, timing, device
        )
        stream = np.concatenate((left, transition, right), axis=0)
        timeline = [
            {"kind": "gloss", "start": 0, "stop": len(left)},
            {"kind": "transition", "start": len(left), "stop": len(left) + span},
            {"kind": "gloss", "start": len(left) + span, "stop": len(stream)},
        ]
        boundary = boundary_diagnostics(stream, timeline)
        bones = hand_bone_diagnostics(stream, timeline)
        composed_right_prediction = recognize(stage1, labels, right)
        summary = Summary()
        summary.add(resample_features(stream, 128))
        motion = summary.result()
        ratios = {
            name: motion["hand_motion"][name]["p95"]
            / motion_reference["hand_motion"][name]["p95"]
            for name in ("speed", "acceleration", "jerk")
        }
        endpoint_sides = [
            left_side or right_side for left_side, right_side in zip(
                hand_participation(left), hand_participation(right)
            )
        ]
        transition_sides = hand_participation(transition)
        gates = {
            "no_invented_hand_side": all(
                not value or allowed for value, allowed in zip(transition_sides, endpoint_sides)
            ),
            "no_handless_transition_frame": boundary["handless_transition_frames"] == 0,
            "no_transition_only_nodes": boundary["nodes_present_only_inside_transition"] == 0,
            "motion_orders_in_global_genuine_range_diagnostic": all(
                0.5 <= value <= 2.0 for value in ratios.values()
            ),
            "hand_bones_in_endpoint_range": (
                0.25 <= bones["transition_over_gloss"]["p05"] <= 4.0
                and 0.5 <= bones["transition_over_gloss"]["p50"] <= 2.0
                and 0.25 <= bones["transition_over_gloss"]["p95"] <= 4.0
            ),
            "transition_motion_orders_in_gloss_range": all(
                0.25 <= value <= 4.0
                for value in boundary["transition_motion_p95_over_gloss_p95"].values()
            ),
            "right_entry_preserves_stage1_label": (
                composed_right_prediction[0] == second
            ),
        }
        required_gates = {
            key: value for key, value in gates.items()
            if not key.endswith("_diagnostic")
        }
        rows.append({
            "first": first, "second": second,
            "pattern": f"{hand_participation(left)}->{hand_participation(right)}",
            "transition_frames": span, "right_entry_trim_frames": entry_trim,
            "transition_hand_participation": transition_sides,
            "motion_p95_over_genuine_local_train": ratios,
            "hand_bone_diagnostics": bones,
            "boundary_diagnostics": boundary, "gates": gates,
            "composed_right_prediction": {
                "label": composed_right_prediction[0],
                "confidence": composed_right_prediction[1],
            },
            "passed": all(required_gates.values()),
        })
        if index % 50 == 0:
            print(json.dumps({"complete": index, "total": len(selected)}), flush=True)

    by_pattern = {}
    for pattern in sorted(grouped):
        values = [row for row in rows if row["pattern"] == pattern]
        by_pattern[pattern] = {
            "available_ordered_pairs": len(grouped[pattern]),
            "audited": len(values), "passed": sum(row["passed"] for row in values),
            "pass_rate": sum(row["passed"] for row in values) / max(1, len(values)),
        }
    failures = Counter(
        gate for row in rows for gate, passed in row["gates"].items()
        if not passed and not gate.endswith("_diagnostic")
    )
    report = {
        "format": "slt_grounded_transition_scalability_audit_v17", "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "signer": args.signer, "recognized_glosses": len(signs),
        "samples_per_pattern": args.samples_per_pattern,
        "seed": args.seed,
        "available_ordered_pairs": sum(map(len, grouped.values())),
        "audited_pairs": len(rows), "passed_pairs": sum(row["passed"] for row in rows),
        "pass_rate": sum(row["passed"] for row in rows) / len(rows),
        "failure_counts": dict(failures), "patterns": by_pattern,
        "rows": rows, "component_predictions": predictions,
        "mean_checkpoint": args.mean_checkpoint.as_posix(),
        "mean_checkpoint_sha256": sha256(args.mean_checkpoint),
        "timing_checkpoint": args.timing_checkpoint.as_posix(),
        "timing_checkpoint_sha256": sha256(args.timing_checkpoint),
        "test_evaluated": False, "citizen_test_accessed": False,
        "how2sign_validation_accessed": False, "how2sign_test_accessed": False,
        "claim_boundary": (
            "Machine landmark gates across sampled boundaries do not establish ASL "
            "coarticulation correctness or human perceptual naturalness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--signer", default="P37")
    value.add_argument("--samples-per-pattern", type=int, default=50)
    value.add_argument("--seed", type=int, default=1701)
    value.add_argument("--provenance", type=Path, default=Path("data/local/citizen100_v17/provenance.csv"))
    value.add_argument("--isolated-root", type=Path, default=Path("data/local/citizen100_v17/landmarks"))
    value.add_argument("--stage1-checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_baseline/best_model.pth"))
    value.add_argument("--mean-checkpoint", type=Path, default=Path("artifacts/models/transition_all_real_v17_v1/model.pth"))
    value.add_argument("--timing-checkpoint", type=Path, default=Path("artifacts/models/transition_span_multicorpus_v17_allvoices_final/model.pth"))
    value.add_argument("--full-manifest", type=Path, default=Path("active/v17/full_trajectory_generation_manifest_v17.json"))
    value.add_argument("--full-root", type=Path, default=Path("data/local/full_trajectory_landmarks_v17"))
    value.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    value.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_grounded_transition_scalability_v1/report.json"))
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({
        key: result[key] for key in ("audited_pairs", "passed_pairs", "pass_rate", "failure_counts")
    }, indent=2))
