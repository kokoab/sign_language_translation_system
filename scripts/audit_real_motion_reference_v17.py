#!/usr/bin/env python3
"""Compare genuine continuous-landmark sources with rejected phrase pilots."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


PHRASES = (
    "GOOD_MORNING",
    "HELLO_HOW_YOU",
    "I_WANT_FOOD",
    "MY_NAME",
    "PLEASE_HELP_ME",
    "SORRY_I_LATE",
    "THANKYOU_FRIEND",
    "TOMORROW_SCHOOL_GO",
    "YESTERDAY_TEACHER_MEET",
)


class Summary:
    def __init__(self) -> None:
        self.archives = 0
        self.trajectories = 0
        self.frames = 0
        self.frame_counts: defaultdict[str, int] = defaultdict(int)
        self.step_count = 0
        self.presence_change_steps = 0
        self.motion: defaultdict[str, list[np.ndarray]] = defaultdict(list)

    def add(self, value: np.ndarray, *, count_archive: bool = False) -> None:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 3 or value.shape[1:] != (61, 5):
            raise ValueError(f"expected [T,61,5], got {value.shape}")
        if count_archive:
            self.archives += 1
        self.trajectories += 1
        self.frames += len(value)
        present = value[..., 3] > 0.5
        left = present[:, :21]
        right = present[:, 21:42]
        face = present[:, 42:57]
        body = present[:, 57:61]
        masks = {
            "left_any": left.any(axis=1),
            "right_any": right.any(axis=1),
            "both_hands_any": left.any(axis=1) & right.any(axis=1),
            "no_hand": ~(left.any(axis=1) | right.any(axis=1)),
            "left_complete": left.all(axis=1),
            "right_complete": right.all(axis=1),
            "both_hands_complete": left.all(axis=1) & right.all(axis=1),
            "face_complete": face.all(axis=1),
            "body_complete": body.all(axis=1),
            "all_nodes": present.all(axis=1),
        }
        for key, mask in masks.items():
            self.frame_counts[key] += int(mask.sum())
        if len(value) > 1:
            self.step_count += len(value) - 1
            self.presence_change_steps += int(np.any(present[1:] != present[:-1], axis=1).sum())

        xyz = value[:, :42, :3]
        hand_present = present[:, :42]
        for name, order in (("speed", 1), ("acceleration", 2), ("jerk", 3)):
            delta = np.diff(xyz, n=order, axis=0)
            valid = np.ones(delta.shape[:2], dtype=np.bool_)
            for offset in range(order + 1):
                valid &= hand_present[offset:offset + len(delta)]
            norms = np.linalg.norm(delta, axis=-1)
            for row, row_valid in zip(norms, valid):
                if row_valid.any():
                    self.motion[name].append(np.array([np.median(row[row_valid])], dtype=np.float32))

    def result(self) -> dict[str, Any]:
        presence = {
            key: count / max(self.frames, 1)
            for key, count in sorted(self.frame_counts.items())
        }
        motion = {}
        for key in ("speed", "acceleration", "jerk"):
            values = np.concatenate(self.motion[key]) if self.motion[key] else np.zeros(0)
            motion[key] = {
                "samples": int(len(values)),
                "mean": float(values.mean()) if len(values) else None,
                "p50": float(np.quantile(values, 0.50)) if len(values) else None,
                "p95": float(np.quantile(values, 0.95)) if len(values) else None,
                "p99": float(np.quantile(values, 0.99)) if len(values) else None,
            }
        return {
            "archives": self.archives,
            "trajectories": self.trajectories,
            "frames": self.frames,
            "presence_frame_fractions": presence,
            "presence_change_step_fraction": (
                self.presence_change_steps / max(self.step_count, 1)
            ),
            "hand_motion": motion,
        }


def metadata(payload: Any) -> dict[str, Any]:
    return json.loads(str(payload["metadata_json"].item()))


def add_v17_tree(
    root: Path, summaries: defaultdict[str, Summary], *, fixed_name: str | None = None
) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for path in sorted(root.glob("**/*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            if "landmarks" not in payload.files or "metadata_json" not in payload.files:
                continue
            meta = metadata(payload)
            if str(meta.get("role", "train")) != "train":
                continue
            values = payload["landmarks"].astype(np.float32)
            if values.ndim == 3:
                valid = np.ones(1, dtype=np.bool_)
                values = values[None]
            elif "window_valid" in payload.files:
                valid = payload["window_valid"].astype(np.bool_)
            elif "landmark_window_valid" in payload.files:
                valid = payload["landmark_window_valid"].astype(np.bool_)
            else:
                valid = np.ones(len(values), dtype=np.bool_)
            name = fixed_name or str(meta.get("source", root.name))
            first = True
            for trajectory in values[valid]:
                summaries[name].add(trajectory, count_archive=first)
                first = False
            counts[name] += 1
    return dict(counts)


def add_legacy_local(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    overall = Summary()
    by_phrase: defaultdict[str, Summary] = defaultdict(Summary)
    for path in sorted(root.glob("*.npy")):
        phrase = next((value for value in PHRASES if path.name.startswith(value + "_")), None)
        if phrase is None:
            continue
        value = np.load(path).astype(np.float32)
        overall.add(value, count_archive=True)
        by_phrase[phrase].add(value, count_archive=True)
    result = overall.result()
    # Coordinates are v16-era and are deliberately not pooled with v17 motion.
    result.pop("hand_motion")
    return result, {
        phrase: {k: v for k, v in summary.result().items() if k != "hand_motion"}
        for phrase, summary in sorted(by_phrase.items())
    }


def raw_inventory(root: Path) -> dict[str, int]:
    return {
        phrase: len(list((root / phrase).glob("*.mp4")))
        for phrase in PHRASES
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Genuine motion reference audit",
        "",
        "Generation remained stopped. This report characterizes real downloaded data and the two rejected pilots; it does not create training samples.",
        "",
        "## Compatible v17 experiment",
        "",
        "| Source | Archives | Trajectories | Frames | Both hands detected | Both hands complete | All 61 nodes | Presence-change steps | Jerk p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["v17_sources"].items():
        p = row["presence_frame_fractions"]
        jerk = row["hand_motion"]["jerk"]["p95"]
        lines.append(
            f"| {name} | {row['archives']} | {row['trajectories']} | {row['frames']} | "
            f"{p['both_hands_any']:.1%} | {p['both_hands_complete']:.1%} | "
            f"{p['all_nodes']:.1%} | {row['presence_change_step_fraction']:.1%} | "
            f"{jerk:.5f} |"
        )
    lines += [
        "",
        "## Local phrase inventory",
        "",
        "All nine raw phrase folders were inspected. The v16-era presence audit below covers all 780 clips but is reference-only because that schema cannot be mixed into v17 training.",
        "",
        "| Phrase | Raw clips | Legacy reference clips | Both hands detected | All 61 nodes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phrase in PHRASES:
        row = report["legacy_local_by_phrase"][phrase]
        p = row["presence_frame_fractions"]
        lines.append(
            f"| {phrase} | {report['raw_local_inventory'][phrase]} | {row['archives']} | "
            f"{p['both_hands_any']:.1%} | {p['all_nodes']:.1%} |"
        )
    if "reconstruction_transfer" in report:
        lines += [
            "",
            "## Frozen transition-model transfer experiment",
            "",
            "A deterministic 4–12 frame interval was masked in every compatible genuine train-side window. The existing frozen How2Sign+YouTube model was compared with endpoint interpolation; no phrase was generated.",
            "",
            "| Source | Windows | Improvement over interpolation | Windows improved |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, row in report["reconstruction_transfer"]["domains"].items():
            lines.append(
                f"| {name} | {row['windows']} | "
                f"{row['relative_improvement_vs_linear']:.1%} | "
                f"{row['windows_improved_fraction']:.1%} |"
            )
    lines += [
        "",
        "## Result",
        "",
        *[f"- {value}" for value in report["findings"]],
        "",
        "The contact sheet is `local_phrases_contact_sheet.png`; each row samples one of the nine genuine local phrases.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    summaries: defaultdict[str, Summary] = defaultdict(Summary)
    trees = (
        (args.local_v17, "local_phrases_v17"),
        (args.asllrp_contiguous, "asllrp_contiguous"),
        (args.asllrp_other, "asllrp_other_ctc"),
        (args.flores, "two_m_flores_asl"),
        (args.how2sign_ncslgr, None),
        (args.youtube, None),
        (args.generated_v1, "rejected_generated_v1"),
        (args.generated_v2, "rejected_generated_v2"),
    )
    discovered = {}
    for root, name in trees:
        discovered[root.as_posix()] = add_v17_tree(root, summaries, fixed_name=name)
    v17 = {name: summary.result() for name, summary in sorted(summaries.items())}
    legacy, legacy_by_phrase = add_legacy_local(args.legacy_local)

    v2_presence = v17["rejected_generated_v2"]["presence_frame_fractions"]
    real_names = [name for name in v17 if not name.startswith("rejected_generated")]
    real_frames = sum(v17[name]["frames"] for name in real_names)
    real_all_nodes = sum(
        v17[name]["presence_frame_fractions"]["all_nodes"] * v17[name]["frames"]
        for name in real_names
    ) / max(real_frames, 1)
    report = {
        "format": "slt_real_motion_reference_audit_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "generation_stopped_both_pilots_rejected",
        "v17_sources": v17,
        "raw_local_inventory": raw_inventory(args.raw_local),
        "legacy_local_reference_only": legacy,
        "legacy_local_by_phrase": legacy_by_phrase,
        "additional_downloaded_reference": {
            "openasl_clips": len(list(args.openasl.glob("*.mp4"))),
            "openasl_status": "raw visual/domain reference; no compatible v17 archive",
        },
        "discovered_archives": discovered,
        "findings": [
            "The v2 anatomy completion is a recognizer-distribution failure: "
            f"all 61 nodes are present in {v2_presence['all_nodes']:.1%} of v2 frames "
            f"versus {real_all_nodes:.1%} across the pooled genuine v17 train-side sources.",
            "A detector mask is an observation, not an anatomy rig. Missing or inactive detected hands in genuine data must not be rewritten as always observed.",
            "The local videos show continuous arm travel, preparation, overlap, retraction, and rest across signs; isolated medoid concatenation plus a short masked gap does not model that full phrase trajectory.",
            "Generated v1 and v2 remain review evidence only and are prohibited from recognition training, validation, and testing.",
            "The next experiment should learn or retrieve full genuine phrase motion with source-balanced sampling, while keeping rendering anatomy and recognizer observation masks as separate outputs.",
        ],
        "limitations": [
            "Within-window finite differences are extractor-space diagnostics, not a native-signer naturalness score.",
            "The three vocabulary-ineligible local phrases have raw-video and v16 reference coverage but no compatible v17 Stage-2 archive in the current cache.",
            "How2Sign and YouTube-ASL have no exact ordered gloss supervision in this project and are motion-only sources.",
            "YouTube-ASL internal validation, project sealed sets, How2Sign validation/test, and 2M-Flores devtest were not accessed.",
        ],
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    reconstruction_path = args.output / "reconstruction_transfer.json"
    if reconstruction_path.is_file():
        report["reconstruction_transfer"] = json.loads(reconstruction_path.read_text())
        local_transfer = report["reconstruction_transfer"]["domains"]["local_phrases"]
        report["findings"].insert(
            3,
            "The frozen How2Sign+YouTube inpainter transfers positively to every compatible genuine corpus, but local phrases are the weakest practical result: "
            f"{local_transfer['relative_improvement_vs_linear']:.1%} aggregate improvement and only "
            f"{local_transfer['windows_improved_fraction']:.1%} of windows improved. It is not ready to drive local phrase generation.",
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "README.md").write_text(markdown(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_real_motion_reference_audit"))
    parser.add_argument("--raw-local", type=Path, default=Path("data/raw_videos/PHRASES"))
    parser.add_argument("--legacy-local", type=Path, default=Path("src_v16/ASL_phrases_v16"))
    parser.add_argument("--local-v17", type=Path, default=Path("data/local/stage2_v17_multimodal/train/local_phrases"))
    parser.add_argument("--asllrp-contiguous", type=Path, default=Path("data/local/stage2_v17_multimodal/train/asllrp_contiguous"))
    parser.add_argument("--asllrp-other", type=Path, default=Path("data/local/stage2_v17_asllrp_other_multimodal/train/asllrp_other_ctc"))
    parser.add_argument("--flores", type=Path, default=Path("data/local/stage2_v17_2m_flores_multimodal/train/two_m_flores_asl"))
    parser.add_argument("--how2sign-ncslgr", type=Path, default=Path("data/local/how2sign_transition_landmarks_v17"))
    parser.add_argument("--youtube", type=Path, default=Path("data/local/youtube_asl_transition_landmarks_v17"))
    parser.add_argument("--openasl", type=Path, default=Path("data/local/openasl_transition_subset_v17/clips"))
    parser.add_argument("--generated-v1", type=Path, default=Path("artifacts/reports/stage2_v17_generated_phrase_review_pilot_v1"))
    parser.add_argument("--generated-v2", type=Path, default=Path("artifacts/reports/stage2_v17_generated_phrase_review_pilot_v2"))
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps({"status": result["status"], "sources": list(result["v17_sources"])}, indent=2))
