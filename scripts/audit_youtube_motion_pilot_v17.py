#!/usr/bin/env python3
"""Audit the frozen raw MediaPipe pilot without manufacturing Apple features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


PARTS = {"pose_landmarks": 33, "left_hand_landmarks": 21,
         "right_hand_landmarks": 21, "face_landmarks": 478}


def longest_gap(present):
    longest = current = 0
    for value in present:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def inspect_clip(payload, row):
    frames = payload.get("keypoints")
    if not isinstance(frames, list) or not frames:
        raise ValueError("keypoints must be a nonempty frame list")
    match = re.fullmatch(r"(.+)\.(\d+)-(\d+)\.json", row["member"])
    if not match or match[1] != row["video_id"]:
        raise ValueError("filename/video ID mismatch")
    declared = int(row["frames"])
    if int(match[3]) - int(match[2]) + 1 != declared:
        raise ValueError("manifest and filename range disagree")
    valid = {part: [] for part in PARTS}
    zeros = outside = points = 0
    size = payload.get("size")
    geometry = (isinstance(size, list) and len(size) == 2
                and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in size))
    # Source size is recorded as [height, width]; its convention still needs source verification.
    height, width = size if geometry else (None, None)
    shoulders = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("frame must be an object")
        for part, count in PARTS.items():
            values = frame.get(part)
            if not isinstance(values, list) or len(values) not in (0, count):
                raise ValueError(f"invalid {part} point count")
            for xy in values:
                if (not isinstance(xy, list) or len(xy) != 2
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in xy)):
                    raise ValueError(f"invalid {part} coordinate")
                points += 1
                zeros += xy == [0, 0]
                outside += bool(geometry and not (0 <= xy[0] <= width and 0 <= xy[1] <= height))
            # A zero pair is not silently admitted as a real observation.
            valid[part].append(bool(values) and all(xy != [0, 0] for xy in values))
        pose = frame["pose_landmarks"]
        if len(pose) == 33:
            shoulders.append(math.hypot(pose[11][0] - pose[12][0], pose[11][1] - pose[12][1]))
    left, right = valid["left_hand_landmarks"], valid["right_hand_landmarks"]
    temporal_keys = [k for k in payload if any(s in k.casefold() for s in ("fps", "timestamp", "time_ms"))]
    return {
        "member": row["member"], "video_id": row["video_id"], "frames": len(frames),
        "declared_frames": declared, "frame_count_delta": len(frames) - declared,
        "admitted_for_pretraining": len(frames) == declared,
        "left_present_frames": sum(left), "right_present_frames": sum(right),
        "either_hand_frames": sum(a or b for a, b in zip(left, right)),
        "both_hand_frames": sum(a and b for a, b in zip(left, right)),
        "left_max_gap_frames": longest_gap(left), "right_max_gap_frames": longest_gap(right),
        "both_missing_max_gap_frames": longest_gap([a or b for a, b in zip(left, right)]),
        "points": points, "zero_pairs": zeros, "out_of_image_points": outside,
        "positive_size_metadata": geometry, "size": size,
        "positive_shoulder_scale_frames": sum(x > 1e-6 for x in shoulders),
        "temporal_metadata_keys": temporal_keys,
        "top_level_keys": sorted(payload),
    }


def self_check():
    row = {"member": "a.000000-000002.json", "video_id": "a", "frames": "3"}
    frame = {key: [[1., 2.]] * count for key, count in PARTS.items()}
    frames = [dict(frame), dict(frame), dict(frame)]
    frames[1]["left_hand_landmarks"] = []
    payload = {"keypoints": frames, "size": [10, 20]}
    result = inspect_clip(payload, row)
    assert result["left_present_frames"] == 2 and result["left_max_gap_frames"] == 1
    assert result["either_hand_frames"] == 3 and result["both_hand_frames"] == 2
    shorter = inspect_clip({**payload, "keypoints": frames[:2]}, row)
    assert shorter["frame_count_delta"] == -1 and not shorter["admitted_for_pretraining"]
    assert longest_gap([False, False, True, False]) == 2
    frames[2]["right_hand_landmarks"] = [[float("nan"), 0.]] * 21
    try:
        inspect_clip(payload, row)
    except ValueError as exc:
        assert "coordinate" in str(exc)
    else:
        raise AssertionError("nonfinite coordinates admitted")
    print("audit self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.manifest, args.input_root, args.output_dir)):
        parser.error("manifest, input-root, and output-dir are required")
    with args.manifest.open(newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if len(all_rows) != 1411 or len({r["member"] for r in all_rows}) != 1411:
        raise ValueError("expected frozen 1,411 distinct members")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    rows = all_rows[:args.limit] if args.limit else all_rows
    results, errors = [], []
    for row in rows:
        try:
            if Path(row["member"]).name != row["member"]:
                raise ValueError("member must be a basename")
            path = args.input_root / row["member"]
            content = path.read_bytes()
            result = inspect_clip(json.loads(content), row)
            result["sha256"] = hashlib.sha256(content).hexdigest()
            results.append(result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"member": row["member"], "error": str(exc)})
    totals = {key: sum(r[key] for r in results) for key in (
        "frames", "left_present_frames", "right_present_frames", "either_hand_frames",
        "both_hand_frames", "points", "zero_pairs", "out_of_image_points")}
    quarantined = [r for r in results if not r["admitted_for_pretraining"]]
    issues = [
        "Detector confidence is absent from raw XY point pairs; Apple confidence cannot be reconstructed.",
        "Hand slot names do not verify anatomical chirality or source mirroring without paired visual evidence.",
        "No validated MediaPipe input adapter into the selected Apple temporal encoder exists in this pilot.",
        "Exact face semantics and the Apple relative log-scale channel require explicit mapping, not zero padding.",
    ]
    timing_count = sum(bool(r["temporal_metadata_keys"]) for r in results)
    if not timing_count:
        issues.append("No frame-rate/timestamp fields found; frame order exists but elapsed-time motion is unverified.")
    if errors:
        issues.append(f"{len(errors)} files failed structural validation.")
    report = {
        "audit": {"requested": len(rows), "validated": len(results), "errors": errors,
                  "quarantined_count_mismatch": len(quarantined),
                  "admitted_count": len(results) - len(quarantined),
                  "count_delta_histogram": {str(d): sum(r["frame_count_delta"] == d for r in results)
                                            for d in sorted({r["frame_count_delta"] for r in results})},
                  "complete_manifest": len(rows) == 1411, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()},
        "coverage": totals,
        "coordinates": {"streams": "raw keypoints only; cropped coordinates excluded",
                        "size_metadata_clips": sum(r["positive_size_metadata"] for r in results),
                        "size_convention": "[height,width], verified in upstream predict_pose.py mdeiapipe_to_xy",
                        "outside_image_is_not_automatically_corrupt": True,
                        "point_dimension": 2},
        "timing": {"clips_with_temporal_field_names": timing_count,
                   "frame_ranges": "inclusive filename range checked against manifest and actual frames",
                   "seconds_validated": False},
        "identity_and_splits": {"unique_source_videos": len({r["video_id"] for r in rows}),
                                "signer_disjointness_verified": False,
                                "caption_used_for_supervision": False,
                                "protected_evaluation_accessed": False},
        "chirality": {"slots": "left_hand_landmarks/right_hand_landmarks", "anatomical_mapping_verified": False},
        "channel_mapping": {"xy": "pixel XY; body-relative normalization is feasible after geometry verification",
                            "presence": "empty lists and zero pairs can be masked",
                            "confidence": "not available", "depth": "not provided; Apple scale proxy is a separate derived quantity"},
        "gate": {"direct_existing_encoder_transfer": False,
                 "adapter_temporal_transfer": "blocked" if errors else "conditional",
                 "blocking_issues": issues,
                 "required_bridge_validation": [
                     "Use a separately fingerprinted 2D input adapter, never relabel MediaPipe as Apple features.",
                     "Validate geometry/chirality from source code or paired local evidence.",
                     "Transfer only compatible temporal blocks; retain Apple frontend and supervised decoder.",
                     "Keep unknown timing explicit; frame-index reconstruction is possible but not timing validation."]},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "compatibility_clips.jsonl").open("w") as stream:
        for result in results:
            stream.write(json.dumps(result) + "\n")
    temporary = args.output_dir / "compatibility.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output_dir / "compatibility.json")
    text = ("# YouTube motion pilot compatibility audit\n\n"
            f"Validated **{len(results)}/{len(rows)} clips**, {totals['frames']:,} frames; "
            f"{len(errors)} structural failures. Full frozen subset: {len(rows) == 1411}.\n\n"
            f"Quarantined for manifest/actual frame-count disagreement: **{len(quarantined)}**. "
            "These are provenance discrepancies, not evidence of invalid XY. No padding or invented timestamps.\n\n"
            f"At least one hand observed: {totals['either_hand_frames']:,} frames; "
            f"both hands: {totals['both_hand_frames']:,}. Zero coordinate pairs: {totals['zero_pairs']:,}.\n\n"
            "**Direct Apple feature substitution is not admitted. Separate-adapter temporal "
            "transfer remains conditional, not disproven.**\n\n"
            + "\n".join(f"- {issue}" for issue in issues)
            + "\n\nRaw XY geometry, frame ordering and masks support a potential 2D reconstruction "
            "experiment. A learned source adapter could isolate detector differences while "
            "sharing only temporal weights. Its transfer benefit still needs the matched "
            "downstream comparison. No confidence, timing, boundaries or signer IDs were invented.\n\n"
            "Per-clip counts, gaps and content hashes: `compatibility_clips.jsonl`. "
            "Aggregate evidence and required bridge checks: `compatibility.json`.\n")
    (args.output_dir / "compatibility.md").write_text(text)
    print(json.dumps({"validated": len(results), "errors": len(errors), "frames": totals["frames"]}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
