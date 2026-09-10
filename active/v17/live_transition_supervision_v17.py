#!/usr/bin/env python3
"""Derive conservative transition-only CTC supervision from ASLLRP annotations.

Only the interior of a gap between two annotated signs is supervised as CTC blank.
Annotations that are outside the locked 100-sign vocabulary remain ``__OTHER__``
for the purpose of excluding blank targets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OTHER = "__OTHER__"
OTHER_CTC_INDEX = 101
CTC_BINS_PER_WINDOW = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ctc_bin_intervals(window: dict[str, Any], bins: int = CTC_BINS_PER_WINDOW) -> list[tuple[float, float]]:
    """Return equal-duration CTC-bin intervals for one accepted live window."""
    start, end = float(window["start"]), float(window["end"])
    if bins <= 0 or end <= start:
        raise ValueError("invalid accepted live window")
    step = (end - start) / bins
    return [(start + step * index, start + step * (index + 1)) for index in range(bins)]


def interior_gaps(intervals: list[dict[str, Any]], guard_seconds: float) -> list[tuple[float, float]]:
    """Gaps strictly between signs, with a guard on both adjacent signs."""
    if guard_seconds < 0:
        raise ValueError("guard_seconds must be non-negative")
    ordered = sorted((float(x["start_seconds"]), float(x["end_seconds"])) for x in intervals)
    gaps: list[tuple[float, float]] = []
    right = None
    for start, end in ordered:
        if end <= start:
            raise ValueError("invalid annotation interval")
        if right is not None and start > right:
            left, gap_right = right + guard_seconds, start - guard_seconds
            if gap_right > left:
                gaps.append((left, gap_right))
        right = max(right if right is not None else end, end)
    return gaps


def blank_positions_for_windows(
    windows: list[dict[str, Any]], intervals: list[dict[str, Any]], guard_seconds: float,
) -> tuple[list[int], list[tuple[float, float]]]:
    """Mark only full CTC bins that are inside guarded, annotation-interior gaps."""
    gaps = interior_gaps(intervals, guard_seconds)
    positions: list[int] = []
    accepted = [window for window in windows if window.get("accepted")]
    for window_index, window in enumerate(accepted):
        for bin_index, (left, right) in enumerate(ctc_bin_intervals(window)):
            if any(left >= gap_left and right <= gap_right for gap_left, gap_right in gaps):
                positions.append(window_index * CTC_BINS_PER_WINDOW + bin_index)
    return positions, gaps


def known_core_positions_for_windows(
    windows: list[dict[str, Any]], intervals: list[dict[str, Any]], guard_seconds: float = 1.0 / 30.0,
) -> dict[str, int]:
    """Map full guarded known-sign CTC bins to their 1..100 targets.

    OTHER and every bin touched by a second annotation are deliberately excluded.
    """
    if guard_seconds < 0:
        raise ValueError("guard_seconds must be non-negative")
    output: dict[str, int] = {}
    accepted = [window for window in windows if window.get("accepted")]
    for window_index, window in enumerate(accepted):
        for bin_index, (left, right) in enumerate(ctc_bin_intervals(window)):
            matching = [event for event in intervals if left >= float(event["start_seconds"]) + guard_seconds and right <= float(event["end_seconds"]) - guard_seconds]
            if len(matching) != 1 or int(matching[0]["ctc_index"]) == OTHER_CTC_INDEX:
                continue
            if any(event is not matching[0] and left < float(event["end_seconds"]) and right > float(event["start_seconds"]) for event in intervals):
                continue
            output[str(window_index * CTC_BINS_PER_WINDOW + bin_index)] = int(matching[0]["ctc_index"])
    return output


def load_supervision(path: Path | str = ROOT / "artifacts/reports/stage2_v17_revisable_v1/supervision.json") -> dict[str, Any]:
    """Load the generated item-id mapping and validate its format."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "slt_stage2_live_transition_supervision_v17" or payload.get("version") != 1:
        raise ValueError("unsupported transition supervision payload")
    if not isinstance(payload.get("items"), dict):
        raise ValueError("transition supervision has no item mapping")
    return payload


def _load_annotation_rows(path: Path) -> tuple[list[dict[str, str]], int]:
    from scripts.prepare_asllrp_continuous_citizen100_v17 import read_sentence_csv
    rows, rejected = read_sentence_csv(path)
    # The source parser has already rejected malformed rows.  They cannot provide
    # targets, but valid rows remain usable and the count is retained in the audit.
    return [row for row in rows if row.get("Hidden", "F") != "T"], len(rejected)


def _known_variants() -> dict[str, tuple[str, int]]:
    from scripts.prepare_asllrp_continuous_citizen100_v17 import load_targets
    targets = load_targets(
        ROOT / "active/v17/citizen100_manifest.json",
        ROOT / "data/local/dataset_metadata/asllex2_official/signdata.csv",
    )
    return {
        str(row["signbank_annotation_id"]): (str(row["canonical_label"]), int(row["class_index"]) + 1)
        for row in targets if row["signbank_annotation_id"]
    }


def _events_for_span(
    span: dict[str, Any], rows_by_parent: dict[str, list[dict[str, str]]], known: dict[str, tuple[str, int]],
) -> tuple[list[dict[str, Any]], bool]:
    """All signs touching a crop, expressed in the emitted-video timebase.

    ``complete`` is false for a sign cut by a crop edge; that makes prefix labels
    unavailable even though the clipped interval still blocks blank supervision.
    """
    filename = str(span["utterance_video_filename"])
    source_rows = rows_by_parent.get(filename, [])
    utterance_starts = {int(row["Start frame of the containing utterance"]) for row in source_rows}
    if len(utterance_starts) != 1:
        raise ValueError(f"{filename}: ambiguous ASLLRP utterance start")
    utterance_start = next(iter(utterance_starts))
    global_start = utterance_start + int(span["crop_start_frame_local"])
    global_end = utterance_start + int(span["crop_end_frame_local"])
    try:
        fps = float(Fraction(str(span["frame_rate"])))
    except (KeyError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{filename}: invalid span frame_rate") from error
    if fps <= 0:
        raise ValueError(f"{filename}: non-positive span frame_rate")
    events: list[dict[str, Any]] = []
    complete = True
    for row in source_rows:
        start, end = int(row["Start frame of the sign video"]), int(row["End frame of the sign video"])
        if end < global_start or start > global_end:
            continue
        variant = row["Entry/variant gloss label"]
        occurrence = row["Occurrence label"].rstrip("+")
        label, ctc_index = known.get(variant, (OTHER, OTHER_CTC_INDEX))
        if label != OTHER and occurrence != variant:
            label, ctc_index = OTHER, OTHER_CTC_INDEX
        clipped_start, clipped_end = max(start, global_start), min(end, global_end)
        if clipped_start != start or clipped_end != end:
            complete = False
        events.append(dict(
            start_seconds=(clipped_start - global_start) / fps,
            end_seconds=(clipped_end - global_start + 1) / fps,
            label=label, ctc_index=ctc_index,
            annotation_start_frame_global=start, annotation_end_frame_global=end,
        ))
    events.sort(key=lambda event: (event["start_seconds"], event["end_seconds"]))
    return events, complete


def _collapse_other_indices(events: list[dict[str, Any]]) -> list[int]:
    values: list[int] = []
    for event in events:
        value = int(event["ctc_index"])
        if value == OTHER_CTC_INDEX and values and values[-1] == OTHER_CTC_INDEX:
            continue
        values.append(value)
    return values


def _ctc_required_steps(values: list[int]) -> int:
    """A repeated adjacent target needs one intervening CTC blank."""
    return len(values) + sum(left == right for left, right in zip(values, values[1:]))


def _prefix_targets(
    windows: list[dict[str, Any]], events: list[dict[str, Any]], expected: list[int], complete: bool,
) -> dict[str, list[int]]:
    if not complete or _collapse_other_indices(events) != expected:
        return {}
    output: dict[str, list[int]] = {}
    accepted_count = 0
    for window in windows:
        if not window.get("accepted"):
            continue
        accepted_count += 1
        endpoint = float(window["end"])
        if any(float(event["start_seconds"]) <= endpoint < float(event["end_seconds"]) for event in events):
            continue
        completed = [event for event in events if float(event["end_seconds"]) <= endpoint]
        prefix = _collapse_other_indices(completed)
        if prefix and _ctc_required_steps(prefix) <= accepted_count * CTC_BINS_PER_WINDOW:
            output[str(accepted_count)] = prefix
    return output


def _span_maps() -> dict[str, dict[str, Any]]:
    paths = [
        ROOT / "data/local/asllrp_contiguous_phrases_v17/manifest.json",
        ROOT / "data/local/asllrp_other_ctc_v17/manifest.json",
    ]
    output: dict[str, dict[str, Any]] = {}
    for path in paths:
        for span in json.loads(path.read_text(encoding="utf-8"))["spans"]:
            filename = str(span["utterance_video_filename"])
            index = int(span["span_index_in_utterance"])
            prefix = "asllrp_other_ctc" if span["source"] == "asllrp_other_ctc" else "asllrp"
            stem = filename.removesuffix(".mp4") if prefix == "asllrp_other_ctc" else filename
            output[f"{prefix}:{stem}:span{index:02d}"] = span
    return output


def build_supervision(cache: dict[str, Any], frozen: dict[str, Any], annotation_path: Path, guard_seconds: float = 0.10, core_supervision: bool = False) -> dict[str, Any]:
    if guard_seconds != 0.10:
        raise ValueError("the frozen contract uses a 0.10-second transition guard")
    frozen_rows = {str(row["source_item_id"]): row for row in frozen["rows"]}
    if set(frozen_rows) != {str(row["item_id"]) for row in cache["rows"]}:
        raise ValueError("cache and frozen input item ids disagree")
    by_parent: dict[str, list[dict[str, str]]] = {}
    annotation_rows, rejected_rows = _load_annotation_rows(annotation_path)
    for row in annotation_rows:
        by_parent.setdefault(row["Utterance video filename"], []).append(row)
    for rows in by_parent.values():
        rows.sort(key=lambda row: int(row["Start frame of the sign video"]))
    known, spans = _known_variants(), _span_maps()
    items: dict[str, Any] = {}
    audit = Counter()
    for cache_row in cache["rows"]:
        item_id = str(cache_row["item_id"])
        source_row = frozen_rows[item_id]
        entry: dict[str, Any] = dict(role=source_row["role"], source=source_row["source"], blank_positions=[], sign_intervals=[], prefix_targets={}, eligible_blank_frame_bins=0, conservative_gap_seconds=0.0)
        if core_supervision:
            entry["known_core_positions"] = {}
        span = spans.get(item_id)
        if span is None:
            entry["annotation_status"] = "unavailable_non_asllrp_source"
            items[item_id] = entry
            audit["unavailable_non_asllrp"] += 1
            continue
        events, complete = _events_for_span(span, by_parent, known)
        if not events:
            raise ValueError(f"{item_id}: span has no ASLLRP annotations")
        expected = [int(value) + 1 for value in source_row["target_indices"]]
        blanks, gaps = blank_positions_for_windows(cache_row["windows"], events, guard_seconds)
        entry.update(annotation_status="available", sign_intervals=events, blank_positions=blanks,
                     prefix_targets=_prefix_targets(cache_row["windows"], events, expected, complete),
                     eligible_blank_frame_bins=len(blanks), conservative_gap_seconds=sum(right - left for left, right in gaps),
                     full_clip_alignment_matches=bool(complete and _collapse_other_indices(events) == expected),
                     annotation_crop_complete=complete)
        if core_supervision:
            entry["known_core_positions"] = known_core_positions_for_windows(cache_row["windows"], events)
        items[item_id] = entry
        audit["annotated_asllrp"] += 1
        audit["blank_positions"] += len(blanks)
        audit["prefix_targets"] += len(entry["prefix_targets"])
        audit[f"{source_row['role']}_blank_positions"] += len(blanks)
        audit[f"{source_row['role']}_annotated_asllrp"] += 1
        if core_supervision:
            audit["known_core_positions"] += len(entry["known_core_positions"])
            audit[f"{source_row['role']}_known_core_positions"] += len(entry["known_core_positions"])
        if entry["full_clip_alignment_matches"]:
            audit["alignment_matches"] += 1
    contract = dict(ctc_bins_per_accepted_window=CTC_BINS_PER_WINDOW, guard_seconds=guard_seconds,
                    blank_definition="only full CTC bins inside gaps strictly between all annotated signs, guarded from both signs",
                    other_policy="unlocked ASLLRP annotations map to OTHER=101 and exclude blank; OTHER is never labeled blank",
                    prefix_definition="only nonempty prefixes at accepted-window endpoints outside a sign, after full clip alignment including collapsed consecutive OTHER")
    if core_supervision:
        contract["known_core_definition"] = "full CTC bins inside one known 1..100 annotation, guarded 1/30 second from both edges and not overlapping any other annotation"
    return dict(
        format="slt_stage2_live_transition_supervision_v17", version=1,
        contract=contract,
        inputs=dict(cache_sha256=None, frozen_inputs_sha256=None, annotation_csv=annotation_path.as_posix(), annotation_csv_sha256=sha256(annotation_path), annotation_rows_rejected=rejected_rows),
        items=items, audit=dict(audit),
        limitations=["Blank supervision is sparse and covers only conservative interior annotation gaps.", "CTC blank remains a no-emission symbol; this does not assert a physical transition class.", "No annotation-derived targets are emitted for local phrase videos or protected evaluation data."],
        protected_test_accessed=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/reports/stage2_v17_revisable_v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/reports/stage2_v17_live_matched_v1/cache.json")
    parser.add_argument("--frozen-inputs", type=Path, default=ROOT / "artifacts/reports/stage2_v17_live_matched_v1/frozen_inputs.json")
    parser.add_argument("--annotations", type=Path, default=ROOT / "data/local/dataset_metadata/asllrp_signbank/asllrp_sentence_signs_2025_06_28.csv")
    parser.add_argument("--core-supervision", action="store_true", help="add guarded known-sign core targets")
    args = parser.parse_args()
    payload = build_supervision(json.loads(args.cache.read_text()), json.loads(args.frozen_inputs.read_text()), args.annotations, core_supervision=args.core_supervision)
    payload["inputs"]["cache_sha256"] = sha256(args.cache)
    payload["inputs"]["frozen_inputs_sha256"] = sha256(args.frozen_inputs)
    args.report.mkdir(parents=True, exist_ok=True)
    output = args.report / "supervision.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
