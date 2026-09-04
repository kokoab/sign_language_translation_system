#!/usr/bin/env python3
"""Build strict, timed NCSLGR supervision without guessing gloss aliases."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np


def main_gloss_events(text: str) -> list[dict[str, int | str]]:
    cells: list[str] = []
    collecting = False
    for line in text.splitlines():
        if line.startswith("main gloss\t"):
            collecting = True
            cells.extend(line.split("\t")[1:])
        elif collecting and line.startswith("\t"):
            cells.extend(line.split("\t")[1:])
        elif collecting:
            break
    events = []
    index = 0
    while index < len(cells):
        token = cells[index].strip()
        if not token or re.fullmatch(r"-?\d+", token):
            index += 1
            continue
        numbers = []
        cursor = index + 1
        while cursor < len(cells):
            value = cells[cursor].strip()
            if value and not re.fullmatch(r"-?\d+", value):
                break
            if value:
                numbers.append(int(value))
            cursor += 1
        if len(numbers) >= 2 and numbers[1] >= numbers[0]:
            events.append({"raw_gloss": token, "start": numbers[0], "end": numbers[1]})
        index = max(cursor, index + 1)
    return events


def archive_index(root: Path) -> dict[str, tuple[Path, int]]:
    output = {}
    for path in sorted(root.glob("ncslgr_*/*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
        item_id = str(metadata["source_item_id"])
        output[item_id] = (path, int(metadata["sampled_source_frames"]))
    return output


def scaled_frame(value: int, utterance_start: int, utterance_end: int, frames: int) -> int:
    fraction = (value - utterance_start) / max(utterance_end - utterance_start, 1)
    return int(np.clip(round(fraction * frames), 0, frames))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/local/ncslgr_continuous_v17_source"
    ))
    parser.add_argument("--landmarks", type=Path, default=Path(
        "data/local/how2sign_transition_landmarks_v17"
    ))
    parser.add_argument("--classes", type=Path, default=Path(
        "active/v17/citizen100_manifest.json"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "active/v17/ncslgr_supervised_manifest_v17.json"
    ))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    source = json.loads((args.source / "manifest.json").read_text())
    classes = json.loads(args.classes.read_text())
    labels = {str(row["canonical_label"]) for row in classes["classes"]}
    archives = archive_index(args.landmarks)
    rows, strict_counts, participant_counts = [], Counter(), Counter()
    for item in source["items"]:
        collection, source_id = str(item["collection"]), str(item["source_id"])
        item_id = f"ncslgr:{collection}:{source_id}"
        if item_id not in archives:
            raise FileNotFoundError(f"missing extracted archive for {item_id}")
        archive, frame_count = archives[item_id]
        annotation = Path(item["annotation_path"])
        events = main_gloss_events(annotation.read_text(encoding="latin1"))
        utterance_start, utterance_end = int(item["start_frame"]), int(item["end_frame"])
        timed = []
        for event in events:
            start = scaled_frame(int(event["start"]), utterance_start, utterance_end, frame_count)
            end = scaled_frame(int(event["end"]), utterance_start, utterance_end, frame_count)
            if end <= start:
                end = min(frame_count, start + 1)
            raw = str(event["raw_gloss"])
            canonical = raw if raw in labels else None
            timed.append({
                **event,
                "source_start_frame": start,
                "source_end_frame_exclusive": end,
                "canonical_label": canonical,
            })
            if canonical:
                strict_counts[canonical] += 1
        known = [event["canonical_label"] for event in timed if event["canonical_label"]]
        participant = str(item["participant_id"])
        role = "train" if participant == "BENJAMIN_JAMES_BAHAN" else "validation"
        participant_counts[f"{participant}:{role}"] += 1
        rows.append({
            "source_item_id": item_id,
            "participant_id": participant,
            "role": role,
            "archive_path": archive.as_posix(),
            "annotation_path": annotation.as_posix(),
            "source_frame_count": frame_count,
            "events": timed,
            "strict_target_sequence": known,
            "has_strict_target": bool(known),
        })

    payload = {
        "format": "slt_ncslgr_strict_timed_supervision_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "matching_policy": "raw main-gloss token must exactly equal a frozen canonical label",
        "timing_policy": "SignStream bounds scaled from utterance interval to decoded video frames",
        "split_policy": "participant-disjoint: Benjamin=train, Norma=validation",
        "rows": rows,
        "utterances": len(rows),
        "utterances_with_strict_target": sum(row["has_strict_target"] for row in rows),
        "strict_target_occurrences": sum(strict_counts.values()),
        "strict_target_class_count": len(strict_counts),
        "strict_target_counts": dict(sorted(strict_counts.items())),
        "participant_role_counts": dict(sorted(participant_counts.items())),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
