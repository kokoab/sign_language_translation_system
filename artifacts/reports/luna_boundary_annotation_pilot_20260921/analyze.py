#!/usr/bin/env python3
"""Validate the one-Luna-per-clip pilot and compare it with source timestamps."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent


def load_annotation(path):
    value = json.loads(path.read_text())
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"expected one annotation: {path}")
        value = value[0]
    return value


def metrics(rows):
    usable = [row for row in rows if row["annotated"]]
    start = [abs(row["start_delta_seconds"]) for row in usable]
    end = [abs(row["end_delta_seconds"]) for row in usable]
    return dict(items=len(rows), annotated=len(usable),
                median_absolute_start_delta_seconds=statistics.median(start) if start else None,
                median_absolute_end_delta_seconds=statistics.median(end) if end else None,
                both_within_100ms=sum(a <= .10 and b <= .10 for a, b in zip(start, end)),
                both_within_200ms=sum(a <= .20 and b <= .20 for a, b in zip(start, end)))


def main():
    blind = {row["item"]: row for row in json.loads((HERE / "blind_manifest.json").read_text())}
    reference = {row["item"]: row for row in json.loads((HERE / "reference.json").read_text())}
    paths = sorted((HERE / "individual").glob("item_*.json"))
    if len(paths) != 24 or len(blind) != 24:
        raise ValueError(f"expected 24 individual annotations, found {len(paths)}")
    output = []
    for path in paths:
        annotation = load_annotation(path)
        item = str(annotation["item"])
        if item not in blind or item != path.stem:
            raise ValueError(f"item mismatch: {path}")
        confidence = str(annotation["confidence"])
        start_frame, end_frame = annotation.get("start_frame"), annotation.get("end_frame")
        annotated = start_frame is not None and end_frame is not None
        if annotated and not (0 <= int(start_frame) < int(end_frame) < 24):
            raise ValueError(f"invalid frame interval: {path}")
        times = blind[item]["frame_times_seconds"]
        start = float(times[int(start_frame)]) if annotated else None
        end = float(times[int(end_frame)]) if annotated else None
        edge_censored = bool(annotated and (int(start_frame) == 0 or int(end_frame) == 23))
        admitted = bool(annotated and confidence in {"high", "medium"} and not edge_censored)
        source = reference[item]
        output.append(dict(item=item, gloss=blind[item]["gloss"], confidence=confidence,
                           reason=annotation["reason"], start_frame=start_frame, end_frame=end_frame,
                           luna_start_seconds=start, luna_end_seconds=end, annotated=annotated,
                           edge_censored=edge_censored, admitted=admitted,
                           source_start_seconds=source["source_start_seconds"],
                           source_end_seconds=source["source_end_seconds"],
                           start_delta_seconds=None if not annotated else start-float(source["source_start_seconds"]),
                           end_delta_seconds=None if not annotated else end-float(source["source_end_seconds"]),
                           identity=source["identity"], source_item_id=source["source_item_id"],
                           video_path=source["video_path"]))
    admitted = [row for row in output if row["admitted"]]
    confidence = Counter(row["confidence"] for row in output)
    audit = dict(all=metrics(output), admitted=metrics(admitted), confidence=dict(confidence),
                 edge_censored=sum(row["edge_censored"] for row in output), admitted_items=len(admitted),
                 one_luna_per_clip=True, independent_consensus_available=False,
                 citizen_test_accessed=False)
    (HERE / "luna_annotations.json").write_text(json.dumps(output, indent=2) + "\n")
    (HERE / "admitted_annotations.json").write_text(json.dumps(admitted, indent=2) + "\n")
    (HERE / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (HERE / "verification.json").write_text(json.dumps(dict(
        status="passed", individual_luna_runs=24, unique_items=len({row["item"] for row in output}),
        valid_intervals=sum(row["annotated"] for row in output), provisional_admitted=len(admitted),
        one_luna_per_clip=True, independent_consensus_available=False,
        citizen_test_accessed=False), indent=2) + "\n")
    (HERE / "REPORT.md").write_text(
        "# One-Luna-per-clip boundary pilot\n\n"
        f"Twenty-four train-only signs were assigned to twenty-four separate Luna-low runs. Confidence: {dict(confidence)}. {audit['edge_censored']} annotations touched the first or last review frame and are treated as censored. {len(admitted)} medium/high, non-censored annotations are admitted as provisional supervision.\n\n"
        f"Across all annotated items, the median absolute difference from the source timestamps is {audit['all']['median_absolute_start_delta_seconds']*1000:.0f} ms at start and {audit['all']['median_absolute_end_delta_seconds']*1000:.0f} ms at end; {audit['all']['both_within_100ms']}/{audit['all']['annotated']} agree at both edges within 100 ms and {audit['all']['both_within_200ms']}/{audit['all']['annotated']} within 200 ms. Among admitted items, agreement is {audit['admitted']['both_within_100ms']}/{audit['admitted']['annotated']} within 100 ms and {audit['admitted']['both_within_200ms']}/{audit['admitted']['annotated']} within 200 ms.\n\n"
        "These are single-reviewer pseudo-labels, not ground truth. Source agreement measures consistency, not correctness. Do not scale or train on them unless this pilot demonstrates adequate consistency and edge coverage. Citizen test remained sealed.\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
