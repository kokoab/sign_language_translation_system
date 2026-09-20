#!/usr/bin/env python3
"""Build a conservative, traceable supervision manifest without copying data."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STRICT = ROOT / "artifacts/reports/clean_boundary_subset_20260920/curated_manifest.json"
SOURCE = ROOT / "artifacts/reports/o5s5_citizen100_v17/combined_supervision.json"
OUTPUT = HERE / "confident_supervision.json"
MIN_RAW_SAMPLES = 6
MIN_HAND_VISIBILITY = 0.80
MAX_DURATION_SECONDS = 1.20
MAX_TIMESTAMP_GAP_SECONDS = 0.08


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    strict = json.loads(STRICT.read_text(encoding="utf-8"))
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_rows = {str(row["source_item_id"]): row for row in source["rows"]}
    strict_cores = {str(row["identity"]): row for row in strict["eligible_cores"]}
    decisions = []
    accepted = []
    counts = Counter()
    class_signers = defaultdict(set)
    for item, rows in _by_item(strict["events"]).items():
        source_row = source_rows[item]
        archive_path = ROOT / source_row["archive_path"]
        with np.load(archive_path, allow_pickle=False) as archive:
            timestamps = archive["timestamps_seconds"].astype(np.float64, copy=False)
            raw = archive["raw_features"]
        for event in rows:
            identity = str(event["identity"])
            reasons = list(event["exclusions"])
            quality = None
            if identity in strict_cores:
                core = strict_cores[identity]
                keep = (timestamps >= float(core["start"]) - 1e-9) & (timestamps <= float(core["end"]) + 1e-9)
                target_times = timestamps[keep]
                target = raw[keep]
                hand_visibility = float((target[:, :42, 4] > 0).any(axis=1).mean())
                maximum_gap = float(np.diff(target_times).max()) if len(target_times) > 1 else float("inf")
                quality = dict(raw_samples=len(target_times), hand_visibility=hand_visibility,
                               maximum_timestamp_gap_seconds=maximum_gap,
                               duration_seconds=float(core["duration"]))
                if len(target_times) < MIN_RAW_SAMPLES:
                    reasons.append("fewer_than_six_raw_target_samples")
                if hand_visibility < MIN_HAND_VISIBILITY:
                    reasons.append("hand_visibility_below_80_percent")
                if float(core["duration"]) > MAX_DURATION_SECONDS:
                    reasons.append("duration_above_1_20_seconds")
                if maximum_gap > MAX_TIMESTAMP_GAP_SECONDS:
                    reasons.append("target_timestamp_gap_above_80ms")
            accepted_event = not reasons
            decision = {key: value for key, value in event.items() if key not in {"eligible", "exclusions"}}
            decision.update(accepted=accepted_event, exclusion_reasons=sorted(set(reasons)), quality=quality)
            decisions.append(decision)
            counts[(event["role"], event["source"], event["kind"], "accepted" if accepted_event else "excluded")] += 1
            if not accepted_event:
                continue
            core = strict_cores[identity]
            row = dict(identity=identity, role=event["role"], source=event["source"], signer_id=event["signer"],
                       source_item_id=item, archive_path=event["archive"], video_path=event["video"],
                       label=event["label"], target_index=core["target"], target_kind=event["kind"],
                       start_seconds=event["start"], end_seconds=event["end"], quality=quality,
                       source_crop_complete=event["complete_crop"],
                       boundary_supervision=dict(start_seconds=event["start"], end_seconds=event["end"],
                                                positive_edges_eligible=True,
                                                negative_regions_eligible=event["source"] != "o5s5"),
                       context_policy="exact core only for O5S5" if event["source"] == "o5s5"
                                      else "complete sign plus context clipped before neighbouring annotations")
            accepted.append(row)
            if event["kind"] == "known" and event["role"] == "train":
                class_signers[event["label"]].add(event["signer"])
    role_signers = {role: {row["signer_id"] for row in accepted if row["role"] == role}
                    for role in ("train", "validation")}
    if role_signers["train"] & role_signers["validation"]:
        raise ValueError("confident manifest has overlapping train/validation signers")
    class_coverage = {}
    for label in sorted({row["label"] for row in accepted if row["target_kind"] == "known"}):
        train = [row for row in accepted if row["target_kind"] == "known" and row["label"] == label and row["role"] == "train"]
        validation = [row for row in accepted if row["target_kind"] == "known" and row["label"] == label and row["role"] == "validation"]
        class_coverage[label] = dict(train_occurrences=len(train), validation_occurrences=len(validation),
                                     train_signers=sorted({row["signer_id"] for row in train}),
                                     validation_signers=sorted({row["signer_id"] for row in validation}))
    payload = dict(format="slt_confident_continuous_supervision_v17", version=1,
                   contract=dict(public_glosses=100, unknown_is_internal=True,
                                 gaps_are_transition_targets=False, model_correctness_used_for_filtering=False,
                                 minimum_raw_target_samples=MIN_RAW_SAMPLES,
                                 minimum_hand_visibility=MIN_HAND_VISIBILITY,
                                 maximum_duration_seconds=MAX_DURATION_SECONDS,
                                 maximum_target_timestamp_gap_seconds=MAX_TIMESTAMP_GAP_SECONDS,
                                 o5s5_negative_regions_eligible=False),
                   provenance=dict(strict_manifest=str(STRICT.relative_to(ROOT)), strict_manifest_sha256=sha(STRICT),
                                   source_manifest=str(SOURCE.relative_to(ROOT)), source_manifest_sha256=sha(SOURCE)),
                   rows=accepted, decisions=decisions, class_coverage=class_coverage,
                   audit=dict(annotated_events=len(decisions), accepted_events=len(accepted),
                              accepted_known=sum(row["target_kind"] == "known" for row in accepted),
                              accepted_unknown=sum(row["target_kind"] == "unknown" for row in accepted),
                              train_known_classes=len(class_signers),
                              train_known_single_signer_classes=sum(len(value) == 1 for value in class_signers.values()),
                              counts={"|".join(key): value for key, value in sorted(counts.items())},
                              train_validation_signer_disjoint=True, citizen_test_accessed=False))
    write_json(OUTPUT, payload)
    write_report(payload)
    verify(payload, source_rows)
    print(json.dumps(payload["audit"], indent=2))


def _by_item(rows):
    output = defaultdict(list)
    for row in rows:
        output[str(row["item"])].append(row)
    return output


def write_report(payload) -> None:
    audit = payload["audit"]
    accepted = Counter((row["role"], row["source"], row["target_kind"]) for row in payload["rows"])
    lines = ["# Confident-only continuous supervision", "",
             "This derived manifest leaves all source data untouched. It accepts only complete, non-overlapping annotations that passed the earlier raw-clock and both-edge checks, then requires at least six raw target observations, at least 80% target-frame hand visibility, no target timestamp gap above 80 ms, and duration no longer than 1.20 seconds.", "",
             "Model predictions were not used for filtering. Difficult valid signs therefore remain. Annotation gaps are not targets. O5S5 provides positive event cores and edges only; its unannotated surrounding regions cannot provide negative/background supervision.", "",
             f"Accepted **{audit['accepted_events']:,}/{audit['annotated_events']:,}** annotations: **{audit['accepted_known']:,} known** and **{audit['accepted_unknown']:,} unknown**. Continuous training covers **{audit['train_known_classes']}/100 classes**; {audit['train_known_single_signer_classes']} still have one continuous training signer.", "",
             "| Split/source/type | Accepted |", "| --- | ---: |"]
    for key, value in sorted(accepted.items()):
        lines.append(f"| {' / '.join(key)} | {value:,} |")
    lines += ["", "`confident_supervision.json` contains every accepted row and a decision with exclusion reasons for every source annotation. It is the only continuous supervision manifest recommended for the next controlled experiment.", "",
              "This is mechanical confidence in crop, timing, sampling and visibility. It is not a claim that every gloss or linguistic boundary was independently re-annotated by an ASL expert. Citizen test remained sealed.", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def verify(payload, source_rows) -> None:
    assert payload["audit"]["accepted_events"] == len(payload["rows"])
    assert len(payload["decisions"]) == payload["audit"]["annotated_events"]
    assert all(row["quality"]["raw_samples"] >= MIN_RAW_SAMPLES for row in payload["rows"])
    assert all(row["quality"]["hand_visibility"] >= MIN_HAND_VISIBILITY for row in payload["rows"])
    assert all(row["quality"]["duration_seconds"] <= MAX_DURATION_SECONDS for row in payload["rows"])
    assert all(row["quality"]["maximum_timestamp_gap_seconds"] <= MAX_TIMESTAMP_GAP_SECONDS for row in payload["rows"])
    assert all(row["identity"].split(":event:")[0] in source_rows for row in payload["rows"])
    assert all(row["source"] != "o5s5" or not row["boundary_supervision"]["negative_regions_eligible"] for row in payload["rows"])
    write_json(HERE / "verification.json", dict(status="passed", accepted_rows=len(payload["rows"]),
               all_quality_rules_checked=True, all_source_items_resolved=True,
               signer_disjoint=True, model_predictions_used=False, citizen_test_accessed=False))


if __name__ == "__main__":
    main()
