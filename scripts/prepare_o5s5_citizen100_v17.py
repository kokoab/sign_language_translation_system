#!/usr/bin/env python3
"""Prepare exact Signbank-ID O5S5 supervision for the frozen Citizen-100 head."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import subprocess

from scripts.audit_open_asl_alternatives_v17 import O5S5_PAIRS, digest, eaf_events
from scripts.prepare_asllrp_continuous_citizen100_v17 import load_targets, sha256_file


ROOT = Path("data/local/open_asl_alternatives_20260913/o5s5")
MANIFEST = Path("active/v17/citizen100_manifest.json")
ASLLEX = Path("data/local/dataset_metadata/asllex2_official/signdata.csv")
REPORT = Path("artifacts/reports/o5s5_citizen100_v17")
FINAL_FRAME_EXCLUSIONS = {"029": 7}


def video_duration(path: Path) -> float:
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], text=True).strip())


def _overlap(left: dict, right: dict) -> int:
    return max(0, min(left["end_ms"], right["end_ms"]) - max(left["start_ms"], right["start_ms"]))


def deduplicate_hand_events(events: list[dict]) -> list[dict]:
    """Merge one left/right copy of the same overlapping ID gloss."""
    candidates = []
    for left_index, left in enumerate(events):
        for right_index, right in enumerate(events):
            if left["tier"] >= right["tier"] or left["value"] != right["value"]:
                continue
            overlap = _overlap(left, right)
            if not overlap:
                continue
            union = max(left["end_ms"], right["end_ms"]) - min(left["start_ms"], right["start_ms"])
            candidates.append((overlap / union, overlap, left_index, right_index))
    paired: set[int] = set()
    pairs: dict[int, int] = {}
    for _, _, left_index, right_index in sorted(candidates, reverse=True):
        if left_index in paired or right_index in paired:
            continue
        paired.update((left_index, right_index))
        pairs[left_index] = right_index

    output = []
    for index, event in enumerate(events):
        if index in paired and index not in pairs:
            continue
        other = events[pairs[index]] if index in pairs else None
        output.append({
            "id_gloss": event["value"],
            "start_ms": min(event["start_ms"], other["start_ms"]) if other else event["start_ms"],
            "end_ms": max(event["end_ms"], other["end_ms"]) if other else event["end_ms"],
            "source_tiers": "+".join(sorted({event["tier"], other["tier"]} if other else {event["tier"]})),
            "tier_event_count": 2 if other else 1,
        })
    return sorted(output, key=lambda row: (row["start_ms"], row["end_ms"], row["id_gloss"]))


def choose_validation_signer(rows: list[dict]) -> str:
    """Choose one signer with broad exact-ID coverage shared by the other signers."""
    signers = sorted({row["signer_id"] for row in rows})
    scored = []
    for signer in signers:
        own = {row["canonical_label"] for row in rows if row["signer_id"] == signer}
        other = {row["canonical_label"] for row in rows if row["signer_id"] != signer}
        events = sum(row["signer_id"] == signer for row in rows)
        scored.append((len(own & other), -len(own - other), events, signer))
    return max(scored)[-1]


def build(root: Path, manifest: Path, asllex: Path) -> tuple[list[dict], list[dict], dict]:
    targets = load_targets(manifest, asllex)
    by_id = {row["signbank_annotation_id"]: row for row in targets if row["signbank_annotation_id"]}
    if len(by_id) != sum(bool(row["signbank_annotation_id"]) for row in targets):
        raise ValueError("frozen Citizen classes have duplicate non-empty Signbank IDs")

    sources, occurrences = [], []
    for session, video_name in O5S5_PAIRS.items():
        eaf = next((root / "annotations").glob(f"O5S5_{session}_*.eaf"))
        video = root / "videos" / video_name
        signer, tier_counts, tier_events = eaf_events(eaf)
        unique = deduplicate_hand_events(tier_events)
        duration_seconds = video_duration(video)
        duration_ms = duration_seconds * 1000
        if max(row["end_ms"] for row in unique) > duration_ms + 250:
            raise ValueError(f"{session}: annotation exceeds video duration")
        source_id = f"O5S5_{session}_{signer.replace(' ', '_')}"
        source = {
            "source_item_id": source_id, "session": session, "signer_id": signer,
            "video_path": str(video), "video_sha256": digest(video),
            "eaf_path": str(eaf), "eaf_sha256": digest(eaf),
            "duration_seconds": duration_seconds,
            "tier_events": sum(tier_counts.values()), "unique_occurrences": len(unique),
            "intervals": unique,
        }
        if session in FINAL_FRAME_EXCLUSIONS:
            source["exclude_final_frames"] = FINAL_FRAME_EXCLUSIONS[session]
        sources.append(source)
        for ordinal, event in enumerate(unique):
            target = by_id.get(event["id_gloss"])
            if target is None:
                continue
            occurrences.append({
                "source_item_id": source_id, "session": session, "signer_id": signer,
                "ordinal": ordinal, "video_path": str(video),
                "start_seconds": event["start_ms"] / 1000,
                "end_seconds": event["end_ms"] / 1000,
                "source_tiers": event["source_tiers"],
                "tier_event_count": event["tier_event_count"], **target,
                "match_contract": "exact SignBankAnnotationID equality",
                "positive_training_eligible": True, "background_training_eligible": False,
            })

    validation_signer = choose_validation_signer(occurrences)
    for row in occurrences:
        row["role"] = "validation" if row["signer_id"] == validation_signer else "train"
    for source in sources:
        source["role"] = "validation" if source["signer_id"] == validation_signer else "train"

    coverage = defaultdict(lambda: {"events": 0, "signers": set()})
    for row in occurrences:
        coverage[row["canonical_label"]]["events"] += 1
        coverage[row["canonical_label"]]["signers"].add(row["signer_id"])
    audit = {
        "format": "o5s5_citizen100_exact_signbank_v17", "version": 1,
        "source": "https://osf.io/769sw/",
        "documentation": "https://ida.gallaudet.edu/o5s5/index.html",
        "license": "CC BY-NC-SA 4.0",
        "locked_manifest": str(manifest), "locked_manifest_sha256": sha256_file(manifest),
        "asllex": str(asllex), "asllex_sha256": sha256_file(asllex),
        "source_videos": len(sources), "signers": sorted({row["signer_id"] for row in sources}),
        "validation_signer": validation_signer, "train_signers": sorted({row["signer_id"] for row in sources if row["role"] == "train"}),
        "signer_disjoint": True,
        "tier_events": sum(row["tier_events"] for row in sources),
        "unique_occurrences": sum(row["unique_occurrences"] for row in sources),
        "deduplicated_tier_copies": sum(row["tier_events"] - row["unique_occurrences"] for row in sources),
        "exact_locked_occurrences": len(occurrences),
        "exact_locked_classes": len(coverage),
        "role_counts": dict(Counter(row["role"] for row in occurrences)),
        "signer_counts": dict(Counter(row["signer_id"] for row in occurrences)),
        "coverage": {label: {"events": item["events"], "signers": sorted(item["signers"])} for label, item in sorted(coverage.items())},
        "empty_signbank_id_classes": [row["canonical_label"] for row in targets if not row["signbank_annotation_id"]],
        "positive_training_eligible": True,
        "background_training_eligible": False,
        "background_exclusion_reason": "O5S5 ID-gloss completeness is not established; unannotated gaps cannot be labeled as no-sign.",
        "source_decode_exclusions": FINAL_FRAME_EXCLUSIONS,
        "protected_test_accessed": False,
    }
    return sources, occurrences, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--asllex", type=Path, default=ASLLEX)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    sources, occurrences, audit = build(args.root, args.manifest, args.asllex)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    with (args.output / "exact_occurrences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(occurrences[0]))
        writer.writeheader()
        writer.writerows(occurrences)
    readme = f"""# O5S5 exact Citizen-100 supervision\n\nSix public O5S5 narratives provide {audit['exact_locked_occurrences']} deduplicated occurrences across {audit['exact_locked_classes']} frozen classes and {len(audit['signers'])} signers. Labels are admitted only by exact equality between the frozen class's official ASL-LEX `SignBankAnnotationID` and the O5S5 hand-tier ID gloss.\n\n`{audit['validation_signer']}` is reserved as a whole validation signer; the other five signers are training-only. O5S5 gaps are excluded from background training because annotation completeness is not established. The source is public under CC BY-NC-SA 4.0; no access request is required.\n\n- `exact_occurrences.csv`: authoritative positive-event manifest\n- `sources.json`: video/EAF provenance and every deduplicated ID-gloss interval\n- `audit.json`: counts, hashes, split, exclusions, and coverage\n"""
    (args.output / "README.md").write_text(readme)
    print(json.dumps({key: audit[key] for key in ("exact_locked_classes", "exact_locked_occurrences", "validation_signer", "role_counts")}, indent=2))


if __name__ == "__main__":
    main()
