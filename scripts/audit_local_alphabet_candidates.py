#!/usr/bin/env python3
"""Build a conservative A-Z video shortlist for a separate fingerspelling track."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_local_citizen100_candidates import (
    Candidate,
    inspect_candidate,
    safe_link,
    select_diverse,
)


HEX = re.compile(r"^[0-9a-f]{8}$", re.IGNORECASE)


def alphabet_source(path: Path, letter: str) -> tuple[str, str | None]:
    stem = path.stem
    lowered = stem.lower()
    if "__from_mariah_" in lowered:
        return "duplicate_mariah_copy", None
    if "__from_dwight_" in lowered:
        return "named_local_single_session", "dwight"
    if HEX.fullmatch(stem):
        return "local_hex_unknown_session", None
    if re.fullmatch(re.escape(letter) + r"_\d+", stem, re.IGNORECASE):
        return "local_numbered_single_session", "numbered"
    for prefix in ("msasl_", "signasl_", "wlasl_", "yt_"):
        if lowered.startswith(prefix):
            return prefix[:-1], None
    return "unknown", None


def cap_named_sessions(
    rows: list[tuple[Candidate, object, str | None]],
) -> list[tuple[Candidate, object]]:
    output: list[tuple[Candidate, object]] = []
    best_session: dict[str, tuple[Candidate, object]] = {}
    for candidate, descriptor, session in rows:
        if session is None:
            output.append((candidate, descriptor))
            continue
        current = best_session.get(session)
        if current is None or candidate.quality_score > current[0].quality_score:
            best_session[session] = (candidate, descriptor)
    output.extend(best_session.values())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=Path("data/raw_videos/ASL VIDEOS")
    )
    parser.add_argument(
        "--crop-root", type=Path, default=Path("data/local/ASL_hand_crops_av")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/local_alphabet_quality_audit")
    )
    parser.add_argument("--cap-per-letter", type=int, default=12)
    parser.add_argument("--minimum-quality-score", type=float, default=0.82)
    parser.add_argument("--minimum-appearance-distance", type=float, default=0.08)
    parser.add_argument("--materialize-symlinks", action="store_true")
    args = parser.parse_args()

    selected: list[Candidate] = []
    exclusions: dict[str, int] = {}
    inventory: dict[str, dict[str, int]] = {}
    inspected = 0
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        candidates: list[tuple[Candidate, object, str | None]] = []
        counts: dict[str, int] = {}
        inventory[letter] = counts
        for path in sorted((args.raw_root / letter).glob("*.mp4")):
            source, session = alphabet_source(path, letter)
            counts[source] = counts.get(source, 0) + 1
            if source not in {
                "local_hex_unknown_session",
                "local_numbered_single_session",
                "named_local_single_session",
            }:
                exclusions[source] = exclusions.get(source, 0) + 1
                continue
            candidate, descriptor, reason = inspect_candidate(
                path,
                args.crop_root / f"{letter}_{path.stem}.jpg",
                {
                    "canonical_label": letter,
                    "citizen_raw_gloss": letter,
                    "citizen_asl_lex_code": f"FINGERSPELL_{letter}",
                },
                source,
                args.minimum_quality_score,
            )
            inspected += 1
            if candidate is None or descriptor is None:
                reason = reason or "quality_failure"
                exclusions[reason] = exclusions.get(reason, 0) + 1
                continue
            candidates.append((candidate, descriptor, session))
        capped = cap_named_sessions(candidates)
        descriptors = {row.raw_path: descriptor for row, descriptor in capped}
        selected.extend(
            select_diverse(
                [row for row, _ in capped],
                descriptors,
                args.cap_per_letter,
                args.minimum_appearance_distance,
            )
        )

    if args.materialize_symlinks:
        for row in selected:
            source = Path(row.raw_path)
            safe_link(source, args.output_root / "raw" / row.canonical_label / source.name)
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "local A-Z quality/diversity shortlist for separate fingerspelling track",
        "training_eligible": False,
        "architecture_boundary": "separate fingerspelling model/head; never alias LETTER_I to lexical ME",
        "signer_warning": "appearance diversity is not signer identity",
        "cap_per_letter": args.cap_per_letter,
        "minimum_quality_score": args.minimum_quality_score,
        "minimum_appearance_distance": args.minimum_appearance_distance,
        "inspected_candidates": inspected,
        "selected_clips": len(selected),
        "selected_letters": len({row.canonical_label for row in selected}),
        "inventory": inventory,
        "exclusions": exclusions,
        "videos": [asdict(row) for row in selected],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / "candidate_selection.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "inspected": inspected,
                "clips": len(selected),
                "letters": output["selected_letters"],
                "exclusions": exclusions,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
