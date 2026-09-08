#!/usr/bin/env python3
"""Fail-closed admission audit for the ASL STEM Wiki manual-review pool."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile


STRONG_SUBJECTS = frozenset((3, 4, 9, 11, 12, 14))
L2_SUBJECTS = frozenset((1, 6, 7, 10, 13, 15, 16))
SUBJECT_RE = re.compile(r"^\s*Subject\s*#?\s*(\d+)\b", re.I)


def tokenize_gloss(value: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    depth = 0
    for char in value.strip():
        if char.isspace() and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(char)
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced gloss annotation: {value!r}")
    if depth:
        raise ValueError(f"unbalanced gloss annotation: {value!r}")
    if current:
        tokens.append("".join(current))
    return tokens


def explicit_subject_map(manual_rows, metadata) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in manual_rows:
        filename = row.get("filename", "").strip()
        match = SUBJECT_RE.search(row.get("Gloss Notes", ""))
        if not filename or not match or filename not in metadata:
            continue
        participant = metadata[filename]["participant"]
        subject = int(match.group(1))
        previous = result.get(participant)
        if previous is not None and previous != subject:
            raise ValueError(
                f"conflicting explicit subject IDs for {participant}: {previous}, {subject}"
            )
        result[participant] = subject
    return result


def event_key(token: str) -> tuple[str, str]:
    if token.startswith("fs-"):
        return token[3:].upper(), "fs"
    return token.upper(), "sign"


def pseudo_event_key(event) -> tuple[str, str]:
    return str(event[0]).upper(), "fs" if str(event[3]).lower() == "fs" else "sign"


def align_manual_to_pseudo(manual_tokens, candidates):
    best = {"candidate_index": None, "manual_to_pseudo": {}, "match_count": 0}
    manual_keys = [event_key(token) for token in manual_tokens]
    for candidate_index, candidate in enumerate(candidates):
        pseudo_keys = [pseudo_event_key(event) for event in candidate]
        rows, columns = len(manual_keys), len(pseudo_keys)
        scores = [[0] * (columns + 1) for _ in range(rows + 1)]
        for row in range(rows - 1, -1, -1):
            for column in range(columns - 1, -1, -1):
                if manual_keys[row] == pseudo_keys[column]:
                    scores[row][column] = 1 + scores[row + 1][column + 1]
                else:
                    scores[row][column] = max(
                        scores[row + 1][column], scores[row][column + 1]
                    )
        mapping = {}
        row = column = 0
        while row < rows and column < columns:
            if manual_keys[row] == pseudo_keys[column]:
                mapping[row] = column
                row += 1
                column += 1
            elif scores[row + 1][column] >= scores[row][column + 1]:
                row += 1
            else:
                column += 1
        candidate_score = (len(mapping), -candidate_index)
        best_score = (best["match_count"], -(best["candidate_index"] or 0))
        if candidate_score > best_score:
            best = {
                "candidate_index": candidate_index,
                "manual_to_pseudo": mapping,
                "match_count": len(mapping),
            }
    return best


def signer_status(participant: str, subjects: dict[str, int]) -> str:
    subject = subjects.get(participant)
    if subject in STRONG_SUBJECTS:
        return "source_review_strong"
    if subject in L2_SUBJECTS:
        return "l2_excluded"
    if subject is not None:
        return "source_review_unclassified"
    return "subject_mapping_unresolved"


def build_admission_rows(downloaded, subjects):
    return [
        {
            "filename": sample["filename"],
            "participant": sample["participant"],
            "explicit_subject_id": subjects.get(sample["participant"]),
            "signer_status": signer_status(sample["participant"], subjects),
            "video_path": sample["path"],
            "video_sha256": sample["sha256"],
            "raw_locked_glosses": sample["gloss_tokens"],
            "signer_quality_verified": False,
            "variant_verified": False,
            "boundary_verified": False,
            "training_eligible": False,
            "rejection_reasons": [
                "signer_quality_not_admitted",
                "exact_variant_unverified",
                "boundaries_unverified",
            ],
        }
        for sample in downloaded
    ]


def build_review_queue(downloaded, targets, pseudo_records, subjects):
    best_by_pair = {}
    for sample in downloaded:
        filename = sample["filename"]
        tokens = sample["gloss_tokens"]
        pseudo = pseudo_records.get(Path(filename).stem, {})
        candidates = pseudo.get("candidates", [])
        alignment = align_manual_to_pseudo(tokens, candidates)
        candidate = (
            candidates[alignment["candidate_index"]]
            if alignment["candidate_index"] is not None
            else []
        )
        for token_index, token in enumerate(tokens):
            target = targets.get(token)
            if target is None:
                continue
            pseudo_index = alignment["manual_to_pseudo"].get(token_index)
            event = candidate[pseudo_index] if pseudo_index is not None else None
            position = event[1] if event else None
            row = {
                "participant": sample["participant"],
                "explicit_subject_id": subjects.get(sample["participant"]),
                "signer_status": signer_status(sample["participant"], subjects),
                "raw_gloss": token,
                "canonical_label": target["canonical_label"],
                "citizen_asl_lex_code": target["citizen_asl_lex_code"],
                "filename": filename,
                "video_path": sample["path"],
                "video_sha256": sample["sha256"],
                "manual_token_index": token_index,
                "proposed_frame": position if isinstance(position, int) else None,
                "proposed_start_frame": position[0] if isinstance(position, list) else None,
                "proposed_end_frame": position[1] if isinstance(position, list) else None,
                "pseudo_confidence": float(event[2]) if event else None,
                "pseudo_candidate_index": alignment["candidate_index"],
                "pseudo_alignment_matches": alignment["match_count"],
                "signer_quality_decision": "",
                "variant_decision": "",
                "verified_start_frame": "",
                "verified_end_frame": "",
                "reviewer_notes": "",
                "signer_quality_verified": False,
                "variant_verified": False,
                "boundary_verified": False,
                "training_eligible": False,
            }
            key = (row["participant"], row["raw_gloss"])
            rank = (event is not None, row["pseudo_confidence"] or -1.0)
            old = best_by_pair.get(key)
            if old is None or rank > old[0]:
                best_by_pair[key] = (rank, row)
    return [best_by_pair[key][1] for key in sorted(best_by_pair)]


def load_pseudo_records(path: Path, needed: set[str]):
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as source:
            payload = json.load(io.TextIOWrapper(source, encoding="utf-8"))
    return {key: payload[key] for key in needed if key in payload}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_video(path: Path):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if decode.returncode:
        raise RuntimeError(f"full decode failed for {path}: {decode.stderr.strip()}")
    stream = json.loads(probe.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream["avg_frame_rate"],
        "decoded_frames": int(stream["nb_read_frames"]),
        "duration_seconds": float(stream.get("duration") or 0.0),
    }


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/local/asl_stem_wiki_bootstrap_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--acquisition-manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/asl_stem_wiki_manual_admission_v17"))
    args = parser.parse_args()

    manual_path = args.root / "42553_supp" / "ASL_STEM_Wiki_ManualAnnotations_CVPR-Final.csv"
    with manual_path.open(encoding="utf-8-sig", newline="") as handle:
        manual = list(csv.DictReader(handle))
    with (args.root / "videos.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = {row["video filename"]: row for row in csv.DictReader(handle)}
    acquisition_path = args.acquisition_manifest or args.root / "manual_candidate_manifest.json"
    acquisition = json.loads(acquisition_path.read_text())
    downloaded = acquisition["downloaded_videos"]
    manual_by_filename = {}
    for row in manual:
        filename = row.get("filename", "").strip()
        if filename and filename not in manual_by_filename:
            manual_by_filename[filename] = row
    for sample in downloaded:
        row = manual_by_filename[sample["filename"]]
        sample["gloss_tokens"] = tokenize_gloss(row["GLOSSED SENTENCE"])

    subjects = explicit_subject_map(manual, metadata)
    classes = json.loads(args.manifest.read_text())["classes"]
    targets = {row["citizen_raw_gloss"]: row for row in classes}
    needed = {Path(row["filename"]).stem for row in downloaded}
    pseudo = load_pseudo_records(
        args.root / "42553_supp" / "ASL_STEM_WIKI_pseudo_annotations.json.zip", needed
    )
    videos = []
    for sample in downloaded:
        path = Path(sample["path"])
        if sha256(path) != sample["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        videos.append({"filename": sample["filename"], **verify_video(path)})
    queue = build_review_queue(downloaded, targets, pseudo, subjects)
    admission_rows = build_admission_rows(downloaded, subjects)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "expert_review_queue.csv", queue)
    (args.output / "admission_manifest.json").write_text(
        json.dumps({"format": "slt_v17_admission_manifest", "rows": admission_rows}, indent=2) + "\n"
    )
    report = {
        "format": "slt_v17_asl_stem_wiki_manual_admission_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "downloaded_videos": len(downloaded),
        "fully_decoded_videos": len(videos),
        "explicit_subject_map": subjects,
        "source_review_strong_participants": sorted(
            participant for participant in subjects if signer_status(participant, subjects) == "source_review_strong"
        ),
        "source_review_l2_excluded_participants": sorted(
            participant for participant in subjects if signer_status(participant, subjects) == "l2_excluded"
        ),
        "unresolved_or_unclassified_participants": sorted(
            {row["participant"] for row in downloaded}
            - {participant for participant in subjects if signer_status(participant, subjects) == "source_review_strong"}
            - {participant for participant in subjects if signer_status(participant, subjects) == "l2_excluded"}
        ),
        "review_pairs": len(queue),
        "review_pairs_with_pseudo_position": sum(
            row["proposed_frame"] is not None or row["proposed_start_frame"] is not None
            for row in queue
        ),
        "eligible_spans": 0,
        "training_experiment_run": False,
        "gate_result": "blocked_pending_expert_signer_variant_boundary_review",
        "video_validation": videos,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "rit_evaluation_accessed": False,
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    status_counts = Counter(row["signer_status"] for row in queue)
    readme = f"""# ASL STEM Wiki manual admission audit

The downloaded pool is **not training eligible**. Source notes explicitly map only P13 to Subject 2 and P28 to Subject 7. The appendix classifies Subject 7 as possible L2 signing, so P28 is excluded. The other downloaded participant-to-subject mappings are unresolved, and none of the manual gloss strings proves the frozen ASL-LEX visual variant.

- Fully decoded videos: {len(videos)}/{len(downloaded)}
- Signer/gloss review pairs: {len(queue)}
- Pairs with a pseudo-position proposal: {report['review_pairs_with_pseudo_position']}
- Queue statuses: {dict(sorted(status_counts.items()))}
- Eligible verified spans: 0
- Training or protected evaluation run: no

Complete `expert_review_queue.csv` by entering signer quality, exact variant, and verified start/end frames. Pseudo positions are review aids only. A later admission step may build short participant-disjoint clips only from rows with all three approvals.

Run `venv/bin/python scripts/review_asl_stem_wiki_manual_v17.py` for the local
side-by-side video review UI. It writes approved decisions back to the queue.
"""
    (args.output / "README.md").write_text(readme)
    print(json.dumps({key: report[key] for key in ("fully_decoded_videos", "review_pairs", "review_pairs_with_pseudo_position", "eligible_spans", "gate_result")}, indent=2))


if __name__ == "__main__":
    main()
