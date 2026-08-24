#!/usr/bin/env python3
"""Build exact-variant Stage 1 and Stage 2 manifests from ASLLRP sentence CSVs.

Admission is intentionally fail-closed.  A segmented sign is a Stage 1 candidate only
when the frozen Citizen class resolves through ASL-LEX to a non-empty Sign Bank
annotation ID, the ASLLRP entry/variant is exactly that ID, and the occurrence differs
only by documented trailing repetition markers.  English-label normalization is never
used to admit a row.

For Stage 2, target-bearing utterances are inventoried, but an utterance is marked CTC
eligible only when every visible non-gesture token maps through the same exact contract
and at least two target tokens remain.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


VIDEO_BASE = "https://dai.cs.rutgers.edu/ss3front"
NUMERIC_FIELDS = (
    "Video ID number",
    "Start frame of the sign video",
    "End frame of the sign video",
    "Start frame of the containing utterance",
    "End frame of the containing utterance",
)
REQUIRED_FIELDS = {
    "Video ID number",
    "Main entry gloss label",
    "Entry/variant gloss label",
    "Occurrence label",
    "Start frame of the sign video",
    "End frame of the sign video",
    "Start frame of the containing utterance",
    "End frame of the containing utterance",
    "Sign video filename",
    "Utterance video filename",
    "Source collection",
    "Utterance number",
    "Master video filename",
    "Sign type",
    "Class label",
    "Hidden",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_row(row: dict[str | None, Any]) -> dict[str, str]:
    return {
        str(key).strip(): str(value or "").strip()
        for key, value in row.items()
        if key is not None and str(key).strip()
    }


def read_sentence_csv(path: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {str(field).strip() for field in (reader.fieldnames or []) if str(field).strip()}
        missing = sorted(REQUIRED_FIELDS - fields)
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        rows: list[dict[str, str]] = []
        rejected: list[dict[str, Any]] = []
        for line_number, source in enumerate(reader, start=2):
            row = clean_row(source)
            if not any(row.values()):
                continue
            invalid = [field for field in NUMERIC_FIELDS if not row.get(field, "").isdigit()]
            if invalid:
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "nonnumeric required frame/id field",
                        "invalid_fields": invalid,
                        "video_id": row.get("Video ID number", ""),
                        "entry_variant": row.get("Entry/variant gloss label", ""),
                    }
                )
                continue
            if not row["Sign video filename"].lower().endswith(".mp4"):
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "unexpected sign video filename",
                        "video_id": row["Video ID number"],
                        "sign_video_filename": row["Sign video filename"],
                    }
                )
                continue
            if not row["Utterance video filename"].lower().endswith(".mp4"):
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "unexpected utterance video filename",
                        "video_id": row["Video ID number"],
                        "utterance_video_filename": row["Utterance video filename"],
                    }
                )
                continue
            rows.append(row)
    return rows, rejected


def load_targets(manifest_path: Path, asllex_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with asllex_path.open(encoding="latin-1", newline="") as handle:
        asllex = {row["Code"]: row for row in csv.DictReader(handle)}
    targets: list[dict[str, Any]] = []
    for item in sorted(manifest["classes"], key=lambda row: int(row["class_index"])):
        code = str(item["citizen_asl_lex_code"])
        if code not in asllex:
            raise ValueError(f"Citizen ASL-LEX code is absent from official table: {code}")
        targets.append(
            {
                "class_index": int(item["class_index"]),
                "canonical_label": str(item["canonical_label"]),
                "citizen_raw_gloss": str(item["citizen_raw_gloss"]),
                "citizen_asl_lex_code": code,
                "asllex_entry_id": str(asllex[code]["EntryID"]).strip(),
                "signbank_annotation_id": str(asllex[code]["SignBankAnnotationID"]).strip(),
            }
        )
    return targets


def occurrence_matches(annotation: str, occurrence: str) -> bool:
    return bool(annotation) and occurrence.rstrip("+") == annotation


def signer_id(source: str, collection: str) -> str:
    if source == "rit":
        match = re.search(r"P\d+", collection.upper())
        if not match:
            raise ValueError(f"cannot derive RIT participant from {collection!r}")
        return f"RIT_{match.group(0)}"
    if re.match(r"^\d+-Ben-", collection, re.I):
        return "BENJAMIN_JAMES_BAHAN"
    match = re.match(r"^([A-Za-z]+)_", collection)
    if not match:
        raise ValueError(f"cannot derive ASLLRP participant from {collection!r}")
    return match.group(1).upper()


def video_url(filename: str) -> str:
    return f"{VIDEO_BASE}/{filename}"


def select_stage1(
    source: str,
    rows: list[dict[str, str]],
    targets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_variant[row["Entry/variant gloss label"]].append(row)
    selected: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    split_role = "train_candidate" if source == "asllrp" else "external_evaluation_reserved"
    for target in targets:
        annotation = str(target["signbank_annotation_id"])
        candidates = [
            row
            for row in by_variant.get(annotation, [])
            if row.get("Hidden", "F") != "T"
            and occurrence_matches(annotation, row["Occurrence label"])
        ]
        summary.append(
            {
                "class_index": target["class_index"],
                "canonical_label": target["canonical_label"],
                "signbank_annotation_id": annotation,
                "candidate_signs": len(candidates),
            }
        )
        for row in candidates:
            selected.append(
                {
                    "source": source,
                    "split_role": split_role,
                    "signer_id": signer_id(source, row["Source collection"]),
                    **target,
                    "asllrp_video_id": row["Video ID number"],
                    "main_entry": row["Main entry gloss label"],
                    "entry_variant": row["Entry/variant gloss label"],
                    "occurrence": row["Occurrence label"],
                    "sign_type": row["Sign type"],
                    "sign_start_frame": int(row["Start frame of the sign video"]),
                    "sign_end_frame": int(row["End frame of the sign video"]),
                    "utterance_start_frame": int(row["Start frame of the containing utterance"]),
                    "utterance_end_frame": int(row["End frame of the containing utterance"]),
                    "sign_video_filename": row["Sign video filename"],
                    "sign_video_url": video_url(row["Sign video filename"]),
                    "utterance_video_filename": row["Utterance video filename"],
                    "utterance_video_url": video_url(row["Utterance video filename"]),
                    "source_collection": row["Source collection"],
                    "utterance_number": row["Utterance number"],
                    "master_video_filename": row["Master video filename"],
                    "training_eligible": source == "asllrp",
                }
            )
    selected.sort(
        key=lambda row: (
            int(row["class_index"]),
            str(row["signer_id"]),
            str(row["sign_video_filename"]),
        )
    )
    names = [str(row["sign_video_filename"]) for row in selected]
    if len(names) != len(set(names)):
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        raise ValueError(f"selected sign filenames are duplicated: {duplicates[:10]}")
    return selected, summary


def build_stage2_inventory(
    source: str,
    rows: list[dict[str, str]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_annotation = {
        str(target["signbank_annotation_id"]): target
        for target in targets
        if target["signbank_annotation_id"]
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("Hidden", "F") != "T":
            grouped[row["Utterance video filename"]].append(row)
    inventory: list[dict[str, Any]] = []
    for filename, utterance_rows in grouped.items():
        ordered = sorted(utterance_rows, key=lambda row: int(row["Start frame of the sign video"]))
        collections = {row["Source collection"] for row in ordered}
        if len(collections) != 1:
            raise ValueError(f"utterance spans source collections: {filename}: {collections}")
        lexical = [row for row in ordered if row["Sign type"] != "Gestures"]
        target_sequence: list[str] = []
        target_variants: list[str] = []
        unmatched: list[str] = []
        for row in lexical:
            variant = row["Entry/variant gloss label"]
            target = by_annotation.get(variant)
            if target is not None and occurrence_matches(variant, row["Occurrence label"]):
                target_sequence.append(str(target["canonical_label"]))
                target_variants.append(variant)
            else:
                unmatched.append(variant)
        if not target_sequence:
            continue
        collection = next(iter(collections))
        inventory.append(
            {
                "source": source,
                "signer_id": signer_id(source, collection),
                "source_collection": collection,
                "utterance_video_filename": filename,
                "utterance_video_url": video_url(filename),
                "target_sequence": target_sequence,
                "target_variants": target_variants,
                "target_token_count": len(target_sequence),
                "non_gesture_token_count": len(lexical),
                "unmatched_non_gesture_variants": unmatched,
                "fully_in_vocab": not unmatched,
                "ctc_eligible": not unmatched and len(target_sequence) >= 2,
            }
        )
    inventory.sort(key=lambda row: (str(row["source"]), str(row["utterance_video_filename"])))
    return inventory


def build_stage2_contiguous_spans(
    source: str,
    rows: list[dict[str, str]],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return multi-token runs interrupted by every non-target visible annotation."""
    by_annotation = {
        str(target["signbank_annotation_id"]): target
        for target in targets
        if target["signbank_annotation_id"]
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("Hidden", "F") != "T":
            grouped[row["Utterance video filename"]].append(row)
    spans: list[dict[str, Any]] = []
    split_role = "train_candidate" if source == "asllrp" else "external_evaluation_reserved"
    for filename, utterance_rows in grouped.items():
        ordered = sorted(utterance_rows, key=lambda row: int(row["Start frame of the sign video"]))
        run: list[tuple[dict[str, str], dict[str, Any]]] = []

        def flush() -> None:
            if len(run) < 2:
                run.clear()
                return
            collections = {row["Source collection"] for row, _ in run}
            utterance_starts = {int(row["Start frame of the containing utterance"]) for row, _ in run}
            utterance_ends = {int(row["End frame of the containing utterance"]) for row, _ in run}
            if len(collections) != 1 or len(utterance_starts) != 1 or len(utterance_ends) != 1:
                raise ValueError(f"inconsistent containing utterance metadata: {filename}")
            collection = next(iter(collections))
            utterance_start = next(iter(utterance_starts))
            utterance_end = next(iter(utterance_ends))
            first_start = int(run[0][0]["Start frame of the sign video"])
            last_end = int(run[-1][0]["End frame of the sign video"])
            context_start = max(utterance_start, first_start - 5)
            context_end = min(utterance_end, last_end + 5)
            spans.append(
                {
                    "source": source,
                    "split_role": split_role,
                    "signer_id": signer_id(source, collection),
                    "source_collection": collection,
                    "utterance_video_filename": filename,
                    "utterance_video_url": video_url(filename),
                    "span_index_in_utterance": sum(
                        item["source"] == source
                        and item["utterance_video_filename"] == filename
                        for item in spans
                    ),
                    "target_sequence": [target["canonical_label"] for _, target in run],
                    "target_variants": [row["Entry/variant gloss label"] for row, _ in run],
                    "target_token_count": len(run),
                    "utterance_start_frame_global": utterance_start,
                    "utterance_end_frame_global": utterance_end,
                    "span_start_frame_global": first_start,
                    "span_end_frame_global": last_end,
                    "crop_start_frame_local": context_start - utterance_start,
                    "crop_end_frame_local": context_end - utterance_start,
                    "context_frames_each_side": 5,
                }
            )
            run.clear()

        for row in ordered:
            variant = row["Entry/variant gloss label"]
            target = by_annotation.get(variant)
            if (
                row["Sign type"] != "Gestures"
                and target is not None
                and occurrence_matches(variant, row["Occurrence label"])
            ):
                run.append((row, target))
            else:
                flush()
        flush()
    spans.sort(
        key=lambda row: (
            str(row["source"]),
            str(row["utterance_video_filename"]),
            int(row["crop_start_frame_local"]),
        )
    )
    return spans


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                for key, value in row.items()
            }
            writer.writerow(encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument(
        "--asllex",
        type=Path,
        default=Path("data/local/dataset_metadata/asllex2_official/signdata.csv"),
    )
    parser.add_argument(
        "--asllrp",
        type=Path,
        default=Path(
            "data/local/dataset_metadata/asllrp_signbank/asllrp_sentence_signs_2025_06_28.csv"
        ),
    )
    parser.add_argument(
        "--rit",
        type=Path,
        default=Path(
            "data/local/dataset_metadata/asllrp_signbank/rit_sentence_signs_2025_11_01.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/asllrp_continuous_citizen100_v17"),
    )
    args = parser.parse_args()

    targets = load_targets(args.manifest, args.asllex)
    source_paths = {"asllrp": args.asllrp, "rit": args.rit}
    all_stage1: list[dict[str, Any]] = []
    all_stage2: list[dict[str, Any]] = []
    all_stage2_spans: list[dict[str, Any]] = []
    source_audits: dict[str, Any] = {}
    class_summaries: dict[str, list[dict[str, Any]]] = {}
    for source, path in source_paths.items():
        rows, rejected = read_sentence_csv(path)
        stage1, class_summary = select_stage1(source, rows, targets)
        stage2 = build_stage2_inventory(source, rows, targets)
        stage2_spans = build_stage2_contiguous_spans(source, rows, targets)
        all_stage1.extend(stage1)
        all_stage2.extend(stage2)
        all_stage2_spans.extend(stage2_spans)
        class_summaries[source] = class_summary
        source_audits[source] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "valid_rows": len(rows),
            "rejected_rows": rejected,
            "stage1_candidate_signs": len(stage1),
            "stage1_candidate_classes": len({row["canonical_label"] for row in stage1}),
            "participants": sorted({row["signer_id"] for row in stage1}),
            "source_collections": len({row["source_collection"] for row in stage1}),
            "target_bearing_utterances": len(stage2),
            "ctc_eligible_utterances": sum(bool(row["ctc_eligible"]) for row in stage2),
            "contiguous_target_spans": len(stage2_spans),
            "contiguous_target_span_tokens": sum(
                int(row["target_token_count"]) for row in stage2_spans
            ),
        }

    train = [row for row in all_stage1 if row["split_role"] == "train_candidate"]
    external = [row for row in all_stage1 if row["split_role"] == "external_evaluation_reserved"]
    ctc = [row for row in all_stage2 if row["ctc_eligible"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage1_asllrp_train_candidates.csv", train)
    write_csv(args.output_dir / "stage1_rit_external_eval_candidates.csv", external)
    write_csv(args.output_dir / "stage2_target_bearing_utterances.csv", all_stage2)
    write_csv(args.output_dir / "stage2_ctc_eligible_utterances.csv", ctc)
    write_csv(args.output_dir / "stage2_contiguous_target_spans.csv", all_stage2_spans)
    (args.output_dir / "stage1_sign_video_urls.txt").write_text(
        "\n".join(str(row["sign_video_url"]) for row in all_stage1) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "stage2_target_bearing_utterance_urls.txt").write_text(
        "\n".join(sorted({str(row["utterance_video_url"]) for row in all_stage2})) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "stage2_contiguous_span_utterance_urls.txt").write_text(
        "\n".join(sorted({str(row["utterance_video_url"]) for row in all_stage2_spans})) + "\n",
        encoding="utf-8",
    )

    payload = {
        "format": "slt_v17_asllrp_continuous_citizen100_audit",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "asllex_path": str(args.asllex),
        "asllex_sha256": sha256_file(args.asllex),
        "variant_contract": (
            "official ASL-LEX SignBankAnnotationID exactly equals ASLLRP entry/variant; "
            "occurrence may differ only by trailing documented repetition plus markers"
        ),
        "sources": source_audits,
        "combined": {
            "stage1_candidate_signs": len(all_stage1),
            "stage1_candidate_classes": len({row["canonical_label"] for row in all_stage1}),
            "stage1_train_candidate_signs": len(train),
            "stage1_train_candidate_classes": len({row["canonical_label"] for row in train}),
            "stage1_external_eval_signs": len(external),
            "stage1_external_eval_classes": len({row["canonical_label"] for row in external}),
            "target_bearing_utterances": len(all_stage2),
            "ctc_eligible_utterances": len(ctc),
            "contiguous_target_spans": len(all_stage2_spans),
            "contiguous_target_span_utterances": len(
                {row["utterance_video_filename"] for row in all_stage2_spans}
            ),
            "contiguous_target_span_tokens": sum(
                int(row["target_token_count"]) for row in all_stage2_spans
            ),
            "contiguous_target_train_spans": sum(
                row["split_role"] == "train_candidate" for row in all_stage2_spans
            ),
            "contiguous_target_external_eval_spans": sum(
                row["split_role"] == "external_evaluation_reserved"
                for row in all_stage2_spans
            ),
        },
        "class_summaries": class_summaries,
        "split_contract": {
            "asllrp": "train candidate only after the selected Stage 1 baseline is measured",
            "rit": "permanently reserved external evaluation; never train or select on it",
            "stage2": (
                "only fully-in-vocabulary multi-token utterances are directly CTC eligible; "
                "target-bearing partial utterances remain contextual/weak-label candidates"
            ),
        },
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    readme = f"""# ASLLRP segmented continuous-sign audit for Citizen100

- ASLLRP exact-variant Stage 1 train candidates: {len(train)} signs across {len({row['canonical_label'] for row in train})} classes.
- RIT exact-variant external evaluation reserve: {len(external)} signs across {len({row['canonical_label'] for row in external})} classes.
- Combined exact-variant Stage 1 coverage: {len(all_stage1)} signs across {len({row['canonical_label'] for row in all_stage1})}/100 classes.
- Target-bearing full utterances: {len(all_stage2)}.
- Directly eligible locked-100-class multi-token CTC utterances: {len(ctc)}.
- Exact contiguous target-only spans: {len(all_stage2_spans)} spans across {len({row['utterance_video_filename'] for row in all_stage2_spans})} parent utterances and {sum(int(row['target_token_count']) for row in all_stage2_spans)} target tokens.

The Stage 1 mapping is official ASL-LEX/Sign Bank exact-variant matching, not English-label normalization. RIT remains a held-out external evaluation source. Most full utterances contain glosses outside the locked 100 classes and therefore must not be presented as fully supervised CTC data. Contiguous target-only spans are cropped at manual sign boundaries plus five context frames and may be evaluated as short real phrases without assigning labels to intervening out-of-vocabulary signs.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(payload["combined"], indent=2))


if __name__ == "__main__":
    main()
