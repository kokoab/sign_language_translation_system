#!/usr/bin/env python3
"""Prepare bounded ASLLRP target-plus-OTHER continuous CTC spans."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_asllrp_continuous_citizen100_v17 import (
    load_targets,
    occurrence_matches,
    read_sentence_csv,
    signer_id,
    video_url,
    write_csv,
)


OTHER = "__OTHER__"
OTHER_INDEX = 101
TRAIN_SIGNERS = {"BENJAMIN_JAMES_BAHAN", "CORY", "RACHEL"}
VALIDATION_SIGNER = "JONATHAN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collapse_other(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in events:
        if output and event["label"] == OTHER and output[-1]["label"] == OTHER:
            output[-1]["end"] = max(output[-1]["end"], event["end"])
            output[-1]["variants"].append(event["variant"])
        else:
            output.append({**event, "variants": [event["variant"]]})
    return output


def chunk_events(events: list[dict[str, Any]], maximum_frames: int) -> list[list[dict[str, Any]]]:
    """Split only between annotations and retain every target-bearing chunk."""
    if maximum_frames < 32:
        raise ValueError("maximum_frames must be at least one model window")
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        if event["end"] - event["start"] + 1 > maximum_frames:
            raise ValueError("one ASLLRP annotation exceeds the mobile temporal cap")
        if current and event["end"] - current[0]["start"] + 1 > maximum_frames:
            chunks.append(current)
            current = []
        current.append(event)
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if any(event["label"] != OTHER for event in chunk)]


def build_rows(
    source_rows: list[dict[str, str]], targets: list[dict[str, Any]],
    maximum_frames: int, context_frames: int,
) -> list[dict[str, Any]]:
    by_variant = {
        str(target["signbank_annotation_id"]): target
        for target in targets if target["signbank_annotation_id"]
    }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if row.get("Hidden", "F") != "T":
            grouped[row["Utterance video filename"]].append(row)
    output = []
    for filename, rows in grouped.items():
        rows.sort(key=lambda row: int(row["Start frame of the sign video"]))
        collections = {row["Source collection"] for row in rows}
        utterance_starts = {int(row["Start frame of the containing utterance"]) for row in rows}
        utterance_ends = {int(row["End frame of the containing utterance"]) for row in rows}
        if len(collections) != 1 or len(utterance_starts) != 1 or len(utterance_ends) != 1:
            raise ValueError(f"inconsistent containing utterance metadata: {filename}")
        collection = next(iter(collections))
        signer = signer_id("asllrp", collection)
        if signer not in TRAIN_SIGNERS | {VALIDATION_SIGNER}:
            continue
        utterance_start = next(iter(utterance_starts))
        utterance_end = next(iter(utterance_ends))
        events = []
        for row in rows:
            variant = row["Entry/variant gloss label"]
            target = by_variant.get(variant)
            exact = target is not None and occurrence_matches(variant, row["Occurrence label"])
            events.append({
                "label": str(target["canonical_label"]) if exact else OTHER,
                "index": int(target["class_index"]) + 1 if exact else OTHER_INDEX,
                "variant": variant,
                "start": int(row["Start frame of the sign video"]) - utterance_start,
                "end": int(row["End frame of the sign video"]) - utterance_start,
                "sign_type": row["Sign type"],
            })
        if not any(event["label"] != OTHER for event in events):
            continue
        for chunk_index, chunk in enumerate(chunk_events(events, maximum_frames)):
            chunk = collapse_other(chunk)
            utterance_last = utterance_end - utterance_start
            core_start = chunk[0]["start"]
            core_end = chunk[-1]["end"]
            crop_start = max(0, core_start - context_frames)
            crop_end = min(utterance_last, core_end + context_frames)
            if crop_end - crop_start + 1 > maximum_frames:
                crop_start = max(0, min(core_start, core_end - maximum_frames + 1))
                crop_end = min(utterance_last, crop_start + maximum_frames - 1)
                if crop_end < core_end:
                    crop_end = core_end
                    crop_start = crop_end - maximum_frames + 1
            if any(event["start"] < crop_start or event["end"] > crop_end for event in chunk):
                raise ValueError(f"bounded crop lost an annotation: {filename}:{chunk_index}")
            labels = [event["label"] for event in chunk]
            indices = [event["index"] for event in chunk]
            output.append({
                "source": "asllrp_other_ctc",
                "split_role": "train" if signer in TRAIN_SIGNERS else "validation",
                "signer_id": signer,
                "source_collection": collection,
                "utterance_video_filename": filename,
                "utterance_video_url": video_url(filename),
                "span_index_in_utterance": chunk_index,
                "target_sequence": labels,
                "target_indices": indices,
                "target_variants": [event["variants"] for event in chunk],
                "target_token_count": len(labels),
                "supported_target_token_count": sum(label != OTHER for label in labels),
                "other_token_count": sum(label == OTHER for label in labels),
                "crop_start_frame_local": crop_start,
                "crop_end_frame_local": crop_end,
                "source_frame_count": crop_end - crop_start + 1,
                "other_index": OTHER_INDEX,
                "drop_other_at_inference": True,
            })
    return sorted(output, key=lambda row: (
        row["split_role"], row["signer_id"], row["utterance_video_filename"],
        row["span_index_in_utterance"],
    ))


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows, rejected = read_sentence_csv(args.asllrp)
    targets = load_targets(args.manifest, args.asllex)
    rows = build_rows(source_rows, targets, args.maximum_frames, args.context_frames)
    train = [row for row in rows if row["split_role"] == "train"]
    validation = [row for row in rows if row["split_role"] == "validation"]
    if not train or not validation:
        raise ValueError("ASLLRP OTHER-CTC signer split is empty")
    train_class_counts = Counter(
        label for row in train for label in row["target_sequence"] if label != OTHER
    )
    validation_class_counts = Counter(
        label for row in validation for label in row["target_sequence"] if label != OTHER
    )
    unseen_validation_classes = set(validation_class_counts) - set(train_class_counts)
    if unseen_validation_classes:
        raise ValueError(
            f"held-out ASLLRP classes absent from training: {sorted(unseen_validation_classes)}"
        )
    output_csv = args.output_dir / "spans.csv"
    write_csv(output_csv, rows)
    signer_counts = Counter(row["signer_id"] for row in rows)
    payload = {
        "format": "slt_stage2_asllrp_other_ctc_plan_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": args.asllrp.as_posix(),
        "source_metadata_sha256": sha256(args.asllrp),
        "citizen_manifest": args.manifest.as_posix(),
        "citizen_manifest_sha256": sha256(args.manifest),
        "variant_contract": "exact ASL-LEX SignBankAnnotationID; no normalized-label merges",
        "output_csv": output_csv.as_posix(),
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_signers": sorted(TRAIN_SIGNERS),
        "validation_signer": VALIDATION_SIGNER,
        "rows_by_signer": dict(sorted(signer_counts.items())),
        "unique_parent_utterances": len({row["utterance_video_filename"] for row in rows}),
        "target_tokens": sum(row["supported_target_token_count"] for row in rows),
        "other_tokens": sum(row["other_token_count"] for row in rows),
        "train_unique_sequences": len({tuple(row["target_sequence"]) for row in train}),
        "validation_unique_sequences": len({
            tuple(row["target_sequence"]) for row in validation
        }),
        "train_target_class_counts": dict(sorted(train_class_counts.items())),
        "validation_target_class_counts": dict(sorted(validation_class_counts.items())),
        "validation_classes_absent_from_train": [],
        "maximum_frames": args.maximum_frames,
        "maximum_observed_frames": max(row["source_frame_count"] for row in rows),
        "other_label": OTHER,
        "other_index": OTHER_INDEX,
        "decoder_policy": "CTC blank=0; locked glosses=1..100; OTHER=101; drop OTHER",
        "malformed_source_rows_rejected": rejected,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "rit_external_evaluation_accessed": False,
        "test_evaluated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    value.add_argument("--asllex", type=Path, default=Path("data/local/dataset_metadata/asllex2_official/signdata.csv"))
    value.add_argument("--asllrp", type=Path, default=Path("data/local/dataset_metadata/asllrp_signbank/asllrp_sentence_signs_2025_06_28.csv"))
    value.add_argument("--maximum-frames", type=int, default=256)
    value.add_argument("--context-frames", type=int, default=5)
    value.add_argument("--output-dir", type=Path, default=Path("artifacts/reports/stage2_v17_asllrp_other_ctc_plan"))
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
