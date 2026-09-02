#!/usr/bin/env python3
"""Finalize acquired ASLLRP target-plus-OTHER spans for v17 extraction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_list(value):
    return json.loads(value) if isinstance(value, str) else value


def run(args: argparse.Namespace) -> dict:
    acquisition = json.loads(args.acquisition_manifest.read_text())
    if acquisition.get("failures") or acquisition.get("verified_spans") != acquisition.get("expected_spans"):
        raise ValueError("ASLLRP OTHER-CTC acquisition is incomplete")
    rows = []
    for span in acquisition["spans"]:
        target_sequence = parse_list(span["target_sequence"])
        ctc_target_indices = [int(value) for value in parse_list(span["target_indices"])]
        if len(target_sequence) != len(ctc_target_indices) or not ctc_target_indices:
            raise ValueError("target sequence/index mismatch")
        if any(index < 1 or index > 101 for index in ctc_target_indices):
            raise ValueError("target index exceeds blank/100/OTHER contract")
        # Stage-2 archives use zero-based class IDs and RealPhraseDataset applies
        # the +1 CTC blank offset.  The acquisition plan deliberately uses CTC
        # IDs, so convert exactly once at this manifest boundary.
        target_indices = [index - 1 for index in ctc_target_indices]
        path = Path(span["path"])
        if not path.exists() or sha256(path) != span["sha256"]:
            raise ValueError(f"acquired span hash mismatch: {path}")
        rows.append({
            "source": "asllrp_other_ctc",
            "role": span["split_role"],
            "source_item_id": (
                f"asllrp_other_ctc:{Path(span['utterance_video_filename']).stem}:"
                f"span{int(span['span_index_in_utterance']):02d}"
            ),
            "video_path": path.as_posix(),
            "video_sha256": span["sha256"],
            "source_group": f"asllrp:{span['signer_id']}",
            "signer_id": span["signer_id"],
            "zero_lip_nodes": False,
            "lip_supervision": "visible_source_video_available",
            "target_sequence": target_sequence,
            "target_indices": target_indices,
            "target_token_count": len(target_indices),
            "supported_target_token_count": int(span["supported_target_token_count"]),
            "other_token_count": int(span["other_token_count"]),
            "other_class_index": 100,
            "other_ctc_index": 101,
            "drop_other_at_inference": True,
            "frame_count": int(span["frames"]),
            "duration_seconds": float(span["duration_seconds"]),
            "parent_utterance_video_filename": span["utterance_video_filename"],
            "parent_sha256": span["parent_sha256"],
        })
    identifiers = [row["source_item_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate ASLLRP OTHER-CTC source item IDs")
    payload = {
        "format": "slt_stage2_asllrp_other_ctc_manifest_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "acquisition_manifest": args.acquisition_manifest.as_posix(),
        "acquisition_manifest_sha256": sha256(args.acquisition_manifest),
        "vocabulary_manifest": "active/v17/citizen100_manifest.json",
        "vocabulary_manifest_sha256": sha256(Path("active/v17/citizen100_manifest.json")),
        "class_count_including_other_excluding_blank": 101,
        "blank_index": 0,
        "other_class_index": 100,
        "other_ctc_index": 101,
        "decoder_policy": "drop OTHER=101 after CTC collapse",
        "rows": rows,
        "row_counts": {
            role: sum(row["role"] == role for row in rows)
            for role in ("train", "validation")
        },
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "rit_external_evaluation_accessed": False,
        "test_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--acquisition-manifest", type=Path, default=Path("data/local/asllrp_other_ctc_v17/manifest.json"))
    value.add_argument("--output", type=Path, default=Path("active/v17/stage2_asllrp_other_ctc_manifest_v17.json"))
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"rows": len(result["rows"]), **result["row_counts"]}, indent=2))
