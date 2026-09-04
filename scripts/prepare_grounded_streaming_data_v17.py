#!/usr/bin/env python3
"""Create lightweight signer-disjoint phrase archives for streaming experiments."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def output_name(source_item_id: str) -> str:
    digest = hashlib.sha256(source_item_id.encode()).hexdigest()[:12]
    return f"{digest}.grounded_streaming_v17.npz"


def write_archive(
    output: Path, landmarks: np.ndarray, ranges: np.ndarray,
    targets: list[int] | np.ndarray, metadata: dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        landmarks=landmarks.astype(np.float16),
        window_source_ranges=ranges.astype(np.int64),
        target_indices=np.asarray(targets, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(
        "data/local/stage2_v17_multimodal"
    ))
    parser.add_argument("--signers", type=Path, default=Path(
        "artifacts/reports/local_phrase_signer_audit_v17_v4_auto/signer_clusters.json"
    ))
    parser.add_argument("--ncslgr", type=Path, default=Path(
        "active/v17/ncslgr_supervised_manifest_v17.json"
    ))
    parser.add_argument("--classes", type=Path, default=Path(
        "active/v17/citizen100_manifest.json"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "data/local/stage2_v17_grounded_signer_split"
    ))
    parser.add_argument("--validation-signer", default="local_signer_02")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    signer_payload = json.loads(args.signers.read_text())
    signer_rows = {row["video_path"]: row for row in signer_payload["rows"]}
    class_payload = json.loads(args.classes.read_text())
    labels = {
        str(row["canonical_label"]): int(row["class_index"])
        for row in class_payload["classes"]
    }
    if len(labels) != 100:
        raise ValueError("expected the frozen 100 glosses")
    counts, excluded = Counter(), []

    # Re-split local archives by anonymous signer rather than their old hash split.
    for source in sorted(args.source_root.glob("*/local_phrases/*.npz")):
        with np.load(source, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            video_path = str(metadata["video_path"])
            signer = signer_rows.get(video_path)
            if signer is None or not signer["identity_confident"]:
                excluded.append(video_path)
                continue
            role = "validation" if signer["signer_id"] == args.validation_signer else "train"
            metadata.update({
                "role": role,
                "signer_id": signer["signer_id"],
                "split_policy": "anonymous face-cluster signer-disjoint",
                "original_role": metadata["role"],
            })
            destination = args.output / role / "local_phrases" / output_name(
                str(metadata["source_item_id"])
            )
            write_archive(
                destination, payload["landmarks"], payload["window_source_ranges"],
                payload["target_indices"], metadata,
            )
            counts[f"local_phrases:{role}"] += 1

    # Preserve the already signer-disjoint, manually aligned ASLLRP exact phrases.
    for source in sorted(args.source_root.glob("*/asllrp_contiguous/*.npz")):
        with np.load(source, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            role = str(metadata["role"])
            destination = args.output / role / "asllrp_contiguous" / output_name(
                str(metadata["source_item_id"])
            )
            write_archive(
                destination, payload["landmarks"], payload["window_source_ranges"],
                payload["target_indices"], metadata,
            )
            counts[f"asllrp_contiguous:{role}"] += 1

    ncslgr = json.loads(args.ncslgr.read_text())
    other_index = len(labels)
    for row in ncslgr["rows"]:
        if not row["has_strict_target"]:
            continue
        targets, target_labels = [], []
        for event in row["events"]:
            target = labels.get(event["canonical_label"], other_index)
            if target != other_index or not targets or targets[-1] != other_index:
                targets.append(target)
                target_labels.append(event["canonical_label"] or "__OTHER__")
        with np.load(row["archive_path"], allow_pickle=False) as payload:
            source_metadata = json.loads(str(payload["metadata_json"].item()))
            metadata = {
                **source_metadata,
                "role": row["role"],
                "source": "ncslgr_strict",
                "source_item_id": row["source_item_id"],
                "target_sequence": target_labels,
                "target_policy": "strict exact raw gloss; consecutive OTHER collapsed",
                "timed_supervision_manifest": args.ncslgr.as_posix(),
            }
            destination = (
                args.output / row["role"] / "ncslgr_strict"
                / output_name(row["source_item_id"])
            )
            write_archive(
                destination, payload["landmarks"], payload["window_source_ranges"],
                targets, metadata,
            )
            counts[f"ncslgr_strict:{row['role']}"] += 1

    manifest = {
        "format": "slt_grounded_streaming_signer_split_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": dict(sorted(counts.items())),
        "local_validation_signer": args.validation_signer,
        "local_training_signers": sorted({
            row["signer_id"] for row in signer_payload["rows"]
            if row["signer_id"] and row["signer_id"] != args.validation_signer
        }),
        "excluded_uncertain_local_videos": sorted(set(excluded)),
        "ncslgr_split_policy": ncslgr["split_policy"],
        "test_accessed": False,
        "external_evaluation_reserved_accessed": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
