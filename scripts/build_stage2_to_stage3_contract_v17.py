#!/usr/bin/env python3
"""Build and hash-verify the locked-100 Stage-2-to-Stage-3 interface contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from active.v17.export_stage1_coreml_v17 import tree_sha256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    forbidden = (
        args.vocabulary_manifest,
        args.checkpoint,
        args.image_package,
        args.encoder_package,
        args.head_package,
    )
    if any("test" in {part.lower() for part in path.parts} for path in forbidden):
        raise ValueError("the Stage-2-to-Stage-3 contract cannot be built from test data")
    vocabulary = json.loads(args.vocabulary_manifest.read_text())
    classes = sorted(vocabulary["classes"], key=lambda row: int(row["class_index"]))
    indices = [int(row["class_index"]) for row in classes]
    labels = [str(row["canonical_label"]) for row in classes]
    if indices != list(range(100)) or len(set(labels)) != 100:
        raise ValueError("the vocabulary manifest is not the locked 100-class sequence")
    contract = {
        "format": "slt_stage2_to_stage3_contract_v17",
        "version": 1,
        "scope": "locked_100_gloss_recognition_only",
        "recognizer": {
            "candidate_id": "stage2_v17_compact_context_student_v1",
            "checkpoint": args.checkpoint.as_posix(),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "coreml_packages": {
                "hand_image_encoder": {
                    "path": args.image_package.as_posix(),
                    "tree_sha256": tree_sha256(args.image_package),
                },
                "frozen_multimodal_encoder": {
                    "path": args.encoder_package.as_posix(),
                    "tree_sha256": tree_sha256(args.encoder_package),
                },
                "compact_context_ctc_head": {
                    "path": args.head_package.as_posix(),
                    "tree_sha256": tree_sha256(args.head_package),
                },
            },
        },
        "vocabulary": {
            "manifest": args.vocabulary_manifest.as_posix(),
            "manifest_sha256": sha256_file(args.vocabulary_manifest),
            "label_count": 100,
            "labels": labels,
            "token_mapping": "ctc_token_i_maps_to_labels_i_minus_1_for_i_in_1_through_100",
            "unknown_token": None,
        },
        "ctc": {
            "blank_index": 0,
            "collapse": "greedy_argmax_then_merge_adjacent_duplicates_then_remove_blank",
            "maximum_windows": 8,
            "frames_per_window": 32,
            "tokens_per_window": 8,
            "maximum_source_frames": 256,
        },
        "output": {
            "format": "slt_stage2_gloss_sequence_v17",
            "version": 1,
            "required_keys": [
                "format", "version", "utterance_id", "token_indices", "glosses",
                "window_count", "blank_index", "vocabulary_manifest_sha256",
                "recognizer_checkpoint_sha256",
            ],
            "gloss_order": "temporal",
            "empty_sequence_allowed": True,
            "confidence_is_not_part_of_frozen_boundary": True,
        },
        "stage3_consumer_rules": [
            "Reject any format, version, vocabulary hash, or recognizer checkpoint mismatch.",
            "Consume glosses in emitted temporal order without relabeling or synonym merging.",
            "Do not invent an unknown token; Stage 2 is closed-vocabulary and may emit an empty sequence.",
            "Treat translation quality as a separate Stage-3 gate; recognition metrics are not translation metrics.",
        ],
        "evidence": {
            "full_coreml_validation": "artifacts/reports/mobile_100gloss_v17/full_coreml_validation.json",
            "simulator_benchmark_latest": "artifacts/reports/orientation_v17_simulator_benchmark/latest_result.json",
            "citizen_test_accessed": False,
            "semlex_test_accessed": False,
            "local_test_accessed": False,
            "two_m_flores_devtest_accessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary-manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/stage2_v17_compact_context_student_v1/model.pth"))
    parser.add_argument("--image-package", type=Path, default=Path("artifacts/coreml/MobileCLIP2S0ImageEncoderV17FP32.mlpackage"))
    parser.add_argument("--encoder-package", type=Path, default=Path("artifacts/coreml/Stage2FrozenEncoderV17FP32.mlpackage"))
    parser.add_argument("--head-package", type=Path, default=Path("artifacts/coreml/Stage2CompactContextV17FP32.mlpackage"))
    parser.add_argument("--output", type=Path, default=Path("active/v17/stage2_to_stage3_contract_v17.json"))
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
