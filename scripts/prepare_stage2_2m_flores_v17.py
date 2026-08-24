#!/usr/bin/env python3
"""Build full-order 2M-Flores Stage-2 training and auxiliary-vocabulary manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any


FINGERSPELLED = re.compile(r"^(?:[A-Z0-9]-){1,}[A-Z0-9]$")
NUMERIC = re.compile(r"^[0-9]+(?:[.:/-][0-9]+)*$")
EDGE_PUNCTUATION = ".,!?;\"'`()[]{}<>/\\|"
RESERVED_TOKENS = ("__UNK__", "__FS__", "__NUM__", "__IX__", "__CL__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def normalize_raw_token(raw: str, locked_by_key: dict[str, str]) -> str | None:
    token = raw.strip(EDGE_PUNCTUATION).upper()
    if not token:
        return None
    token = re.sub(r"\++$", "", token)
    key = locked_key(token)
    if key in locked_by_key:
        return locked_by_key[key]
    if token.startswith("#") or FINGERSPELLED.fullmatch(token):
        return "__FS__"
    if NUMERIC.fullmatch(token):
        return "__NUM__"
    if token.startswith("IX") or token.startswith("POSS-"):
        return "__IX__"
    if token.startswith("CL:") or token.startswith("CL-"):
        return "__CL__"
    value = locked_key(token)
    return value or None


def normalize_gloss(gloss: str, locked_by_key: dict[str, str]) -> list[str]:
    return [
        value for raw in gloss.split()
        if (value := normalize_raw_token(raw, locked_by_key)) is not None
    ]


def video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "default=nk=1:nw=1", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    value = result.stdout.strip()
    if not value or value == "N/A":
        raise ValueError(f"video has no indexed frame count: {path}")
    return int(value)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build(args: argparse.Namespace) -> dict[str, Any]:
    selection = json.loads(args.selection.read_text())
    state = json.loads(args.acquisition_state.read_text())
    citizen = json.loads(args.citizen_vocabulary.read_text())
    if selection.get("reserved_devtest_accessed") is not False:
        raise ValueError("2M-Flores devtest must remain sealed")
    if len(selection["rows"]) != len(state["completed_rows"]):
        raise ValueError("2M-Flores acquisition is incomplete")

    locked_labels = [
        str(row["canonical_label"]) for row in sorted(
            citizen["classes"], key=lambda row: int(row["class_index"])
        )
    ]
    if len(locked_labels) != 100 or len(set(locked_labels)) != 100:
        raise ValueError("Citizen locked vocabulary is not exactly 100 unique labels")
    locked_by_key = {locked_key(label): label for label in locked_labels}
    if len(locked_by_key) != len(locked_labels):
        raise ValueError("Citizen labels collide after normalization")

    normalized_sequences: dict[int, list[str]] = {}
    frequencies: Counter[str] = Counter()
    for row in selection["rows"]:
        sequence = normalize_gloss(str(row["gloss"]), locked_by_key)
        if not sequence:
            raise ValueError(f"row {row['id']} has an empty normalized gloss")
        normalized_sequences[int(row["id"])] = sequence
        frequencies.update(sequence)

    extra_lexical = sorted(
        token for token, count in frequencies.items()
        if token not in locked_labels and token not in RESERVED_TOKENS
        and count >= args.minimum_lexical_frequency
    )
    vocabulary = list(locked_labels) + list(RESERVED_TOKENS) + extra_lexical
    label_to_index = {label: index for index, label in enumerate(vocabulary)}
    vocabulary_payload = {
        "format": "slt_stage2_2m_flores_auxiliary_vocabulary_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "locked_prefix_vocabulary": args.citizen_vocabulary.as_posix(),
        "locked_prefix_vocabulary_sha256": sha256(args.citizen_vocabulary),
        "locked_prefix_count": 100,
        "minimum_lexical_frequency": args.minimum_lexical_frequency,
        "reserved_tokens": list(RESERVED_TOKENS),
        "classes": [
            {
                "class_index": index,
                "token": token,
                "kind": (
                    "locked100" if index < 100 else
                    "reserved" if token in RESERVED_TOKENS else "expanded_lexical"
                ),
                "selected_corpus_occurrences": int(frequencies.get(token, 0)),
            }
            for index, token in enumerate(vocabulary)
        ],
        "raw_normalized_unique_tokens": len(frequencies),
        "expanded_vocabulary_count": len(vocabulary),
        "rare_lexical_tokens_mapped_to_unknown": sum(
            1 for token, count in frequencies.items()
            if token not in locked_labels and token not in RESERVED_TOKENS
            and count < args.minimum_lexical_frequency
        ),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    args.output_vocabulary.parent.mkdir(parents=True, exist_ok=True)
    args.output_vocabulary.write_text(json.dumps(vocabulary_payload, indent=2) + "\n")
    vocabulary_sha = sha256(args.output_vocabulary)

    manifest_rows = []
    unknown_occurrences = 0
    maximum_frames = 0
    maximum_windows = 0
    for row in selection["rows"]:
        row_id = int(row["id"])
        completed = state["completed_rows"].get(str(row_id))
        if completed is None:
            raise ValueError(f"missing completed acquisition row {row_id}")
        video_path = Path(str(completed["derived_path"]))
        if not video_path.exists() or sha256(video_path) != completed["derived_sha256"]:
            raise ValueError(f"derived video integrity failure for row {row_id}")
        normalized = normalized_sequences[row_id]
        mapped = [
            token if token in label_to_index else "__UNK__" for token in normalized
        ]
        unknown_occurrences += mapped.count("__UNK__")
        frame_count = video_frame_count(video_path)
        windows = math.ceil(frame_count / 32)
        if len(mapped) > windows * args.tokens_per_window:
            raise ValueError(
                f"row {row_id}: {len(mapped)} targets exceed {windows * args.tokens_per_window} CTC steps"
            )
        maximum_frames = max(maximum_frames, frame_count)
        maximum_windows = max(maximum_windows, windows)
        manifest_rows.append({
            "source": "two_m_flores_asl",
            "role": "train",
            "source_item_id": f"2m_flores_dev:{row_id}",
            "video_path": portable_path(video_path),
            "video_sha256": str(completed["derived_sha256"]),
            "source_group": f"2m_flores_dev:local_signer_{row['signer_local_id']}",
            "signer_id": None,
            "signer_local_id": str(row["signer_local_id"]),
            "signer_identity_contract": "dataset_local_id_not_global_identity",
            "zero_lip_nodes": False,
            "lip_supervision": "visible_source_video_available",
            "raw_gloss": str(row["gloss"]),
            "normalized_full_gloss_sequence": normalized,
            "target_sequence": mapped,
            "target_indices": [label_to_index[token] for token in mapped],
            "target_token_count": len(mapped),
            "matched_locked_labels": row["matched_locked_labels"],
            "frame_count": frame_count,
            "window_count": windows,
            "domain": row["domain"],
            "topic": row["topic"],
            "sentence": row["sentence"],
        })

    required_maximum_source_frames = int(math.ceil(maximum_frames / 32) * 32)
    manifest_payload = {
        "format": "slt_stage2_2m_flores_training_manifest_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_manifest": args.selection.as_posix(),
        "selection_manifest_sha256": sha256(args.selection),
        "acquisition_state": args.acquisition_state.as_posix(),
        "acquisition_state_sha256": sha256(args.acquisition_state),
        "auxiliary_vocabulary": args.output_vocabulary.as_posix(),
        "auxiliary_vocabulary_sha256": vocabulary_sha,
        "locked_prefix_count": 100,
        "expanded_vocabulary_count": len(vocabulary),
        "minimum_lexical_frequency": args.minimum_lexical_frequency,
        "unknown_token_occurrences": unknown_occurrences,
        "maximum_source_frames": maximum_frames,
        "required_maximum_source_frames": required_maximum_source_frames,
        "maximum_windows": maximum_windows,
        "tokens_per_window": args.tokens_per_window,
        "rows": manifest_rows,
        "split_contract": (
            "2M-Flores dev is training-only; devtest remains sealed; local signer IDs "
            "are not treated as globally identifying signers"
        ),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    result = {
        "manifest": args.output_manifest.as_posix(),
        "manifest_sha256": sha256(args.output_manifest),
        "vocabulary": args.output_vocabulary.as_posix(),
        "vocabulary_sha256": vocabulary_sha,
        "rows": len(manifest_rows),
        "expanded_vocabulary_count": len(vocabulary),
        "raw_normalized_unique_tokens": len(frequencies),
        "unknown_token_occurrences": unknown_occurrences,
        "maximum_source_frames": maximum_frames,
        "required_maximum_source_frames": required_maximum_source_frames,
        "maximum_windows": maximum_windows,
        "maximum_target_tokens": max(len(row["target_sequence"]) for row in manifest_rows),
        "reserved_devtest_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path,
        default=Path("data/local/dataset_metadata/2m_flores_asl/dev_selected_v17.json"),
    )
    parser.add_argument(
        "--acquisition-state", type=Path,
        default=Path("data/local/2m_flores_asl_stage2_v17/acquisition_state.json"),
    )
    parser.add_argument(
        "--citizen-vocabulary", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument("--minimum-lexical-frequency", type=int, default=2)
    parser.add_argument("--tokens-per-window", type=int, default=8)
    parser.add_argument(
        "--output-vocabulary", type=Path,
        default=Path("active/v17/stage2_2m_flores_vocabulary_v17.json"),
    )
    parser.add_argument(
        "--output-manifest", type=Path,
        default=Path("active/v17/stage2_2m_flores_training_manifest_v17.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_2m_flores_preparation/report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.minimum_lexical_frequency < 2:
        raise ValueError("minimum lexical frequency must be at least two")
    if args.tokens_per_window < 1:
        raise ValueError("tokens per window must be positive")
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
