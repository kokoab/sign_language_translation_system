#!/usr/bin/env python3
"""Build the labeled real-video corpus for whole-utterance landmark generation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


RESERVED = ("<PAD>", "<BOS>", "<EOS>")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text())
    return payload, list(payload["rows"])


def run(args: argparse.Namespace) -> dict[str, object]:
    local_manifest, local_rows = load_rows(args.local_manifest)
    other_manifest, other_rows = load_rows(args.asllrp_other_manifest)
    flores_manifest, flores_rows = load_rows(args.flores_manifest)
    exact_manifest, exact_rows = load_rows(args.asllrp_exact_manifest)

    rows = []
    for row in local_rows:
        sequence = str(row["phrase_prompt"]).split("_")
        rows.append({**row, "source": "local_phrase_full", "target_sequence": sequence})
    rows.extend(other_rows)
    rows.extend(flores_rows)
    rows.extend(
        row for row in exact_rows
        if row["source"] == "asllrp_contiguous" and row["role"] in {"train", "validation"}
    )

    seen_ids = set()
    for row in rows:
        item_id = str(row["source_item_id"])
        if item_id in seen_ids:
            raise ValueError(f"duplicate source item ID: {item_id}")
        seen_ids.add(item_id)
        video = Path(row["video_path"])
        if not video.is_file():
            raise FileNotFoundError(video)
        if row.get("video_sha256") and sha256(video) != row["video_sha256"]:
            raise ValueError(f"source video changed: {video}")
        sequence = [str(token) for token in row["target_sequence"]]
        if not sequence or len(sequence) > args.maximum_tokens:
            raise ValueError(f"invalid target sequence for {item_id}: {sequence}")
        row["target_sequence"] = sequence
        row["video_sha256"] = row.get("video_sha256") or sha256(video)
        row["duration_seconds"] = row.get("duration_seconds")

    vocabulary = RESERVED + tuple(sorted({
        token for row in rows for token in row["target_sequence"]
    }))
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    output_rows = []
    for row in rows:
        output_rows.append({
            "source_item_id": row["source_item_id"],
            "source": row["source"],
            "role": row["role"],
            "video_path": row["video_path"],
            "video_sha256": row["video_sha256"],
            "source_group": row.get("source_group"),
            "signer_id": row.get("signer_id"),
            "target_sequence": row["target_sequence"],
            "target_token_ids": [
                token_to_index[token] for token in row["target_sequence"]
            ],
            "duration_seconds": row["duration_seconds"],
            "license": row.get("license"),
        })

    counts = Counter((str(row["source"]), str(row["role"])) for row in output_rows)
    payload = {
        "format": "slt_full_trajectory_generation_manifest_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "gloss-conditioned whole-utterance landmark generation; every video is "
            "normalized once as a complete trajectory"
        ),
        "source_manifests": {
            path.as_posix(): sha256(path) for path in (
                args.local_manifest, args.asllrp_other_manifest,
                args.flores_manifest, args.asllrp_exact_manifest,
            )
        },
        "split_contract": (
            "preserve every source role; ASLLRP validation remains signer-held-out; "
            "local validation is familiar-source every-fifth capture; 2M dev clips "
            "remain train-role only"
        ),
        "trajectory_contract": (
            "never concatenate independently normalized windows; retain genuine sparse "
            "observation presence; render completion is a separate downstream step"
        ),
        "reserved_tokens": list(RESERVED),
        "token_to_index": token_to_index,
        "maximum_tokens": args.maximum_tokens,
        "row_count": len(output_rows),
        "source_role_counts": {
            f"{source}:{role}": count
            for (source, role), count in sorted(counts.items())
        },
        "rows": output_rows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "how2sign_validation_accessed": False,
        "how2sign_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--local-manifest", type=Path, default=Path("active/v17/local_phrase_motion_manifest_v17.json"))
    value.add_argument("--asllrp-other-manifest", type=Path, default=Path("active/v17/stage2_asllrp_other_ctc_manifest_v17.json"))
    value.add_argument("--flores-manifest", type=Path, default=Path("active/v17/stage2_2m_flores_training_manifest_v17.json"))
    value.add_argument("--asllrp-exact-manifest", type=Path, default=Path("active/v17/stage2_training_manifest_v17.json"))
    value.add_argument("--maximum-tokens", type=int, default=40)
    value.add_argument("--output", type=Path, default=Path("active/v17/full_trajectory_generation_manifest_v17.json"))
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({
        "rows": result["row_count"],
        "vocabulary": len(result["token_to_index"]),
        "counts": result["source_role_counts"],
    }, indent=2))
