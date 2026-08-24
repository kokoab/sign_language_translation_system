#!/usr/bin/env python3
"""Audit train-side 2M-Flores-ASL gloss metadata against the locked 100 signs."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DATASET = "facebook/2M-Flores-ASL"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "SLT-v17-dataset-audit/1.0"
FINGERSPELLED = re.compile(r"^(?:[A-Z0-9]-){1,}[A-Z0-9](?:[.,!?])?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_gloss_tokens(gloss: str) -> list[str]:
    output = []
    for raw in gloss.split():
        if FINGERSPELLED.fullmatch(raw):
            continue
        token = re.sub(r"[^A-Z0-9]", "", raw.upper())
        if token:
            output.append(token)
    return output


def fetch_rows(split: str, page_size: int = 100) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    total = None
    while total is None or offset < total:
        query = urlencode({
            "dataset": DATASET,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": page_size,
        })
        request = Request(f"{ROWS_ENDPOINT}?{query}", headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
        total = int(payload["num_rows_total"])
        page = [item["row"] for item in payload["rows"]]
        if not page:
            break
        rows.extend(page)
        offset += len(page)
    if total is None or len(rows) != total:
        raise RuntimeError(f"incomplete {split} metadata: {len(rows)}/{total}")
    return rows


def run(args: argparse.Namespace) -> dict[str, object]:
    vocabulary = json.loads(args.vocabulary_manifest.read_text())
    labels = [
        str(row["canonical_label"]) for row in sorted(
            vocabulary["classes"], key=lambda item: int(item["class_index"])
        )
    ]
    normalized_to_label = {
        re.sub(r"[^A-Z0-9]", "", label.upper()): label for label in labels
    }
    rows = fetch_rows(args.split)
    label_counts = Counter()
    candidate_rows = []
    gloss_token_counts = Counter()
    signer_counts = Counter()
    for row in rows:
        tokens = normalized_gloss_tokens(str(row["gloss"]))
        gloss_token_counts.update(tokens)
        matched = sorted({normalized_to_label[token] for token in tokens if token in normalized_to_label})
        signer_counts[str(row["signer"])] += 1
        label_counts.update(matched)
        if matched:
            video = row["video"]
            if not isinstance(video, dict) or not video.get("src"):
                raise ValueError(f"row {row['id']}: missing video URL")
            candidate_rows.append({
                "dataset": DATASET,
                "split": args.split,
                "id": int(row["id"]),
                "signer_local_id": str(row["signer"]),
                "domain": str(row["domain"]),
                "topic": str(row["topic"]),
                "sentence": str(row["sentence"]),
                "gloss": str(row["gloss"]),
                "video_url": str(video["src"]),
                "matched_locked_labels": matched,
            })
    payload = {
        "format": "slt_stage2_2m_flores_asl_metadata_audit_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "dataset_card": "https://huggingface.co/datasets/facebook/2M-Flores-ASL",
        "license": "CC-BY-SA-4.0",
        "split": args.split,
        "reserved_split_not_accessed": "devtest",
        "vocabulary_manifest": args.vocabulary_manifest.as_posix(),
        "vocabulary_manifest_sha256": sha256(args.vocabulary_manifest),
        "rows": len(rows),
        "signer_local_id_counts": dict(sorted(signer_counts.items())),
        "unique_normalized_gloss_tokens": len(gloss_token_counts),
        "locked_labels_covered": sorted(label_counts),
        "locked_label_row_counts": {label: label_counts[label] for label in labels},
        "candidate_rows": candidate_rows,
        "annotation_contract": (
            "human-created sentence glosses with an additional expert harmonization pass; "
            "fingerspelled letter sequences are excluded from locked-sign overlap matching"
        ),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "split_rows": len(rows),
        "candidate_rows": len(candidate_rows),
        "locked_labels_covered": len(label_counts),
        "missing_locked_labels": [label for label in labels if not label_counts[label]],
        "signer_local_id_counts": payload["signer_local_id_counts"],
        "reserved_devtest_accessed": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev",))
    parser.add_argument(
        "--vocabulary-manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/local/dataset_metadata/2m_flores_asl/dev_locked100_audit.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_new_dataset_search/2m_flores_dev_audit.json"),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
