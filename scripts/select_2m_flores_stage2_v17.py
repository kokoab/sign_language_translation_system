#!/usr/bin/env python3
"""Select a compact, hash-pinned 2M-Flores dev subset for Stage 2 training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


DATASET = "facebook/2M-Flores-ASL"
REVISION = "b450c1a427738e78f06362fc4619674f5d74f774"
TREE_ENDPOINT = "https://huggingface.co/api/datasets"
USER_AGENT = "SLT-v17-dataset-selector/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_dev_tree() -> dict[str, dict[str, object]]:
    dataset_path = quote(DATASET, safe="/")
    revision = quote(REVISION, safe="")
    url = (
        f"{TREE_ENDPOINT}/{dataset_path}/tree/{revision}/data/dev"
        "?recursive=true&expand=false&limit=1000"
    )
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        entries = json.load(response)
    files: dict[str, dict[str, object]] = {}
    for entry in entries:
        if entry.get("type") != "file" or not str(entry.get("path", "")).lower().endswith((".mov", ".mp4")):
            continue
        lfs = entry.get("lfs")
        if not isinstance(lfs, dict) or not lfs.get("oid"):
            raise ValueError(f"missing LFS digest: {entry.get('path')}")
        files[str(entry["path"])] = {
            "source_bytes": int(entry["size"]),
            "source_sha256": str(lfs["oid"]),
        }
    return files


def source_path(video_url: str) -> str:
    marker = "/data/dev/"
    path = urlparse(video_url).path
    if marker not in path:
        raise ValueError(f"unexpected video URL: {video_url}")
    return "data/dev/" + path.split(marker, 1)[1]


def select_rows(
    rows: list[dict[str, object]],
    label_counts: dict[str, int],
    file_metadata: dict[str, dict[str, object]],
    quota: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    labels = sorted(label for label, count in label_counts.items() if count > 0)
    targets = {label: min(quota, int(label_counts[label])) for label in labels}
    matrix = np.zeros((len(labels), len(rows)), dtype=np.float64)
    label_index = {label: index for index, label in enumerate(labels)}
    sizes = np.empty(len(rows), dtype=np.float64)
    enriched: list[dict[str, object]] = []
    for column, row in enumerate(rows):
        path = source_path(str(row["video_url"]))
        if path not in file_metadata:
            raise ValueError(f"missing tree metadata: {path}")
        metadata = file_metadata[path]
        sizes[column] = int(metadata["source_bytes"])
        for label in row["matched_locked_labels"]:
            matrix[label_index[str(label)], column] = 1.0
        enriched.append({
            **row,
            "source_path": path,
            "source_bytes": int(metadata["source_bytes"]),
            "source_sha256": str(metadata["source_sha256"]),
            "derived_relative_path": f"data/dev/{Path(path).stem}.mp4",
        })

    # Minimize source bytes. A very small row-id term makes equal-size outcomes stable.
    objective = sizes / max(float(sizes.max()), 1.0)
    objective += np.arange(len(rows), dtype=np.float64) * 1e-10
    lower = np.asarray([targets[label] for label in labels], dtype=np.float64)
    result = milp(
        c=objective,
        integrality=np.ones(len(rows), dtype=np.int8),
        bounds=Bounds(np.zeros(len(rows)), np.ones(len(rows))),
        constraints=LinearConstraint(csr_matrix(matrix), lower, np.full(len(labels), np.inf)),
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"selection optimization failed: {result.message}")
    selected = [row for row, value in zip(enriched, result.x) if value >= 0.5]
    achieved = {
        label: sum(label in row["matched_locked_labels"] for row in selected)
        for label in labels
    }
    if any(achieved[label] < targets[label] for label in labels):
        raise RuntimeError("optimizer result does not satisfy the label quotas")
    return sorted(selected, key=lambda row: int(row["id"])), targets


def run(args: argparse.Namespace) -> dict[str, object]:
    audit = json.loads(args.audit.read_text())
    if audit.get("split") != "dev" or audit.get("reserved_split_not_accessed") != "devtest":
        raise ValueError("audit must be the dev-only artifact with devtest reserved")
    tree = fetch_dev_tree()
    selected, targets = select_rows(
        audit["candidate_rows"], audit["locked_label_row_counts"], tree, args.quota
    )
    achieved = {
        label: sum(label in row["matched_locked_labels"] for row in selected)
        for label in targets
    }
    total_bytes = sum(int(row["source_bytes"]) for row in selected)
    payload = {
        "format": "slt_stage2_2m_flores_asl_selection_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "revision": REVISION,
        "license": "CC-BY-SA-4.0",
        "source_split": "dev",
        "reserved_devtest_accessed": False,
        "selection_policy": (
            "minimum-source-byte binary optimization satisfying up to the requested "
            "per-label quota; rare labels retain every available matching row"
        ),
        "requested_per_label_quota": args.quota,
        "effective_label_targets": targets,
        "achieved_label_counts": achieved,
        "selected_rows": len(selected),
        "selected_source_bytes": total_bytes,
        "selected_source_gib": total_bytes / (1024 ** 3),
        "training_contract": (
            "preserve complete ordered gloss sequences and train an expanded Stage 2 "
            "vocabulary; matched locked labels are selection metadata only"
        ),
        "derived_video_contract": (
            "aspect-preserving scale to fit within 1280x720, 30 fps, H.264; never crop "
            "or stretch; verify duration and decode; retain source and derived hashes"
        ),
        "rows": selected,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    result = {
        "output": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "selected_rows": len(selected),
        "selected_source_bytes": total_bytes,
        "selected_source_gib": total_bytes / (1024 ** 3),
        "covered_labels": len(targets),
        "minimum_achieved_quota": min(achieved.values()),
        "reserved_devtest_accessed": False,
    }
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path,
        default=Path("data/local/dataset_metadata/2m_flores_asl/dev_locked100_audit.json"),
    )
    parser.add_argument("--quota", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/local/dataset_metadata/2m_flores_asl/dev_selected_v17.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.quota < 1:
        raise ValueError("--quota must be positive")
    run(args)


if __name__ == "__main__":
    main()
