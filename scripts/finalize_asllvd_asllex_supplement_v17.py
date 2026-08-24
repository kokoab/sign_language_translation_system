#!/usr/bin/env python3
"""Finalize an exact-variant ASLLVD manifest after v17 extraction.

Rows whose declared feature archive is absent or invalid are retained for provenance
but made ineligible. The trainer can then consume only verified feature archives.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from active.v17.extract_v17 import load_v17_result  # noqa: E402
from active.v17.schema_v17 import V17Config, schema_fingerprint  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path("data/local/asllvd_asllex_v17/exact_variant_manifest.json"),
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if payload.get("format") != "slt_v17_asllvd_asllex_exact_supplement":
        raise ValueError("not an ASLLVD exact-variant supplement manifest")
    if payload.get("citizen_test_accessed") is not False:
        raise ValueError("Citizen test isolation is not proven")
    if payload.get("semlex_test_accessed") is not False:
        raise ValueError("SemLex test isolation is not proven")

    valid = 0
    rejected = 0
    download_rejected = 0
    extraction_rejected = 0
    errors: list[dict[str, str]] = []
    config = V17Config()
    for row in payload.get("videos", []):
        feature_path = Path(str(row.get("feature_path", "")))
        if row.get("consensus_tier") == "download_failed":
            reason = str(row.get("download_error", "download failed validation"))
            row["training_eligible"] = False
            row["feature_sha256"] = ""
            row["feature_rejection_reason"] = reason
            errors.append(
                {
                    "canonical_label": str(row.get("canonical_label", "")),
                    "clip_filename": str(row.get("clip_filename", "")),
                    "reason": reason,
                }
            )
            download_rejected += 1
            rejected += 1
            continue
        try:
            if not feature_path.is_file():
                raise FileNotFoundError("feature archive was not produced")
            load_v17_result(feature_path, config)
        except Exception as exc:
            row["training_eligible"] = False
            row["consensus_tier"] = "extraction_failed"
            row["feature_sha256"] = ""
            row["feature_rejection_reason"] = str(exc)
            errors.append(
                {
                    "canonical_label": str(row.get("canonical_label", "")),
                    "clip_filename": str(row.get("clip_filename", "")),
                    "reason": str(exc),
                }
            )
            rejected += 1
            extraction_rejected += 1
            continue
        row["training_eligible"] = True
        row["consensus_tier"] = "official_asllex_signbank_exact"
        row["feature_sha256"] = sha256_file(feature_path)
        row.pop("feature_rejection_reason", None)
        valid += 1

    payload["feature_finalized_utc"] = datetime.now(timezone.utc).isoformat()
    payload["feature_schema_fingerprint"] = schema_fingerprint(config)
    payload["training_eligible_clips"] = valid
    payload["feature_rejected_clips"] = rejected
    payload["download_rejected_clips"] = download_rejected
    payload["extraction_rejected_clips"] = extraction_rejected
    payload["feature_errors"] = errors
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "training_eligible_clips": valid,
                "feature_rejected_clips": rejected,
                "download_rejected_clips": download_rejected,
                "extraction_rejected_clips": extraction_rejected,
            },
            indent=2,
        )
    )
    if valid == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
