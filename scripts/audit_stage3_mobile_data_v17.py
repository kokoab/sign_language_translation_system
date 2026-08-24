#!/usr/bin/env python3
"""Audit the evidence boundary for the locked-100 mobile Stage-3 naturalizer."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "active/v17/stage2_to_stage3_contract_v17.json"
NATURALIZER = ROOT / "active/v17/stage3_mobile_naturalizer_manifest_v17.json"
FLORES = ROOT / "data/local/dataset_metadata/2m_flores_asl/dev_all_metadata_v17.json"
NCSLGR = ROOT / "data/local/ncslgr_continuous_v17_source/manifest.json"
SYNTHETIC = ROOT / "artifacts/reports/slt_stage3_dataset_final.csv"
STAGE2 = ROOT / "active/v17/stage2_training_manifest_v17.json"
DEFAULT_OUTPUT = ROOT / "artifacts/reports/stage3_mobile_v17/data_and_coverage_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conservative_tokens(text: str) -> list[str]:
    output: list[str] = []
    for raw in text.split():
        token = raw.upper().strip(".,!?;:\"'()[]{}+…")
        token = token.replace("THANK-YOU", "THANKYOU")
        if token:
            output.append(token)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text())
    naturalizer = json.loads(NATURALIZER.read_text())
    vocabulary = set(contract["vocabulary"]["labels"])
    template_keys = {
        tuple(row["glosses"]) for row in naturalizer["reviewed_templates"]
    }

    flores = json.loads(FLORES.read_text())
    ncslgr = json.loads(NCSLGR.read_text())
    if flores.get("source_split") != "dev" or flores.get("reserved_devtest_accessed") is not False:
        raise ValueError("2M-Flores metadata violates the dev-only contract")
    genuine = [
        ("2m_flores_dev", str(row["gloss"]), str(row["sentence"]))
        for row in flores["rows"]
    ] + [
        ("ncslgr", " ".join(map(str, row["main_glosses"])), str(row["english_translation"]))
        for row in ncslgr["items"]
    ]
    genuine_by_source: dict[str, dict[str, int]] = {}
    for source, gloss, text in genuine:
        values = conservative_tokens(gloss)
        payload = genuine_by_source.setdefault(source, {"pairs": 0, "fully_locked_100": 0})
        payload["pairs"] += int(bool(values and text.strip()))
        payload["fully_locked_100"] += int(bool(values) and all(value in vocabulary for value in values))

    with SYNTHETIC.open(encoding="utf-8-sig", newline="") as handle:
        synthetic_rows = list(csv.DictReader(handle))
    synthetic_eligible = [
        row for row in synthetic_rows
        if 1 <= len(row["gloss"].split()) <= 8
        and all(token in vocabulary for token in row["gloss"].split())
    ]

    stage2 = json.loads(STAGE2.read_text())
    coverage: dict[str, dict[str, int]] = {}
    validation_rows = [row for row in stage2["rows"] if row["role"] == "validation"]
    for row in validation_rows:
        source = str(row["source"])
        payload = coverage.setdefault(source, {"rows": 0, "reviewed_template_rows": 0})
        payload["rows"] += 1
        payload["reviewed_template_rows"] += int(tuple(row["target_sequence"]) in template_keys)
    for payload in coverage.values():
        payload["literal_fallback_rows"] = payload["rows"] - payload["reviewed_template_rows"]

    genuine_fully = sum(value["fully_locked_100"] for value in genuine_by_source.values())
    report: dict[str, Any] = {
        "format": "slt_v17_stage3_mobile_data_and_coverage_audit",
        "version": 1,
        "createdUTC": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "decision": "bounded_reviewed_templates_plus_literal_fail_safe",
        "genuinePairs": {
            "total": len(genuine),
            "fullyLocked100": genuine_fully,
            "bySource": genuine_by_source,
        },
        "legacySynthetic": {
            "rows": len(synthetic_rows),
            "fullyLocked100AtMostEightTokens": len(synthetic_eligible),
            "role": "development_reference_only_not_genuine_validation",
        },
        "reviewedTemplates": len(template_keys),
        "stage2GroundTruthValidationCoverage": coverage,
        "stage2GroundTruthValidationRows": len(validation_rows),
        "reviewedTemplateValidationRows": sum(
            value["reviewed_template_rows"] for value in coverage.values()
        ),
        "literalFallbackValidationRows": sum(
            value["literal_fallback_rows"] for value in coverage.values()
        ),
        "sourceSHA256": {
            "stage2Contract": sha256(CONTRACT),
            "naturalizerManifest": sha256(NATURALIZER),
            "floresDevMetadata": sha256(FLORES),
            "ncslgrManifest": sha256(NCSLGR),
            "legacySynthetic": sha256(SYNTHETIC),
            "stage2TrainingManifest": sha256(STAGE2),
        },
        "claimsRejected": [
            "No general or open-domain ASL-to-English translation claim.",
            "No neural Stage-3 promotion from synthetic-only in-vocabulary pairs.",
            "No OOV gloss deletion or partial-reference training.",
            "No physical-iPhone performance claim.",
        ],
        "citizenTestAccessed": False,
        "semlexTestAccessed": False,
        "localTestAccessed": False,
        "twoMFloresDevtestAccessed": False,
    }
    if genuine_fully != 0:
        report["status"] = "fail"
        raise ValueError("new fully in-vocabulary genuine pairs require design review")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": report["status"],
        "genuinePairs": report["genuinePairs"],
        "reviewedTemplates": report["reviewedTemplates"],
        "stage2GroundTruthValidationCoverage": coverage,
    }, indent=2))


if __name__ == "__main__":
    main()
