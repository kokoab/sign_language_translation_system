#!/usr/bin/env python3
"""Build a signer-coverage report for the utility-focused iOS-100 proposal.

The report uses cached metadata only. It does not download dataset videos.
Counts across independent datasets are working coverage estimates because the
public anonymous participant namespaces cannot be cross-deduplicated.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def normalize_gloss(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", value.upper())
    return re.sub(r"(?<=[A-Z])\d+$", "", normalized)


def load_citizen(cache_dir: Path) -> dict[str, dict[str, object]]:
    coverage: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            split: {
                "participants": set(),
                "videos": 0,
                "raw_glosses": set(),
                "lex_codes": set(),
            }
            for split in SPLITS
        }
    )
    for split in SPLITS:
        path = cache_dir / f"asl_citizen_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run scripts/build_ios100_dataset_coverage.py first."
            )
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = normalize_gloss(row["Gloss"])
                item = coverage[key][split]
                item["participants"].add(row["Participant ID"].strip())
                item["videos"] += 1
                item["raw_glosses"].add(row["Gloss"].strip())
                item["lex_codes"].add(row["ASL-LEX Code"].strip())
    return coverage


def load_popsign(cache_dir: Path) -> dict[str, tuple[str, dict[str, object]]]:
    path = cache_dir / "popsign_v1_game_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_ios100_dataset_coverage.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalize_gloss(raw_gloss): (raw_gloss, split_data)
        for raw_gloss, split_data in payload["signs"].items()
    }


def flatten_labels(config: dict[str, object]) -> list[tuple[str, str]]:
    labels = [
        (category, label)
        for category, values in config["canonical_labels"].items()
        for label in values
    ]
    names = [label for _, label in labels]
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError(
            f"Expected exactly 100 unique canonical labels; got "
            f"{len(names)} labels and {len(set(names))} unique labels."
        )
    return labels


def source_keys(config: dict[str, object], label: str, source: str) -> set[str]:
    aliases = config.get("source_aliases", {}).get(label, {}).get(source, [])
    return {normalize_gloss(label), *(normalize_gloss(alias) for alias in aliases)}


def union_citizen(
    coverage: dict[str, dict[str, object]], keys: set[str], split: str
) -> dict[str, object]:
    result = {
        "participants": set(),
        "videos": 0,
        "raw_glosses": set(),
        "lex_codes": set(),
    }
    for key in keys:
        item = coverage.get(key, {}).get(split)
        if not item:
            continue
        result["participants"].update(item["participants"])
        result["videos"] += item["videos"]
        result["raw_glosses"].update(item["raw_glosses"])
        result["lex_codes"].update(item["lex_codes"])
    return result


def union_popsign(
    coverage: dict[str, tuple[str, dict[str, object]]], keys: set[str], split: str
) -> dict[str, object]:
    result = {"participants": set(), "videos": 0, "raw_glosses": set()}
    for key in keys:
        item = coverage.get(key)
        if not item:
            continue
        raw_gloss, split_map = item
        result["raw_glosses"].add(raw_gloss)
        result["participants"].update(split_map[split]["participants"])
        result["videos"] += split_map[split]["videos"]
    return result


def build_rows(
    config: dict[str, object],
    citizen: dict[str, dict[str, object]],
    popsign: dict[str, tuple[str, dict[str, object]]],
    current_vocabulary: set[str],
) -> list[dict[str, object]]:
    assumptions = config["coverage_assumptions"]
    local_train = int(assumptions["current_local_train_signers"])
    minimums = {
        "train": int(assumptions["minimum_train_signers"]),
        "val": int(assumptions["minimum_validation_signers"]),
        "test": int(assumptions["minimum_test_signers"]),
    }
    rows: list[dict[str, object]] = []
    for category, label in flatten_labels(config):
        citizen_keys = source_keys(config, label, "asl_citizen")
        popsign_keys = source_keys(config, label, "popsign")
        current_keys = source_keys(config, label, "current_v16")
        present_current_labels = sorted(current_keys & current_vocabulary)
        local_signers = local_train if present_current_labels else 0
        citizen_data = {
            split: union_citizen(citizen, citizen_keys, split) for split in SPLITS
        }
        popsign_data = {
            split: union_popsign(popsign, popsign_keys, split) for split in SPLITS
        }
        working = {
            split: (
                len(citizen_data[split]["participants"])
                + len(popsign_data[split]["participants"])
                + (local_signers if split == "train" else 0)
            )
            for split in SPLITS
        }
        deficits = {
            split: max(0, minimums[split] - working[split]) for split in SPLITS
        }
        raw_glosses = sorted(
            {
                raw
                for split in SPLITS
                for raw in citizen_data[split]["raw_glosses"]
            }
        )
        lex_codes = sorted(
            {
                code
                for split in SPLITS
                for code in citizen_data[split]["lex_codes"]
                if code
            }
        )
        popsign_glosses = sorted(
            {
                raw
                for split in SPLITS
                for raw in popsign_data[split]["raw_glosses"]
            },
            key=str.casefold,
        )
        variant_audit = len(raw_glosses) > 1 or len(lex_codes) > 1
        rows.append(
            {
                "category": category,
                "canonical_gloss": label,
                "in_current_v16": bool(present_current_labels),
                "current_v16_glosses": "|".join(present_current_labels),
                "citizen_raw_glosses": "|".join(raw_glosses),
                "citizen_lex_codes": "|".join(lex_codes),
                "popsign_glosses": "|".join(popsign_glosses),
                **{
                    f"citizen_{split}_signers": len(
                        citizen_data[split]["participants"]
                    )
                    for split in SPLITS
                },
                **{
                    f"popsign_{split}_signers": len(
                        popsign_data[split]["participants"]
                    )
                    for split in SPLITS
                },
                "assumed_local_train_signers": local_signers,
                **{f"working_{split}_signers": working[split] for split in SPLITS},
                **{f"additional_{split}_signers_needed": deficits[split] for split in SPLITS},
                "meets_working_20_5_5": all(value == 0 for value in deficits.values()),
                "requires_variant_audit": variant_audit,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path, config: dict[str, object], rows: list[dict[str, object]]
) -> None:
    passed = sum(bool(row["meets_working_20_5_5"]) for row in rows)
    popsign_covered = sum(bool(row["popsign_glosses"]) for row in rows)
    citizen_covered = sum(bool(row["citizen_raw_glosses"]) for row in rows)
    variant_count = sum(bool(row["requires_variant_audit"]) for row in rows)
    current_count = sum(bool(row["in_current_v16"]) for row in rows)
    train_deficit = sum(int(row["additional_train_signers_needed"]) for row in rows)
    val_deficit = sum(int(row["additional_val_signers_needed"]) for row in rows)
    test_deficit = sum(int(row["additional_test_signers_needed"]) for row in rows)

    lines = [
        "# iOS-100 utility vocabulary proposal",
        "",
        f"**Status:** `{config['status']}`  ",
        "**Basis:** cached ASL Citizen and PopSign metadata; no videos downloaded",
        "",
        "This is an accuracy-first, conversational vocabulary. It deliberately keeps "
        "high-utility signs that are absent from PopSign instead of optimizing only for "
        "the easiest metadata intersection.",
        "",
        "## Coverage summary",
        "",
        f"- Canonical labels: **{len(rows)}**",
        f"- Reuse an exact or declared-equivalent current v16 class: **{current_count}**",
        f"- Covered by ASL Citizen metadata: **{citizen_covered}**",
        f"- Covered by PopSign game metadata: **{popsign_covered}**",
        f"- Meet the working 20 train / 5 validation / 5 test estimate: **{passed}/{len(rows)}**",
        f"- Require ASL-LEX/raw-label variant review: **{variant_count}**",
        f"- Sum of per-class signer deficits: train **{train_deficit}**, validation "
        f"**{val_deficit}**, test **{test_deficit}**",
        "",
        "The deficit totals are planning units, not necessarily that many unique new "
        "people: one newly recorded signer can fill a deficit for many signs. Counts "
        "from independent anonymous datasets are added only as a working estimate; "
        "cross-dataset identity overlap cannot be proven from public metadata.",
        "",
        "## Proposed 100",
        "",
        "| Category | Signs |",
        "| --- | --- |",
    ]
    for category, labels in config["canonical_labels"].items():
        lines.append(f"| {category.replace('_', ' ').title()} | {', '.join(labels)} |")

    lines.extend(
        [
            "",
            "## Signs still below the working threshold",
            "",
            "| Sign | Working train/val/test | Additional train/val/test signers |",
            "| --- | ---: | ---: |",
        ]
    )
    deficit_rows = [row for row in rows if not row["meets_working_20_5_5"]]
    if deficit_rows:
        for row in deficit_rows:
            lines.append(
                f"| {row['canonical_gloss']} | "
                f"{row['working_train_signers']}/{row['working_val_signers']}/"
                f"{row['working_test_signers']} | "
                f"{row['additional_train_signers_needed']}/"
                f"{row['additional_val_signers_needed']}/"
                f"{row['additional_test_signers_needed']} |"
            )
    else:
        lines.append("| None | - | - |")

    lines.extend(
        [
            "",
            "## Mandatory review before training",
            "",
            *[f"- {item}" for item in config["required_review"]],
            "",
            "## Machine-readable detail",
            "",
            "See `artifacts/reports/ios100_vocabulary_proposal.csv` for per-source "
            "signer counts, aliases, ASL-LEX codes, and deficits for all 100 signs.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("active/v16/ios100_vocabulary_proposal.json")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/local/dataset_metadata")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("models/manifest_v16.json")
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/IOS100_VOCABULARY_PROPOSAL.md")
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("artifacts/reports/ios100_vocabulary_proposal.csv")
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    current_vocabulary = set(manifest if isinstance(manifest, dict) else manifest)
    rows = build_rows(
        config,
        load_citizen(args.cache_dir),
        load_popsign(args.cache_dir),
        current_vocabulary,
    )
    write_csv(args.csv, rows)
    write_markdown(args.report, config, rows)
    print(f"Wrote {args.report}")
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
