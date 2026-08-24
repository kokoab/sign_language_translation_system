#!/usr/bin/env python3
"""Freeze a 100-class ASL Citizen manifest at exact ASL-LEX variant level."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


SPLITS = ("train", "val", "test")


def normalize_gloss(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", value.upper())
    return re.sub(r"(?<=[A-Z])\d+$", "", normalized)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class VariantCoverage:
    normalized_gloss: str
    raw_gloss: str
    lex_code: str
    participants: dict[str, set[str]]
    videos: dict[str, int]

    def signer_counts(self) -> dict[str, int]:
        return {split: len(self.participants[split]) for split in SPLITS}


def load_variants(cache_dir: Path) -> tuple[dict[tuple[str, str, str], VariantCoverage], dict[str, str]]:
    grouped: dict[tuple[str, str, str], VariantCoverage] = {}
    source_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = cache_dir / f"asl_citizen_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run scripts/build_ios100_dataset_coverage.py first"
            )
        source_hashes[split] = sha256_file(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_gloss = row["Gloss"].strip()
                lex_code = row["ASL-LEX Code"].strip()
                key = (normalize_gloss(raw_gloss), raw_gloss, lex_code)
                if key not in grouped:
                    grouped[key] = VariantCoverage(
                        normalized_gloss=key[0],
                        raw_gloss=raw_gloss,
                        lex_code=lex_code,
                        participants={name: set() for name in SPLITS},
                        videos={name: 0 for name in SPLITS},
                    )
                grouped[key].participants[split].add(row["Participant ID"].strip())
                grouped[key].videos[split] += 1
    return grouped, source_hashes


def flatten_seed(seed: dict[str, object]) -> list[tuple[str, str]]:
    labels = [
        (category, label)
        for category, values in seed["categories"].items()
        for label in values
    ]
    names = [label for _, label in labels]
    if len(names) != 100 or len(set(names)) != 100:
        raise ValueError(
            f"Citizen100 seed must contain 100 unique labels; got {len(names)} labels "
            f"and {len(set(names))} unique"
        )
    return labels


def choose_variant(
    canonical: str,
    variants: dict[tuple[str, str, str], VariantCoverage],
    aliases: dict[str, list[str]],
    minimums: dict[str, int],
) -> VariantCoverage:
    allowed = {
        normalize_gloss(canonical),
        *(normalize_gloss(value) for value in aliases.get(canonical, [])),
    }
    candidates = [item for key, item in variants.items() if key[0] in allowed]
    eligible = [
        item
        for item in candidates
        if all(len(item.participants[split]) >= minimums[split] for split in SPLITS)
    ]
    if not eligible:
        available = [
            {
                "raw_gloss": item.raw_gloss,
                "lex_code": item.lex_code,
                "signers": item.signer_counts(),
            }
            for item in candidates
        ]
        raise ValueError(
            f"No eligible exact ASL Citizen variant for {canonical}: {available}"
        )

    # Citizen uses dotted raw glosses such as W.H.A.T for fingerspelling. A
    # product label like WHAT should select the lexical sign when one satisfies
    # the same coverage floor, not silently become a fingerspelling class.
    lexical_candidates = [item for item in eligible if "." not in item.raw_gloss]
    if lexical_candidates:
        eligible = lexical_candidates

    def rank(item: VariantCoverage) -> tuple[float, int, int, int, str, str]:
        counts = item.signer_counts()
        balance = min(counts[split] / minimums[split] for split in SPLITS)
        exact = int(item.normalized_gloss == normalize_gloss(canonical))
        return (
            balance,
            sum(counts.values()),
            sum(item.videos.values()),
            exact,
            item.raw_gloss,
            item.lex_code,
        )

    return max(eligible, key=rank)


def build_manifest(seed: dict[str, object], cache_dir: Path) -> dict[str, object]:
    variants, source_hashes = load_variants(cache_dir)
    minimums = {split: int(seed["minimum_signers_per_class"][split]) for split in SPLITS}
    aliases = seed.get("citizen_normalized_aliases", {})
    classes: list[dict[str, object]] = []
    selected_sources: set[tuple[str, str]] = set()
    for class_index, (category, canonical) in enumerate(flatten_seed(seed)):
        chosen = choose_variant(canonical, variants, aliases, minimums)
        source_key = (chosen.raw_gloss, chosen.lex_code)
        if source_key in selected_sources:
            raise ValueError(f"Source variant selected twice: {source_key}")
        selected_sources.add(source_key)
        classes.append(
            {
                "class_index": class_index,
                "canonical_label": canonical,
                "category": category,
                "citizen_raw_gloss": chosen.raw_gloss,
                "citizen_normalized_gloss": chosen.normalized_gloss,
                "citizen_asl_lex_code": chosen.lex_code,
                "signer_counts": chosen.signer_counts(),
                "video_counts": chosen.videos,
            }
        )
    return {
        "name": "citizen100_v17_manifest",
        "status": "metadata_frozen_pending_asl_review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "ASL Citizen v1.0",
        "split_contract": "official participant-disjoint train/val/test",
        "selection_unit": "one exact raw gloss plus ASL-LEX code per class",
        "minimum_signers_per_class": minimums,
        "class_count": len(classes),
        "seed_sha256": hashlib.sha256(
            json.dumps(seed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_csv_sha256": source_hashes,
        "classes": classes,
    }


def write_report(path: Path, manifest: dict[str, object]) -> None:
    classes = manifest["classes"]
    totals = {
        split: sum(int(item["video_counts"][split]) for item in classes)
        for split in SPLITS
    }
    unique_signers = {
        split: sorted({int(item["signer_counts"][split]) for item in classes})
        for split in SPLITS
    }
    lines = [
        "# Citizen100 v17 manifest",
        "",
        f"**Status:** `{manifest['status']}`",
        "",
        "ASL Citizen is the sole primary dataset. Each class maps to exactly one raw",
        "gloss and ASL-LEX code; numeric/lexical variants are not merged.",
        "",
        f"- Classes: {manifest['class_count']}",
        f"- Minimum signer floor: `{json.dumps(manifest['minimum_signers_per_class'])}`",
        f"- Planned videos by official split: `{json.dumps(totals)}`",
        f"- Per-class signer-count values by split: `{json.dumps(unique_signers)}`",
        "- Manual ASL/variant review remains required before a final accuracy claim.",
        "",
        "| # | Canonical | Citizen raw gloss | ASL-LEX | Train/val/test signers | Train/val/test videos |",
        "| ---: | --- | --- | --- | ---: | ---: |",
    ]
    for item in classes:
        signers = item["signer_counts"]
        videos = item["video_counts"]
        lines.append(
            f"| {item['class_index']} | {item['canonical_label']} | "
            f"{item['citizen_raw_gloss']} | `{item['citizen_asl_lex_code']}` | "
            f"{signers['train']}/{signers['val']}/{signers['test']} | "
            f"{videos['train']}/{videos['val']}/{videos['test']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=Path, default=Path("active/v17/citizen100_seed.json")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/local/dataset_metadata")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("artifacts/reports/CITIZEN100_V17_MANIFEST.md")
    )
    args = parser.parse_args()
    seed = json.loads(args.seed.read_text(encoding="utf-8"))
    manifest = build_manifest(seed, args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(args.report, manifest)
    totals = {
        split: sum(item["video_counts"][split] for item in manifest["classes"])
        for split in SPLITS
    }
    print(json.dumps({"classes": manifest["class_count"], "videos": totals}, indent=2))


if __name__ == "__main__":
    main()
