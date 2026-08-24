#!/usr/bin/env python3
"""Build the iOS-100 dataset coverage report from official remote metadata.

This script deliberately avoids downloading dataset videos. It reads the three
small ASL Citizen split CSVs from the remote ZIP via HTTP byte-range requests
and reads PopSign's public per-sign preview metadata.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ASL_CITIZEN_URL = (
    "https://download.microsoft.com/download/b/8/8/"
    "b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
)
ASL_CITIZEN_SIZE = 45_924_134_223
ASL_CITIZEN_SPLITS = {
    "train": "ASL_Citizen/splits/train.csv",
    "val": "ASL_Citizen/splits/val.csv",
    "test": "ASL_Citizen/splits/test.csv",
}
POPSIGN_INDEX_URL = "https://signdata.cc.gatech.edu/data/popsign_v1_0/game/train/"
POPSIGN_PAGE = (
    "https://signdata.cc.gatech.edu/view/datasets/"
    "popsign_v1_0/game/{sign}/index.html"
)
USER_AGENT = "SLT-iOS100-metadata-audit/1.0"
SPLITS = ("train", "val", "test")


def fetch(url: str, *, byte_range: tuple[int, int] | None = None,
          timeout: int = 120, retries: int = 3) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # Network retries are intentionally broad.
            error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {error}") from error


class HTTPRangeReader(io.RawIOBase):
    """Minimal seekable byte-range reader used by Python's zipfile module."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.size - self.position
        if size == 0 or self.position >= self.size:
            return b""
        end = min(self.size - 1, self.position + size - 1)
        data = fetch(self.url, byte_range=(self.position, end))
        self.position += len(data)
        return data


def normalize_gloss(value: str) -> str:
    """Normalize formatting only; do not silently merge semantic aliases."""
    normalized = re.sub(r"[^A-Z0-9]", "", value.upper())
    # ASL Citizen uses suffixes such as SPECIAL1 for lexical variants. PopSign
    # does not use numeric vocabulary labels, so retain the raw label elsewhere
    # but use the base form for a candidate match that still requires audit.
    return re.sub(r"(?<=[A-Z])\d+$", "", normalized)


def load_asl_citizen(cache_dir: Path, refresh: bool) -> list[dict[str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        split for split in SPLITS
        if refresh or not (cache_dir / f"asl_citizen_{split}.csv").exists()
    ]
    if missing:
        with zipfile.ZipFile(HTTPRangeReader(ASL_CITIZEN_URL, ASL_CITIZEN_SIZE)) as archive:
            for split in missing:
                payload = archive.read(ASL_CITIZEN_SPLITS[split])
                (cache_dir / f"asl_citizen_{split}.csv").write_bytes(payload)

    rows: list[dict[str, str]] = []
    for split in SPLITS:
        path = cache_dir / f"asl_citizen_{split}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "split": split,
                    "participant": row["Participant ID"].strip(),
                    "video": row["Video file"].strip(),
                    "gloss": row["Gloss"].strip(),
                    "lex_code": row["ASL-LEX Code"].strip(),
                })
    return rows


def parse_popsign_index() -> list[str]:
    page = fetch(POPSIGN_INDEX_URL).decode("utf-8", "replace")
    signs = re.findall(r'href="([^"/]+)\.tar/"', page)
    return sorted(set(signs), key=str.casefold)


def participant_from_popsign_filename(filename: str, sign: str) -> str:
    marker = f"-{sign}-"
    if marker in filename:
        return filename.split(marker, 1)[0]
    match = re.match(r"^(.*?)-[^-]+-20\d{2}_", filename)
    if not match:
        raise ValueError(f"Unrecognized PopSign filename: {filename}")
    return match.group(1)


def parse_popsign_page(sign: str) -> dict[str, dict[str, list[str] | int]]:
    url = POPSIGN_PAGE.format(sign=sign.lower())
    page = fetch(url).decode("utf-8", "replace")
    match = re.search(r"sourceMap=JSON\.parse\('(.*?)'\),", page, re.DOTALL)
    if not match:
        raise ValueError(f"sourceMap not found for {sign}: {url}")
    source_map = json.loads(html.unescape(match.group(1)))
    result: dict[str, dict[str, list[str] | int]] = {}
    for split in SPLITS:
        original_names = list(source_map.get(split, {}).get("orig_name", {}).values())
        participants = sorted({
            participant_from_popsign_filename(name, sign)
            for name in original_names
        })
        result[split] = {
            "videos": len(original_names),
            "participants": participants,
        }
    return result


def load_popsign(cache_dir: Path, refresh: bool, workers: int) -> dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "popsign_v1_game_metadata.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    signs = parse_popsign_index()
    records: dict[str, object] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_popsign_page, sign): sign for sign in signs}
        for future in as_completed(futures):
            sign = futures[future]
            try:
                records[sign] = future.result()
            except Exception as exc:
                errors[sign] = str(exc)
    if errors:
        details = "\n".join(f"{key}: {value}" for key, value in sorted(errors.items()))
        raise RuntimeError(f"Failed to read {len(errors)} PopSign pages:\n{details}")

    payload: dict[str, object] = {
        "source": "PopSign ASL v1.0 public game preview metadata",
        "sign_count": len(records),
        "signs": dict(sorted(records.items(), key=lambda item: item[0].casefold())),
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def citizen_coverage(rows: list[dict[str, str]]) -> dict[str, object]:
    coverage: dict[str, object] = defaultdict(
        lambda: {
            split: {"participants": set(), "videos": 0, "raw_glosses": set(), "lex_codes": set()}
            for split in SPLITS
        }
    )
    for row in rows:
        key = normalize_gloss(row["gloss"])
        split_data = coverage[key][row["split"]]
        split_data["participants"].add(row["participant"])
        split_data["videos"] += 1
        split_data["raw_glosses"].add(row["gloss"])
        split_data["lex_codes"].add(row["lex_code"])

    serialized: dict[str, object] = {}
    for gloss, split_map in coverage.items():
        serialized[gloss] = {}
        for split, data in split_map.items():
            serialized[gloss][split] = {
                "participants": sorted(data["participants"]),
                "videos": data["videos"],
                "raw_glosses": sorted(data["raw_glosses"]),
                "lex_codes": sorted(data["lex_codes"]),
            }
    return serialized


def load_current_vocabulary(manifest_path: Path) -> dict[str, list[str]]:
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = defaultdict(list)
    for gloss in manifest:
        result[normalize_gloss(gloss)].append(gloss)
    return dict(result)


def build_rows(popsign: dict[str, object], citizen: dict[str, object],
               current: dict[str, list[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for popsign_gloss, popsign_data in popsign["signs"].items():
        normalized = normalize_gloss(popsign_gloss)
        if normalized not in citizen:
            continue
        citizen_data = citizen[normalized]
        row: dict[str, object] = {
            "normalized_gloss": normalized,
            "popsign_gloss": popsign_gloss,
            "citizen_raw_glosses": sorted({
                raw
                for split in SPLITS
                for raw in citizen_data[split]["raw_glosses"]
            }),
            "citizen_lex_codes": sorted({
                code
                for split in SPLITS
                for code in citizen_data[split]["lex_codes"]
            }),
            "current_v16_glosses": current.get(normalized, []),
            "in_current_v16": normalized in current,
        }
        for split in SPLITS:
            row[f"popsign_{split}_signers"] = len(popsign_data[split]["participants"])
            row[f"popsign_{split}_videos"] = popsign_data[split]["videos"]
            row[f"citizen_{split}_signers"] = len(citizen_data[split]["participants"])
            row[f"citizen_{split}_videos"] = citizen_data[split]["videos"]
            row[f"combined_{split}_signers"] = (
                row[f"popsign_{split}_signers"] + row[f"citizen_{split}_signers"]
            )
        row["meets_20_5_5_combined"] = (
            row["combined_train_signers"] >= 20
            and row["combined_val_signers"] >= 5
            and row["combined_test_signers"] >= 5
        )
        row["requires_variant_audit"] = (
            len(row["citizen_raw_glosses"]) > 1
            or len(row["citizen_lex_codes"]) > 1
        )
        row["coverage_score"] = (
            10_000 * int(row["in_current_v16"])
            + 1_000 * int(row["meets_20_5_5_combined"])
            + min(row["combined_train_signers"], 50) * 10
            + min(row["combined_val_signers"], 20) * 5
            + min(row["combined_test_signers"], 20) * 5
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (-row["coverage_score"], row["normalized_gloss"]))


CSV_FIELDS = [
    "candidate_rank", "selected_candidate_100", "normalized_gloss", "popsign_gloss",
    "citizen_raw_glosses", "citizen_lex_codes", "current_v16_glosses", "in_current_v16",
    "popsign_train_signers", "popsign_val_signers", "popsign_test_signers",
    "citizen_train_signers", "citizen_val_signers", "citizen_test_signers",
    "combined_train_signers", "combined_val_signers", "combined_test_signers",
    "popsign_train_videos", "popsign_val_videos", "popsign_test_videos",
    "citizen_train_videos", "citizen_val_videos", "citizen_test_videos",
    "meets_20_5_5_combined", "requires_variant_audit", "coverage_score",
]


def csv_value(value: object) -> object:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def write_reports(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [row for row in rows if row["meets_20_5_5_combined"]]
    candidates = eligible[:100]
    selected = {row["normalized_gloss"] for row in candidates}
    rank = {row["normalized_gloss"]: index + 1 for index, row in enumerate(candidates)}

    csv_path = output_dir / "ios100_dataset_coverage.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            record = {field: csv_value(row.get(field, "")) for field in CSV_FIELDS}
            record["candidate_rank"] = rank.get(row["normalized_gloss"], "")
            record["selected_candidate_100"] = row["normalized_gloss"] in selected
            writer.writerow(record)

    json_path = output_dir / "ios100_candidate_100.json"
    summary = {
        "status": "metadata candidate list; manual ASL variant and utility audit still required",
        "intersection_signs": len(rows),
        "eligible_20_5_5_signs": len(eligible),
        "candidate_count": len(candidates),
        "candidate_current_v16_overlap": sum(bool(row["in_current_v16"]) for row in candidates),
        "candidate_requires_variant_audit": sum(bool(row["requires_variant_audit"]) for row in candidates),
        "candidates": candidates,
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_path = output_dir / "IOS100_DATASET_COVERAGE_REPORT.md"
    lines = [
        "# iOS-100 Dataset Coverage Report",
        "",
        "**Status:** Metadata-derived candidate set; not the final vocabulary.",
        "",
        "## Result",
        "",
        f"- Exact normalized PopSign/ASL Citizen intersection: **{len(rows)} signs**",
        f"- Signs meeting combined 20 train / 5 validation / 5 test signer coverage: **{len(eligible)}**",
        f"- Candidate list produced: **{len(candidates)} signs**",
        f"- Candidates already present in v16: **{summary['candidate_current_v16_overlap']}**",
        f"- Candidates with multiple ASL Citizen labels or lexical codes: **{summary['candidate_requires_variant_audit']}**",
        "",
        "PopSign and ASL Citizen participant IDs are separate namespaces, so combined counts are sums",
        "across the two datasets. Dataset identity must remain attached to every sample.",
        "",
        "## Candidate 100",
        "",
        "| Rank | Gloss | Current v16 | Train signers P/C/Total | Val P/C/Total | Test P/C/Total | Variant audit |",
        "| ---: | --- | :---: | ---: | ---: | ---: | :---: |",
    ]
    for index, row in enumerate(candidates, 1):
        lines.append(
            f"| {index} | {row['normalized_gloss']} | "
            f"{'yes' if row['in_current_v16'] else 'no'} | "
            f"{row['popsign_train_signers']}/{row['citizen_train_signers']}/{row['combined_train_signers']} | "
            f"{row['popsign_val_signers']}/{row['citizen_val_signers']}/{row['combined_val_signers']} | "
            f"{row['popsign_test_signers']}/{row['citizen_test_signers']}/{row['combined_test_signers']} | "
            f"{'yes' if row['requires_variant_audit'] else 'no'} |"
        )
    lines += [
        "",
        "## Required review before download",
        "",
        "1. Confirm the PopSign game sign and every included ASL Citizen ASL-LEX code are the same sign variant.",
        "2. Replace low-utility or child-vocabulary signs with eligible alternatives where appropriate.",
        "3. Mark one-handed versus two-handed forms and avoid silently treating them as equivalent.",
        "4. Run Apple Vision detection coverage after downloading a small audit sample.",
        "5. Keep the official signer partitions locked.",
        "",
        "The complete machine-readable coverage table is `ios100_dataset_coverage.csv`.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("data/local/dataset_metadata"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/reports"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("models/manifest_v16.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    citizen_rows = load_asl_citizen(args.cache_dir, args.refresh)
    popsign = load_popsign(args.cache_dir, args.refresh, max(1, args.workers))
    citizen = citizen_coverage(citizen_rows)
    current = load_current_vocabulary(args.manifest)
    rows = build_rows(popsign, citizen, current)
    paths = write_reports(rows, args.output_dir)
    print(f"ASL Citizen rows: {len(citizen_rows)}")
    print(f"PopSign signs: {popsign['sign_count']}")
    print(f"Intersection signs: {len(rows)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
