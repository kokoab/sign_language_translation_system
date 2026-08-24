#!/usr/bin/env python3
"""Download official ASLLVD clips with exact ASL-LEX/SignBank variants.

ASL-LEX 2.0 supplies the SignBank annotation ID pinned by each Citizen class.
ASLLVD's official workbook supplies a unique variant gloss, signer identity, and
direct per-signer movie URL. A row is admitted only when the nonempty annotation ID
equals the ASLLVD variant after removing repetition markers documented as equivalent.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.request import Request, urlopen

import cv2
import openpyxl


WORKBOOK_PAGE = "https://www.bu.edu/asllrp/av/dai-asllvd.html"
WORKBOOK_URL = (
    "https://www.bu.edu/asllrp/"
    "dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx"
)
HYPERLINK_RE = re.compile(r'^=HYPERLINK\("([^"]+)",\s*"[^"]*"\)$')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def formula_url(value: object) -> str:
    match = HYPERLINK_RE.match(str(value or ""))
    if match is None:
        return ""
    return match.group(1).replace("http://", "https://", 1)


def probe_video(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"could not open downloaded video: {path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames < 4:
        raise ValueError(f"invalid downloaded video: {path}")
    return {"width": width, "height": height, "fps": fps, "frames": frames}


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "SLT-v17-research/1.0"})
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temporary)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, destination)


def workbook_rows(path: Path) -> list[dict[str, object]]:
    workbook = openpyxl.load_workbook(path, read_only=False, data_only=False)
    sheet = workbook["Sheet1"]
    rows: list[dict[str, object]] = []
    for row_number, values in enumerate(
        sheet.iter_rows(min_row=2, values_only=True), start=2
    ):
        signer = str(values[2] or "").strip()
        main_gloss = str(values[3] or "").strip()
        variant = str(values[4] or "").strip()
        url = formula_url(values[11])
        if signer and main_gloss and variant and url:
            rows.append(
                {
                    "workbook_row": row_number,
                    "signer": signer,
                    "main_gloss": main_gloss,
                    "variant_gloss": variant,
                    "url": url,
                    "session": str(values[12] or "").strip(),
                    "scene": values[13],
                    "start_frame": values[14],
                    "end_frame": values[15],
                }
            )
    workbook.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--asllex", type=Path,
        default=Path("data/local/dataset_metadata/asllex2_official/signdata.csv"),
    )
    parser.add_argument(
        "--workbook", type=Path,
        default=Path(
            "data/local/dataset_metadata/asllrp_signbank/"
            "dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/local/asllvd_asllex_v17")
    )
    parser.add_argument("--cap-per-class", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.cap_per_class < 1:
        raise ValueError("cap-per-class must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    asllex_by_code = {
        row["Code"]: row for row in read_csv(args.asllex, "latin-1")
    }
    asllvd = workbook_rows(args.workbook)
    selected: list[dict[str, object]] = []
    class_summary: list[dict[str, object]] = []
    for item in sorted(manifest["classes"], key=lambda value: int(value["class_index"])):
        code = str(item["citizen_asl_lex_code"])
        asllex = asllex_by_code.get(code)
        if asllex is None:
            raise ValueError(f"ASL-LEX code is absent: {code}")
        annotation = asllex["SignBankAnnotationID"].strip()
        candidates = [
            row for row in asllvd
            if annotation and str(row["variant_gloss"]).rstrip("+") == annotation
        ]
        by_signer: dict[str, dict[str, object]] = {}
        for row in sorted(
            candidates,
            key=lambda value: (
                str(value["variant_gloss"]).count("+"),
                str(value["signer"]),
                str(value["url"]),
            ),
        ):
            by_signer.setdefault(str(row["signer"]), row)
        chosen = list(by_signer.values())[: args.cap_per_class]
        class_summary.append(
            {
                "class_index": int(item["class_index"]),
                "canonical_label": item["canonical_label"],
                "citizen_raw_gloss": item["citizen_raw_gloss"],
                "citizen_asl_lex_code": code,
                "asllex_entry_id": asllex["EntryID"],
                "signbank_annotation_id": annotation,
                "exact_workbook_candidates": len(candidates),
                "distinct_signers": len(by_signer),
                "selected_clips": len(chosen),
            }
        )
        for row in chosen:
            source_name = Path(str(row["url"])).name
            signer = str(row["signer"])
            filename = f"{signer}_{source_name}"
            selected.append(
                {
                    **row,
                    "class_index": int(item["class_index"]),
                    "canonical_label": item["canonical_label"],
                    "citizen_raw_gloss": item["citizen_raw_gloss"],
                    "citizen_asl_lex_code": code,
                    "asllex_entry_id": asllex["EntryID"],
                    "signbank_annotation_id": annotation,
                    "clip_filename": filename,
                }
            )

    def fetch(row: dict[str, object]) -> str:
        label = str(row["canonical_label"])
        destination = args.output_root / "raw" / label / str(row["clip_filename"])
        media: dict[str, int | float] | None = None
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                if destination.is_file():
                    media = probe_video(destination)
                else:
                    download(str(row["url"]), destination)
                    media = probe_video(destination)
                break
            except Exception as exc:
                last_error = exc
                destination.unlink(missing_ok=True)
                if attempt < 3:
                    time.sleep(float(attempt))
        if media is None:
            row.update(
                {
                    "raw_path": str(destination),
                    "feature_path": str(
                        args.output_root / "landmarks" / label
                        / f"{destination.stem}.v17.npz"
                    ),
                    "bytes": 0,
                    "sha256": "",
                    "download_error": str(last_error),
                    "consensus_tier": "download_failed",
                    "training_eligible": False,
                }
            )
            return f"REJECTED {label}/{destination.name}: {last_error}"
        row.update(
            {
                "raw_path": str(destination),
                "feature_path": str(
                    args.output_root / "landmarks" / label / f"{destination.stem}.v17.npz"
                ),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                **media,
                "consensus_tier": "official_asllex_signbank_exact",
                "training_eligible": True,
            }
        )
        return f"{label}/{destination.name}"

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, name in enumerate(executor.map(fetch, selected), start=1):
            print(f"[{index}/{len(selected)}] {name}", flush=True)

    payload = {
        "format": "slt_v17_asllvd_asllex_exact_supplement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "official-ASL-LEX-cross-referenced ASLLVD train-only supplement",
        "split_eligibility": "train_only_official_asllex_signbank_cross_reference",
        "license": "ASLLRP research-only noncommercial terms; do not redistribute clips",
        "source_page": WORKBOOK_PAGE,
        "workbook_url": WORKBOOK_URL,
        "manifest_sha256": sha256_file(args.manifest),
        "asllex_metadata_sha256": sha256_file(args.asllex),
        "asllvd_workbook_sha256": sha256_file(args.workbook),
        "variant_contract": (
            "nonempty ASL-LEX SignBankAnnotationID equals ASLLVD Gloss Variant after "
            "removing only repetition plus markers documented as equivalent"
        ),
        "signer_policy": "at most one clip per named ASLLVD consultant per class",
        "cap_per_class": args.cap_per_class,
        "selected_clips": len(selected),
        "selected_classes": len({row["canonical_label"] for row in selected}),
        "downloaded_clips": sum(
            row.get("consensus_tier") == "official_asllex_signbank_exact"
            for row in selected
        ),
        "download_rejected_clips": sum(
            row.get("consensus_tier") == "download_failed" for row in selected
        ),
        "signers": sorted({str(row["signer"]) for row in selected}),
        "classes": class_summary,
        "videos": selected,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
    }
    output = args.output_root / "exact_variant_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "clips": payload["selected_clips"],
        "classes": payload["selected_classes"],
        "signers": payload["signers"],
    }, indent=2))


if __name__ == "__main__":
    main()
