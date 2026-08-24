#!/usr/bin/env python3
"""Download tiny PopSign previews for Citizen100 lexical-variant auditing.

The preview videos are downsampled and speed-normalized by the PopSign website.
They are suitable for visual/model-assisted compatibility checks only and must
never be placed in a training corpus. Exact Citizen variants remain pinned by
``active/v17/citizen100_manifest.json``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

import cv2

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from scripts.download_popsign_v17 import fetch_source_map, source_records


PREVIEW_URL = (
    "https://signdata.cc.gatech.edu/res/downsampled/"
    "popsign_v1_0/game/{split}/{sign}/{archive_name}"
)
ALIASES = {"GOODBYE": "bye", "MOTHER": "mom", "FATHER": "dad"}


def normalize(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def overlapping_classes(
    manifest: dict[str, object], popsign_metadata: dict[str, object]
) -> list[dict[str, str]]:
    popsign_names = {
        normalize(str(name)): str(name) for name in popsign_metadata["signs"]
    }
    overlap: list[dict[str, str]] = []
    for item in sorted(manifest["classes"], key=lambda row: row["class_index"]):
        canonical = str(item["canonical_label"])
        candidate = ALIASES.get(canonical, canonical)
        popsign = popsign_names.get(normalize(candidate))
        if popsign is None:
            continue
        overlap.append(
            {
                "canonical_label": canonical,
                "citizen_raw_gloss": str(item["citizen_raw_gloss"]),
                "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
                "popsign_gloss": popsign,
            }
        )
    return overlap


def choose_participant_distinct(
    records: dict[str, dict[str, str]], count: int
) -> list[tuple[str, dict[str, str]]]:
    by_participant: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for archive_name, record in records.items():
        by_participant.setdefault(record["participant"], []).append(
            (archive_name, record)
        )
    selected: list[tuple[str, dict[str, str]]] = []
    for participant in sorted(by_participant, key=str.casefold):
        choices = sorted(
            by_participant[participant],
            key=lambda pair: (pair[1]["original_name"].casefold(), pair[0]),
        )
        selected.append(choices[0])
        if len(selected) == count:
            break
    if len(selected) < count:
        raise ValueError(
            f"only {len(selected)} distinct PopSign participants; requested {count}"
        )
    return selected


def download(url: str, destination: Path, timeout: int) -> tuple[bytes, str]:
    if destination.exists():
        return destination.read_bytes(), "existing"
    request = Request(url, headers={"User-Agent": "SLT-v17-variant-audit/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return payload, "downloaded"


def validate_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not ok or frame is None or frames < 1:
        raise RuntimeError(f"preview did not decode: {path}")
    height, width = frame.shape[:2]
    return {
        "decoded_width": width,
        "decoded_height": height,
        "frames": frames,
        "fps": fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--popsign-metadata",
        type=Path,
        default=Path("data/local/dataset_metadata/popsign_v1_game_metadata.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/local/popsign_citizen100_variant_audit/raw"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    popsign = json.loads(args.popsign_metadata.read_text(encoding="utf-8"))
    overlap = overlapping_classes(manifest, popsign)
    records_out: list[dict[str, object]] = []
    for index, item in enumerate(overlap, start=1):
        sign = item["popsign_gloss"].lower()
        source_map, source_map_bytes = fetch_source_map(sign, timeout=args.timeout)
        records = source_records(source_map, args.split, sign)
        selected = choose_participant_distinct(records, args.samples_per_class)
        for archive_name, record in selected:
            url = PREVIEW_URL.format(
                split=args.split, sign=sign, archive_name=archive_name
            )
            destination = args.output_root / item["canonical_label"] / archive_name
            row: dict[str, object] = {
                **item,
                "split": args.split,
                "participant": record["participant"],
                "archive_name": archive_name,
                "original_name": record["original_name"],
                "preview_url": url,
                "destination": str(destination),
                "source_map_sha256": hashlib.sha256(source_map_bytes).hexdigest(),
                "training_eligible": False,
            }
            if not args.dry_run:
                payload, status = download(url, destination, args.timeout)
                row.update(
                    {
                        "status": status,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        **validate_video(destination),
                    }
                )
            records_out.append(row)
        print(
            f"[{index}/{len(overlap)}] {item['canonical_label']}: "
            f"{len(selected)} distinct participants"
        )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "PopSign/Citizen100 lexical-variant preview audit only",
        "training_eligible": False,
        "warning": (
            "PopSign website previews are downsampled and speed-normalized; "
            "download original game archives only after variant approval."
        ),
        "split": args.split,
        "overlap_class_count": len(overlap),
        "samples_per_class": args.samples_per_class,
        "video_count": len(records_out),
        "videos": records_out,
    }
    if args.dry_run:
        print(
            json.dumps(
                {key: provenance[key] for key in provenance if key != "videos"}, indent=2
            )
        )
        return
    provenance_path = args.output_root.parent / "preview_provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"provenance": str(provenance_path), "videos": len(records_out)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
