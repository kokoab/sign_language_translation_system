#!/usr/bin/env python3
"""Acquire the public NCSLGR continuous-ASL subset for v17 Stage 2.

The Boston University pages explicitly expose downloadable utterance videos and
SignStream export annotations.  Metadata-only mode is the default so coverage and
disk requirements are known before downloading the uncompressed frontal videos.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAGE_TEMPLATE = "https://www.bu.edu/asllrp/cslgr/display/ncslgr10{}/"
PUBLIC_PAGES = ("a", "b", "c", "d")
USER_AGENT = "SLT-v17-stage2-data-acquisition/1.0"


def request_bytes(url: str, method: str = "GET") -> tuple[bytes, Any]:
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    response = urllib.request.urlopen(request, timeout=120)
    return response.read() if method == "GET" else b"", response.headers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def page_items(letter: str) -> list[dict[str, Any]]:
    page_url = PAGE_TEMPLATE.format(letter)
    source = request_bytes(page_url)[0].decode("latin1")
    items = []
    for block in re.findall(r"<TR\b[^>]*>(.*?)</TR>", source, re.I | re.S):
        compressed = re.search(r'HREF="([^"]*/compressed/master/[^"]+\.mov)"', block, re.I)
        annotation = re.search(r'HREF="([^"]*/sstxt/([^"]+))"', block, re.I)
        uncompressed = re.search(r'HREF="([^"]*/uncompressed/master/[^"]+\.avi)"', block, re.I)
        utterance = re.search(r"Utt\s*#:\s*(\d+)", block, re.I)
        if not (compressed and annotation and utterance):
            continue
        items.append(
            {
                "collection": f"ncslgr10{letter}",
                "utterance_number": int(utterance.group(1)),
                "source_id": annotation.group(2),
                "page_url": page_url,
                "annotation_url": urllib.parse.urljoin(page_url, html.unescape(annotation.group(1))),
                "compressed_front_url": urllib.parse.urljoin(page_url, html.unescape(compressed.group(1))),
                "uncompressed_front_url": (
                    urllib.parse.urljoin(page_url, html.unescape(uncompressed.group(1)))
                    if uncompressed
                    else None
                ),
            }
        )
    return items


def parse_annotation(text: str) -> dict[str, Any]:
    participant = re.search(r"^Participant:\s*(.*?)\s*$", text, re.M)
    start = re.search(r"^Start frame:\s*(\d+)", text, re.M)
    end = re.search(r"^End frame:\s*(\d+)", text, re.M)
    english = re.search(r"^English translation\s*\t(.*?)\s*$", text, re.M)
    lines = text.splitlines()
    gloss_cells: list[str] = []
    collecting = False
    for line in lines:
        if line.startswith("main gloss\t"):
            collecting = True
            gloss_cells.extend(line.split("\t")[1:])
            continue
        if collecting and line.startswith("\t"):
            gloss_cells.extend(line.split("\t")[1:])
            continue
        if collecting:
            break
    glosses = []
    for cell in gloss_cells:
        token = cell.strip()
        if not token or re.fullmatch(r"-?\d+", token):
            continue
        glosses.append(token)
    return {
        "participant": participant.group(1) if participant else None,
        "start_frame": int(start.group(1)) if start else None,
        "end_frame": int(end.group(1)) if end else None,
        "main_glosses": glosses,
        "english_translation": english.group(1) if english else None,
    }


def acquire_annotation(item: dict[str, Any], root: Path) -> dict[str, Any]:
    destination = root / "annotations" / item["collection"] / f"{item['source_id']}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = request_bytes(item["annotation_url"])[0]
    if not data.startswith(b"SIGNSTREAM EXPORT DATA FILE"):
        raise RuntimeError(f"unexpected annotation payload: {item['annotation_url']}")
    if not destination.exists() or destination.read_bytes() != data:
        destination.write_bytes(data)
    parsed = parse_annotation(data.decode("latin1"))
    return {
        **item,
        **parsed,
        "annotation_path": destination.as_posix(),
        "annotation_bytes": len(data),
        "annotation_sha256": sha256_bytes(data),
    }


def remote_size(url: str) -> int | None:
    _, headers = request_bytes(url, method="HEAD")
    value = headers.get("Content-Length")
    return int(value) if value is not None else None


def download_video(item: dict[str, Any], root: Path, quality: str) -> dict[str, Any]:
    url_key = f"{quality}_front_url"
    if not item.get(url_key):
        return {"video_path": None, "video_sha256": None}
    extension = ".avi" if quality == "uncompressed" else ".mov"
    destination = root / "raw" / item["collection"] / f"{item['source_id']}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = item.get(f"{quality}_front_bytes")
    if destination.exists() and expected is not None and destination.stat().st_size == expected:
        return {"video_path": destination.as_posix(), "video_sha256": sha256(destination)}
    data = request_bytes(item[url_key])[0]
    if expected is not None and len(data) != expected:
        raise RuntimeError(f"size mismatch for {item[url_key]}: {len(data)} != {expected}")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(destination)
    return {"video_path": destination.as_posix(), "video_sha256": sha256_bytes(data)}


def normalized_gloss(value: str) -> str:
    value = re.sub(r"^fs-", "", value, flags=re.I)
    value = re.sub(r"[+^]$", "", value)
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def participant_id(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/local/ncslgr_continuous_v17_source"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--download-videos", choices=("compressed", "uncompressed"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    items = [item for letter in PUBLIC_PAGES for item in page_items(letter)]
    if len(items) != 166:
        raise RuntimeError(f"expected 166 public utterances, found {len(items)}")

    acquired = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(acquire_annotation, item, args.root): item for item in items}
        for future in as_completed(futures):
            acquired.append(future.result())
    acquired.sort(key=lambda row: (row["collection"], row["utterance_number"]))

    for quality in ("compressed", "uncompressed"):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(remote_size, item[f"{quality}_front_url"]): index
                for index, item in enumerate(acquired)
                if item.get(f"{quality}_front_url")
            }
            for future in as_completed(futures):
                acquired[futures[future]][f"{quality}_front_bytes"] = future.result()
        for item in acquired:
            item.setdefault(f"{quality}_front_bytes", None)

    if args.download_videos:
        quality = args.download_videos
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
            futures = {
                pool.submit(download_video, item, args.root, quality): index
                for index, item in enumerate(acquired)
            }
            for future in as_completed(futures):
                acquired[futures[future]].update(future.result())

    class_payload = json.loads(args.manifest.read_text())
    labels = {normalized_gloss(item["canonical_label"]) for item in class_payload["classes"]}
    target_counts: dict[str, int] = {}
    for item in acquired:
        item["participant_id"] = participant_id(item["participant"])
        targets = [gloss for gloss in item["main_glosses"] if normalized_gloss(gloss) in labels]
        item["target_vocabulary_glosses"] = targets
        for gloss in targets:
            key = normalized_gloss(gloss)
            target_counts[key] = target_counts.get(key, 0) + 1

    payload = {
        "format": "slt_v17_ncslgr_continuous_source_manifest",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Boston University NCSLGR public downloadable pages",
        "source_pages": [PAGE_TEMPLATE.format(letter) for letter in PUBLIC_PAGES],
        "collections": list(f"ncslgr10{letter}" for letter in PUBLIC_PAGES),
        "utterances": len(acquired),
        "participant_ids": sorted({item["participant_id"] for item in acquired if item["participant_id"]}),
        "participant_name_variants": sorted({item["participant"] for item in acquired if item["participant"]}),
        "annotation_bytes": sum(item["annotation_bytes"] for item in acquired),
        "compressed_front_bytes": sum(item["compressed_front_bytes"] or 0 for item in acquired),
        "uncompressed_front_bytes": sum(item["uncompressed_front_bytes"] or 0 for item in acquired),
        "uncompressed_front_available": sum(bool(item["uncompressed_front_url"]) for item in acquired),
        "utterances_with_target_vocabulary_gloss": sum(bool(item["target_vocabulary_glosses"]) for item in acquired),
        "target_vocabulary_gloss_occurrences": sum(target_counts.values()),
        "target_vocabulary_counts": dict(sorted(target_counts.items())),
        "downloaded_video_quality": args.download_videos,
        "items": acquired,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    manifest_path = args.root / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": manifest_path.as_posix(),
                "utterances": payload["utterances"],
                "participant_ids": payload["participant_ids"],
                "utterances_with_target_vocabulary_gloss": payload["utterances_with_target_vocabulary_gloss"],
                "target_vocabulary_gloss_occurrences": payload["target_vocabulary_gloss_occurrences"],
                "compressed_front_gib": payload["compressed_front_bytes"] / 2**30,
                "uncompressed_front_gib": payload["uncompressed_front_bytes"] / 2**30,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
