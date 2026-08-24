#!/usr/bin/env python3
"""Audit local and public continuous-ASL sources for v17 Stage 2.

This is deliberately an inventory/selection tool.  It does not download videos,
touch any sealed test split, or treat English translations as gloss sequences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import statistics
import subprocess
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ASLLRP_SEARCH = "https://dai.cs.rutgers.edu/dai/s/"
MEANING_SAFE_ALIASES = {"ME": "I"}
REVIEW_ONLY_ALIASES = {"FOOD": "EAT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames:"
        "stream_tags=rotate:stream_side_data=rotation:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    rotation = stream.get("tags", {}).get("rotate", 0)
    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            rotation = item["rotation"]
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "duration_seconds": float(payload["format"]["duration"]),
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("avg_frame_rate") or stream.get("r_frame_rate"),
        "frame_count": int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") != "N/A" else None,
        "rotation_degrees": float(rotation or 0),
    }


def summarize_local_videos(root: Path, workers: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    videos = sorted(root.glob("*/*.mp4"))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_paths = {pool.submit(ffprobe, path): path for path in videos}
        for future in as_completed(future_paths):
            path = future_paths[future]
            try:
                row = future.result()
                row["phrase"] = path.parent.name
                row["source_filename_kind"] = (
                    "numbered"
                    if re.fullmatch(re.escape(path.parent.name) + r"_\d+\.mp4", path.name)
                    else "opaque"
                )
                rows.append(row)
            except Exception as exc:  # fail is recorded, not hidden
                failures.append({"path": path.as_posix(), "error": str(exc)})
    rows.sort(key=lambda row: row["path"])
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        hash_groups[row["sha256"]].append(row["path"])
    durations = [row["duration_seconds"] for row in rows]
    by_phrase = Counter(row["phrase"] for row in rows)
    by_resolution = Counter(f"{row['width']}x{row['height']}" for row in rows)
    by_rotation = Counter(str(row["rotation_degrees"]) for row in rows)
    by_kind = Counter(row["source_filename_kind"] for row in rows)
    summary = {
        "root": root.as_posix(),
        "videos": len(videos),
        "probed": len(rows),
        "failures": failures,
        "bytes": sum(row["bytes"] for row in rows),
        "phrases": dict(sorted(by_phrase.items())),
        "source_filename_kinds": dict(sorted(by_kind.items())),
        "unique_sha256": len(hash_groups),
        "exact_duplicate_groups": [paths for paths in hash_groups.values() if len(paths) > 1],
        "duration_seconds": {
            "total": sum(durations),
            "minimum": min(durations) if durations else None,
            "median": statistics.median(durations) if durations else None,
            "maximum": max(durations) if durations else None,
        },
        "resolutions": dict(sorted(by_resolution.items())),
        "rotation_metadata_degrees": dict(sorted(by_rotation.items())),
        "signer_metadata_available": False,
    }
    return summary, rows


def load_classes(manifest_path: Path) -> tuple[list[str], dict[str, list[str]]]:
    payload = json.loads(manifest_path.read_text())
    labels: list[str] = []
    aliases: dict[str, list[str]] = {}
    for item in payload["classes"]:
        label = str(item["canonical_label"]).upper()
        labels.append(label)
        candidates = {label, str(item.get("citizen_raw_gloss", label)).upper()}
        aliases[label] = sorted(candidates)
    return labels, aliases


def audit_phrase_vocabulary(phrases: dict[str, int], labels: list[str]) -> dict[str, Any]:
    vocabulary = set(labels)
    rows = []
    for phrase, count in sorted(phrases.items()):
        original = phrase.split("_")
        safe = [MEANING_SAFE_ALIASES.get(token, token) for token in original]
        strict_oov = sorted({token for token in safe if token not in vocabulary})
        reviewed = [REVIEW_ONLY_ALIASES.get(token, token) for token in safe]
        reviewed_oov = sorted({token for token in reviewed if token not in vocabulary})
        rows.append(
            {
                "phrase": phrase,
                "videos": count,
                "original_tokens": original,
                "safe_target_tokens": safe,
                "strict_in_vocabulary": not strict_oov,
                "strict_oov": strict_oov,
                "review_candidate_tokens": reviewed,
                "in_vocabulary_if_review_alias_approved": not reviewed_oov,
                "reviewed_oov": reviewed_oov,
            }
        )
    return {
        "meaning_safe_aliases": MEANING_SAFE_ALIASES,
        "review_only_aliases": REVIEW_ONLY_ALIASES,
        "strict_usable_videos": sum(row["videos"] for row in rows if row["strict_in_vocabulary"]),
        "usable_if_review_alias_approved": sum(
            row["videos"] for row in rows if row["in_vocabulary_if_review_alias_approved"]
        ),
        "phrases": rows,
    }


def audit_legacy_archives(paths: list[Path]) -> list[dict[str, Any]]:
    results = []
    for root in paths:
        arrays = sorted(root.glob("*.npy"))
        shapes = Counter()
        dtypes = Counter()
        for path in arrays[: min(256, len(arrays))]:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            shapes[str(tuple(array.shape))] += 1
            dtypes[str(array.dtype)] += 1
        manifest = root / "manifest.json"
        results.append(
            {
                "root": root.as_posix(),
                "arrays": len(arrays),
                "manifest_entries": len(json.loads(manifest.read_text())) if manifest.exists() else None,
                "sampled_arrays": min(256, len(arrays)),
                "sample_shapes": dict(shapes),
                "sample_dtypes": dict(dtypes),
                "v17_compatible": False,
                "reason": "legacy schema/version and temporal preprocessing are incompatible with current v17",
            }
        )
    return results


def words(text: str) -> set[str]:
    return set(re.findall(r"[A-Z]+", text.upper()))


def audit_how2sign(paths: list[Path], labels: list[str]) -> dict[str, Any]:
    label_words = {
        label: ({"THANK", "YOU"} if label == "THANKYOU" else {label}) for label in labels
    }
    splits: dict[str, Any] = {}
    combined_counts = Counter()
    for path in paths:
        split = "val" if "val" in path.stem else "train"
        count = 0
        occurrences = Counter()
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                count += 1
                sentence_words = words(row["SENTENCE"])
                for label, required in label_words.items():
                    if required <= sentence_words:
                        occurrences[label] += 1
                        combined_counts[label] += 1
        splits[split] = {
            "path": path.as_posix(),
            "sha256": sha256(path),
            "sentences": count,
            "labels_with_english_word_hit": sum(value > 0 for value in occurrences.values()),
            "english_word_hits": dict(sorted(occurrences.items())),
        }
    return {
        "splits": splits,
        "combined_labels_with_english_word_hit": sum(combined_counts[label] > 0 for label in labels),
        "combined_english_word_hits": {label: combined_counts[label] for label in labels},
        "supervision_status": "weak_translation_only_not_ctc_gloss",
        "warning": "English word occurrence does not prove the corresponding ASL gloss or gloss order.",
    }


def default_asllrp_params() -> list[tuple[str, str]]:
    request = urllib.request.Request(ASLLRP_SEARCH + "dai", headers={"User-Agent": "SLT-v17-data-audit/1.0"})
    source = urllib.request.urlopen(request, timeout=30).read().decode("utf-8", "replace")
    params: list[tuple[str, str]] = []
    for tag in re.findall(r"<input[^>]+>", source, re.I):
        if "checked" not in tag.lower():
            continue
        name = re.search(r"name=[\"']?([^\"' >]+)", tag, re.I)
        value = re.search(r"value=[\"']?([^\"'>]+)", tag, re.I)
        if name and value and name.group(1) in {"datasource", "rit3datasource", "participant"}:
            params.append((name.group(1), html.unescape(value.group(1).strip())))
    return params


def query_asllrp(label: str, common: list[tuple[str, str]]) -> dict[str, Any]:
    params = list(common)
    params.extend(
        [
            ("signName", label),
            ("full_partial", "1"),
            ("hand", "0,1"),
            ("all_signs", "all_signs_egcl"),
            ("SignTag", "1"),
            ("SignTag", "5"),
            ("SignTag", "3"),
            ("SignTag", "9"),
            ("incl_comp", "either"),
            ("video_views", "noCare"),
            ("color", "noCare"),
            ("sleeves", "noCare"),
            ("glasses", "noCare"),
            ("minOccur", "-1"),
            ("data_source", ""),
        ]
    )
    url = ASLLRP_SEARCH + "searchsummary?" + urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(url, headers={"User-Agent": "SLT-v17-data-audit/1.0"})
    source = urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")
    rows = []
    pattern = re.compile(
        r'<td id="cs_(?P<canonical>\d+)".*?<b>(?P<gloss>.*?)</b></td>\s*'
        r'<td.*?<a href="(?P<href>searchresult.*?)">(?P<count>\d+)</a>',
        re.S,
    )
    for match in pattern.finditer(source):
        gloss = re.sub(r"<.*?>", "", html.unescape(match.group("gloss"))).strip()
        rows.append(
            {
                "canonical_sign_id": int(match.group("canonical")),
                "gloss": gloss,
                "occurrences": int(match.group("count")),
                "result_url": urllib.parse.urljoin(ASLLRP_SEARCH, html.unescape(match.group("href"))),
            }
        )
    exact = [row for row in rows if normalized_token(row["gloss"]) == normalized_token(label)]
    return {
        "label": label,
        "matches": rows,
        "exact_normalized_occurrences": sum(row["occurrences"] for row in exact),
        "requires_login_for_bulk_download": True,
    }


def audit_asllrp(labels: list[str], workers: int) -> dict[str, Any]:
    common = default_asllrp_params()
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(query_asllrp, label, common): label for label in labels}
        for future in as_completed(futures):
            label = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"label": label, "error": str(exc)})
    rows.sort(key=lambda row: labels.index(row["label"]))
    return {
        "source": ASLLRP_SEARCH + "dai",
        "queried_labels": len(labels),
        "labels_with_exact_normalized_occurrence": sum(
            row["exact_normalized_occurrences"] > 0 for row in rows
        ),
        "total_exact_normalized_occurrences": sum(
            row["exact_normalized_occurrences"] for row in rows
        ),
        "rows": rows,
        "failures": failures,
        "acquisition_status": "metadata_public_bulk_video_and_xml_download_requires_account",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def markdown_report(payload: dict[str, Any]) -> str:
    local = payload["local_raw_phrases"]
    vocabulary = payload["local_phrase_vocabulary"]
    how2 = payload["how2sign"]
    asllrp = payload.get("asllrp")
    ncslgr = payload.get("ncslgr_public_subset")
    lines = [
        "# Stage 2 v17 phrase-source audit",
        "",
        f"Generated: `{payload['created_utc']}`",
        "",
        "## Local corpus",
        "",
        f"- {local['probed']}/{local['videos']} videos probed; {local['unique_sha256']} unique SHA-256 values.",
        f"- Total duration: {local['duration_seconds']['total'] / 3600:.2f} hours across {len(local['phrases'])} fixed phrases.",
        f"- Strict 100-class coverage: {vocabulary['strict_usable_videos']} videos.",
        f"- Coverage if the `FOOD -> EAT` variant is manually approved: {vocabulary['usable_if_review_alias_approved']} videos.",
        "- No signer IDs are present, so a signer-disjoint split cannot be reconstructed from filenames alone.",
        "- Existing phrase/synthetic arrays use legacy v16 schemas and temporal preprocessing and must not feed v17.",
        "",
        "## External sources",
        "",
        f"- How2Sign metadata: {sum(v['sentences'] for v in how2['splits'].values())} train/validation sentences; {how2['combined_labels_with_english_word_hit']}/100 labels appear as English words.",
        "- How2Sign public files provide English translations, not released CTC gloss sequences; use only for weak/self-supervised work unless labels are created.",
    ]
    if asllrp:
        lines.extend(
            [
                f"- ASLLRP exact query coverage: {asllrp['labels_with_exact_normalized_occurrence']}/100 labels and {asllrp['total_exact_normalized_occurrences']} matching sign occurrences inside real utterances.",
                "- ASLLRP is the preferred supervised source because it has real continuous utterances plus linguistic XML; bulk download requires an ASLLRP account.",
            ]
        )
    if ncslgr:
        lines.extend(
            [
                f"- NCSLGR public subset acquired: {ncslgr['utterances']} utterances, {ncslgr['downloaded_videos']} verified frontal videos, {ncslgr['target_vocabulary_gloss_occurrences']} target-vocabulary gloss occurrences across {ncslgr['target_vocabulary_labels']} labels.",
                "- NCSLGR is real frame-aligned supervision but low-resolution and narrow-vocabulary; retain it as supplemental training data.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use the local raw videos after full-length v17 re-extraction, add the acquired NCSLGR subset as supplemental real data, regenerate synthetic sequences from current v17 isolated archives, and acquire modern ASLLRP utterance videos/XML as the primary broad supervised source. Do not spend disk on How2Sign RGB until a weak-label or self-supervised experiment is predeclared.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase-root", type=Path, default=Path("data/raw_videos/PHRASES"))
    parser.add_argument("--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument(
        "--how2sign",
        type=Path,
        nargs="+",
        default=[
            Path("data/local/dataset_metadata/how2sign/how2sign_realigned_train.csv"),
            Path("data/local/dataset_metadata/how2sign/how2sign_realigned_val.csv"),
        ],
    )
    parser.add_argument("--query-asllrp", action="store_true")
    parser.add_argument(
        "--ncslgr-manifest",
        type=Path,
        default=Path("data/local/ncslgr_continuous_v17_source/manifest.json"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--asllrp-workers", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/stage2_v17_data_audit"),
    )
    args = parser.parse_args()

    labels, _ = load_classes(args.manifest)
    local, local_rows = summarize_local_videos(args.phrase_root, args.workers)
    payload: dict[str, Any] = {
        "format": "slt_stage2_v17_phrase_source_audit",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "vocabulary_manifest": args.manifest.as_posix(),
        "vocabulary_manifest_sha256": sha256(args.manifest),
        "class_count": len(labels),
        "local_raw_phrases": local,
        "local_phrase_vocabulary": audit_phrase_vocabulary(local["phrases"], labels),
        "legacy_archives": audit_legacy_archives(
            [
                Path("data/local/ASL_phrases_apple_vision"),
                Path("data/local/ASL_phrases_apple_vision_fixed"),
                Path("data/local/ASL_phrases_reextracted"),
                Path("data/local/ASL_continuous_synthetic"),
                Path("src_v16/ASL_phrases_v16"),
            ]
        ),
        "how2sign": audit_how2sign(args.how2sign, labels),
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
    }
    if args.query_asllrp:
        payload["asllrp"] = audit_asllrp(labels, args.asllrp_workers)
    if args.ncslgr_manifest.exists():
        ncslgr = json.loads(args.ncslgr_manifest.read_text())
        downloaded = [item for item in ncslgr["items"] if item.get("video_path")]
        payload["ncslgr_public_subset"] = {
            "manifest": args.ncslgr_manifest.as_posix(),
            "manifest_sha256": sha256(args.ncslgr_manifest),
            "utterances": ncslgr["utterances"],
            "participant_ids": ncslgr["participant_ids"],
            "downloaded_videos": len(downloaded),
            "downloaded_bytes": sum(Path(item["video_path"]).stat().st_size for item in downloaded),
            "target_vocabulary_gloss_occurrences": ncslgr["target_vocabulary_gloss_occurrences"],
            "target_vocabulary_labels": len(ncslgr["target_vocabulary_counts"]),
            "resolution": "324x312",
            "supervision_status": "real_frame_aligned_signstream_gloss",
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "audit.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(
        args.output_dir / "local_videos.csv",
        local_rows,
        [
            "phrase",
            "path",
            "source_filename_kind",
            "bytes",
            "sha256",
            "duration_seconds",
            "codec",
            "width",
            "height",
            "frame_rate",
            "frame_count",
            "rotation_degrees",
        ],
    )
    if "asllrp" in payload:
        asllrp_rows = [
            {
                "label": row["label"],
                "exact_normalized_occurrences": row["exact_normalized_occurrences"],
                "matched_glosses": "|".join(match["gloss"] for match in row["matches"]),
            }
            for row in payload["asllrp"]["rows"]
        ]
        write_csv(
            args.output_dir / "asllrp_coverage.csv",
            asllrp_rows,
            ["label", "exact_normalized_occurrences", "matched_glosses"],
        )
    (args.output_dir / "README.md").write_text(markdown_report(payload))
    print(json.dumps({"audit": json_path.as_posix(), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
