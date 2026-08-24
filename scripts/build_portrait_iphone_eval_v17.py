#!/usr/bin/env python3
"""Build and audit the frozen v17 portrait-iPhone confirmation capture pack.

The workflow is deliberately fail-closed:

1. ``init-review`` creates an exact-variant review sheet from the frozen Citizen100
   manifest and the ASL-LEX code mapping.
2. An ASL-fluent reviewer must approve every pinned variant.
3. ``build-pack`` creates reproducible, separately randomized capture schedules and
   a pre-populated ledger. It refuses pending or rejected review rows.
4. ``audit-pack`` validates the untouched setup or the completed pre-inference set.

This script never runs a model and never reads either frozen test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "active/v17/citizen100_manifest.json"
DEFAULT_PHONOLOGY = REPO_ROOT / "active/v17/citizen100_phonology.json"
DEFAULT_CANDIDATES = REPO_ROOT / "active/v17/portrait_iphone_candidates_v17.json"
EXPECTED_CLASS_COUNT = 100
REQUIRED_CANDIDATE_IDS = {
    "compact_landmark_partwise",
    "teacher_landmark_flat",
    "teacher_hand_multisource",
    "teacher_mouth_auto_avsr",
    "teacher_lower_face_auto_avsr",
    "diagnostic_mouth_lower_learned",
}
REQUIRED_FUSIONS = {
    "landmark_hand_75_25": {
        "teacher_landmark_flat": 0.75,
        "teacher_hand_multisource": 0.25,
    },
    "four_stream_teacher_30_15_35_20": {
        "teacher_landmark_flat": 0.30,
        "teacher_mouth_auto_avsr": 0.15,
        "teacher_lower_face_auto_avsr": 0.35,
        "teacher_hand_multisource": 0.20,
    },
    "partwise_teacher_substitution_fixed": {
        "compact_landmark_partwise": 0.30,
        "teacher_mouth_auto_avsr": 0.15,
        "teacher_lower_face_auto_avsr": 0.35,
        "teacher_hand_multisource": 0.20,
    },
}
REVIEW_FIELDS = (
    "class_index",
    "canonical_label",
    "citizen_raw_gloss",
    "citizen_asl_lex_code",
    "asllex_entry_id",
    "asllex_reference_url",
    "review_status",
    "reviewer_id",
    "reviewed_utc",
    "review_notes",
)
LEDGER_FIELDS = (
    "capture_id",
    "planned_id",
    "attempt",
    "signer_id",
    "session_id",
    "class_index",
    "canonical_label",
    "expected_raw_gloss",
    "citizen_asl_lex_code",
    "performed_gloss",
    "repetition",
    "prompt_order",
    "prompt_hidden_before_capture",
    "video_path",
    "video_sha256",
    "recorded_utc",
    "device_model",
    "ios_version",
    "camera",
    "width",
    "height",
    "fps",
    "orientation",
    "mirrored",
    "lighting",
    "background",
    "objective_qc_status",
    "objective_qc_reason",
)
SCHEDULE_FIELDS = (
    "prompt_order",
    "planned_id",
    "canonical_label",
    "citizen_raw_gloss",
    "citizen_asl_lex_code",
    "asllex_reference_url",
    "operator_note",
)
FORBIDDEN_QC_TERMS = (
    "confidence",
    "prediction",
    "misclass",
    "model error",
    "model disagreement",
    "low score",
    "wrong top",
)
SIGNER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_CAPTURE_ORIENTATIONS = {
    "portrait",
    "portrait_upside_down",
    "landscape_left",
    "landscape_right",
}
ORIENTATION_PROTOCOL = {
    "capture_orientation_restricted": False,
    "accepted_interface_orientations": sorted(ALLOWED_CAPTURE_ORIENTATIONS),
    "native_aspect_ratio_required": True,
    "pixel_stretching_allowed": False,
}


class PortraitPackError(ValueError):
    """Raised when an immutable protocol gate is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PortraitPackError(f"Expected a JSON object: {path}")
    return payload


def _parse_utc(value: str) -> bool:
    if not value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def load_frozen_classes(manifest_path: Path) -> list[dict[str, object]]:
    manifest = _read_json(manifest_path)
    classes_value = manifest.get("classes")
    if not isinstance(classes_value, list):
        raise PortraitPackError("Citizen manifest has no classes list")
    classes = sorted(classes_value, key=lambda row: int(row["class_index"]))
    if len(classes) != EXPECTED_CLASS_COUNT:
        raise PortraitPackError(
            f"Expected {EXPECTED_CLASS_COUNT} frozen classes, found {len(classes)}"
        )
    indices = [int(row["class_index"]) for row in classes]
    if indices != list(range(EXPECTED_CLASS_COUNT)):
        raise PortraitPackError("Frozen class indices must be exactly 0..99")
    for key in ("canonical_label", "citizen_raw_gloss", "citizen_asl_lex_code"):
        values = [str(row[key]).strip() for row in classes]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise PortraitPackError(f"Frozen class field must be nonempty and unique: {key}")
    return classes


def load_asllex_entries(phonology_path: Path) -> dict[int, str]:
    payload = _read_json(phonology_path)
    rows = payload.get("classes")
    if not isinstance(rows, list):
        raise PortraitPackError("Phonology mapping has no classes list")
    result: dict[int, str] = {}
    for row in rows:
        index = int(row["class_index"])
        entry = str(row["asllex_entry_id"]).strip()
        if not entry or index in result:
            raise PortraitPackError("Invalid or duplicate ASL-LEX entry mapping")
        result[index] = entry
    if set(result) != set(range(EXPECTED_CLASS_COUNT)):
        raise PortraitPackError("ASL-LEX mapping does not cover the frozen 100 classes")
    return result


def _resolve_repository_file(
    candidate_manifest_path: Path,
    repository_root_value: object,
    relative_value: object,
) -> Path:
    repository_root = (candidate_manifest_path.parent / str(repository_root_value)).resolve()
    relative = Path(str(relative_value))
    if relative.is_absolute():
        raise PortraitPackError(f"Frozen artifact path must be relative: {relative}")
    resolved = (repository_root / relative).resolve()
    try:
        resolved.relative_to(repository_root)
    except ValueError as exc:
        raise PortraitPackError(f"Frozen artifact escapes repository root: {relative}") from exc
    if not resolved.is_file():
        raise PortraitPackError(f"Frozen artifact is missing: {relative}")
    return resolved


def validate_candidates(candidate_manifest_path: Path) -> dict[str, object]:
    payload = _read_json(candidate_manifest_path)
    if (
        payload.get("format") != "slt_v17_portrait_iphone_frozen_candidates"
        or payload.get("version") != 1
    ):
        raise PortraitPackError("Frozen candidate manifest format/version is invalid")
    required_flags = {
        "selection_locked": True,
        "allow_recalibration": False,
        "independent_confirmation_required": True,
        "test_evaluated": False,
        "score_normalization": "per_sample_zscore",
    }
    for key, expected in required_flags.items():
        if payload.get(key) != expected:
            raise PortraitPackError(f"Frozen candidate safety field changed: {key}")
    repository_root_value = payload.get("repository_root")
    if not isinstance(repository_root_value, str) or not repository_root_value:
        raise PortraitPackError("Frozen candidate repository_root is missing")

    candidate_rows = payload.get("candidates")
    if not isinstance(candidate_rows, list):
        raise PortraitPackError("Frozen candidate list is missing")
    candidates: dict[str, dict[str, object]] = {}
    for row in candidate_rows:
        if not isinstance(row, dict):
            raise PortraitPackError("Frozen candidate row is invalid")
        candidate_id = str(row.get("id", ""))
        if not candidate_id or candidate_id in candidates:
            raise PortraitPackError(f"Duplicate or empty frozen candidate id: {candidate_id!r}")
        expected_hash = str(row.get("sha256", "")).casefold()
        if not SHA256_RE.fullmatch(expected_hash):
            raise PortraitPackError(f"Frozen candidate hash is invalid: {candidate_id}")
        artifact_path = _resolve_repository_file(
            candidate_manifest_path, repository_root_value, row.get("path", "")
        )
        if sha256_file(artifact_path) != expected_hash:
            raise PortraitPackError(f"Frozen candidate hash mismatch: {candidate_id}")
        candidates[candidate_id] = row
    if set(candidates) != REQUIRED_CANDIDATE_IDS:
        raise PortraitPackError("Frozen candidate IDs do not match the declared evaluation set")

    compact = candidates["compact_landmark_partwise"]
    if (
        compact.get("citizen_validation_correct") != 366
        or compact.get("citizen_validation_total") != 378
        or compact.get("semlex_validation_correct") != 853
        or compact.get("semlex_validation_total") != 978
    ):
        raise PortraitPackError("Compact landmark development evidence changed")

    fusion_rows = payload.get("fusions")
    if not isinstance(fusion_rows, dict) or set(fusion_rows) != set(REQUIRED_FUSIONS):
        raise PortraitPackError("Frozen fusion set changed")
    for fusion_id, expected_members in REQUIRED_FUSIONS.items():
        fusion = fusion_rows[fusion_id]
        if not isinstance(fusion, dict) or fusion.get("recalibration_allowed") is not False:
            raise PortraitPackError(f"Fusion is not locked against recalibration: {fusion_id}")
        members = fusion.get("members")
        if not isinstance(members, dict) or set(members) != set(expected_members):
            raise PortraitPackError(f"Frozen fusion members changed: {fusion_id}")
        for candidate_id, expected_weight in expected_members.items():
            try:
                actual_weight = float(members[candidate_id])
            except (TypeError, ValueError) as exc:
                raise PortraitPackError(f"Invalid fusion weight: {fusion_id}") from exc
            if abs(actual_weight - expected_weight) > 1e-12:
                raise PortraitPackError(f"Frozen fusion weight changed: {fusion_id}/{candidate_id}")
        if abs(sum(float(value) for value in members.values()) - 1.0) > 1e-12:
            raise PortraitPackError(f"Frozen fusion weights do not sum to one: {fusion_id}")

    teacher = fusion_rows["four_stream_teacher_30_15_35_20"]
    if (
        teacher.get("citizen_validation_correct") != 370
        or teacher.get("citizen_validation_total") != 378
        or teacher.get("semlex_validation_correct") != 882
        or teacher.get("semlex_validation_total") != 978
    ):
        raise PortraitPackError("Four-stream teacher development evidence changed")

    for group_name in ("runtime_sources", "evidence_artifacts"):
        rows = payload.get(group_name)
        if not isinstance(rows, list) or not rows:
            raise PortraitPackError(f"Frozen {group_name} list is missing")
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise PortraitPackError(f"Invalid frozen {group_name} row")
            relative = str(row.get("path", ""))
            if not relative or relative in seen:
                raise PortraitPackError(f"Duplicate or empty frozen {group_name} path")
            seen.add(relative)
            expected_hash = str(row.get("sha256", "")).casefold()
            if not SHA256_RE.fullmatch(expected_hash):
                raise PortraitPackError(f"Invalid frozen {group_name} hash: {relative}")
            artifact_path = _resolve_repository_file(
                candidate_manifest_path, repository_root_value, relative
            )
            if sha256_file(artifact_path) != expected_hash:
                raise PortraitPackError(f"Frozen {group_name} hash mismatch: {relative}")

    external_assets = payload.get("external_assets")
    if not isinstance(external_assets, list) or not external_assets:
        raise PortraitPackError("Frozen external asset declarations are missing")
    for row in external_assets:
        if not isinstance(row, dict) or not SHA256_RE.fullmatch(
            str(row.get("sha256", "")).casefold()
        ):
            raise PortraitPackError("Frozen external asset declaration is invalid")
    return payload


def reference_url(entry_id: str) -> str:
    return f"https://asl-lex.org/visualization/?sign={quote(entry_id, safe='')}"


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def init_review(
    manifest_path: Path,
    phonology_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    if output_path.exists() and not overwrite:
        raise PortraitPackError(f"Refusing to overwrite existing review sheet: {output_path}")
    classes = load_frozen_classes(manifest_path)
    entries = load_asllex_entries(phonology_path)
    rows: list[dict[str, object]] = []
    for item in classes:
        index = int(item["class_index"])
        entry = entries[index]
        rows.append(
            {
                "class_index": index,
                "canonical_label": str(item["canonical_label"]),
                "citizen_raw_gloss": str(item["citizen_raw_gloss"]),
                "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
                "asllex_entry_id": entry,
                "asllex_reference_url": reference_url(entry),
                "review_status": "pending",
                "reviewer_id": "",
                "reviewed_utc": "",
                "review_notes": "",
            }
        )
    write_csv(output_path, REVIEW_FIELDS, rows)
    return rows


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return fields, list(reader)


def validate_review(
    review_path: Path,
    classes: list[dict[str, object]],
    entries: dict[int, str],
    *,
    require_approved: bool,
) -> list[dict[str, str]]:
    fields, rows = _read_csv(review_path)
    if fields != list(REVIEW_FIELDS):
        raise PortraitPackError("Variant review columns do not match the frozen schema")
    if len(rows) != EXPECTED_CLASS_COUNT:
        raise PortraitPackError("Variant review must contain exactly 100 rows")
    by_index: dict[int, dict[str, str]] = {}
    for row in rows:
        index = int(row["class_index"])
        if index in by_index:
            raise PortraitPackError(f"Duplicate review class index: {index}")
        by_index[index] = row
    for item in classes:
        index = int(item["class_index"])
        row = by_index.get(index)
        if row is None:
            raise PortraitPackError(f"Missing review class index: {index}")
        expected = {
            "canonical_label": str(item["canonical_label"]),
            "citizen_raw_gloss": str(item["citizen_raw_gloss"]),
            "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
            "asllex_entry_id": entries[index],
            "asllex_reference_url": reference_url(entries[index]),
        }
        for key, value in expected.items():
            if row[key] != value:
                raise PortraitPackError(f"Review row {index} changed pinned field {key}")
        status = row["review_status"].strip().casefold()
        if status not in {"pending", "approved", "rejected"}:
            raise PortraitPackError(f"Invalid review status for class {index}: {status}")
        if status == "approved" and (
            not row["reviewer_id"].strip() or not _parse_utc(row["reviewed_utc"])
        ):
            raise PortraitPackError(
                f"Approved class {index} needs reviewer_id and timezone-aware reviewed_utc"
            )
        if require_approved and status != "approved":
            raise PortraitPackError(
                f"Capture is blocked: class {index} review status is {status}"
            )
    return [by_index[index] for index in range(EXPECTED_CLASS_COUNT)]


def _stable_rng(seed: int, signer_id: str, repetition: int) -> random.Random:
    material = f"portrait-iphone-v17\0{seed}\0{signer_id}\0{repetition}".encode()
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def _blank_capture_row() -> dict[str, object]:
    return {field: "" for field in LEDGER_FIELDS}


def build_pack(
    manifest_path: Path,
    phonology_path: Path,
    review_path: Path,
    output_dir: Path,
    signer_ids: list[str],
    *,
    candidate_manifest_path: Path = DEFAULT_CANDIDATES,
    seed: int = 1701,
    repetitions: int = 2,
    oov_per_signer: int = 20,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PortraitPackError(f"Refusing to overwrite nonempty pack directory: {output_dir}")
    if len(signer_ids) < 5 or len(set(signer_ids)) != len(signer_ids):
        raise PortraitPackError("At least five unique signer IDs are required")
    if any(not SIGNER_RE.fullmatch(signer) for signer in signer_ids):
        raise PortraitPackError("Signer IDs must be short pseudonyms using letters/numbers/_/-")
    if repetitions < 2:
        raise PortraitPackError("At least two repetitions are required")
    if oov_per_signer < 0:
        raise PortraitPackError("OOV count cannot be negative")

    classes = load_frozen_classes(manifest_path)
    entries = load_asllex_entries(phonology_path)
    validate_candidates(candidate_manifest_path)
    review_rows = validate_review(
        review_path, classes, entries, require_approved=True
    )
    review_by_index = {int(row["class_index"]): row for row in review_rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_rows: list[dict[str, object]] = []
    schedule_hashes: dict[str, str] = {}

    for signer_id in signer_ids:
        prior_order: list[int] | None = None
        for repetition in range(1, repetitions + 1):
            order = list(range(EXPECTED_CLASS_COUNT))
            _stable_rng(seed, signer_id, repetition).shuffle(order)
            if prior_order == order:
                order = order[1:] + order[:1]
            prior_order = order
            session_id = f"{signer_id}_r{repetition}"
            schedule_rows: list[dict[str, object]] = []
            for prompt_order, index in enumerate(order, start=1):
                item = classes[index]
                review = review_by_index[index]
                label = str(item["canonical_label"])
                planned_id = f"{signer_id}-r{repetition}-{index:03d}"
                capture_id = f"{planned_id}-a01"
                video_rel = (
                    Path("videos")
                    / signer_id
                    / session_id
                    / f"{prompt_order:03d}_{_safe_label(label)}_{capture_id}.mov"
                )
                row = _blank_capture_row()
                row.update(
                    {
                        "capture_id": capture_id,
                        "planned_id": planned_id,
                        "attempt": 1,
                        "signer_id": signer_id,
                        "session_id": session_id,
                        "class_index": index,
                        "canonical_label": label,
                        "expected_raw_gloss": str(item["citizen_raw_gloss"]),
                        "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
                        "repetition": repetition,
                        "prompt_order": prompt_order,
                        "video_path": video_rel.as_posix(),
                        "objective_qc_status": "pending",
                    }
                )
                ledger_rows.append(row)
                schedule_rows.append(
                    {
                        "prompt_order": prompt_order,
                        "planned_id": planned_id,
                        "canonical_label": label,
                        "citizen_raw_gloss": str(item["citizen_raw_gloss"]),
                        "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
                        "asllex_reference_url": review["asllex_reference_url"],
                        "operator_note": "Hide prompt/reference before recording",
                    }
                )
            schedule_path = output_dir / "schedules" / f"{session_id}.csv"
            write_csv(schedule_path, SCHEDULE_FIELDS, schedule_rows)
            schedule_hashes[schedule_path.relative_to(output_dir).as_posix()] = sha256_file(
                schedule_path
            )

        if oov_per_signer:
            session_id = f"{signer_id}_oov"
            schedule_rows = []
            for prompt_order in range(1, oov_per_signer + 1):
                planned_id = f"{signer_id}-oov-{prompt_order:03d}"
                capture_id = f"{planned_id}-a01"
                video_rel = (
                    Path("videos")
                    / signer_id
                    / session_id
                    / f"{prompt_order:03d}_UNKNOWN_{capture_id}.mov"
                )
                row = _blank_capture_row()
                row.update(
                    {
                        "capture_id": capture_id,
                        "planned_id": planned_id,
                        "attempt": 1,
                        "signer_id": signer_id,
                        "session_id": session_id,
                        "canonical_label": "UNKNOWN",
                        "prompt_order": prompt_order,
                        "video_path": video_rel.as_posix(),
                        "objective_qc_status": "pending",
                    }
                )
                ledger_rows.append(row)
                schedule_rows.append(
                    {
                        "prompt_order": prompt_order,
                        "planned_id": planned_id,
                        "canonical_label": "UNKNOWN",
                        "citizen_raw_gloss": "",
                        "citizen_asl_lex_code": "",
                        "asllex_reference_url": "",
                        "operator_note": (
                            "Choose one natural non-target sign or non-sign gesture; "
                            "record its description in performed_gloss"
                        ),
                    }
                )
            schedule_path = output_dir / "schedules" / f"{session_id}.csv"
            write_csv(schedule_path, SCHEDULE_FIELDS, schedule_rows)
            schedule_hashes[schedule_path.relative_to(output_dir).as_posix()] = sha256_file(
                schedule_path
            )

    ledger_path = output_dir / "capture_ledger.csv"
    write_csv(ledger_path, LEDGER_FIELDS, ledger_rows)
    pack_manifest: dict[str, object] = {
        "format": "slt_v17_portrait_iphone_capture_pack",
        "version": 1,
        "status": "capture_pending",
        "evaluation_allowed": False,
        "model_inference_accessed": False,
        "test_splits_accessed": False,
        "seed": seed,
        "signer_ids": signer_ids,
        "repetitions": repetitions,
        "class_count": EXPECTED_CLASS_COUNT,
        "oov_per_signer": oov_per_signer,
        "expected_target_records": len(signer_ids) * repetitions * EXPECTED_CLASS_COUNT,
        "expected_oov_records": len(signer_ids) * oov_per_signer,
        "citizen_manifest_sha256": sha256_file(manifest_path),
        "phonology_mapping_sha256": sha256_file(phonology_path),
        "variant_review_sha256": sha256_file(review_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "ledger_sha256_at_creation": sha256_file(ledger_path),
        "schedule_sha256": dict(sorted(schedule_hashes.items())),
        "orientation_protocol": ORIENTATION_PROTOCOL,
        "asllex_reference_policy": "links_only_do_not_copy_or_embed_reference_videos",
    }
    manifest_out = output_dir / "capture_pack_manifest.json"
    manifest_out.write_text(
        json.dumps(pack_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pack_manifest


def _int_value(row: dict[str, str], key: str, errors: list[str]) -> int | None:
    try:
        return int(row[key])
    except (KeyError, ValueError):
        errors.append(f"{row.get('capture_id', '<unknown>')}: invalid integer {key}")
        return None


def probe_video_full_decode(
    path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
    ffmpeg_binary: str = "ffmpeg",
) -> dict[str, object]:
    """Read container metadata and fully decode the first video stream.

    Audio is neither selected nor decoded. ``-xerror`` makes corrupt video packets a
    hard failure instead of a warning that could be missed in a large capture set.
    """
    ffprobe_path = shutil.which(ffprobe_binary)
    ffmpeg_path = shutil.which(ffmpeg_binary)
    if ffprobe_path is None or ffmpeg_path is None:
        raise PortraitPackError("ffprobe and ffmpeg are required for pre-inference media audit")
    metadata_result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if metadata_result.returncode != 0:
        detail = metadata_result.stderr.strip() or "ffprobe failed"
        raise PortraitPackError(f"{path}: {detail}")
    try:
        metadata = json.loads(metadata_result.stdout)
        stream = metadata["streams"][0]
        raw_width = int(stream["width"])
        raw_height = int(stream["height"])
        fps = float(Fraction(str(stream["avg_frame_rate"])))
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise PortraitPackError(f"{path}: invalid video metadata") from exc
    if raw_width <= 0 or raw_height <= 0 or fps <= 0:
        raise PortraitPackError(f"{path}: nonpositive video dimensions or frame rate")
    rotation = 0
    tags = stream.get("tags")
    if isinstance(tags, dict) and str(tags.get("rotate", "")).strip():
        try:
            rotation = int(round(float(tags["rotate"])))
        except ValueError as exc:
            raise PortraitPackError(f"{path}: invalid rotation metadata") from exc
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and "rotation" in item:
                try:
                    rotation = int(round(float(item["rotation"])))
                except ValueError as exc:
                    raise PortraitPackError(f"{path}: invalid rotation side data") from exc
                break
    normalized_rotation = rotation % 360
    if normalized_rotation in {90, 270}:
        oriented_width, oriented_height = raw_height, raw_width
    elif normalized_rotation in {0, 180}:
        oriented_width, oriented_height = raw_width, raw_height
    else:
        raise PortraitPackError(f"{path}: unsupported non-right-angle rotation {rotation}")

    decode_result = subprocess.run(
        [
            ffmpeg_path,
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if decode_result.returncode != 0:
        detail = decode_result.stderr.strip() or "full video decode failed"
        raise PortraitPackError(f"{path}: {detail}")
    frame_count_value = stream.get("nb_frames")
    try:
        reported_frames = int(frame_count_value) if frame_count_value not in (None, "N/A") else None
    except ValueError:
        reported_frames = None
    return {
        "raw_width": raw_width,
        "raw_height": raw_height,
        "oriented_width": oriented_width,
        "oriented_height": oriented_height,
        "rotation_degrees": rotation,
        "fps": fps,
        "reported_frames": reported_frames,
        "full_video_decode_passed": True,
        "audio_accessed": False,
    }


def audit_pack(
    pack_dir: Path,
    manifest_path: Path,
    phonology_path: Path,
    review_path: Path,
    *,
    phase: str,
    candidate_manifest_path: Path = DEFAULT_CANDIDATES,
    decode_workers: int = 4,
) -> dict[str, object]:
    if phase not in {"setup", "pre-inference"}:
        raise PortraitPackError(f"Unknown audit phase: {phase}")
    errors: list[str] = []
    warnings: list[str] = []
    pack_manifest_path = pack_dir / "capture_pack_manifest.json"
    ledger_path = pack_dir / "capture_ledger.csv"
    pack = _read_json(pack_manifest_path)
    if pack.get("format") != "slt_v17_portrait_iphone_capture_pack" or pack.get("version") != 1:
        errors.append("Capture pack format/version is not supported")
    if pack.get("orientation_protocol") != ORIENTATION_PROTOCOL:
        errors.append("Capture pack orientation protocol changed")
    for key in ("evaluation_allowed", "model_inference_accessed", "test_splits_accessed"):
        if pack.get(key) is not False:
            errors.append(f"Capture pack safety flag must remain false: {key}")
    for key, path in (
        ("citizen_manifest_sha256", manifest_path),
        ("phonology_mapping_sha256", phonology_path),
        ("variant_review_sha256", review_path),
        ("candidate_manifest_sha256", candidate_manifest_path),
    ):
        if str(pack.get(key, "")) != sha256_file(path):
            errors.append(f"Input hash mismatch: {key}")
    classes = load_frozen_classes(manifest_path)
    entries = load_asllex_entries(phonology_path)
    try:
        validate_candidates(candidate_manifest_path)
    except PortraitPackError as exc:
        errors.append(str(exc))
    try:
        validate_review(review_path, classes, entries, require_approved=True)
    except PortraitPackError as exc:
        errors.append(str(exc))

    fields, rows = _read_csv(ledger_path)
    if fields != list(LEDGER_FIELDS):
        errors.append("Capture ledger columns do not match the frozen schema")
    expected_target = int(pack.get("expected_target_records", -1))
    expected_oov = int(pack.get("expected_oov_records", -1))
    signer_ids = [str(value) for value in pack.get("signer_ids", [])]
    repetitions = int(pack.get("repetitions", 0))
    class_by_index = {int(item["class_index"]): item for item in classes}
    target_plans: dict[str, list[dict[str, str]]] = defaultdict(list)
    oov_plans: dict[str, list[dict[str, str]]] = defaultdict(list)
    target_keys: Counter[tuple[str, int, int]] = Counter()
    oov_keys: Counter[tuple[str, int]] = Counter()
    prompt_keys: Counter[tuple[str, str, int]] = Counter()
    prompt_plans: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    capture_ids: set[str] = set()
    paths: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    media_tasks: list[tuple[str, dict[str, str], Path]] = []

    for row in rows:
        capture_id = row.get("capture_id", "")
        planned_id = row.get("planned_id", "")
        if not capture_id or capture_id in capture_ids:
            errors.append(f"Duplicate or empty capture_id: {capture_id!r}")
        capture_ids.add(capture_id)
        if row.get("signer_id") not in signer_ids:
            errors.append(f"{capture_id}: signer not declared in pack")
        signer_id = row.get("signer_id", "")
        attempt = _int_value(row, "attempt", errors)
        prompt_order = _int_value(row, "prompt_order", errors)
        if prompt_order is not None:
            prompt_key = (signer_id, row.get("session_id", ""), prompt_order)
            prompt_keys[prompt_key] += 1
            prompt_plans[prompt_key].add(planned_id)
        if attempt is not None and (attempt < 1 or capture_id != f"{planned_id}-a{attempt:02d}"):
            errors.append(f"{capture_id}: attempt/capture_id mismatch")

        label = row.get("canonical_label", "")
        if label == "UNKNOWN":
            oov_plans[planned_id].append(row)
            if prompt_order is not None:
                oov_keys[(signer_id, prompt_order)] += 1
                if planned_id != f"{signer_id}-oov-{prompt_order:03d}":
                    errors.append(f"{capture_id}: OOV planned_id does not match its slot")
                if not 1 <= prompt_order <= int(pack.get("oov_per_signer", 0)):
                    errors.append(f"{capture_id}: OOV prompt_order is outside pack range")
            for key in ("class_index", "expected_raw_gloss", "citizen_asl_lex_code", "repetition"):
                if row.get(key, "").strip():
                    errors.append(f"{capture_id}: UNKNOWN row must leave {key} empty")
        else:
            target_plans[planned_id].append(row)
            index = _int_value(row, "class_index", errors)
            repetition = _int_value(row, "repetition", errors)
            if index is not None and index in class_by_index:
                item = class_by_index[index]
                expected = {
                    "canonical_label": str(item["canonical_label"]),
                    "expected_raw_gloss": str(item["citizen_raw_gloss"]),
                    "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
                }
                for key, value in expected.items():
                    if row.get(key) != value:
                        errors.append(f"{capture_id}: changed pinned target field {key}")
            else:
                errors.append(f"{capture_id}: class_index is outside frozen vocabulary")
            if repetition is not None and not 1 <= repetition <= repetitions:
                errors.append(f"{capture_id}: repetition is outside pack range")
            if index is not None and repetition is not None:
                target_keys[(signer_id, repetition, index)] += 1
                if planned_id != f"{signer_id}-r{repetition}-{index:03d}":
                    errors.append(f"{capture_id}: target planned_id does not match its slot")
            if prompt_order is not None and not 1 <= prompt_order <= EXPECTED_CLASS_COUNT:
                errors.append(f"{capture_id}: target prompt_order is outside 1..100")

        status = row.get("objective_qc_status", "").strip().casefold()
        if status not in {"pending", "accepted", "rejected"}:
            errors.append(f"{capture_id}: invalid objective_qc_status {status!r}")
        reason = row.get("objective_qc_reason", "").strip()
        if status == "rejected" and not reason:
            errors.append(f"{capture_id}: rejected capture needs an objective reason")
        if any(term in reason.casefold() for term in FORBIDDEN_QC_TERMS):
            errors.append(f"{capture_id}: QC reason appears model-derived")

        if phase == "pre-inference" and status == "accepted":
            required = (
                "performed_gloss",
                "recorded_utc",
                "device_model",
                "ios_version",
                "camera",
                "width",
                "height",
                "fps",
                "orientation",
                "mirrored",
                "lighting",
                "background",
                "video_path",
                "video_sha256",
            )
            for key in required:
                if not row.get(key, "").strip():
                    errors.append(f"{capture_id}: accepted capture missing {key}")
            if row.get("prompt_hidden_before_capture", "").casefold() != "true":
                errors.append(f"{capture_id}: prompt was not confirmed hidden")
            if not _parse_utc(row.get("recorded_utc", "")):
                errors.append(f"{capture_id}: recorded_utc is not timezone-aware ISO-8601")
            if row.get("camera", "").casefold() not in {"front", "back"}:
                errors.append(f"{capture_id}: camera must be front or back")
            orientation = row.get("orientation", "").casefold()
            if orientation not in ALLOWED_CAPTURE_ORIENTATIONS:
                errors.append(f"{capture_id}: unsupported phone orientation")
            if row.get("mirrored", "").casefold() not in {"true", "false"}:
                errors.append(f"{capture_id}: mirrored must be true or false")
            video_hash = row.get("video_sha256", "").casefold()
            if video_hash and not SHA256_RE.fullmatch(video_hash):
                errors.append(f"{capture_id}: invalid video_sha256")
            if video_hash:
                hashes[video_hash] += 1
            video_value = row.get("video_path", "")
            if video_value:
                video_path = Path(video_value)
                safe_video_path = True
                if video_path.is_absolute():
                    errors.append(f"{capture_id}: video_path must be relative to the pack")
                    safe_video_path = False
                resolved = (pack_dir / video_path).resolve()
                try:
                    resolved.relative_to(pack_dir.resolve())
                except ValueError:
                    errors.append(f"{capture_id}: video_path escapes the pack directory")
                    safe_video_path = False
                if not resolved.is_file():
                    errors.append(f"{capture_id}: video file does not exist")
                elif video_hash and sha256_file(resolved) != video_hash:
                    errors.append(f"{capture_id}: video file hash mismatch")
                elif safe_video_path and video_hash:
                    media_tasks.append((capture_id, row, resolved))
                paths[str(resolved.resolve())] += 1
            for key in ("width", "height"):
                value = _int_value(row, key, errors)
                if value is not None and value <= 0:
                    errors.append(f"{capture_id}: {key} must be positive")
            try:
                if float(row.get("fps", "")) <= 0:
                    raise ValueError
            except ValueError:
                errors.append(f"{capture_id}: fps must be positive")
            if label != "UNKNOWN" and row.get("performed_gloss") != row.get("expected_raw_gloss"):
                errors.append(f"{capture_id}: performed target variant was not confirmed exactly")

    if len(target_plans) != expected_target:
        errors.append(
            f"Expected {expected_target} target plans, found {len(target_plans)}"
        )
    if len(oov_plans) != expected_oov:
        errors.append(f"Expected {expected_oov} OOV plans, found {len(oov_plans)}")
    expected_target_keys = {
        (signer_id, repetition, index)
        for signer_id in signer_ids
        for repetition in range(1, repetitions + 1)
        for index in range(EXPECTED_CLASS_COUNT)
    }
    expected_oov_keys = {
        (signer_id, slot)
        for signer_id in signer_ids
        for slot in range(1, int(pack.get("oov_per_signer", 0)) + 1)
    }
    if set(target_keys) != expected_target_keys:
        errors.append("Target signer/repetition/class coverage is not exact")
    if set(oov_keys) != expected_oov_keys:
        errors.append("OOV signer/slot coverage is not exact")
    if any(count < 1 for count in target_keys.values()) or any(
        count < 1 for count in oov_keys.values()
    ):
        errors.append("A planned capture has no attempt")
    for key in prompt_keys:
        # Multiple prompt-order rows are permitted only as recapture attempts for
        # one planned item; distinct plans sharing a prompt order are caught below.
        if len(prompt_plans[key]) != 1:
            errors.append(f"Prompt order assigned to multiple plans: {key}")

    if phase == "setup":
        if len(rows) != expected_target + expected_oov:
            errors.append("Setup ledger must contain exactly one attempt per planned capture")
        if any(row.get("objective_qc_status") != "pending" for row in rows):
            errors.append("Setup ledger must begin with every capture pending")
        if str(pack.get("ledger_sha256_at_creation", "")) != sha256_file(ledger_path):
            errors.append("Setup ledger changed after pack creation")
        schedule_map = pack.get("schedule_sha256", {})
        if not isinstance(schedule_map, dict):
            errors.append("Pack schedule hash map is invalid")
        else:
            for relative, expected_hash in schedule_map.items():
                path = pack_dir / str(relative)
                if not path.is_file() or sha256_file(path) != str(expected_hash):
                    errors.append(f"Schedule missing or changed: {relative}")
    else:
        for planned_id, attempts in {**target_plans, **oov_plans}.items():
            accepted = [
                row for row in attempts
                if row.get("objective_qc_status", "").casefold() == "accepted"
            ]
            pending = [
                row for row in attempts
                if row.get("objective_qc_status", "").casefold() == "pending"
            ]
            if len(accepted) != 1:
                errors.append(f"{planned_id}: expected exactly one accepted attempt")
            if pending:
                errors.append(f"{planned_id}: unresolved pending attempt")
        for value, count in hashes.items():
            if count > 1:
                errors.append(f"Duplicate accepted video hash: {value}")
        for value, count in paths.items():
            if count > 1:
                errors.append(f"Duplicate accepted video path: {value}")

    media_results: list[dict[str, object]] = []
    if phase == "pre-inference" and not errors:
        if not 1 <= decode_workers <= 16:
            errors.append("decode_workers must be in 1..16")
        elif len(media_tasks) != expected_target + expected_oov:
            errors.append("Every accepted plan must have one hash-verified video for full decode")
        else:
            with ThreadPoolExecutor(max_workers=decode_workers) as executor:
                futures = {
                    executor.submit(probe_video_full_decode, path): (capture_id, row, path)
                    for capture_id, row, path in media_tasks
                }
                for future in as_completed(futures):
                    capture_id, row, path = futures[future]
                    try:
                        metadata = future.result()
                    except (OSError, PortraitPackError) as exc:
                        errors.append(f"{capture_id}: {exc}")
                        continue
                    ledger_width = int(row["width"])
                    ledger_height = int(row["height"])
                    if (
                        ledger_width != metadata["oriented_width"]
                        or ledger_height != metadata["oriented_height"]
                    ):
                        errors.append(f"{capture_id}: oriented dimensions disagree with ledger")
                    orientation = row["orientation"].casefold()
                    if orientation.startswith("portrait") and metadata["oriented_height"] <= metadata["oriented_width"]:
                        errors.append(f"{capture_id}: decoded dimensions disagree with portrait metadata")
                    if orientation.startswith("landscape") and metadata["oriented_width"] <= metadata["oriented_height"]:
                        errors.append(f"{capture_id}: decoded dimensions disagree with landscape metadata")
                    ledger_fps = float(row["fps"])
                    measured_fps = float(metadata["fps"])
                    if abs(ledger_fps - measured_fps) > max(0.5, measured_fps * 0.03):
                        errors.append(f"{capture_id}: frame rate disagrees with ledger")
                    media_results.append(
                        {
                            "capture_id": capture_id,
                            "video_path": path.relative_to(pack_dir.resolve()).as_posix(),
                            "video_sha256": row["video_sha256"],
                            **metadata,
                        }
                    )

    status_counts = Counter(
        row.get("objective_qc_status", "").strip().casefold() for row in rows
    )
    result: dict[str, object] = {
        "format": "slt_v17_portrait_iphone_pack_audit",
        "phase": phase,
        "pack_dir": str(pack_dir),
        "rows": len(rows),
        "target_plans": len(target_plans),
        "oov_plans": len(oov_plans),
        "status_counts": dict(sorted(status_counts.items())),
        "ledger_sha256": sha256_file(ledger_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "media_full_decode_count": len(media_results),
        "media_full_decode_passed": (
            phase == "pre-inference"
            and len(media_results) == expected_target + expected_oov
            and not errors
        ),
        "media": sorted(media_results, key=lambda item: str(item["capture_id"])),
        "errors": errors,
        "warnings": warnings,
        "pass": not errors,
        "ready_for_first_inference": (
            phase == "pre-inference"
            and not errors
            and len(media_results) == expected_target + expected_oov
        ),
        "test_splits_accessed": False,
        "model_inference_accessed": False,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    review = commands.add_parser("init-review", help="Create the 100-row variant review sheet")
    review.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    review.add_argument("--phonology", type=Path, default=DEFAULT_PHONOLOGY)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--overwrite", action="store_true")

    pack = commands.add_parser("build-pack", help="Create schedules after all variants are approved")
    pack.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    pack.add_argument("--phonology", type=Path, default=DEFAULT_PHONOLOGY)
    pack.add_argument("--review", type=Path, required=True)
    pack.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--signer-id", action="append", required=True)
    pack.add_argument("--seed", type=int, default=1701)
    pack.add_argument("--repetitions", type=int, default=2)
    pack.add_argument("--oov-per-signer", type=int, default=20)

    audit = commands.add_parser("audit-pack", help="Audit setup or completed pre-inference pack")
    audit.add_argument("--pack-dir", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    audit.add_argument("--phonology", type=Path, default=DEFAULT_PHONOLOGY)
    audit.add_argument("--review", type=Path, required=True)
    audit.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    audit.add_argument("--phase", choices=("setup", "pre-inference"), required=True)
    audit.add_argument("--decode-workers", type=int, default=4)
    audit.add_argument("--report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "init-review":
            rows = init_review(
                args.manifest, args.phonology, args.output, overwrite=args.overwrite
            )
            print(json.dumps({"output": str(args.output), "rows": len(rows), "status": "pending"}))
            return 0
        if args.command == "build-pack":
            result = build_pack(
                args.manifest,
                args.phonology,
                args.review,
                args.output_dir,
                args.signer_id,
                candidate_manifest_path=args.candidates,
                seed=args.seed,
                repetitions=args.repetitions,
                oov_per_signer=args.oov_per_signer,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        result = audit_pack(
            args.pack_dir,
            args.manifest,
            args.phonology,
            args.review,
            phase=args.phase,
            candidate_manifest_path=args.candidates,
            decode_workers=args.decode_workers,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["pass"] else 1
    except (OSError, KeyError, TypeError, PortraitPackError) as exc:
        print(json.dumps({"pass": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
