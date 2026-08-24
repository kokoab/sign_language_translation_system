#!/usr/bin/env python3
"""Fail-closed import of PortraitCaptureV17 iPhone exports.

The phone app records video and immutable capture metadata, but it never accepts a
sample for evaluation. This importer validates the complete export first, copies new
videos without overwriting existing evidence, and then atomically updates the pack
ledger. Objective human QC remains a separate pre-inference gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_portrait_iphone_eval_v17 import (
    DEFAULT_CANDIDATES,
    DEFAULT_MANIFEST,
    DEFAULT_PHONOLOGY,
    EXPECTED_CLASS_COUNT,
    LEDGER_FIELDS,
    ORIENTATION_PROTOCOL,
    SHA256_RE,
    PortraitPackError,
    _parse_utc,
    _read_csv,
    _read_json,
    _safe_label,
    load_asllex_entries,
    load_frozen_classes,
    sha256_file,
    validate_candidates,
    validate_review,
)


DEFAULT_REVIEW = Path("active/v17/portrait_iphone_variant_review_v17.csv")
DEFAULT_PACK = Path("data/local/portrait_iphone_eval_v17")
DEFAULT_REPORT = Path("artifacts/reports/portrait_iphone_capture_import_v17.json")
ATTEMPT_SUFFIX_RE = re.compile(r"-a\d{2}\.mov$")

IMMUTABLE_PLAN_FIELDS = (
    "planned_id",
    "signer_id",
    "session_id",
    "class_index",
    "canonical_label",
    "expected_raw_gloss",
    "citizen_asl_lex_code",
    "repetition",
    "prompt_order",
)


def _safe_child(root: Path, relative_value: str, *, kind: str) -> Path:
    relative = Path(relative_value)
    if not relative_value or relative.is_absolute() or ".." in relative.parts:
        raise PortraitPackError(f"Unsafe {kind} path: {relative_value!r}")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PortraitPackError(f"{kind.capitalize()} path escapes its root: {relative_value}") from exc
    return resolved


def _expected_video_path(base: dict[str, str], attempt: int) -> str:
    original = base["video_path"]
    if not ATTEMPT_SUFFIX_RE.search(original):
        raise PortraitPackError(f"Frozen plan has an invalid attempt filename: {original}")
    return ATTEMPT_SUFFIX_RE.sub(f"-a{attempt:02d}.mov", original)


def _validate_pack_contract(
    pack_dir: Path,
    manifest_path: Path,
    phonology_path: Path,
    review_path: Path,
    candidate_manifest_path: Path,
) -> tuple[dict[str, object], list[dict[str, str]], dict[str, dict[str, str]]]:
    pack_path = pack_dir / "capture_pack_manifest.json"
    ledger_path = pack_dir / "capture_ledger.csv"
    if not pack_path.is_file() or not ledger_path.is_file():
        raise PortraitPackError("Capture pack manifest or ledger is missing")
    pack = _read_json(pack_path)
    if (
        pack.get("format") != "slt_v17_portrait_iphone_capture_pack"
        or pack.get("version") != 1
        or pack.get("status") != "capture_pending"
    ):
        raise PortraitPackError("Capture pack format, version, or status changed")
    for key in ("evaluation_allowed", "model_inference_accessed", "test_splits_accessed"):
        if pack.get(key) is not False:
            raise PortraitPackError(f"Capture pack safety flag must remain false: {key}")
    if pack.get("orientation_protocol") != ORIENTATION_PROTOCOL:
        raise PortraitPackError("Capture pack orientation protocol changed")
    for key, path in (
        ("citizen_manifest_sha256", manifest_path),
        ("phonology_mapping_sha256", phonology_path),
        ("variant_review_sha256", review_path),
        ("candidate_manifest_sha256", candidate_manifest_path),
    ):
        if not path.is_file() or str(pack.get(key, "")) != sha256_file(path):
            raise PortraitPackError(f"Frozen input hash mismatch: {key}")

    classes = load_frozen_classes(manifest_path)
    entries = load_asllex_entries(phonology_path)
    validate_review(review_path, classes, entries, require_approved=True)
    validate_candidates(candidate_manifest_path)
    class_by_index = {int(item["class_index"]): item for item in classes}

    schedule_values = pack.get("schedule_sha256")
    if not isinstance(schedule_values, dict):
        raise PortraitPackError("Capture pack schedule hash map is invalid")
    scheduled: dict[str, tuple[str, int]] = {}
    for relative, expected_hash in schedule_values.items():
        schedule_path = _safe_child(pack_dir, str(relative), kind="schedule")
        if not schedule_path.is_file() or sha256_file(schedule_path) != str(expected_hash):
            raise PortraitPackError(f"Frozen schedule missing or changed: {relative}")
        fields, schedule_rows = _read_csv(schedule_path)
        if "planned_id" not in fields or "prompt_order" not in fields:
            raise PortraitPackError(f"Frozen schedule schema changed: {relative}")
        session_id = schedule_path.stem
        for row in schedule_rows:
            planned_id = row["planned_id"]
            if planned_id in scheduled:
                raise PortraitPackError(f"Duplicate planned_id in schedules: {planned_id}")
            try:
                prompt_order = int(row["prompt_order"])
            except ValueError as exc:
                raise PortraitPackError(f"Invalid schedule prompt order: {planned_id}") from exc
            scheduled[planned_id] = (session_id, prompt_order)

    fields, rows = _read_csv(ledger_path)
    if fields != list(LEDGER_FIELDS):
        raise PortraitPackError("Capture ledger columns do not match the frozen schema")
    signer_ids = [str(value) for value in pack.get("signer_ids", [])]
    repetitions = int(pack.get("repetitions", 0))
    oov_per_signer = int(pack.get("oov_per_signer", 0))
    expected_plans = int(pack.get("expected_target_records", -1)) + int(
        pack.get("expected_oov_records", -1)
    )
    by_capture: dict[str, dict[str, str]] = {}
    base_by_plan: dict[str, dict[str, str]] = {}
    attempts_by_plan: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        capture_id = row["capture_id"]
        planned_id = row["planned_id"]
        try:
            attempt = int(row["attempt"])
            prompt_order = int(row["prompt_order"])
        except ValueError as exc:
            raise PortraitPackError(f"Invalid numeric plan field: {capture_id}") from exc
        if not 1 <= attempt <= 99 or capture_id != f"{planned_id}-a{attempt:02d}":
            raise PortraitPackError(f"Attempt/capture_id mismatch: {capture_id}")
        if capture_id in by_capture or attempt in attempts_by_plan[planned_id]:
            raise PortraitPackError(f"Duplicate capture or attempt in ledger: {capture_id}")
        if row["objective_qc_status"].casefold() not in {"pending", "accepted", "rejected"}:
            raise PortraitPackError(f"Invalid QC status in ledger: {capture_id}")
        schedule = scheduled.get(planned_id)
        if schedule != (row["session_id"], prompt_order):
            raise PortraitPackError(f"Ledger disagrees with frozen schedule: {capture_id}")

        signer_id = row["signer_id"]
        if signer_id not in signer_ids:
            raise PortraitPackError(f"Signer is outside the frozen pack: {capture_id}")
        if row["canonical_label"] == "UNKNOWN":
            expected_planned = f"{signer_id}-oov-{prompt_order:03d}"
            if (
                planned_id != expected_planned
                or row["session_id"] != f"{signer_id}_oov"
                or not 1 <= prompt_order <= oov_per_signer
                or any(
                    row[key]
                    for key in (
                        "class_index",
                        "expected_raw_gloss",
                        "citizen_asl_lex_code",
                        "repetition",
                    )
                )
            ):
                raise PortraitPackError(f"OOV plan fields changed: {capture_id}")
        else:
            try:
                index = int(row["class_index"])
                repetition = int(row["repetition"])
            except ValueError as exc:
                raise PortraitPackError(f"Invalid target class/repetition: {capture_id}") from exc
            item = class_by_index.get(index)
            if item is None or not 1 <= repetition <= repetitions:
                raise PortraitPackError(f"Target plan is outside the frozen design: {capture_id}")
            expected = {
                "planned_id": f"{signer_id}-r{repetition}-{index:03d}",
                "session_id": f"{signer_id}_r{repetition}",
                "canonical_label": str(item["canonical_label"]),
                "expected_raw_gloss": str(item["citizen_raw_gloss"]),
                "citizen_asl_lex_code": str(item["citizen_asl_lex_code"]),
            }
            if any(row[key] != value for key, value in expected.items()):
                raise PortraitPackError(f"Pinned target plan fields changed: {capture_id}")

        expected_path = (
            Path("videos")
            / signer_id
            / row["session_id"]
            / f"{prompt_order:03d}_{_safe_label(row['canonical_label'])}_{capture_id}.mov"
        ).as_posix()
        if row["video_path"] != expected_path:
            raise PortraitPackError(f"Ledger video path changed: {capture_id}")
        by_capture[capture_id] = row
        attempts_by_plan[planned_id].add(attempt)
        if attempt == 1:
            base_by_plan[planned_id] = row

    if len(base_by_plan) != expected_plans or set(base_by_plan) != set(scheduled):
        raise PortraitPackError("Ledger does not retain exactly one base row per frozen plan")
    for planned_id, attempts in attempts_by_plan.items():
        if attempts != set(range(1, max(attempts) + 1)):
            raise PortraitPackError(f"Ledger attempts are not contiguous: {planned_id}")
    return pack, rows, base_by_plan


def _validate_update_row(
    row: dict[str, str],
    base: dict[str, str],
    export_root: Path,
) -> tuple[Path, int]:
    capture_id = row["capture_id"]
    try:
        attempt = int(row["attempt"])
    except ValueError as exc:
        raise PortraitPackError(f"Invalid attempt in export: {capture_id}") from exc
    if not 1 <= attempt <= 99 or capture_id != f"{base['planned_id']}-a{attempt:02d}":
        raise PortraitPackError(f"Export attempt/capture_id mismatch: {capture_id}")
    for field in IMMUTABLE_PLAN_FIELDS:
        if row[field] != base[field]:
            raise PortraitPackError(f"{capture_id}: changed frozen field {field}")
    expected_path = _expected_video_path(base, attempt)
    if row["video_path"] != expected_path:
        raise PortraitPackError(f"{capture_id}: unexpected video_path")
    if row["prompt_hidden_before_capture"].casefold() != "true":
        raise PortraitPackError(f"{capture_id}: prompt was not confirmed hidden")
    if row["objective_qc_status"].casefold() != "pending" or row["objective_qc_reason"].strip():
        raise PortraitPackError(f"{capture_id}: phone export cannot make a QC decision")
    if base["canonical_label"] == "UNKNOWN":
        if not row["performed_gloss"].strip():
            raise PortraitPackError(f"{capture_id}: OOV description is missing")
    elif row["performed_gloss"] != base["expected_raw_gloss"]:
        raise PortraitPackError(f"{capture_id}: performed target variant is not exact")
    required = (
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
        "video_sha256",
    )
    if any(not row[field].strip() for field in required):
        raise PortraitPackError(f"{capture_id}: required recording metadata is missing")
    if not _parse_utc(row["recorded_utc"]):
        raise PortraitPackError(f"{capture_id}: recorded_utc is not timezone-aware ISO-8601")
    if row["camera"].casefold() != "front":
        raise PortraitPackError(f"{capture_id}: capture must use the front camera")
    orientation = row["orientation"].casefold()
    allowed = {
        "portrait",
        "portrait_upside_down",
        "landscape_left",
        "landscape_right",
    }
    if orientation not in allowed:
        raise PortraitPackError(f"{capture_id}: unsupported phone orientation")
    family = "portrait" if orientation.startswith("portrait") else "landscape"
    if row["mirrored"].casefold() != "true":
        raise PortraitPackError(f"{capture_id}: front-camera capture must be declared mirrored")
    try:
        width, height, fps = int(row["width"]), int(row["height"]), float(row["fps"])
    except ValueError as exc:
        raise PortraitPackError(f"{capture_id}: invalid dimensions or frame rate") from exc
    if width <= 0 or height <= 0 or fps <= 0:
        raise PortraitPackError(f"{capture_id}: media metadata is not positive video")
    if family == "portrait" and height <= width:
        raise PortraitPackError(f"{capture_id}: dimensions disagree with portrait metadata")
    if family == "landscape" and width <= height:
        raise PortraitPackError(f"{capture_id}: dimensions disagree with landscape metadata")
    video_hash = row["video_sha256"].casefold()
    if not SHA256_RE.fullmatch(video_hash):
        raise PortraitPackError(f"{capture_id}: invalid video_sha256")
    source = _safe_child(export_root, row["video_path"], kind="export video")
    if not source.is_file() or sha256_file(source) != video_hash:
        raise PortraitPackError(f"{capture_id}: exported video is missing or hash-mismatched")
    return source, attempt


def _atomic_write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=".capture_ledger.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def import_captures(
    pack_dir: Path,
    export_root: Path,
    updates_path: Path,
    report_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    phonology_path: Path = DEFAULT_PHONOLOGY,
    review_path: Path = DEFAULT_REVIEW,
    candidate_manifest_path: Path = DEFAULT_CANDIDATES,
) -> dict[str, object]:
    """Validate an entire export, copy new evidence, and atomically update the ledger."""
    pack_dir = pack_dir.resolve()
    export_root = export_root.resolve()
    if pack_dir == export_root:
        raise PortraitPackError("Export root must be separate from the capture pack")
    _, ledger_rows, base_by_plan = _validate_pack_contract(
        pack_dir, manifest_path, phonology_path, review_path, candidate_manifest_path
    )
    fields, updates = _read_csv(updates_path)
    if fields != list(LEDGER_FIELDS):
        raise PortraitPackError("Capture update columns do not match the frozen ledger schema")
    if not updates:
        raise PortraitPackError("Capture update export is empty")
    existing_by_capture = {row["capture_id"]: row for row in ledger_rows}
    seen_updates: set[str] = set()
    validated: list[tuple[dict[str, str], Path, Path, int, bool]] = []
    combined_attempts: dict[str, set[int]] = defaultdict(set)
    for row in ledger_rows:
        if row["video_sha256"].strip():
            combined_attempts[row["planned_id"]].add(int(row["attempt"]))

    for row in updates:
        capture_id = row["capture_id"]
        planned_id = row["planned_id"]
        if not capture_id or capture_id in seen_updates:
            raise PortraitPackError(f"Duplicate or empty capture_id in export: {capture_id!r}")
        seen_updates.add(capture_id)
        base = base_by_plan.get(planned_id)
        if base is None:
            raise PortraitPackError(f"Export references an unknown planned_id: {planned_id}")
        source, attempt = _validate_update_row(row, base, export_root)
        destination = _safe_child(pack_dir, row["video_path"], kind="pack video")
        existing = existing_by_capture.get(capture_id)
        placeholder = (
            existing is not None
            and attempt == 1
            and not existing["video_sha256"].strip()
            and existing["objective_qc_status"].casefold() == "pending"
        )
        idempotent = existing == row
        if existing is not None and not placeholder and not idempotent:
            raise PortraitPackError(f"Existing ledger row differs from export: {capture_id}")
        if destination.exists() and sha256_file(destination) != row["video_sha256"].casefold():
            raise PortraitPackError(f"Refusing to overwrite a different video: {capture_id}")
        combined_attempts[planned_id].add(attempt)
        validated.append((row, source, destination, attempt, idempotent))

    for planned_id, attempts in combined_attempts.items():
        if attempts != set(range(1, max(attempts) + 1)):
            raise PortraitPackError(f"Export would create a non-contiguous attempt sequence: {planned_id}")

    imported_ids: list[str] = []
    idempotent_ids: list[str] = []
    copied_ids: list[str] = []
    for row, source, destination, _, idempotent in validated:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
            try:
                shutil.copyfile(source, temporary)
                if sha256_file(temporary) != row["video_sha256"].casefold():
                    raise PortraitPackError(f"Copied video hash mismatch: {row['capture_id']}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            copied_ids.append(row["capture_id"])
        if idempotent:
            idempotent_ids.append(row["capture_id"])
        else:
            existing_by_capture[row["capture_id"]] = row
            imported_ids.append(row["capture_id"])

    ordered_rows = sorted(
        existing_by_capture.values(),
        key=lambda row: (
            row["signer_id"],
            row["session_id"],
            int(row["prompt_order"]),
            int(row["attempt"]),
        ),
    )
    _atomic_write_ledger(pack_dir / "capture_ledger.csv", ordered_rows)
    result: dict[str, object] = {
        "format": "slt_v17_portrait_iphone_capture_import",
        "version": 1,
        "pack_dir": str(pack_dir),
        "export_root": str(export_root),
        "updates_path": str(updates_path.resolve()),
        "updates_sha256": sha256_file(updates_path),
        "validated_update_count": len(updates),
        "new_ledger_rows": len(imported_ids),
        "new_video_copies": len(copied_ids),
        "idempotent_rows": len(idempotent_ids),
        "capture_ids_imported": imported_ids,
        "capture_ids_idempotent": idempotent_ids,
        "ledger_rows_after_import": len(ordered_rows),
        "ledger_sha256_after_import": sha256_file(pack_dir / "capture_ledger.csv"),
        "objective_qc_performed": False,
        "model_inference_accessed": False,
        "test_splits_accessed": False,
        "pass": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--updates", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phonology", type=Path, default=DEFAULT_PHONOLOGY)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    updates = args.updates or args.export_root / "capture_updates.csv"
    try:
        result = import_captures(
            args.pack_dir,
            args.export_root,
            updates,
            args.report,
            manifest_path=args.manifest,
            phonology_path=args.phonology,
            review_path=args.review,
            candidate_manifest_path=args.candidates,
        )
    except (OSError, PortraitPackError, json.JSONDecodeError) as exc:
        raise SystemExit(f"capture import failed: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
