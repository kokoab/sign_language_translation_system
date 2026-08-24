"""Strict resolver for the frozen 1,021-clip local exact-text audit pool."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


SOURCE = "local_audit"
SPLIT = "train_only_review_diagnostic"
EXACT_TEXT_TIER = "canonical_and_pinned_raw_text_equal"


@dataclass(frozen=True)
class LocalAuditItem:
    label: str
    item_id: str
    raw_path: Path
    landmark_path: Path
    source_row: dict[str, object]


def local_audit_items(manifest_path: Path) -> tuple[list[LocalAuditItem], dict[str, object]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("training_eligible") is not False
        or manifest.get("split_eligibility") != "train_only_after_exact_variant_review"
        or int(manifest.get("selected_clips", -1)) != 1021
    ):
        raise ValueError("not the frozen 1,021-clip local exact-text audit manifest")
    rows = manifest.get("videos")
    if not isinstance(rows, list) or len(rows) != 1021:
        raise ValueError("local audit manifest count mismatch")
    landmark_root = manifest_path.parent / "landmarks"
    items: list[LocalAuditItem] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if (
            row.get("training_eligible") is not False
            or row.get("lexical_tier") != EXACT_TEXT_TIER
        ):
            raise ValueError("local audit row violates exact-text/non-training contract")
        label = str(row.get("canonical_label", ""))
        raw_path = Path(str(row.get("raw_path", "")))
        item_id = raw_path.stem
        if not label or not item_id or Path(item_id).name != item_id:
            raise ValueError("unsafe local audit label or item ID")
        key = (label, item_id)
        if key in seen:
            raise ValueError(f"duplicate local audit item: {label}/{item_id}")
        seen.add(key)
        landmark_path = landmark_root / label / f"{item_id}.v17.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if not landmark_path.is_file():
            raise FileNotFoundError(landmark_path)
        items.append(LocalAuditItem(label, item_id, raw_path, landmark_path, row))
    return items, manifest
