#!/usr/bin/env python3
"""Audit finalized local hand crops and optional MobileCLIP2 embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.extract_hand_rgb_supplement_v17 import selection_items
from active.v17.schema_hand_mobileclip2_v17 import (
    HandMobileCLIP2V17Config,
    schema_fingerprint as embedding_schema_fingerprint,
)
from active.v17.schema_hand_rgb_v17 import (
    HandRGBV17Config,
    schema_fingerprint as crop_schema_fingerprint,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_provenance(source: str) -> tuple[str, bool]:
    if source == "local_deep_clean":
        return "train_only", True
    if source == "local_deep_clean_val":
        return "validation_nonsigner_disjoint_user_approved", False
    raise ValueError("audit only accepts finalized local train or validation")


def audit_crop(
    path: Path,
    *,
    source: str,
    item_id: str,
    label: str,
    manifest_sha256: str,
    split: str,
    training_eligible: bool,
) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        blob = payload["jpeg_blob"]
        offsets = payload["jpeg_offsets"]
        valid = payload["valid"].astype(np.bool_)
        boxes = payload["boxes_normalized"].astype(np.float32)
        selected = payload["selected_raw_frame_indices"]
    if (
        blob.dtype != np.uint8
        or offsets.shape != (16, 3, 2)
        or valid.shape != (16, 3)
        or boxes.shape != (16, 3, 4)
        or selected.shape != (16,)
    ):
        raise ValueError(f"{path}: crop shape/dtype contract mismatch")
    if not np.isfinite(boxes).all() or (selected < 0).any() or (np.diff(selected) < 0).any():
        raise ValueError(f"{path}: invalid box/frame values")
    if not np.all(boxes[~valid] == 0):
        raise ValueError(f"{path}: invalid crop boxes must be zero")
    starts, lengths = offsets[..., 0], offsets[..., 1]
    if not np.all((starts[~valid] == -1) & (lengths[~valid] == 0)):
        raise ValueError(f"{path}: invalid crop offsets are not explicit")
    if not np.all((starts[valid] >= 0) & (lengths[valid] > 0)):
        raise ValueError(f"{path}: valid crop offsets are empty")
    if valid.any() and int(np.max(starts[valid] + lengths[valid])) > len(blob):
        raise ValueError(f"{path}: crop offset exceeds JPEG blob")
    if (
        metadata.get("schema_fingerprint")
        != crop_schema_fingerprint(HandRGBV17Config())
        or metadata.get("source") != source
        or metadata.get("source_item_id") != item_id
        or metadata.get("canonical_label") != label
        or metadata.get("selection_manifest_sha256") != manifest_sha256
        or metadata.get("split") != split
        or metadata.get("training_eligible") is not training_eligible
        or metadata.get("test_accessed") is not False
    ):
        raise ValueError(f"{path}: crop provenance mismatch")
    return {
        "valid_views": float(valid.sum()),
        "total_views": float(valid.size),
        "jpeg_bytes": float(len(blob)),
    }


def audit_embedding(
    path: Path,
    *,
    source: str,
    item_id: str,
    label: str,
    manifest_sha256: str,
    split: str,
    training_eligible: bool,
) -> dict[str, float]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        embeddings = payload["embeddings"].astype(np.float32)
        valid = payload["valid"].astype(np.bool_)
        boxes = payload["boxes_normalized"].astype(np.float32)
    if embeddings.shape != (16, 3, 512) or valid.shape != (16, 3) or boxes.shape != (16, 3, 4):
        raise ValueError(f"{path}: embedding shape mismatch")
    if not np.isfinite(embeddings).all() or not np.isfinite(boxes).all():
        raise ValueError(f"{path}: non-finite embedding values")
    if not np.all(embeddings[~valid] == 0) or not np.all(boxes[~valid] == 0):
        raise ValueError(f"{path}: invalid embedding views must be zero")
    norms = np.linalg.norm(embeddings[valid], axis=-1)
    if len(norms) and not np.allclose(norms, 1.0, atol=0.01, rtol=0.01):
        raise ValueError(f"{path}: valid embeddings are not unit normalized")
    if (
        metadata.get("schema_fingerprint")
        != embedding_schema_fingerprint(HandMobileCLIP2V17Config())
        or metadata.get("source") != source
        or metadata.get("source_item_id") != item_id
        or metadata.get("canonical_label") != label
        or metadata.get("selection_manifest_sha256") != manifest_sha256
        or metadata.get("split") != split
        or metadata.get("training_eligible") is not training_eligible
        or metadata.get("test_accessed") is not False
    ):
        raise ValueError(f"{path}: embedding provenance mismatch")
    return {"valid_views": float(valid.sum()), "total_views": float(valid.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("local_deep_clean", "local_deep_clean_val"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items, _ = selection_items(args.manifest, args.source)
    split, training_eligible = expected_provenance(args.source)
    manifest_sha256 = sha256_file(args.manifest)
    crop_expected = {
        args.crop_root / args.source / item.label / f"{item.item_id}.hand_rgb_v17.npz"
        for item in items
    }
    crop_actual = set((args.crop_root / args.source).glob("*/*.hand_rgb_v17.npz"))
    if crop_actual != crop_expected:
        raise ValueError(
            f"crop inventory mismatch missing={len(crop_expected-crop_actual)} "
            f"extra={len(crop_actual-crop_expected)}"
        )
    if args.embedding_root is not None:
        embedding_expected = {
            args.embedding_root / item.label
            / f"{item.item_id}.hand_mobileclip2_v17.npz"
            for item in items
        }
        embedding_actual = set(
            args.embedding_root.glob("*/*.hand_mobileclip2_v17.npz")
        )
        if embedding_actual != embedding_expected:
            raise ValueError(
                "embedding inventory mismatch "
                f"missing={len(embedding_expected-embedding_actual)} "
                f"extra={len(embedding_actual-embedding_expected)}"
            )
    crop_totals = {"valid_views": 0.0, "total_views": 0.0, "jpeg_bytes": 0.0}
    embedding_totals = {"valid_views": 0.0, "total_views": 0.0}
    for item in items:
        crop_path = args.crop_root / args.source / item.label / f"{item.item_id}.hand_rgb_v17.npz"
        values = audit_crop(
            crop_path, source=args.source, item_id=item.item_id, label=item.label,
            manifest_sha256=manifest_sha256, split=split,
            training_eligible=training_eligible,
        )
        for key, value in values.items():
            crop_totals[key] += value
        if args.embedding_root is not None:
            embedding_path = args.embedding_root / item.label / f"{item.item_id}.hand_mobileclip2_v17.npz"
            values = audit_embedding(
                embedding_path, source=args.source, item_id=item.item_id,
                label=item.label, manifest_sha256=manifest_sha256, split=split,
                training_eligible=training_eligible,
            )
            for key, value in values.items():
                embedding_totals[key] += value

    result = {
        "format": "slt_v17_local_deep_clean_hand_feature_audit",
        "source": args.source,
        "clips": len(items),
        "classes": len({item.label for item in items}),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "crop_schema_fingerprint": crop_schema_fingerprint(HandRGBV17Config()),
        "crop_valid_view_fraction": crop_totals["valid_views"] / crop_totals["total_views"],
        "crop_jpeg_bytes": int(crop_totals["jpeg_bytes"]),
        "embedding_audited": args.embedding_root is not None,
        "embedding_schema_fingerprint": (
            embedding_schema_fingerprint(HandMobileCLIP2V17Config())
            if args.embedding_root is not None else None
        ),
        "embedding_valid_view_fraction": (
            embedding_totals["valid_views"] / embedding_totals["total_views"]
            if args.embedding_root is not None else None
        ),
        "split": split,
        "training_eligible": training_eligible,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "errors": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
