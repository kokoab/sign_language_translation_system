#!/usr/bin/env python3
"""Freeze non-test inputs for the Stage-2 STEM transition adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_ENCODER_SHA256 = "1caeadf4b3ca620aa9fef00b35c012b39d7c093f67da1ee2f6987d2c2297906b"
EXPECTED_QUEUE_SHA256 = "38437d0afd506d050b0b89a452ae0a9d769d72e28496d9ab3b6c92235d64650a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(sha256(path).encode() + b"\n")
    return digest.hexdigest()


def participant_split(participants: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(set(participants), key=lambda value: hashlib.sha256(
        f"1701:{value}".encode("utf-8")).hexdigest())
    return ordered[:4], ordered[4:]


def frame_interval_times(start_frame: int, end_frame_inclusive: int, source_fps: float) -> dict[str, Any]:
    if start_frame < 0 or end_frame_inclusive < start_frame or source_fps <= 0:
        raise ValueError("invalid inclusive interval or source fps")
    return {
        "source_start_frame_inclusive": start_frame,
        "source_end_frame_inclusive": end_frame_inclusive,
        "source_fps": source_fps,
        "start_seconds": start_frame / source_fps,
        "end_seconds_exclusive": (end_frame_inclusive + 1) / source_fps,
        "sample_rate_fps": 30.0,
        "geometry_transform": "none",
    }


def validate_queue(path: Path, expected_sha: str = EXPECTED_QUEUE_SHA256, *, expected_count: int = 111, expected_participants: int = 18) -> list[dict[str, str]]:
    if sha256(path) != expected_sha:
        raise ValueError("queue sha256 does not match frozen final queue")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["training_eligible"] == "True"]
    if len(rows) != expected_count or len({row["participant"] for row in rows}) != expected_participants:
        raise ValueError("frozen queue eligibility count or participant count differs")
    if any(not row["verified_start_frame"] or not row["verified_end_frame"] for row in rows):
        raise ValueError("eligible row lacks verified inclusive bounds")
    return rows


def validate_encoder_sha256(value: str) -> str:
    if value != EXPECTED_ENCODER_SHA256:
        raise ValueError("frozen archive encoder hash does not match pinned encoder")
    return value


def validate_target_index(index: int) -> int:
    if not 0 <= index < 100:
        raise ValueError("stored locked gloss index must be in [0, 99]")
    return index


def validate_no_overlap(rows: list[dict[str, Any]]) -> None:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("parent_video_sha256"):
            by_parent[row["parent_video_sha256"]].append(row)
    for parent, members in by_parent.items():
        roles = {row["role"] for row in members}
        if len(roles) > 1:
            raise ValueError(f"train/validation parent leakage: {parent}")


def load_vocab(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text())
    labels = [str(row["canonical_label"]).upper() for row in payload["classes"]]
    if len(labels) != 100 or len(set(labels)) != 100:
        raise ValueError("expected exactly 100 locked vocabulary labels")
    return {label: index for index, label in enumerate(labels)}


def validate_pool(path: Path, expected_split: str, class_count: int, labels: dict[str, int] | None = None) -> dict[str, Any]:
    import numpy as np
    with np.load(path, allow_pickle=False) as payload:
        targets = payload["target_indices"].astype(np.int64)
        metadata = json.loads(str(payload["metadata_json"]))
        item_ids = payload["item_ids"].tolist() if "item_ids" in payload.files else metadata.get("item_ids", [])
    if len(item_ids) != len(targets) or any(not str(item_id) for item_id in item_ids):
        raise ValueError("pool requires one nonempty item IDs entry per target")
    validate_encoder_sha256(str(metadata.get("stage1_checkpoint_sha256", "")))
    actual_split = metadata.get("source_split", metadata.get("role"))
    if actual_split != expected_split or len(set(targets.tolist())) != class_count:
        raise ValueError("pool role or label contract mismatch")
    if targets.min() != 0 or targets.max() != class_count - 1:
        raise ValueError("pool label indices mismatch")
    if labels is not None and set(metadata.get("class_counts", {})) != set(labels):
        raise ValueError("pool vocabulary identity mismatch")
    ordered_labels = [label for label, _ in sorted(labels.items(), key=lambda pair: pair[1])] if labels else []
    if labels is not None:
        # Pool IDs are `CANONICAL/...`; this pins their stored 0-based label index.
        for item_id, index in zip(item_ids, targets.tolist()):
            if str(item_id).split("/", 1)[0] != ordered_labels[index]:
                raise ValueError("pool item-id label order mismatch")
    return {"path": path.as_posix(), "sha256": sha256(path), "source_split": expected_split, "items": len(targets), "item_ids": [str(value) for value in item_ids]}


def validate_pool_pair(train: dict[str, Any], validation: dict[str, Any]) -> None:
    if set(train["item_ids"]) & set(validation["item_ids"]):
        raise ValueError("Citizen item leakage across train/validation")


def audit_archives(root: Path, source_manifest: Path, canonical_manifest: Path, multimodal_root: Path, vocabulary: dict[str, int]) -> dict[str, Any]:
    import numpy as np
    source = json.loads(source_manifest.read_text())
    canonical_sha = sha256(canonical_manifest)
    canonical = {row["source_item_id"]: row for row in json.loads(canonical_manifest.read_text())["rows"]}
    expected = {}
    for span in source["spans"]:
        item_id = f"{span['source']}:{Path(span['utterance_video_filename']).stem}:span{int(span['span_index_in_utterance']):02d}"
        expected[item_id] = span
    multimodal = {}
    for path in multimodal_root.rglob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        multimodal[str(metadata.get("source_item_id"))] = metadata
    counts = Counter()
    observed = set()
    identities = []
    for path in sorted(root.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        validate_encoder_sha256(str(metadata.get("stage1_checkpoint_sha256", "")))
        item_id = str(metadata.get("source_item_id", ""))
        span = expected.get(item_id)
        if span is None or item_id in observed:
            raise ValueError(f"archive identity mismatch: {path}")
        role = str(metadata.get("role"))
        if role != str(span["split_role"]) or list(metadata.get("target_sequence", [])) != list(span["target_sequence"]):
            raise ValueError(f"archive split or ordered targets mismatch: {path}")
        if metadata.get("video_sha256") != span["sha256"]:
            raise ValueError(f"archive video provenance mismatch: {path}")
        source_metadata = multimodal.get(item_id)
        canonical_row = canonical.get(item_id)
        if canonical_row is None or canonical_row["parent_sha256"] != span["parent_sha256"] or canonical_row["video_path"] != span["path"]:
            raise ValueError(f"canonical parent or video provenance mismatch: {path}")
        if source_metadata is None or source_metadata.get("signer_id") != canonical_row["signer_id"] or source_metadata.get("video_sha256") != metadata.get("video_sha256"):
            raise ValueError(f"archive signer or source-video provenance mismatch: {path}")
        for actual in (metadata, source_metadata):
            if actual.get("training_manifest_sha256") != canonical_sha:
                raise ValueError(f"archive canonical training-manifest hash mismatch: {path}")
        if any(token != "__OTHER__" and token not in vocabulary for token in span["target_sequence"]):
            raise ValueError(f"archive vocabulary mismatch: {path}")
        observed.add(item_id)
        counts[role] += 1
        identities.append({"role": role, "signer_id": span["signer_id"], "parent_video_sha256": span["parent_sha256"], "interval": [span["crop_start_frame_local"], span["crop_end_frame_local"]]})
    if counts != Counter({"train": 879, "validation": 225}):
        raise ValueError(f"unexpected ASLLRP frozen archive roles: {dict(counts)}")
    if observed != set(expected):
        raise ValueError("ASLLRP archive set differs from frozen manifest")
    validate_no_overlap(identities)
    return {"root": root.as_posix(), "directory_sha256": directory_digest(root), "archives": sum(counts.values()), "role_counts": dict(counts), "encoder_sha256": EXPECTED_ENCODER_SHA256, "canonical_manifest": canonical_manifest.as_posix(), "canonical_manifest_sha256": canonical_sha, "identity_validation": "frozen/multimodal metadata pinned to canonical manifest; parent/interval provenance is indirect through canonical/source-manifest joins"}


def validate_identity_ledger(rows: list[dict[str, Any]]) -> None:
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        identity = row.get("identity")
        if not identity:
            continue
        key = (str(row.get("namespace")), str(identity))
        role = str(row["role"])
        previous = seen.get(key)
        if previous is not None and previous != role:
            raise ValueError(f"experiment-wide train/validation leakage: {key}")
        seen[key] = role


def video_metadata(path: Path) -> dict[str, Any]:
    import cv2
    capture = cv2.VideoCapture(path.as_posix())
    if not capture.isOpened():
        raise ValueError(f"cannot open STEM source video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid STEM source metadata: {path}")
    return {"source_fps": fps, "source_width": width, "source_height": height, "aspect_ratio": width / height}


def validate_phrase_manifest(path: Path, vocabulary: dict[str, int], *, required_source: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    records = []
    for span in payload["spans"]:
        if span["source"] != required_source:
            if span["split_role"] != "external_evaluation_reserved":
                raise ValueError(f"unexpected phrase source in {path}")
            continue
        role = str(span["split_role"])
        if role not in {"train", "validation", "train_candidate", "external_evaluation_reserved"}:
            raise ValueError(f"unexpected phrase role: {role}")
        if any(token != "__OTHER__" and token not in vocabulary for token in span["target_sequence"]):
            raise ValueError("phrase vocabulary mismatch")
        records.append({"role": "train" if role == "train_candidate" else role, "parent_video_sha256": span["parent_sha256"], "interval": [span.get("crop_start_frame_local", 0), span.get("crop_end_frame_local", 0)]})
    validate_no_overlap([row for row in records if row["role"] in {"train", "validation"}])
    return {"path": path.as_posix(), "sha256": sha256(path), "spans": len(records), "source": required_source}


def validate_local_audit(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"path", "sha256"} <= set(rows[0]):
        raise ValueError("local phrase audit lacks provenance fields")
    return {"path": path.as_posix(), "sha256": sha256(path), "rows": len(rows), "inherited_checkpoint_exposure": True}


def audit_phrase_frozen(root: Path, training_manifest: Path) -> dict[str, Any]:
    import numpy as np
    expected = {row["source_item_id"]: row for row in json.loads(training_manifest.read_text())["rows"] if row["role"] in {"train", "validation"}}
    observed = set()
    ledger = []
    for path in root.rglob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        item_id = metadata.get("source_item_id")
        row = expected.get(item_id)
        if row is None or item_id in observed:
            raise ValueError(f"phrase frozen identity mismatch: {path}")
        if metadata.get("role") != row["role"] or metadata.get("video_sha256") != row["video_sha256"] or list(metadata.get("target_sequence", [])) != row["target_sequence"]:
            raise ValueError(f"phrase frozen role, target, or video mismatch: {path}")
        validate_encoder_sha256(str(metadata.get("stage1_checkpoint_sha256", "")))
        observed.add(item_id)
        ledger.append({"role": row["role"], "parent_video_sha256": row.get("parent_video_sha256", row["video_sha256"]), "interval": [0, int(row.get("frame_count", 0))]})
    if observed != set(expected):
        raise ValueError("phrase frozen archive set differs from training manifest")
    validate_no_overlap(ledger)
    return {"root": root.as_posix(), "directory_sha256": directory_digest(root), "archives": len(observed), "training_manifest_sha256": sha256(training_manifest), "identity_validation": "manifest item, role, ordered targets, video, encoder"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if sha256(args.encoder) != EXPECTED_ENCODER_SHA256:
        raise ValueError("pinned encoder file hash mismatch")
    queue = validate_queue(args.queue)
    vocabulary = load_vocab(args.vocabulary)
    validation, train = participant_split([row["participant"] for row in queue])
    rows = []
    for ordinal, item in enumerate(queue):
        role = "validation" if item["participant"] in validation else "train"
        source = Path(item["video_path"])
        if not source.exists() or sha256(source) != item["video_sha256"]:
            raise ValueError(f"missing or changed STEM source: {source}")
        label = item["canonical_label"].upper()
        if label not in vocabulary:
            raise ValueError(f"STEM label outside locked vocabulary: {label}")
        source_video_metadata = video_metadata(source)
        timing = frame_interval_times(int(item["verified_start_frame"]), int(item["verified_end_frame"]), source_video_metadata["source_fps"])
        row = {"source": "asl_stem_wiki_verified_interval", "role": role, "source_item_id": f"stem:{item['participant']}:{ordinal:03d}", "participant": item["participant"], "signer_id": item["participant"], "source_group": f"asl_stem_wiki:participant:{item['participant']}", "video_path": source.as_posix(), "video_sha256": item["video_sha256"], "parent_video_sha256": item["video_sha256"], "target_sequence": [label], "target_indices": [validate_target_index(vocabulary[label])], "interval": [timing["source_start_frame_inclusive"], timing["source_end_frame_inclusive"]], "timing": {**timing, **source_video_metadata}}
        rows.append(row)
    validate_no_overlap(rows)
    archive_audit = audit_archives(args.asllrp_archives, args.asllrp_other_manifest, args.asllrp_other_canonical_manifest, args.asllrp_other_multimodal, vocabulary)
    phrase_audit = validate_phrase_manifest(args.asllrp_phrase_manifest, vocabulary, required_source="asllrp")
    phrase_frozen_audit = audit_phrase_frozen(args.phrase_frozen_archives, args.phrase_training_manifest)
    local_audit = validate_local_audit(args.local_phrase_audit)
    pool_audit = {"citizen_train": validate_pool(args.citizen_train_pool, "citizen_official_train_only", 100, vocabulary), "citizen_validation": validate_pool(args.citizen_validation_pool, "validation", 100, vocabulary)}
    validate_pool_pair(pool_audit["citizen_train"], pool_audit["citizen_validation"])
    ledger = [{"namespace": "parent_video_sha256", "identity": row["parent_video_sha256"], "role": row["role"]} for row in rows]
    other_source = json.loads(args.asllrp_other_manifest.read_text())["spans"]
    ledger += [{"namespace": "parent_video_sha256", "identity": span["parent_sha256"], "role": span["split_role"]} for span in other_source]
    phrase_source = json.loads(args.phrase_training_manifest.read_text())["rows"]
    ledger += [{"namespace": "parent_video_sha256", "identity": row.get("parent_video_sha256", row["video_sha256"]), "role": row["role"]} for row in phrase_source if row["role"] in {"train", "validation"}]
    ledger += [{"namespace": "citizen_item_id", "identity": item, "role": "train"} for item in pool_audit["citizen_train"]["item_ids"]]
    ledger += [{"namespace": "citizen_item_id", "identity": item, "role": "validation"} for item in pool_audit["citizen_validation"]["item_ids"]]
    validate_identity_ledger(ledger)
    inputs = [args.encoder, args.queue, args.vocabulary, args.asllrp_archives, args.asllrp_other_manifest, args.asllrp_other_canonical_manifest, args.asllrp_other_multimodal, args.asllrp_phrase_manifest, args.phrase_frozen_archives, args.phrase_training_manifest, args.local_phrase_audit, args.citizen_train_pool, args.citizen_validation_pool]
    for path in inputs:
        if not path.exists():
            raise ValueError(f"required input is absent: {path}")
    coverage = {role: dict(Counter(row["target_sequence"][0] for row in rows if row["role"] == role)) for role in ("train", "validation")}
    audit_results = {"stem": {"rows": len(rows), "role_counts": dict(Counter(row["role"] for row in rows)), "per_gloss_coverage": coverage}, "asllrp_frozen": archive_audit, "asllrp_phrase": phrase_audit, "phrase_frozen": phrase_frozen_audit, "local_phrase": local_audit, "citizen_pools": {key: {field: value for field, value in audit.items() if field != "item_ids"} for key, audit in pool_audit.items()}, "experiment_wide_leakage": "passed"}
    roles = {args.encoder: "frozen encoder", args.queue: "verified STEM supervision", args.vocabulary: "locked vocabulary", args.asllrp_archives: "ASLLRP OTHER train/validation replay", args.asllrp_other_manifest: "ASLLRP OTHER source provenance", args.asllrp_other_canonical_manifest: "ASLLRP OTHER canonical extraction manifest", args.asllrp_other_multimodal: "ASLLRP OTHER multimodal provenance", args.asllrp_phrase_manifest: "exact ASLLRP phrase provenance", args.phrase_frozen_archives: "exact/local phrase frozen replay", args.phrase_training_manifest: "exact/local phrase frozen manifest", args.local_phrase_audit: "local familiar-domain retention provenance", args.citizen_train_pool: "Citizen official-training replay", args.citizen_validation_pool: "Citizen official-validation retention"}
    manifest = {"format": "slt_v17_stage2_transition_adapt", "version": 1, "encoder": {"path": args.encoder.as_posix(), "sha256": EXPECTED_ENCODER_SHA256}, "vocabulary": {"path": args.vocabulary.as_posix(), "sha256": sha256(args.vocabulary), "blank_index": 0, "locked_gloss_indices": "1-100", "other_index": 101}, "inputs": [{"path": path.as_posix(), "sha256": sha256(path) if path.is_file() else directory_digest(path), "source_role": roles.get(path, "frozen provenance")} for path in inputs], "split_rule": "sort SHA-256(UTF-8 '1701:<participant>'); first four validation", "validation_participants": validation, "training_participants": train, "rows": rows, "audit_results": audit_results, "exclusions": ["NCSLGR", "2M-Flores", "unreviewed surrounding STEM content"], "local_phrase_evaluation": "familiar-domain retention; inherited checkpoint exposure", "citizen_test_accessed": False, "protected_test_flags": {"citizen_test_closed": True, "semlex_test_accessed": False, "local_test_accessed": False}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    audit = {"manifest": args.output.as_posix(), "manifest_sha256": sha256(args.output), "participants": len(validation) + len(train), "audit_results": audit_results, "citizen_test_accessed": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path, default=Path("artifacts/models/stage1_v17_unified_multimodal_student_v1/best_model.pth"))
    parser.add_argument("--queue", type=Path, default=Path("artifacts/reports/asl_stem_wiki_manual_expansion_admission_v17/expert_review_queue.final.csv"))
    parser.add_argument("--vocabulary", type=Path, default=Path("active/v17/citizen100_manifest.json"))
    parser.add_argument("--asllrp-archives", type=Path, default=Path("data/local/stage2_v17_asllrp_other_frozen_features"))
    parser.add_argument("--asllrp-other-manifest", type=Path, default=Path("data/local/asllrp_other_ctc_v17/manifest.json"))
    parser.add_argument("--asllrp-other-canonical-manifest", type=Path, default=Path("active/v17/stage2_asllrp_other_ctc_manifest_v17.json"))
    parser.add_argument("--asllrp-other-multimodal", type=Path, default=Path("data/local/stage2_v17_asllrp_other_multimodal"))
    parser.add_argument("--phrase-frozen-archives", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument("--phrase-training-manifest", type=Path, default=Path("active/v17/stage2_training_manifest_v17.json"))
    parser.add_argument("--asllrp-phrase-manifest", type=Path, default=Path("data/local/asllrp_contiguous_phrases_v17/manifest.json"))
    parser.add_argument("--local-phrase-audit", type=Path, default=Path("artifacts/reports/stage2_v17_data_audit/local_videos.csv"))
    parser.add_argument("--citizen-train-pool", type=Path, default=Path("data/local/stage2_v17_synthetic/citizen_train_isolated_pool.npz"))
    parser.add_argument("--citizen-validation-pool", type=Path, default=Path("data/local/stage2_v17_isolated_replay/citizen_validation.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/stage2_v17_transition_adapt_v1/manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_transition_adapt_v1/audit.json"))
    return parser


if __name__ == "__main__":
    print(json.dumps(run(build_parser().parse_args()), indent=2))
