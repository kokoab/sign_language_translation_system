#!/usr/bin/env python3
"""Replay exact-ID O5S5 supervision through the existing live Apple Vision observer."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from active.v17.stage1_window_v17 import RAW_FORMAT, raw_observation_features
from active.v17.train_stage1_window_v17 import _sha256, load_context_samples
from scripts.cache_stage2_live_matched_v17 import input_contract
from scripts.diagnose_stage1_window_v17 import recording_observations
from scripts.live_reel_continuous_v17 import parser as live_parser


REPORT = Path("artifacts/reports/o5s5_citizen100_v17")
RAW_ROOT = Path("data/local/stage1_window_o5s5_v17/raw_observations")
BASE_SUPERVISION = Path("artifacts/reports/stage1_window_v17/supervision_manifest.json")
CITIZEN_MANIFEST = Path("active/v17/citizen100_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--base-supervision", type=Path, default=BASE_SUPERVISION)
    parser.add_argument("--citizen-manifest", type=Path, default=CITIZEN_MANIFEST)
    args = parser.parse_args()
    sources_path = args.report / "sources.json"
    occurrences_path = args.report / "exact_occurrences.csv"
    sources = json.loads(sources_path.read_text())
    with occurrences_path.open(newline="") as handle:
        exact = list(csv.DictReader(handle))
    exact_by_item_ordinal = {(row["source_item_id"], int(row["ordinal"])): row for row in exact}

    live_args = live_parser().parse_args(["--no-display", "--no-speech", "--naturalizer", "literal"])
    contract = input_contract(live_args)
    rows, diagnostics = [], []
    started = time.monotonic()
    for index, source in enumerate(sources, 1):
        video = Path(source["video_path"])
        if _sha256(video) != source["video_sha256"]:
            raise ValueError(f"source video changed: {video}")
        output = args.raw_root / source["role"] / f"{source['source_item_id']}.stage1_window_raw_v17.npz"
        metadata = {
            "source": "o5s5", "source_item_id": source["source_item_id"],
            "role": source["role"], "signer_id": source["signer_id"],
            "video_path": str(video), "video_sha256": source["video_sha256"],
            "eaf_sha256": source["eaf_sha256"], "observer_contract": contract,
            **({"exclude_final_frames": source["exclude_final_frames"]} if source.get("exclude_final_frames") else {}),
        }
        if output.exists():
            with np.load(output, allow_pickle=False) as payload:
                raw = payload["raw_features"].astype(np.float32, copy=False)
                timestamps = payload["timestamps_seconds"].astype(np.float64, copy=False)
                if str(payload["raw_format"].item()) != RAW_FORMAT or json.loads(str(payload["metadata_json"].item())) != metadata:
                    raise ValueError(f"cached archive provenance changed: {output}")
        else:
            observations, _ = recording_observations({
                "video": str(video), "discard_pixels": True,
                "exclude_final_frames": source.get("exclude_final_frames", 0),
            }, live_args)
            raw, timestamps = raw_observation_features(observations)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary, raw_features=raw, timestamps_seconds=timestamps,
                raw_format=np.array(RAW_FORMAT), metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
            )
            temporary.replace(output)

        intervals = []
        usable_exact = 0
        for ordinal, interval in enumerate(source["intervals"]):
            target = exact_by_item_ordinal.get((source["source_item_id"], ordinal))
            start, end = interval["start_ms"] / 1000, interval["end_ms"] / 1000
            keep = (timestamps >= start) & (timestamps <= end)
            hand = (raw[keep, :42, 4] > 0).any(axis=1) if keep.any() else np.zeros(0, dtype=bool)
            if target and hand.any():
                usable_exact += 1
            intervals.append({
                "start_seconds": start, "end_seconds": end,
                "label": target["canonical_label"] if target else "__OTHER__",
                "id_gloss": interval["id_gloss"], "exact_locked_match": target is not None,
            })
        left = (raw[:, :21, 4] > 0).any(axis=1)
        right = (raw[:, 21:42, 4] > 0).any(axis=1)
        expected_exact = sum(item == source["source_item_id"] for item, _ in exact_by_item_ordinal)
        diagnostics.append({
            "source_item_id": source["source_item_id"], "role": source["role"],
            "signer_id": source["signer_id"], "observations": len(raw),
            "timeline_seconds": float(timestamps[-1] - timestamps[0]),
            "any_hand_frames": int((left | right).sum()), "both_hand_frames": int((left & right).sum()),
            "expected_exact_occurrences": expected_exact, "usable_exact_occurrences": usable_exact,
        })
        rows.append({
            "role": source["role"], "source": "o5s5", "signer_id": source["signer_id"],
            "source_item_id": source["source_item_id"], "archive_path": str(output),
            "all_signs_annotated": False, "background_training_eligible": False,
            "intervals": intervals, "video_sha256": source["video_sha256"],
        })
        progress = {"completed": index, "total": len(sources), "elapsed_seconds": time.monotonic() - started, "diagnostics": diagnostics}
        (args.report / "apple_vision_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(f"[{index}/{len(sources)}] {source['source_item_id']}: {usable_exact}/{expected_exact} exact occurrences have Vision hands", flush=True)

    payload = {
        "format": "slt_stage1_window_supervision_v17", "version": 1, "raw_format": RAW_FORMAT,
        "rows": rows, "observer_contract": contract,
        "sources_sha256": _sha256(sources_path), "exact_occurrences_sha256": _sha256(occurrences_path),
        "background_training_eligible": False, "citizen_test_accessed": False,
    }
    (args.report / "apple_vision_supervision.json").write_text(json.dumps(payload, indent=2) + "\n")
    base = json.loads(args.base_supervision.read_text())
    if base.get("format") != payload["format"] or base.get("version") != payload["version"]:
        raise ValueError("incompatible base supervision manifest")
    combined = {
        **base, "rows": [*base["rows"], *rows],
        "base_supervision_sha256": _sha256(args.base_supervision),
        "o5s5_sources_sha256": payload["sources_sha256"],
        "o5s5_exact_occurrences_sha256": payload["exact_occurrences_sha256"],
        "citizen_test_accessed": False,
    }
    role_signers = {
        role: {row["signer_id"] for row in combined["rows"] if row["role"] == role}
        for role in ("train", "validation")
    }
    if role_signers["train"] & role_signers["validation"]:
        raise ValueError("combined supervision is not signer-disjoint")
    (args.report / "combined_supervision.json").write_text(json.dumps(combined, indent=2) + "\n")
    audit = {
        "format": "o5s5_apple_vision_audit_v17", "observer_contract": contract,
        "rows": diagnostics, "source_videos": len(rows),
        "observations": sum(row["observations"] for row in diagnostics),
        "expected_exact_occurrences": sum(row["expected_exact_occurrences"] for row in diagnostics),
        "usable_exact_occurrences": sum(row["usable_exact_occurrences"] for row in diagnostics),
        "role_counts": dict(Counter(row["role"] for row in diagnostics)),
        "background_training_eligible": False, "protected_test_accessed": False,
    }
    (args.report / "apple_vision_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    classes = json.loads(args.citizen_manifest.read_text())["classes"]
    labels = {row["canonical_label"]: int(row["class_index"]) for row in classes}
    loader_audit = {}
    for role in ("train", "validation"):
        samples, role_audit = load_context_samples(args.report / "combined_supervision.json", labels, role)
        o5s5 = [row for row in samples if row.source == "o5s5"]
        loader_audit[role] = {
            "audit": role_audit, "o5s5_windows": len(o5s5),
            "o5s5_targets": len({row.target for row in o5s5}),
            "o5s5_background": sum(row.category == "background" for row in o5s5),
        }
    (args.report / "loader_audit.json").write_text(json.dumps(loader_audit, indent=2) + "\n")
    readme = f"""# O5S5 exact Citizen-100 supervision\n\nThe six public, frontal O5S5 narratives provide {sum(row['duration_seconds'] for row in sources) / 60:.2f} minutes from six signers. Exact ASL-LEX-to-Signbank matching admits {audit['expected_exact_occurrences']} deduplicated occurrences across 53 frozen Citizen classes; all {audit['usable_exact_occurrences']} contain Apple Vision hand detections. These are genuine Apple Vision observations extracted from source pixels, not converted MediaPipe coordinates.\n\n`LG` is reserved as a whole validation signer. O5S5 adds {loader_audit['train']['o5s5_windows']} positive training windows across {loader_audit['train']['o5s5_targets']} classes and {loader_audit['validation']['o5s5_windows']} validation windows across {loader_audit['validation']['o5s5_targets']} classes. Combined with the existing ASLLRP supervision, training has {loader_audit['train']['audit']['context_windows']} contextual windows across {loader_audit['train']['audit']['distinct_signs']}/100 classes plus {loader_audit['train']['audit']['background_windows']} verified background windows. O5S5 itself contributes zero background windows because ID-gloss completeness is not established.\n\nThe source is public under CC BY-NC-SA 4.0; no account or access request is required.\n\n- `exact_occurrences.csv`: authoritative exact positive-event manifest\n- `apple_vision_supervision.json`: O5S5 positive-only Apple Vision supervision\n- `combined_supervision.json`: ready-to-load ASLLRP + O5S5 supervision\n- `audit.json`, `apple_vision_audit.json`, `loader_audit.json`: provenance, coverage, and loader checks\n- `sources.json`: source video/EAF hashes and deduplicated ID-gloss intervals\n"""
    (args.report / "README.md").write_text(readme)
    print(json.dumps({key: audit[key] for key in ("observations", "expected_exact_occurrences", "usable_exact_occurrences")}, indent=2))


if __name__ == "__main__":
    main()
