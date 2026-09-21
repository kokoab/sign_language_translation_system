#!/usr/bin/env python3
"""Reuse exact-content caches for the user-reviewed PHRASES FIXED subset."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_grounded_streaming_data_v17 import write_archive
from scripts.prepare_stage2_training_manifest_v17 import LOCAL_TARGETS, sha256

REPORT = ROOT / "artifacts/reports/local_phrases_fixed_audit_20260921"
SOURCE = ROOT / "data/local/stage2_v17_grounded_signer_split"
OUTPUT = ROOT / "data/local/stage2_v17_grounded_phrases_fixed_20260921"


def main():
    inventory = json.loads((REPORT / "inventory.json").read_text())
    fixed = {row["sha256"]: row for row in inventory["retained"]}
    if len(fixed) != len(inventory["retained"]) or inventory["summary"]["new_or_changed"] or inventory["summary"]["relabeled"]:
        raise ValueError("only unique unchanged subset caches are supported")
    for row in fixed.values():
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("fixed video changed since inventory")
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    counts = Counter(); excluded = []; manifest = []; seen = set(); signers = {}
    for role in ("train", "validation"):
        signers[role] = set()
        for path in sorted((SOURCE / role).glob("*/*.npz")):
            target = OUTPUT / path.relative_to(SOURCE)
            with np.load(path, allow_pickle=False) as payload:
                m = json.loads(str(payload["metadata_json"].item()))
                if m["role"] != role:
                    raise ValueError("source role mismatch")
                if m["source"] != "local_phrases":
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    counts[f"{role}:{m['source']}"] += 1
                    continue
                digest = m["video_sha256"]
                row = fixed.get(digest)
                if row is None:
                    excluded.append({"archive": str(path.relative_to(ROOT)), "role": role,
                                     "video_path": m["video_path"], "sha256": digest})
                    continue
                expected = list(LOCAL_TARGETS[row["phrase"]])
                if expected != m["target_sequence"]:
                    raise ValueError(f"cached targets disagree with corrected folder: {row['path']}")
                if digest in seen:
                    raise ValueError("same content occurs more than once across train/validation")
                seen.add(digest)
                original_path = m["video_path"]
                m.update(video_path=row["path"], fixed_original_video_path=original_path,
                         phrase_review="user confirms every retained video checked against its phrase",
                         phrase_review_date="2026-09-21", fixed_inventory_sha256=sha256(REPORT / "inventory.json"))
                write_archive(target, payload["landmarks"], payload["window_source_ranges"], payload["target_indices"], m)
                with np.load(target, allow_pickle=False) as check:
                    for key in ("landmarks", "window_source_ranges", "target_indices"):
                        if not np.array_equal(check[key], payload[key]):
                            raise ValueError(f"reuse changed {key}")
                counts[f"{role}:local_phrases"] += 1
                counts[f"{role}:phrase:{row['phrase']}"] += 1
                signers[role].add(m["signer_id"])
                manifest.append({"archive": str(target.relative_to(ROOT)), "source_archive": str(path.relative_to(ROOT)),
                                 "video_path": row["path"], "sha256": digest, "role": role,
                                 "phrase": row["phrase"], "signer": m["signer_id"], "targets": expected})
    if signers["train"] & signers["validation"]:
        raise ValueError("local signer overlap")
    unused = [row for digest, row in fixed.items() if digest not in seen]
    summary = {"status": "passed", "source": str(SOURCE.relative_to(ROOT)), "output": str(OUTPUT.relative_to(ROOT)),
               "label_review": "user confirmed every retained video reviewed; no machine semantic-validation claim",
               "counts": dict(counts), "removed_from_cached_local_set": len(excluded),
               "signers": {role: sorted(values) for role, values in signers.items()},
               "fixed_without_admitted_cache": len(unused),
               "unused_by_phrase": dict(Counter(row["phrase"] for row in unused)),
               "features_and_targets_identical_for_retained_clips": True,
               "no_train_validation_content_overlap": True}
    (REPORT / "cache_verification.json").write_text(json.dumps({**summary, "retained": manifest, "excluded": excluded, "unused": unused}, indent=2) + "\n")
    text = "# Corrected local phrase audit\n\nUser confirms every retained clip was checked against its folder phrase.\n\n"
    text += "685 videos are unchanged original content;95 originals were removed, none relabeled or edited.\n\n"
    text += f"Retained cached local clips: {counts['train:local_phrases']} train / {counts['validation:local_phrases']} validation. Removed from this supervised subset: {len(excluded)}.\n\n"
    text += "Features and target arrays are bitwise identical for retained clips; source paths and review provenance are updated. Signer roles are preserved and disjoint. ASLLRP and NCSLGR archives are copied unchanged.\n\n"
    text += f"Fixed videos outside the admitted cache: {len(unused)}; see `cache_verification.json` for per-phrase counts. These include phrases outside the existing locked-label recipe and prior cache-admission exclusions; no new labels or unverified aliases are added.\n"
    (REPORT / "REPORT.md").write_text(text)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
