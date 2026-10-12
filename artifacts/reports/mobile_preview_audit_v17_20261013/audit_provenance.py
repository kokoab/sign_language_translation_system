"""Read-only preview membership audit. Reads metadata and hashes local preview sources.

No official test video is opened. Run from any directory with repository venv Python.
"""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CATALOG = OUT / "original_catalog.json"
TRAIN = "data/local/web_variant_supplement_v17_20261008/local_train_citizen_variant_v2.json"
VAL = "data/local/web_variant_supplement_v17_20261008/local_val_citizen_variant_v2.json"


def read(path):
    return json.loads((ROOT / path).read_text())


def index(path):
    rows = read(path)["videos"]
    return (
        {str((ROOT / row[key]).resolve()): row for row in rows
         for key in ("source_raw_path", "raw_path") if row.get(key)},
        {row["raw_sha256"]: row for row in rows if row.get("raw_sha256")},
    )


def main():
    catalog = json.loads(CATALOG.read_text())
    indexes = {"train": index(TRAIN), "validation": index(VAL)}
    rows = []
    for label, preview in catalog["examples"].items():
        if preview["source"] != "local":
            continue
        source = preview["source_video"]
        assert source.startswith("data/raw_videos/ASL VIDEOS/"), source
        resolved = str((ROOT / source).resolve())
        digest = hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
        memberships = {}
        for split, (paths, hashes) in indexes.items():
            path_row, hash_row = paths.get(resolved), hashes.get(digest)
            assert bool(path_row) == bool(hash_row), (label, split)
            if path_row:
                assert path_row["canonical_label"] == label, (label, split)
                memberships[split] = path_row["canonical_label"]
        assert len(memberships) <= 1, label
        status = next(iter(memberships), "absent")
        reason = "admitted_with_same_label" if memberships else "not_admitted_reason_unestablished"
        if label in {"BIG", "SIGN", "COME", "CHILD"}:
            reason = "local_class_excluded_for_differing_citizen_variant"
        rows.append({"label": label, "source_video": source, "source_sha256": digest,
                     "membership": status, "reason": reason})
    counts = dict(collections.Counter(row["membership"] for row in rows))
    assert counts == {"train": 50, "validation": 7, "absent": 20}, counts
    evidence = [TRAIN, VAL,
        "ATLAS/mobile_app/slt_mobile_app/assets/gloss_examples/catalog.json",
        "ATLAS/mobile_app/slt_mobile_app/ios/Runner/LiveReel/gloss_examples/catalog.json",
        "data/local/local_deep_clean_v17/train_final_manifest.json",
        "data/local/local_deep_clean_v17/val_final_manifest.json",
        "data/local/local_citizen100_quality_audit_q82_cap14_exact/candidate_selection.json",
        "artifacts/reports/citizen_variant_local_filter_v17_20261008/build_manifests.py",
        "artifacts/reports/citizen_variant_local_filter_v17_20261008/run_experiment_v2.py"]
    report = {"date": "2026-10-13", "catalog_snapshot": str(CATALOG.relative_to(ROOT)),
        "local_preview_count": len(rows), "counts": counts, "source_label_mismatches": 0,
        "path_and_hash_membership_agree": True, "official_test_videos_accessed": False,
        "findings": ["BIG and SIGN were original deep-clean train inputs, removed for Citizen variant mismatch.",
                     "COME and CHILD previews were not original admitted inputs; their local classes are variant-excluded.",
                     "ASK was initially excluded, restored in the v2 training recipe.",
                     "The preview quality-pool manifest declares training_eligible=false.",
                     "Original Flutter and iOS catalogs were identical at audit time."],
        "evidence": evidence, "previews": rows}
    (OUT / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Original mobile preview provenance audit — 2026-10-13", "",
        "The original catalog contains 77 local previews: **50 current train**, **7 validation**, "
        "and **20 absent**. Source paths and SHA256 membership agree; all admitted labels agree. "
        "This is a membership audit, not a recognition score. No official test video was accessed.", "",
        "BIG and SIGN were original deep-clean training inputs removed for differing Citizen variants. "
        "COME and CHILD previews were never admitted in examined deep-clean metadata; their local "
        "classes are also explicitly variant-excluded. ASK was restored in v2. "
        "The preview quality pool itself declares `training_eligible=false`.", "",
        "| Label | Membership | Original source video |", "|---|---|---|"]
    lines.extend(f"| {r['label']} | {r['membership']} | `{r['source_video']}` |" for r in rows)
    lines.extend(["", "Other absent previews have no established exclusion reason in the examined "
        "pre/post deep-clean manifests, quarantine or extraction-rejection metadata. "
        "Absence alone does not prove an incorrect sign.", "", "Evidence:", ""])
    lines.extend(f"- `{path}`" for path in evidence)
    lines.extend(["", "Rerun: `venv/bin/python artifacts/reports/mobile_preview_audit_v17_20261013/audit_provenance.py`", ""])
    (OUT / "provenance.md").write_text("\n".join(lines))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
