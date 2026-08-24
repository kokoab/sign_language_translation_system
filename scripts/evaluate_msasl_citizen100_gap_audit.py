#!/usr/bin/env python3
"""Use the frozen Citizen100 model to triage retained MS-ASL train candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.schema_v17 import V17Config, schema_fingerprint
from scripts.evaluate_rit_citizen100_variant_audit import load_features, sha256_file


def aggregate_classes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["true_label"]), []).append(row)
    output: list[dict[str, object]] = []
    for label in sorted(grouped):
        items = grouped[label]
        clips = len(items)
        signers = len({str(item["msasl_signer_id"]) for item in items})
        top1_hits = sum(bool(item["top1_hit"]) for item in items)
        top5_hits = sum(bool(item["top5_hit"]) for item in items)
        top1_rate = top1_hits / clips
        top5_rate = top5_hits / clips
        if signers >= 2 and top1_rate >= 0.5 and top5_rate >= 0.8:
            triage = "model_consistent_manual_variant_review_required"
        elif top5_rate >= 0.5:
            triage = "ambiguous_manual_variant_review_required"
        else:
            triage = "high_risk_manual_variant_review_required"
        output.append(
            {
                "canonical_label": label,
                "citizen_raw_gloss": items[0]["citizen_raw_gloss"],
                "citizen_asl_lex_code": items[0]["citizen_asl_lex_code"],
                "clips": clips,
                "signers": signers,
                "top1_hits": top1_hits,
                "top5_hits": top5_hits,
                "mean_true_probability": float(
                    np.mean([float(item["true_probability"]) for item in items])
                ),
                "triage": triage,
                "training_approved": False,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        default=Path("artifacts/models/stage1_v17_baseline/best_model.pth"),
        nargs="?",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data/local/msasl_citizen100_gap_audit/landmarks"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/local/msasl_citizen100_gap_audit/candidate_provenance.json"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/msasl_citizen100_gap_audit"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance.get("training_eligible") is not False or provenance.get("split") != "train_only":
        raise ValueError("MS-ASL provenance must be unapproved and train-only")
    retained = [
        row
        for row in provenance["videos"]
        if row["status"] in {"downloaded", "existing_verified"}
    ]
    metadata_by_stem = {
        (str(row["canonical_label"]), Path(str(row["destination"])).stem): row
        for row in retained
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("not a v17 Stage 1 checkpoint")
    if checkpoint["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("checkpoint manifest mismatch")
    expected_schema = schema_fingerprint(V17Config())
    if checkpoint["schema_fingerprint"] != expected_schema:
        raise ValueError("checkpoint extractor schema mismatch")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels = {
        str(item["canonical_label"]): int(item["class_index"])
        for item in manifest["classes"]
    }
    index_to_label = {index: label for label, index in labels.items()}
    files = sorted(args.feature_root.glob("*/*.v17.npz"))
    features: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for path in files:
        key = (path.parent.name, path.name.removesuffix(".v17.npz"))
        if key not in metadata_by_stem:
            raise ValueError(f"feature absent from retained MS-ASL provenance: {path}")
        features.append(load_features(path, expected_schema))
        records.append(metadata_by_stem[key])
    if len(files) != len(metadata_by_stem):
        missing = sorted(
            set(metadata_by_stem)
            - {
                (path.parent.name, path.name.removesuffix(".v17.npz"))
                for path in files
            }
        )
        raise ValueError(f"missing {len(missing)} MS-ASL features; first={missing[:1]}")

    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        logits = torch.cat(
            [
                model(torch.from_numpy(np.stack(features[start : start + args.batch_size]))).float()
                for start in range(0, len(features), args.batch_size)
            ]
        )
    probabilities = logits.softmax(dim=1)
    top5 = logits.topk(5, dim=1).indices
    rows: list[dict[str, object]] = []
    for index, (path, record) in enumerate(zip(files, records)):
        target = labels[str(record["canonical_label"])]
        predicted = int(logits[index].argmax())
        top5_indices = [int(value) for value in top5[index].tolist()]
        rows.append(
            {
                "feature_path": str(path),
                "msasl_signer_id": record["msasl_signer_id"],
                "true_label": record["canonical_label"],
                "citizen_raw_gloss": record["citizen_raw_gloss"],
                "citizen_asl_lex_code": record["citizen_asl_lex_code"],
                "predicted_label": index_to_label[predicted],
                "top5_labels": "|".join(index_to_label[value] for value in top5_indices),
                "top1_hit": predicted == target,
                "top5_hit": target in top5_indices,
                "true_probability": float(probabilities[index, target]),
                "training_approved": False,
            }
        )
    classes = aggregate_classes(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, output_rows in (("predictions.csv", rows), ("class_triage.csv", classes)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    counts: dict[str, int] = {}
    for item in classes:
        counts[str(item["triage"])] = counts.get(str(item["triage"]), 0) + 1
    summary = {
        "purpose": "model-assisted MS-ASL label triage; not an accuracy benchmark",
        "clips": len(rows),
        "classes": len(classes),
        "clip_top1": float(np.mean([bool(row["top1_hit"]) for row in rows])),
        "clip_top5": float(np.mean([bool(row["top5_hit"]) for row in rows])),
        "triage_counts": counts,
        "training_approved_classes": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# MS-ASL/Citizen100 gap-candidate triage",
        "",
        "**Status:** model-assisted audit only; no class is automatically approved.",
        "",
        f"- Clips/classes: {len(rows)}/{len(classes)}",
        f"- Clip top-1/top-5 agreement: {100 * summary['clip_top1']:.2f}% / {100 * summary['clip_top5']:.2f}%",
        f"- Triage: `{json.dumps(counts, sort_keys=True)}`",
        "",
        "| Class | Clips/signers | Top-1 | Top-5 | Triage |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in classes:
        lines.append(
            f"| {item['canonical_label']} | {item['clips']}/{item['signers']} | "
            f"{item['top1_hits']}/{item['clips']} | {item['top5_hits']}/{item['clips']} | "
            f"{item['triage']} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
