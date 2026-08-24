#!/usr/bin/env python3
"""Use the frozen Citizen100 model to triage PopSign preview compatibility.

This is not an accuracy benchmark and cannot prove lexical equivalence. PopSign
website previews are speed-normalized and downsampled. Results only prioritize
which exact Citizen/PopSign pairs require human ASL-variant review before any
original PopSign archive is admitted to training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_features(path: Path, expected_schema: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32, copy=False)
        metadata = json.loads(str(payload["metadata_json"]))
    if features.shape != (32, 61, 5):
        raise ValueError(f"{path}: unexpected shape {features.shape}")
    if metadata.get("schema_fingerprint") != expected_schema:
        raise ValueError(f"{path}: v17 schema mismatch")
    if not np.isfinite(features).all():
        raise ValueError(f"{path}: non-finite features")
    return features


def aggregate_classes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["true_label"]), []).append(row)
    output: list[dict[str, object]] = []
    for label in sorted(grouped):
        items = grouped[label]
        top1_hits = sum(bool(item["top1_hit"]) for item in items)
        top5_hits = sum(bool(item["top5_hit"]) for item in items)
        if top1_hits >= 2 and top5_hits == len(items):
            triage = "model_consistent_manual_review_required"
        elif top5_hits >= 2:
            triage = "ambiguous_manual_review_required"
        else:
            triage = "high_risk_manual_review_required"
        output.append(
            {
                "canonical_label": label,
                "citizen_raw_gloss": items[0]["citizen_raw_gloss"],
                "citizen_asl_lex_code": items[0]["citizen_asl_lex_code"],
                "popsign_gloss": items[0]["popsign_gloss"],
                "clips": len(items),
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
        "checkpoint", type=Path, default=Path("artifacts/models/stage1_v17_baseline/best_model.pth"), nargs="?"
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data/local/popsign_citizen100_variant_audit/landmarks"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/local/popsign_citizen100_variant_audit/preview_provenance.json"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/popsign_citizen100_variant_audit"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance.get("training_eligible") is not False:
        raise ValueError("preview provenance must explicitly prohibit training use")
    metadata_by_stem = {
        (str(row["canonical_label"]), Path(str(row["destination"])).stem): row
        for row in provenance["videos"]
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
    if not files:
        raise ValueError(f"no features under {args.feature_root}")
    features: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for path in files:
        key = (path.parent.name, path.name.removesuffix(".v17.npz"))
        if key not in metadata_by_stem:
            raise ValueError(f"feature absent from preview provenance: {path}")
        features.append(load_features(path, expected_schema))
        records.append(metadata_by_stem[key])

    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), args.batch_size):
            batch = torch.from_numpy(np.stack(features[start : start + args.batch_size]))
            logits_parts.append(model(batch).float())
    logits = torch.cat(logits_parts)
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
                "participant": record["participant"],
                "true_label": record["canonical_label"],
                "citizen_raw_gloss": record["citizen_raw_gloss"],
                "citizen_asl_lex_code": record["citizen_asl_lex_code"],
                "popsign_gloss": record["popsign_gloss"],
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
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "class_triage.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(classes[0]))
        writer.writeheader()
        writer.writerows(classes)

    counts: dict[str, int] = {}
    for item in classes:
        counts[str(item["triage"])] = counts.get(str(item["triage"]), 0) + 1
    summary = {
        "purpose": "model-assisted lexical-variant triage; not an accuracy benchmark",
        "checkpoint": str(args.checkpoint),
        "preview_clips": len(rows),
        "classes": len(classes),
        "clip_top1": float(np.mean([bool(row["top1_hit"]) for row in rows])),
        "clip_top5": float(np.mean([bool(row["top5_hit"]) for row in rows])),
        "triage_counts": counts,
        "training_approved_classes": 0,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# PopSign/Citizen100 exact-variant triage",
        "",
        "**Status:** model-assisted preview audit only; no PopSign class is approved for training.",
        "",
        "PopSign previews are downsampled and speed-normalized. Frozen-model agreement can",
        "prioritize review but cannot establish that a PopSign gloss is the same lexical",
        "variant as the pinned Citizen raw gloss and ASL-LEX code.",
        "",
        f"- Preview clips: {len(rows)}",
        f"- Overlap classes: {len(classes)}",
        f"- Clip top-1 label agreement: {100 * summary['clip_top1']:.2f}%",
        f"- Clip top-5 label agreement: {100 * summary['clip_top5']:.2f}%",
        f"- Triage counts: `{json.dumps(counts, sort_keys=True)}`",
        "",
        "| Canonical | Citizen raw | ASL-LEX | PopSign | Top-1 | Top-5 | Triage |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in classes:
        lines.append(
            f"| {item['canonical_label']} | {item['citizen_raw_gloss']} | "
            f"{item['citizen_asl_lex_code']} | {item['popsign_gloss']} | "
            f"{item['top1_hits']}/{item['clips']} | {item['top5_hits']}/{item['clips']} | "
            f"{item['triage']} |"
        )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
