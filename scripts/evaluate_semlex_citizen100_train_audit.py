#!/usr/bin/env python3
"""Use the frozen Citizen100 model only as a SemLex mismatch diagnostic."""

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
        signers = len({str(item["semlex_signer_id"]) for item in items})
        top1_hits = sum(bool(item["top1_hit"]) for item in items)
        top5_hits = sum(bool(item["top5_hit"]) for item in items)
        top1_rate = top1_hits / clips
        top5_rate = top5_hits / clips
        if signers >= 2 and top1_rate >= 0.5 and top5_rate >= 0.8:
            triage = "model_consistent"
        elif top5_rate >= 0.5:
            triage = "cross_domain_ambiguous"
        else:
            triage = "mismatch_review_priority"
        output.append(
            {
                "canonical_label": label,
                "citizen_raw_gloss": items[0]["citizen_raw_gloss"],
                "citizen_asl_lex_code": items[0]["citizen_asl_lex_code"],
                "asllex_entry_id": items[0]["asllex_entry_id"],
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
        default=Path("data/local/semlex_citizen100_train_audit/landmarks_v17"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/local/semlex_citizen100_train_audit/download_provenance.json"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("active/v17/citizen100_manifest.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/semlex_citizen100_train_audit/model_triage"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--expected-split", choices=("train", "val", "test"), default="train"
    )
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if provenance.get("training_eligible") is not False:
        raise ValueError("SemLex provenance must explicitly prohibit unreviewed training use")
    retained = list(provenance["videos"])
    if provenance.get("split", "train") != args.expected_split:
        raise ValueError("SemLex provenance split does not match --expected-split")
    if any(row.get("semlex_split") != args.expected_split for row in retained):
        raise ValueError("SemLex provenance contains mixed or unexpected splits")
    if args.expected_split != "train" and provenance.get("split_eligibility") != (
        "evaluation_only_never_training"
    ):
        raise ValueError("SemLex evaluation split is not locked evaluation-only")
    metadata_by_stem = {
        (str(row["canonical_label"]), str(row["semlex_video_id"])): row for row in retained
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
            raise ValueError(f"feature absent from SemLex provenance: {path}")
        features.append(load_features(path, expected_schema))
        records.append(metadata_by_stem[key])
    if len(files) != len(metadata_by_stem):
        raise ValueError(f"feature/provenance count mismatch: {len(files)} != {len(metadata_by_stem)}")

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
                "raw_path": record["raw_path"],
                "semlex_video_id": record["semlex_video_id"],
                "semlex_signer_id": record["semlex_signer_id"],
                "semlex_split": record["semlex_split"],
                "true_label": record["canonical_label"],
                "citizen_raw_gloss": record["citizen_raw_gloss"],
                "citizen_asl_lex_code": record["citizen_asl_lex_code"],
                "asllex_entry_id": record["asllex_entry_id"],
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
    np.savez_compressed(
        args.output_dir / "logits.npz",
        logits=logits.numpy().astype(np.float32),
        targets=np.asarray(
            [labels[str(record["canonical_label"])] for record in records], dtype=np.int64
        ),
        item_ids=np.asarray([str(path) for path in files]),
    )
    for filename, output_rows in (("predictions.csv", rows), ("class_triage.csv", classes)):
        with (args.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    counts: dict[str, int] = {}
    for item in classes:
        counts[str(item["triage"])] = counts.get(str(item["triage"]), 0) + 1
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    targets = np.asarray([labels[str(row["true_label"])] for row in rows])
    predictions = np.asarray([labels[str(row["predicted_label"])] for row in rows])
    np.add.at(confusion, (targets, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / np.maximum(confusion.sum(axis=0), 1)
    recall = true_positive / np.maximum(confusion.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    present = confusion.sum(axis=1) > 0
    summary = {
        "purpose": (
            "secondary cross-domain evaluation"
            if args.expected_split != "train"
            else "cross-domain mismatch diagnostic; not SemLex accuracy"
        ),
        "split": args.expected_split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "clips": len(rows),
        "classes": len(classes),
        "clip_top1": float(np.mean([bool(row["top1_hit"]) for row in rows])),
        "clip_top5": float(np.mean([bool(row["top5_hit"]) for row in rows])),
        "macro_f1_present_classes": float(f1[present].mean()),
        "triage_counts": counts,
        "training_approved_classes": 0,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        f"# SemLex {args.expected_split} secondary-domain evaluation",
        "",
        "**Status:** evaluation only; this split is never training data.",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Clips/classes: {len(rows)}/{len(classes)}",
        f"- Top-1: {100 * summary['clip_top1']:.2f}%",
        f"- Top-5: {100 * summary['clip_top5']:.2f}%",
        f"- Macro F1 over present classes: "
        f"{100 * summary['macro_f1_present_classes']:.2f}%",
        "",
        "SemLex validation mostly reuses SemLex train signer identities, so this is a",
        "cross-domain clip diagnostic rather than an unseen-signer production test.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
