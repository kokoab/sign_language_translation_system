#!/usr/bin/env python3
"""Evaluate the current Stage 1 checkpoint on the external audit subset.

This is a domain-shift diagnostic, not a valid replacement for a locked
100-class evaluation. The current checkpoint has 310 labels and was trained on
the legacy coordinate schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("src_v16/output_v16_d384/best_model.pth"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("data/local/ios100_audit/asl_citizen/provenance.csv"),
    )
    parser.add_argument(
        "--landmark-dir",
        type=Path,
        default=Path("data/local/ios100_audit/landmarks_legacy"),
    )
    parser.add_argument(
        "--coordinate-schema", default="legacy_anisotropic"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/reports/ios100_stage1_external_audit.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/reports/IOS100_STAGE1_EXTERNAL_AUDIT.md"),
    )
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from active.v16.inference_v16 import load_models, mirror_tta_v16

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model, index_to_label, in_channels, _, _ = load_models(
        str(args.checkpoint), None, device
    )
    with args.provenance.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    results: list[dict[str, object]] = []
    for row in source_rows:
        video_path = Path(row["destination"])
        feature_path = (
            args.landmark_dir
            / row["split"]
            / row["canonical_gloss"]
            / f"{video_path.stem}.npy"
        )
        features = np.load(feature_path).astype(np.float32)
        x = torch.from_numpy(features[..., :in_channels])[None].to(device)
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=-1)
            if not args.no_tta:
                mirrored = mirror_tta_v16(x)
                probabilities = (
                    probabilities + F.softmax(model(mirrored), dim=-1)
                ) / 2
        top5 = probabilities[0].topk(5)
        predictions = [
            (index_to_label[int(index)], float(confidence))
            for confidence, index in zip(top5.values, top5.indices)
        ]
        expected = row["canonical_gloss"]
        top1_correct = predictions[0][0] == expected
        top5_correct = expected in {label for label, _ in predictions}
        results.append(
            {
                "split": row["split"],
                "canonical_gloss": expected,
                "raw_gloss": row["raw_gloss"],
                "participant": row["participant"],
                "video": row["video"],
                "top1": predictions[0][0],
                "top1_confidence": round(predictions[0][1], 8),
                "top1_correct": top1_correct,
                "top5": "|".join(label for label, _ in predictions),
                "top5_correct": top5_correct,
            }
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    count = len(results)
    top1 = sum(bool(row["top1_correct"]) for row in results) / count
    top5 = sum(bool(row["top5_correct"]) for row in results) / count
    by_sign: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_sign[str(row["canonical_gloss"])].append(row)
    predicted = Counter(str(row["top1"]) for row in results)
    by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        by_split[str(row["split"])].append(row)
    lines = [
        "# Current Stage 1 external-signer audit",
        "",
        "**Purpose:** domain-shift diagnostic only  ",
        f"**Checkpoint:** `{args.checkpoint}`  ",
        f"**Coordinate schema:** `{args.coordinate_schema}`  ",
        f"**Videos:** {count}",
        "",
        "The checkpoint was selected using the existing 310-class, random-file "
        "evaluation. These ASL Citizen samples contain public participant IDs not "
        "used to construct that local split. Dataset labels and variants have not "
        "yet received the required ASL review.",
        "",
        "## Results",
        "",
        f"- Top-1 accuracy: **{top1 * 100:.2f}%**",
        f"- Top-5 accuracy: **{top5 * 100:.2f}%**",
        f"- Most frequent top-1 outputs: `{json.dumps(dict(predicted.most_common(10)))}`",
        "",
        "### Citizen official-split identities",
        "",
        "All three groups are external to the local seven-person dataset. The split "
        "names are preserved only for future dataset construction.",
        "",
        "| Citizen split | Videos | Top-1 | Top-5 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        split_rows = by_split[split]
        split_top1 = sum(bool(row["top1_correct"]) for row in split_rows) / len(split_rows)
        split_top5 = sum(bool(row["top5_correct"]) for row in split_rows) / len(split_rows)
        lines.append(
            f"| {split} | {len(split_rows)} | {split_top1 * 100:.1f}% | "
            f"{split_top5 * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "### Per-sign results",
            "",
            "| Sign | Top-1 | Top-5 | Observed top-1 outputs |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for sign, sign_rows in sorted(by_sign.items()):
        sign_top1 = sum(bool(row["top1_correct"]) for row in sign_rows) / len(sign_rows)
        sign_top5 = sum(bool(row["top5_correct"]) for row in sign_rows) / len(sign_rows)
        outputs = Counter(str(row["top1"]) for row in sign_rows)
        rendered_outputs = ", ".join(
            f"{label} ({count})" for label, count in outputs.most_common()
        )
        lines.append(
            f"| {sign} | {sign_top1 * 100:.1f}% | "
            f"{sign_top5 * 100:.1f}% | {rendered_outputs} |"
        )
    lines.extend(
        [
            "",
            "This result must not be compared directly with a future signer-locked "
            "100-class score. It measures the current checkpoint under external "
            "dataset shift and is useful mainly for deciding whether retraining is "
            "necessary.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Top-1: {top1 * 100:.2f}%  Top-5: {top5 * 100:.2f}%")
    print(f"Wrote {args.csv}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
