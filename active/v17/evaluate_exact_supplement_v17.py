#!/usr/bin/env python3
"""Evaluate a checkpoint on exact-variant external v17 landmarks it never trained on."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_stage_1_v17 import LocalReviewSupplementV17Dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--feature-root", type=Path,
        default=Path("data/local/asllvd_asllex_v17/landmarks"),
    )
    parser.add_argument(
        "--supplement-manifest", type=Path,
        default=Path("data/local/asllvd_asllex_v17/exact_variant_manifest.json"),
    )
    parser.add_argument(
        "--class-manifest", type=Path,
        default=Path("active/v17/citizen100_manifest.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/reports/asllvd_asllex_v17_external_baseline.json"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    supplement = json.loads(args.supplement_manifest.read_text(encoding="utf-8"))
    if supplement.get("format") != "slt_v17_asllvd_asllex_exact_supplement":
        raise ValueError("supplement is not the exact ASLLVD format")
    if supplement.get("citizen_test_accessed") is not False:
        raise ValueError("Citizen test isolation is not proven")
    if supplement.get("semlex_test_accessed") is not False:
        raise ValueError("SemLex test isolation is not proven")
    if supplement.get("training_eligible_clips") != 175:
        raise ValueError("exact supplement is not fully finalized")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "slt_stage1_v17":
        raise ValueError("not a v17 Stage-1 checkpoint")
    provenance = checkpoint.get("training_data_provenance", {})
    supplement_sha = sha256_file(args.supplement_manifest)
    if (
        provenance.get("local_manifest_sha256") == supplement_sha
        or provenance.get("local_manifest_format")
        == "slt_v17_asllvd_asllex_exact_supplement"
        or provenance.get("local_train_samples")
    ):
        raise ValueError("checkpoint has local-supplement training provenance")

    class_payload = json.loads(args.class_manifest.read_text(encoding="utf-8"))
    label_to_index = {
        str(row["canonical_label"]): int(row["class_index"])
        for row in class_payload["classes"]
    }
    index_to_label = {value: key for key, value in label_to_index.items()}
    dataset = LocalReviewSupplementV17Dataset(
        args.feature_root,
        args.supplement_manifest,
        label_to_index,
        allowed_tiers=("official_asllex_signbank_exact",),
        cache=True,
        expected_schema=checkpoint["schema_fingerprint"],
    )
    model = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logits_parts: list[torch.Tensor] = []
    targets_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for features, targets in DataLoader(dataset, batch_size=args.batch_size):
            logits_parts.append(model(features).float())
            targets_parts.append(targets)
    logits = torch.cat(logits_parts)
    targets = torch.cat(targets_parts)
    predictions = logits.argmax(dim=1)
    top5 = logits.topk(5, dim=1).indices
    present = sorted(set(targets.tolist()))
    classes = []
    for target in present:
        mask = targets == target
        classes.append(
            {
                "canonical_label": index_to_label[target],
                "clips": int(mask.sum()),
                "top1_correct": int((predictions[mask] == target).sum()),
                "top1": float((predictions[mask] == target).float().mean()),
            }
        )
    top1 = float((predictions == targets).float().mean())
    top5_rate = float((top5 == targets[:, None]).any(dim=1).float().mean())
    payload = {
        "format": "slt_v17_exact_external_baseline_evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "supplement_manifest": str(args.supplement_manifest),
        "supplement_manifest_sha256": supplement_sha,
        "checkpoint_never_trained_on_this_supplement": True,
        "clips": len(dataset),
        "classes": len(present),
        "signers": list(dataset.known_signers),
        "top1_correct": int((predictions == targets).sum()),
        "top1": top1,
        "top5": top5_rate,
        "macro_class_top1": float(np.mean([row["top1"] for row in classes])),
        "class_results": classes,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "raw_video_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "clips", "classes", "top1_correct", "top1", "top5", "macro_class_top1"
    )}, indent=2))


if __name__ == "__main__":
    main()
