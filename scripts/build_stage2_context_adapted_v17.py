#!/usr/bin/env python3
"""Build and validate the v17 CTC head with a compact contextual residual."""

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
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    load_stage2_context_adapted,
    Stage2ContextAdapterV17,
    Stage2TemporalHeadV17,
    Stage2V17Config,
)
from active.v17.train_stage_2_v17 import RealPhraseDataset, collate, evaluate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return [label for values in payload["categories"].values() for label in values]


def loader(root: Path, role: str, batch_size: int) -> tuple[RealPhraseDataset, DataLoader]:
    dataset = RealPhraseDataset(root, role)
    return dataset, DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if base_payload.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("base checkpoint is not a v17 Stage-2 CTC head")
    base = Stage2TemporalHeadV17(Stage2V17Config(**base_payload["model_config"]))
    base.load_state_dict(base_payload["model_state_dict"], strict=True)

    with np.load(args.adapter, allow_pickle=False) as payload:
        adapter_values = {name: payload[name].copy() for name in payload.files}
    labels = locked_labels(args.labels)
    if len(labels) != base.config.num_classes or len(set(labels)) != len(labels):
        raise ValueError("locked label file does not define exactly 100 unique classes")
    target_indices = tuple(labels.index(label) for label in args.target_labels)
    adapted = Stage2ContextAdapterV17(
        base,
        feature_mode=str(adapter_values["feature_mode"]),
        scaler_mean=torch.from_numpy(adapter_values["scaler_mean"]),
        scaler_scale=torch.from_numpy(adapter_values["scaler_scale"]),
        coefficients=torch.from_numpy(adapter_values["coefficients"]),
        intercept=torch.from_numpy(adapter_values["intercept"]),
        class_indices=torch.from_numpy(adapter_values["class_indices"]),
        target_class_indices=target_indices,
        weight=args.weight,
    )
    device = torch.device(args.device)
    base.to(device).eval()
    adapted.to(device).eval()

    context_dataset, context_loader = loader(
        args.context_validation_root, "validation", args.batch_size
    )
    phrase_dataset, phrase_loader = loader(
        args.phrase_validation_root, "validation", args.batch_size
    )
    base_context = evaluate(base, context_loader, device)
    adapted_context = evaluate(adapted, context_loader, device)
    base_phrases = evaluate(base, phrase_loader, device)
    adapted_phrases = evaluate(adapted, phrase_loader, device)

    context_domain = adapted_context["domains"]["asllrp_segmented_validation"]
    local_unchanged = (
        base_phrases["domains"]["local_phrases"]
        == adapted_phrases["domains"]["local_phrases"]
    )
    asllrp_phrase_unchanged = (
        base_phrases["domains"]["asllrp_contiguous"]
        == adapted_phrases["domains"]["asllrp_contiguous"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "format": "slt_stage2_context_adapted_ctc_v17",
        "format_version": 1,
        "model_config": base.config.to_dict(),
        "model_state_dict": adapted.state_dict(),
        "context_adapter_config": {
            "feature_mode": adapted.feature_mode,
            "weight": adapted.weight,
            "target_labels": list(args.target_labels),
            "target_class_indices": list(target_indices),
            "normalization": "per-window population z-score over fitted adapter classes",
        },
        "base_checkpoint": args.base_checkpoint.as_posix(),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "adapter": args.adapter.as_posix(),
        "adapter_sha256": sha256(args.adapter),
        "validation_metrics": {
            "contextual": adapted_context,
            "phrases": adapted_phrases,
        },
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    torch.save(combined, args.output)
    reloaded, reloaded_payload = load_stage2_context_adapted(args.output)
    reloaded.to(device).eval()
    reloaded_context = evaluate(reloaded, context_loader, device)
    reloaded_phrases = evaluate(reloaded, phrase_loader, device)
    reload_verified = (
        reloaded_payload["base_checkpoint_sha256"] == sha256(args.base_checkpoint)
        and reloaded_context == adapted_context
        and reloaded_phrases == adapted_phrases
    )

    report = {
        "format": "slt_stage2_context_adapted_validation_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": args.output.as_posix(),
        "artifact_sha256": sha256(args.output),
        "base_checkpoint": args.base_checkpoint.as_posix(),
        "base_checkpoint_sha256": combined["base_checkpoint_sha256"],
        "adapter": args.adapter.as_posix(),
        "adapter_sha256": combined["adapter_sha256"],
        "target_labels": list(args.target_labels),
        "target_class_indices": list(target_indices),
        "residual_weight": args.weight,
        "selection_disclosure": (
            "HOME/WHERE and the smallest tested weight clearing 20% were selected on "
            "development validation after the base error audit; this is not independent-test evidence"
        ),
        "context_validation_samples": len(context_dataset),
        "phrase_validation_samples": len(phrase_dataset),
        "base_context_validation": base_context,
        "adapted_context_validation": adapted_context,
        "base_phrase_validation": base_phrases,
        "adapted_phrase_validation": adapted_phrases,
        "gates": {
            "contextual_wer_below_20_percent": context_domain["wer"] < 0.20,
            "local_phrase_metrics_unchanged": local_unchanged,
            "asllrp_phrase_metrics_unchanged": asllrp_phrase_unchanged,
            "saved_artifact_reload_verified": reload_verified,
            "all_required_validation_gates_pass": (
                context_domain["wer"] < 0.20
                and local_unchanged
                and asllrp_phrase_unchanged
                and reload_verified
            ),
        },
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "test_evaluated": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", type=Path,
        default=Path("artifacts/models/stage2_v17_unified_ctc_v2/best_model.pth"),
    )
    parser.add_argument(
        "--adapter", type=Path,
        default=Path("artifacts/models/stage2_v17_context_adapter_v1/adapter.npz"),
    )
    parser.add_argument(
        "--labels", type=Path, default=Path("active/v17/citizen100_seed.json")
    )
    parser.add_argument(
        "--context-validation-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"),
    )
    parser.add_argument(
        "--phrase-validation-root", type=Path,
        default=Path("data/local/stage2_v17_frozen_features"),
    )
    parser.add_argument("--target-labels", nargs="+", default=("HOME", "WHERE"))
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_context_adapted_ctc_v1/model.pth"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_context_adapted_ctc_v1/validation.json"),
    )
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
