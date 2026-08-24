#!/usr/bin/env python3
"""Package the phrase-agnostic multi-voice/direct-transition Stage-2 selector."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import (
    Stage2GeneralCTCSelectorV17,
    Stage2TemporalHeadV17,
    Stage2V17Config,
    load_stage2_context_adapted,
    load_stage2_general_ctc_selector,
)
from active.v17.train_stage_2_v17 import (
    CombinedDataset,
    RealPhraseDataset,
    collate,
    evaluate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_specialist(path: Path) -> tuple[Stage2TemporalHeadV17, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "slt_stage2_ctc_v17":
        raise ValueError(f"{path}: expected a plain v17 CTC checkpoint")
    model = Stage2TemporalHeadV17(Stage2V17Config(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def loader(datasets: list[RealPhraseDataset], batch_size: int) -> DataLoader:
    return DataLoader(
        CombinedDataset(datasets), batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )


def domain_edits(metrics: dict[str, object], domain: str) -> int:
    values = metrics["domains"][domain]
    return int(round(float(values["wer"]) * int(values["tokens"])))


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    primary, primary_payload = load_stage2_context_adapted(args.primary)
    specialist, _ = load_specialist(args.specialist)
    model = Stage2GeneralCTCSelectorV17(
        primary,
        specialist,
        blend_weight=args.blend_weight,
        blank_bias=args.blank_bias,
        score_margin=args.score_margin,
        minimum_tokens=args.minimum_tokens,
    ).to(device).eval()
    primary.to(device).eval()

    train_loader = loader([
        RealPhraseDataset(args.phrase_root, "train"),
        RealPhraseDataset(args.context_train_root, "train"),
    ], args.batch_size)
    validation_loader = loader([
        RealPhraseDataset(args.phrase_root, "validation"),
        RealPhraseDataset(args.context_validation_root, "validation"),
    ], args.batch_size)
    primary_train = evaluate(primary, train_loader, device)
    candidate_train = evaluate(model, train_loader, device)
    primary_validation = evaluate(primary, validation_loader, device)
    candidate_validation = evaluate(model, validation_loader, device)

    compared_domains = (
        "asllrp_contiguous", "local_phrases", "asllrp_segmented_validation",
    )
    primary_edits = {
        domain: domain_edits(primary_validation, domain) for domain in compared_domains
    }
    candidate_edits = {
        domain: domain_edits(candidate_validation, domain) for domain in compared_domains
    }
    gates = {
        "all_validation_domains_not_worse": all(
            candidate_edits[domain] <= primary_edits[domain]
            for domain in compared_domains
        ),
        "asllrp_phrases_improved": (
            candidate_edits["asllrp_contiguous"]
            < primary_edits["asllrp_contiguous"]
        ),
        "local_phrases_improved": (
            candidate_edits["local_phrases"] < primary_edits["local_phrases"]
        ),
        "contextual_improved": (
            candidate_edits["asllrp_segmented_validation"]
            < primary_edits["asllrp_segmented_validation"]
        ),
        "phrase_identity_used": False,
    }
    if not all(value for key, value in gates.items() if key != "phrase_identity_used"):
        raise RuntimeError(f"general selector did not clear required gates: {gates}")

    checkpoint = {
        "format": "slt_stage2_general_ctc_selector_v17",
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "primary_context_adapter_config": primary_payload["context_adapter_config"],
        "selector_config": {
            "blend_weight": args.blend_weight,
            "blank_bias": args.blank_bias,
            "score_margin": args.score_margin,
            "minimum_tokens": args.minimum_tokens,
            "same_length_required": True,
            "phrase_identity_used": False,
            "rule": (
                "specialist owns a different same-length multi-token row only when "
                "its own exact CTC path probability is no lower"
            ),
        },
        "primary": args.primary.as_posix(),
        "primary_sha256": sha256(args.primary),
        "specialist": args.specialist.as_posix(),
        "specialist_sha256": sha256(args.specialist),
        "voice_evidence": {
            "primary_usable_style_voices": 63,
            "primary_dataset_local_voice_counts": {
                "citizen": 29, "semlex": 31, "asllrp": 3,
            },
            "claim_boundary": (
                "dataset-local IDs are not asserted to be globally unique people"
            ),
        },
        "validation_metrics": candidate_validation,
        "selection_disclosure": (
            "The 0.10 blend and +0.30 blank bias came from development exploration. "
            "The equal-length, >=2-token, nonnegative specialist CTC likelihood-ratio "
            "rule is phrase-agnostic. Metrics remain development evidence."
        ),
        "accuracy_research_model": True,
        "coreml_export_ready": False,
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)

    reloaded, reloaded_payload = load_stage2_general_ctc_selector(args.output)
    reloaded.to(device).eval()
    reloaded_train = evaluate(reloaded, train_loader, device)
    reloaded_validation = evaluate(reloaded, validation_loader, device)
    if reloaded_train != candidate_train or reloaded_validation != candidate_validation:
        raise RuntimeError("cold-reloaded general selector metrics changed")

    gates["saved_artifact_reload_verified"] = True
    report = {
        "format": "stage2_general_ctc_selector_validation_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": args.output.as_posix(),
        "artifact_sha256": sha256(args.output),
        "artifact_format": reloaded_payload["format"],
        "primary": args.primary.as_posix(),
        "primary_sha256": sha256(args.primary),
        "specialist": args.specialist.as_posix(),
        "specialist_sha256": sha256(args.specialist),
        "selector_config": checkpoint["selector_config"],
        "primary_train": primary_train,
        "candidate_train": candidate_train,
        "primary_validation": primary_validation,
        "candidate_validation": candidate_validation,
        "primary_validation_edits": primary_edits,
        "candidate_validation_edits": candidate_edits,
        "gates": gates,
        "selection_disclosure": checkpoint["selection_disclosure"],
        "limitations": [
            "No independent signer/capture evaluation has been performed.",
            "Recognition WER does not establish perceptually natural generated motion.",
            "The dynamic selector must be distilled before a compact Core ML export.",
        ],
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "consumed_rit_test_accessed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary", type=Path,
        default=Path(
            "artifacts/models/stage2_v17_multivoice_transfer_context_adapted_v3/model.pth"
        ),
    )
    parser.add_argument(
        "--specialist", type=Path,
        default=Path("artifacts/models/stage2_v17_signer_voice_ctc_pilot_v1/best_model.pth"),
    )
    parser.add_argument(
        "--phrase-root", type=Path,
        default=Path("data/local/stage2_v17_frozen_features"),
    )
    parser.add_argument(
        "--context-train-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_train_frozen_features"),
    )
    parser.add_argument(
        "--context-validation-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path(
            "artifacts/reports/stage2_v17_general_ctc_selector_v1/validation.json"
        ),
    )
    parser.add_argument("--blend-weight", type=float, default=0.10)
    parser.add_argument("--blank-bias", type=float, default=0.30)
    parser.add_argument("--score-margin", type=float, default=0.0)
    parser.add_argument("--minimum-tokens", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
