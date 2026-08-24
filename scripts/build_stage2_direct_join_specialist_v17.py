#!/usr/bin/env python3
"""Build and validate a gated direct-isolated-join Stage-2 specialist."""

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
    load_stage2_context_adapted,
    load_stage2_direct_join_specialist,
    Stage2DirectJoinSpecialistV17,
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


def labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    return [label for values in payload["categories"].values() for label in values]


def loader(root: Path, batch_size: int):
    dataset = RealPhraseDataset(root, "validation")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def domain_edits(metrics: dict[str, object], domain: str) -> int:
    row = metrics["domains"][domain]
    return int(round(float(row["wer"]) * int(row["tokens"])))


def run(args: argparse.Namespace) -> dict[str, object]:
    primary, primary_payload = load_stage2_context_adapted(args.primary)
    specialist_payload = torch.load(args.specialist, map_location="cpu", weights_only=False)
    if specialist_payload.get("format") != "slt_stage2_ctc_v17":
        raise ValueError("specialist is not a v17 CTC checkpoint")
    specialist = Stage2TemporalHeadV17(
        Stage2V17Config(**specialist_payload["model_config"])
    )
    specialist.load_state_dict(specialist_payload["model_state_dict"], strict=True)
    locked_labels = labels(args.labels)
    gate_class_indices = tuple(locked_labels.index(label) for label in args.gate_labels)
    gate_ctc_tokens = tuple(index + 1 for index in gate_class_indices)
    model = Stage2DirectJoinSpecialistV17(
        primary,
        specialist,
        blend_weight=args.blend_weight,
        gate_ctc_tokens=gate_ctc_tokens,
    )
    device = torch.device(args.device)
    primary.to(device).eval()
    model.to(device).eval()
    phrase_loader = loader(args.phrase_validation_root, args.batch_size)
    context_loader = loader(args.context_validation_root, args.batch_size)
    primary_phrase = evaluate(primary, phrase_loader, device)
    primary_context = evaluate(primary, context_loader, device)
    candidate_phrase = evaluate(model, phrase_loader, device)
    candidate_context = evaluate(model, context_loader, device)

    primary_asllrp = domain_edits(primary_phrase, "asllrp_contiguous")
    candidate_asllrp = domain_edits(candidate_phrase, "asllrp_contiguous")
    primary_local = domain_edits(primary_phrase, "local_phrases")
    candidate_local = domain_edits(candidate_phrase, "local_phrases")
    primary_context_edits = domain_edits(primary_context, "asllrp_segmented_validation")
    candidate_context_edits = domain_edits(candidate_context, "asllrp_segmented_validation")
    gates = {
        "asllrp_phrase_edits_improved": candidate_asllrp < primary_asllrp,
        "contextual_edits_improved": candidate_context_edits < primary_context_edits,
        "local_phrase_edits_not_worse": candidate_local <= primary_local,
    }
    if not all(gates.values()):
        raise RuntimeError(f"direct-join specialist did not clear gates: {gates}")

    combined = {
        "format": "slt_stage2_direct_join_specialist_ctc_v17",
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "primary_context_adapter_config": primary_payload["context_adapter_config"],
        "specialist_config": {
            "blend_weight": args.blend_weight,
            "gate_labels": list(args.gate_labels),
            "gate_class_indices": list(gate_class_indices),
            "gate_ctc_tokens": list(gate_ctc_tokens),
            "gate_rule": "specialist greedy CTC collapse exactly equals gate_labels",
        },
        "primary": args.primary.as_posix(),
        "primary_sha256": sha256(args.primary),
        "specialist": args.specialist.as_posix(),
        "specialist_sha256": sha256(args.specialist),
        "validation_metrics": {
            "phrases": candidate_phrase,
            "contextual": candidate_context,
        },
        "selection_disclosure": (
            "FRIEND NOW gate and the smallest tested blend weight improving contextual edits "
            "(0.03 from 0.00..0.30 in 0.01 increments) were selected on development validation; "
            "this is not independent-test evidence"
        ),
        "test_evaluated": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(combined, args.output)
    reloaded, _ = load_stage2_direct_join_specialist(args.output)
    reloaded.to(device).eval()
    reload_phrase = evaluate(reloaded, phrase_loader, device)
    reload_context = evaluate(reloaded, context_loader, device)
    reload_verified = reload_phrase == candidate_phrase and reload_context == candidate_context
    if not reload_verified:
        raise RuntimeError("cold-reloaded direct-join specialist metrics changed")
    gates["saved_artifact_reload_verified"] = True
    gates["all_required_validation_gates_pass"] = all(gates.values())
    report = {
        "format": "slt_stage2_direct_join_specialist_validation_v17",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": args.output.as_posix(),
        "artifact_sha256": sha256(args.output),
        "primary": args.primary.as_posix(),
        "primary_sha256": sha256(args.primary),
        "specialist": args.specialist.as_posix(),
        "specialist_sha256": sha256(args.specialist),
        "blend_weight": args.blend_weight,
        "gate_labels": list(args.gate_labels),
        "primary_phrase_validation": primary_phrase,
        "candidate_phrase_validation": candidate_phrase,
        "primary_context_validation": primary_context,
        "candidate_context_validation": candidate_context,
        "edit_counts": {
            "primary_asllrp_phrases": primary_asllrp,
            "candidate_asllrp_phrases": candidate_asllrp,
            "primary_local_phrases": primary_local,
            "candidate_local_phrases": candidate_local,
            "primary_contextual": primary_context_edits,
            "candidate_contextual": candidate_context_edits,
        },
        "gates": gates,
        "selection_disclosure": combined["selection_disclosure"],
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
    parser.add_argument("--primary", type=Path, default=Path("artifacts/models/stage2_v17_multivoice_transfer_context_adapted_v3/model.pth"))
    parser.add_argument("--specialist", type=Path, default=Path("artifacts/models/stage2_v17_signer_voice_ctc_pilot_v1/best_model.pth"))
    parser.add_argument("--labels", type=Path, default=Path("active/v17/citizen100_seed.json"))
    parser.add_argument("--gate-labels", nargs="+", default=("FRIEND", "NOW"))
    parser.add_argument("--blend-weight", type=float, default=0.03)
    parser.add_argument("--phrase-validation-root", type=Path, default=Path("data/local/stage2_v17_frozen_features"))
    parser.add_argument("--context-validation-root", type=Path, default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/stage2_v17_direct_join_specialist_v1/model.pth"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_direct_join_specialist_v1/validation.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
