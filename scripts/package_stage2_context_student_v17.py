#!/usr/bin/env python3
"""Package an already-trained compact Stage-2 head with its evaluated context residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from active.v17.model_stage2_v17 import load_stage2_general_ctc_selector
from active.v17.train_stage_2_general_selector_distill_v17 import (
    package_context_adapted_student,
    sha256,
    validation_edits,
)
from active.v17.train_stage_2_v17 import RealPhraseDataset, collate


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
    )
    teacher, _ = load_stage2_general_ctc_selector(args.teacher)
    teacher.to(device).eval()
    phrase_loader = DataLoader(
        RealPhraseDataset(args.phrase_root, "validation"),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    )
    context_loader = DataLoader(
        RealPhraseDataset(args.context_root, "validation"),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate,
    )
    _, phrase, context = package_context_adapted_student(
        args.student, teacher, output=args.output,
        phrase_loader=phrase_loader, context_loader=context_loader, device=device,
        extra={
            "packaged_from_existing_student": True,
            "distillation_teacher": args.teacher.as_posix(),
            "distillation_teacher_sha256": sha256(args.teacher),
        },
    )
    report = {
        "format": "slt_stage2_compact_context_student_package_v17",
        "version": 1,
        "artifact": args.output.as_posix(),
        "artifact_sha256": sha256(args.output),
        "student": args.student.as_posix(),
        "student_sha256": sha256(args.student),
        "teacher": args.teacher.as_posix(),
        "teacher_sha256": sha256(args.teacher),
        "validation_edits": list(validation_edits(phrase, context)),
        "phrase_validation": phrase,
        "context_validation": context,
        "cold_reload_verified": True,
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
        "--student", type=Path,
        default=Path("artifacts/models/stage2_v17_general_selector_distill_head_v1/best_model.pth"),
    )
    parser.add_argument(
        "--teacher", type=Path,
        default=Path("artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"),
    )
    parser.add_argument(
        "--phrase-root", type=Path,
        default=Path("data/local/stage2_v17_frozen_features"),
    )
    parser.add_argument(
        "--context-root", type=Path,
        default=Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/models/stage2_v17_compact_context_student_v1/model.pth"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_compact_context_student_v1/package.json"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
