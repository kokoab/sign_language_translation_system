#!/usr/bin/env python3
"""Finalize one validation-selected Stage 3 tiny checkpoint.

This is intentionally separate from training so an interrupted run can evaluate its
already-saved best checkpoint without repeating optimization. The fixed synthetic test
split must be opened only once per checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from active.v17.train_stage3_tiny_v17 import (
    BASE_MODEL,
    BASE_REVISION,
    CONTRACT,
    DIALOGUE_CSV,
    LEGACY_CSV,
    NATURALIZER,
    OUTPUT,
    REPORT,
    evaluate,
    load_pairs,
    sha256_file,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--seed", type=int, default=17032)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-input-length", type=int, default=48)
    parser.add_argument("--max-target-length", type=int, default=48)
    parser.add_argument("--selected-epoch", type=int, default=1)
    args = parser.parse_args()

    result_path = args.report_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to access the test split again: {result_path}")
    rows, data_plan = load_pairs(args.seed)
    validation = [row for row in rows if row.split == "validation"]
    test = [row for row in rows if row.split == "test"]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, local_files_only=True)
    model.eval()
    validation_metrics, validation_details = evaluate(
        model, tokenizer, validation, torch.device("cpu"), args.batch_size,
        args.max_input_length, args.max_target_length,
    )

    # This is the checkpoint's first and only fixed-test access.
    test_metrics, test_details = evaluate(
        model, tokenizer, test, torch.device("cpu"), args.batch_size,
        args.max_input_length, args.max_target_length,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.report_dir / "validation_predictions.jsonl", validation_details)
    write_jsonl(args.report_dir / "test_predictions.jsonl", test_details)

    validation_gate = (
        validation_metrics["locked100"]["normalized_exact_rate"] >= 0.90
        and validation_metrics["five_or_more"]["normalized_exact_rate"] >= 0.90
        and validation_metrics["controlled_five_or_more"]["normalized_exact_rate"] >= 0.95
    )
    test_gate = (
        test_metrics["locked100"]["normalized_exact_rate"] >= 0.85
        and test_metrics["five_or_more"]["normalized_exact_rate"] >= 0.85
        and test_metrics["controlled_five_or_more"]["normalized_exact_rate"] >= 0.90
    )
    result = {
        "format": "slt_stage3_t5_efficient_tiny_locked100_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "selected_model_sha256": sha256_file(args.checkpoint / "model.safetensors"),
        "selected_tokenizer_sha256": sha256_file(args.checkpoint / "tokenizer.json"),
        "source_sha256": {
            "legacy_csv": sha256_file(LEGACY_CSV),
            "dialogue_csv": sha256_file(DIALOGUE_CSV),
            "contract": sha256_file(CONTRACT),
            "naturalizer": sha256_file(NATURALIZER),
        },
        "configuration": {**vars(args), "checkpoint": str(args.checkpoint), "report_dir": str(args.report_dir)},
        "data_plan": {
            **data_plan,
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "selected_epoch": args.selected_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "promotion_gate": {
            "validation_passed": validation_gate,
            "test_passed": test_gate,
            "passed": validation_gate and test_gate,
        },
        "training_note": "Epoch 1 checkpoint selected; later epoch was interrupted after validation had saturated.",
        "selection_used_test": False,
        "test_split_accessed_once_after_selection": True,
        "citizen_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "claim_scope": "synthetic/reviewed locked-vocabulary naturalization; not general ASL translation",
    }
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"promotion_gate": result["promotion_gate"], "validation": validation_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
