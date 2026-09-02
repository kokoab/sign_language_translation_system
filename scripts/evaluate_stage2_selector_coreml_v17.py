#!/usr/bin/env python3
"""Verify the two exported Core ML heads reproduce the v17 general CTC selector."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import coremltools as ct
import numpy as np
import torch

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from active.v17.export_stage1_coreml_v17 import tree_sha256
from active.v17.export_stage2_coreml_v17 import fixed_input
from active.v17.model_stage2_v17 import (
    ctc_sequence_log_probability,
    greedy_ctc_tokens,
    load_stage2_general_ctc_selector,
)
from active.v17.train_stage_2_v17 import RealPhraseDataset, edit_distance


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select(primary: np.ndarray, specialist: np.ndarray, length: int):
    primary = torch.from_numpy(primary.reshape(-1, 101)[:length].astype(np.float32))
    specialist = torch.from_numpy(specialist.reshape(-1, 101)[:length].astype(np.float32))
    calibrated = primary * 0.9 + specialist * 0.1
    calibrated[:, 0] += 0.3
    base = greedy_ctc_tokens(calibrated, length)
    alternate = greedy_ctc_tokens(specialist, length)
    selected = False
    if base != alternate and len(base) >= 2 and len(base) == len(alternate):
        selected = bool(
            ctc_sequence_log_probability(specialist, alternate)
            >= ctc_sequence_log_probability(specialist, base)
        )
    return (specialist if selected else calibrated), selected


def run(args):
    guarded = (args.selector, args.primary_package, args.specialist_package, *args.roots)
    if any("test" in {part.lower() for part in path.parts} for path in guarded):
        raise ValueError("selector parity is restricted to train/validation data")
    selector, _ = load_stage2_general_ctc_selector(args.selector)
    selector.eval()
    primary = ct.models.MLModel(str(args.primary_package), compute_units=ct.ComputeUnit.ALL)
    specialist = ct.models.MLModel(str(args.specialist_package), compute_units=ct.ComputeUnit.ALL)
    primary_output = primary.get_spec().description.output[0].name
    specialist_output = specialist.get_spec().description.output[0].name
    datasets = [RealPhraseDataset(root, "validation") for root in args.roots]
    totals = defaultdict(lambda: {"edits": 0, "tokens": 0, "exact": 0, "samples": 0})
    decode_mismatches = selection_mismatches = 0
    with torch.inference_mode():
        for dataset in datasets:
            for row in dataset.samples:
                arrays = fixed_input(row.features.astype(np.float32), 8)
                provider = {"frozen_features": arrays[0], "window_mask": arrays[1]}
                primary_logits = np.asarray(primary.predict(provider)[primary_output])
                specialist_logits = np.asarray(specialist.predict(provider)[specialist_output])
                length = len(row.features) * 8
                coreml_logits, coreml_selected = select(
                    primary_logits, specialist_logits, length
                )
                reference_logits, _, reference_selected = selector.forward_with_selection(
                    torch.from_numpy(arrays[0]), torch.from_numpy(arrays[1]) > 0.5
                )
                coreml_tokens = tuple(value - 1 for value in greedy_ctc_tokens(coreml_logits, length))
                reference_tokens = tuple(
                    value - 1 for value in greedy_ctc_tokens(reference_logits[0], length)
                )
                decode_mismatches += int(coreml_tokens != reference_tokens)
                selection_mismatches += int(coreml_selected != bool(reference_selected[0]))
                expected = tuple(int(value) - 1 for value in row.targets.tolist())
                stats = totals[row.source]
                stats["edits"] += edit_distance(list(expected), list(coreml_tokens))
                stats["tokens"] += len(expected)
                stats["exact"] += int(expected == coreml_tokens)
                stats["samples"] += 1
    domains = {
        name: values | {
            "wer": values["edits"] / values["tokens"],
            "sequence_accuracy": values["exact"] / values["samples"],
        }
        for name, values in sorted(totals.items())
    }
    report = {
        "format": "slt_stage2_general_selector_coreml_parity_v17",
        "version": 1,
        "selector": args.selector.as_posix(),
        "selector_sha256": sha256(args.selector),
        "primary_package_tree_sha256": tree_sha256(args.primary_package),
        "specialist_package_tree_sha256": tree_sha256(args.specialist_package),
        "samples": sum(value["samples"] for value in totals.values()),
        "coreml_vs_pytorch_decode_mismatches": decode_mismatches,
        "coreml_vs_pytorch_selection_mismatches": selection_mismatches,
        "metrics": domains,
        "execution_environment": "mac_host_coreml",
        "hardware_performance_claim": False,
        "thermals_interpretable": False,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "test_evaluated": False,
    }
    if decode_mismatches or selection_mismatches:
        raise RuntimeError(f"Core ML selector parity failed: {report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--selector", type=Path, default=Path("artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth"))
    value.add_argument("--primary-package", type=Path, default=Path("artifacts/coreml/Stage2SelectorPrimaryV17FP32.mlpackage"))
    value.add_argument("--specialist-package", type=Path, default=Path("artifacts/coreml/Stage2SelectorSpecialistV17FP32.mlpackage"))
    value.add_argument("--roots", nargs="+", type=Path, default=[Path("data/local/stage2_v17_frozen_features"), Path("data/local/stage2_v17_asllrp_segmented_validation_frozen_features")])
    value.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_general_ctc_selector_v1/coreml_pipeline_validation.json"))
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2))
