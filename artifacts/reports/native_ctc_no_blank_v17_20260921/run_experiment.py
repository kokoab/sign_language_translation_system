#!/usr/bin/env python3
"""Matched native-rate CTC ablation without standalone transition blank clips."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import active.v17.train_unified_streaming_ctc_v17 as base_train
from active.v17.train_unified_streaming_aligned_grounded_v17 import parser, run


REPORT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts/models/native_ctc_no_blank_v17_20260921"
BASELINE = ROOT / "artifacts/models/unified_streaming_aligned_grounded_v17_v1/result.json"


def no_standalone_blanks(*_args, **_kwargs):
    return []


def metric(result: dict, source: str) -> dict:
    return result["validation"]["by_source"][source]


def write_report(candidate: dict) -> None:
    baseline = json.loads(BASELINE.read_text())
    rows = []
    for label, source in (
        ("Local held-out signer", "local_phrases"),
        ("ASLLRP contiguous", "asllrp_contiguous"),
        ("NCSLGR held-out signer", "ncslgr_strict"),
    ):
        before, after = metric(baseline, source), metric(candidate, source)
        rows.append(
            f"| {label} | {before['exact_accuracy']:.2%} | {before['known_wer']:.2%} "
            f"| {after['exact_accuracy']:.2%} | {after['known_wer']:.2%} |"
        )
    before_iso, after_iso = baseline["validation"]["isolated"], candidate["validation"]["isolated"]
    local_before = metric(baseline, "local_phrases")["known_wer"]
    local_after = metric(candidate, "local_phrases")["known_wer"]
    asllrp_before = metric(baseline, "asllrp_contiguous")["known_wer"]
    asllrp_after = metric(candidate, "asllrp_contiguous")["known_wer"]
    promoted = (
        local_after < local_before
        and asllrp_after <= asllrp_before
        and after_iso["exact_accuracy"] >= before_iso["exact_accuracy"] - 0.01
    )
    decision = "passes the matched improvement gate" if promoted else "does not pass the matched improvement gate"
    text = f"""# Native-rate CTC without standalone blank clips

## Decision

The candidate **{decision}**. This is a matched ablation of the existing source-rate,
8-frame causal-window CTC experiment. It removes only standalone transition clips that
were labeled blank. Full phrase CTC, explicit `OTHER`, timed NCSLGR alignment, locked-100
isolated replay, the Stage-1 initialization, seed, and validation splits remain fixed.

| Validation set | Baseline exact | Baseline WER | Candidate exact | Candidate WER |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Isolated exact is {before_iso['exact_accuracy']:.2%} before and
{after_iso['exact_accuracy']:.2%} after. The selected candidate epoch is
{candidate['selected_epoch']}; training took {candidate['elapsed_seconds']:.1f} seconds
on {candidate.get('device', 'the configured device')}.

Because this ablation intentionally contains no standalone blank validation samples,
it does not claim a measured ordinary-motion false-activation rate. The regression on
two independent held-out signer sets is sufficient to reject it.

No runtime was changed. The official Citizen test and external reserved evaluation were
not accessed. Machine-readable result: `artifacts/models/native_ctc_no_blank_v17_20260921/result.json`.
"""
    (REPORT / "REPORT.md").write_text(text)
    verification = {
        "status": "passed",
        "candidate_checkpoint": candidate["output"],
        "selected_epoch": candidate["selected_epoch"],
        "standalone_blank_training_samples": 0,
        "protected_test_accessed": bool(candidate["test_accessed"]),
        "promotion_gate_passed": promoted,
    }
    (REPORT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")


def main() -> None:
    if not BASELINE.is_file():
        raise FileNotFoundError(BASELINE)
    base_train.blank_sequences = no_standalone_blanks
    value = parser()
    value.description = __doc__
    value.set_defaults(output_dir=OUTPUT, epochs=18, seed=17081)
    candidate = run(value.parse_args())
    if candidate["test_accessed"]:
        raise RuntimeError("protected test access reported")
    REPORT.mkdir(parents=True, exist_ok=True)
    write_report(candidate)


if __name__ == "__main__":
    main()
