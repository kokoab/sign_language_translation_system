#!/usr/bin/env python3
"""Run the separate learned-emission Reel Stage-1 experiment.

This keeps the accepted Reel entry point and its model unchanged.  A 101st Core ML
output suppresses incomplete prefixes/transitions; the original 100-gloss logits and
the existing full visual verifier are reused unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.live_reel_stage1_v17 import parser as reel_parser, run


DEFAULT_EMISSION_CHECKPOINT = REPO / (
    "artifacts/models/stage1_v17_reel_emission_v3/best_model.pth"
)
DEFAULT_EMISSION_COREML = REPO / (
    "artifacts/coreml/Stage1ReelEmissionV17FP16.mlpackage"
)


def parser() -> argparse.ArgumentParser:
    value = reel_parser()
    value.description = __doc__
    value.set_defaults(
        output_root=REPO / "artifacts/reports/live_reel_emission_stage1_v17",
        orientation_coreml=DEFAULT_EMISSION_COREML,
        candidate_minimum_seconds=0.32,
        probe_interval_seconds=0.08,
        stability_hits=2,
        no_emit_probability_threshold=0.95,
        landmark_only_commit=True,
    )
    value.add_argument(
        "--emission-checkpoint", type=Path, default=DEFAULT_EMISSION_CHECKPOINT,
        help="metadata source for the calibrated NO_EMIT probability threshold",
    )
    value.add_argument(
        "--no-emit-probability-threshold", type=float,
        help="override the validation-selected checkpoint threshold",
    )
    value.add_argument(
        "--full-visual-verifier", dest="landmark_only_commit",
        action="store_false",
        help="restore the slower hand-image verifier for comparison",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    checkpoint = torch.load(
        args.emission_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("format") != "slt_stage1_reel_emission_v17":
        raise ValueError("emission checkpoint has the wrong format")
    if args.no_emit_probability_threshold is None:
        args.no_emit_probability_threshold = float(
            checkpoint["reel_emission"][
                "runtime_no_emit_probability_threshold"
            ]
        )
    if not 0.0 < args.no_emit_probability_threshold < 1.0:
        raise ValueError("NO_EMIT probability threshold must be between zero and one")
    positive = (
        args.processing_fps, args.candidate_minimum_seconds,
        args.candidate_maximum_seconds, args.probe_interval_seconds,
        args.stability_hits, args.release_hits, args.start_frames,
        args.commit_hits, args.no_hand_release_seconds, args.preroll_seconds,
        args.stage2_minimum_phrase_seconds, args.stage2_window_seconds,
    )
    if min(positive) <= 0:
        raise ValueError("timing, frame, and stability values must be positive")
    if args.candidate_minimum_seconds >= args.candidate_maximum_seconds:
        raise ValueError("candidate minimum must be below candidate maximum")
    if not 0 <= args.transition_overlap_seconds < args.candidate_minimum_seconds:
        raise ValueError("transition overlap must be shorter than a candidate")
    run(args)


if __name__ == "__main__":
    main()
