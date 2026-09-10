#!/usr/bin/env python3
"""Reel preview with preserved incoming motion and a Finish-time sequence suggestion.

Amber glosses ending in ? are tentative. Keep signing; press F/Finish once per
utterance. The sequence suggestion is for review and never replaces or speaks over
the verified transcript. The original Reel command retains its existing defaults.

Pass --revisable-transcript for a live CTC gloss transcript that may change until
Finish performs its final visual re-decode; grammar runs only after that pass.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.live_reel_stage1_v17 import parser as reel_parser, run, validate_args


def parser():
    value = reel_parser()
    value.description = __doc__
    value.set_defaults(
        output_root=REPO / "artifacts/reports/live_reel_continuous_v17",
        preserve_pending_frames=True,
        provisional_glosses=True,
        stage2_at_finish=True,
        stage2_review_only=True,
        no_stage2_arbiter=False,
        no_finish_gesture=True,
        stage2_other_preservation=REPO / 'artifacts/models/stage2_v17_transition_repair_v3/seed_1702.pth',
    )
    return value


def main():
    args = parser().parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
