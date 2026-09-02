#!/usr/bin/env python3
"""Low-motion sign boundaries without requiring hands to return to neutral.

This is a separate experimental entry point over the proven isolated pipeline. A sign
starts on motion and closes after a short low-motion hold at any hand position.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.live_isolated_v17 import parser as isolated_parser, run


def parser():
    value = isolated_parser()
    value.description = __doc__
    value.set_defaults(
        mode="cascade",
        quiet_motion=0.010,
        quiet_seconds=0.12,
        output_root=REPO / "artifacts/reports/live_motion_valley_v17",
        end_on_low_motion=True,
        segment_video=True,
    )
    return value


def main() -> None:
    args = parser().parse_args()
    if args.processing_fps <= 0 or args.quiet_seconds <= 0:
        raise ValueError("processing FPS and quiet duration must be positive")
    if args.expected_label:
        args.expected_label = args.expected_label.upper()
    run(args)


if __name__ == "__main__":
    main()
