#!/usr/bin/env python3
"""Source-balanced v2 experiment focused on genuine continuous generalization."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from active.v17.train_unified_streaming_ctc_v17 import parser, run


if __name__ == "__main__":
    value = parser()
    value.description = __doc__
    value.set_defaults(
        base=Path("artifacts/models/stage1_v17_asllrp_core_adapt_v1/best_model.pth"),
        output_dir=Path(
            "artifacts/models/unified_streaming_ctc_v17_experiment_v2_balanced"
        ),
        evidence_level="window",
        rolling_stride=4,
        blank_policy="transitions_only",
        full_local_root=Path("data/local/full_trajectory_landmarks_v17"),
        source_balanced_groups=True,
        selection_policy="continuous_balanced",
        epochs=16,
    )
    run(value.parse_args())
