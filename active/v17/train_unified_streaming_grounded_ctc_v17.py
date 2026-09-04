#!/usr/bin/env python3
"""Train the causal head with signer-disjoint local and strict NCSLGR phrases."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

import active.v17.train_unified_streaming_ctc_v17 as trainer


_evaluate = trainer.evaluate
_group = trainer.group


def grounded_evaluate(*args, **kwargs):
    result = _evaluate(*args, **kwargs)
    buckets = [
        value for key, value in result["by_source"].items()
        if key in {"local_phrases", "asllrp_contiguous", "ncslgr_strict"}
    ]
    keys = (
        "samples", "exact", "known_edits", "known_target_tokens",
        "known_predicted_tokens", "false_emission_samples",
    )
    total = {key: sum(value[key] for value in buckets) for key in keys}
    total["exact_accuracy"] = total["exact"] / max(total["samples"], 1)
    total["known_wer"] = total["known_edits"] / max(total["known_target_tokens"], 1)
    total["false_emission_rate"] = (
        total["false_emission_samples"] / max(total["samples"], 1)
    )
    result["exact_phrases"] = total
    return result


def grounded_group(sample, other_index: int, source_balanced: bool = False) -> str:
    if not source_balanced:
        return _group(sample, other_index, False)
    if not sample.targets:
        return "blank"
    if other_index in sample.targets:
        # Keep partial-vocabulary corpora in one OOV task bucket. Giving the small
        # NCSLGR subset equal mass to the 879-span OOV corpus overfit its two signers.
        return "other"
    if sample.source.startswith("isolated:"):
        return sample.source
    return f"phrase:{sample.source}"


if __name__ == "__main__":
    trainer.evaluate = grounded_evaluate
    trainer.group = grounded_group
    parser = trainer.parser()
    parser.add_argument(
        "--exclude-ncslgr", action="store_true",
        help="matched ablation using the same signer split without NCSLGR rows",
    )
    parser.add_argument(
        "--exclude-asllrp-other", action="store_true",
        help="exclude the noisy ASLLRP OTHER expansion while retaining exact spans",
    )
    parser.description = __doc__
    parser.set_defaults(
        base=Path("artifacts/models/stage1_v17_asllrp_core_adapt_v1/best_model.pth"),
        phrase_root=Path("data/local/stage2_v17_grounded_signer_split"),
        output_dir=Path("artifacts/models/unified_streaming_grounded_ctc_v17_v1"),
        evidence_level="window",
        rolling_stride=4,
        rolling_window_frames=8,
        blank_policy="transitions_only",
        full_local_root=None,
        source_balanced_groups=True,
        selection_policy="aggregate",
        epochs=18,
    )
    args = parser.parse_args()
    if args.exclude_asllrp_other:
        original_other = trainer.phrase_sequences

        def skip_asllrp_other(root, *values, **options):
            if root == args.other_root:
                return []
            return original_other(root, *values, **options)

        trainer.phrase_sequences = skip_asllrp_other
    if args.exclude_ncslgr:
        original_phrase_sequences = trainer.phrase_sequences

        def without_ncslgr(*values, **options):
            return [
                sample for sample in original_phrase_sequences(*values, **options)
                if sample.source != "ncslgr_strict"
            ]

        trainer.phrase_sequences = without_ncslgr
    trainer.run(args)
