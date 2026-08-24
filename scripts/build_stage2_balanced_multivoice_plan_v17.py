#!/usr/bin/env python3
"""Build exhaustive, signer-coherent bigram replay across all locked signs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    with np.load(args.pool, allow_pickle=False) as payload:
        targets = payload["target_indices"].astype(np.int64)
        sources = payload["source_codes"].astype(np.uint8)
        signers = payload["signer_ids"].astype(str)
        durations = payload["observed_frames"].astype(np.int16)
        metadata = json.loads(str(payload["metadata_json"]))
    if metadata.get("source_split") != "citizen_semlex_asllrp_train_only_replay":
        raise ValueError("balanced replay requires the locked train-only multivoice pool")
    if any(signer == "asllrp:JONATHAN" for signer in signers):
        raise ValueError("held-out signer entered balanced replay")

    by_signer_class: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    signer_source: dict[str, int] = {}
    for index, (target, source, signer) in enumerate(zip(targets, sources, signers)):
        signer = str(signer)
        by_signer_class[signer][int(target)].append(index)
        previous = signer_source.setdefault(signer, int(source))
        if previous != int(source):
            raise ValueError(f"{signer}: identity spans multiple datasets")
    eligible = {
        signer: classes for signer, classes in by_signer_class.items()
        if len(classes) >= args.minimum_classes_per_voice
    }
    if len(eligible) < 60:
        raise ValueError(f"insufficient signer diversity: {len(eligible)}")

    pair_signers: dict[tuple[int, int], list[str]] = {}
    for first in range(args.num_classes):
        for second in range(args.num_classes):
            if first == second:
                continue
            candidates = sorted(
                signer for signer, classes in eligible.items()
                if first in classes and second in classes
            )
            if len(candidates) < args.minimum_voices_per_pair:
                raise ValueError(f"pair {(first, second)} has only {len(candidates)} voices")
            pair_signers[first, second] = candidates

    rng = np.random.default_rng(args.seed)
    voice_counts = Counter()
    rows = []
    pair_voice_sets: dict[tuple[int, int], set[str]] = defaultdict(set)
    pairs = list(pair_signers)
    for repetition in range(args.repetitions_per_pair):
        rng.shuffle(pairs)
        for first, second in pairs:
            candidates = [
                signer for signer in pair_signers[first, second]
                if signer not in pair_voice_sets[first, second]
            ]
            if not candidates:
                candidates = pair_signers[first, second]
            minimum = min(voice_counts[signer] for signer in candidates)
            least_used = [signer for signer in candidates if voice_counts[signer] == minimum]
            signer = str(rng.choice(least_used))
            labels = [first, second]
            pool_indices = [
                int(rng.choice(eligible[signer][label])) for label in labels
            ]
            source_code = signer_source[signer]
            rows.append({
                "sequence_id": (
                    f"balanced_bigram_r{repetition:02d}_{first:03d}_{second:03d}"
                ),
                "source": "synthetic_balanced_multivoice_train",
                "pool_source_code": source_code,
                "target_indices": labels,
                "pool_indices": pool_indices,
                "signer_voice_synthesis": {
                    "signer_id": signer,
                    "pool_source_code": source_code,
                    "token_duration_frames": [
                        int(durations[index]) for index in pool_indices
                    ],
                    "context_frames": args.context_frames,
                    "bridge_frames": args.bridge_frames,
                    "max_trim_frames": args.max_trim_frames,
                    "minimum_keep_frames": args.minimum_keep_frames,
                },
            })
            voice_counts[signer] += 1
            pair_voice_sets[first, second].add(signer)

    transfer = json.loads(args.transfer_plan.read_text())
    if transfer.get("pool_sha256") != sha256(args.pool):
        raise ValueError("transfer plan does not use the same pinned pool")
    if any(
        row.get("signer_voice_synthesis", {}).get("signer_id") == "asllrp:JONATHAN"
        for row in transfer["rows"]
    ):
        raise ValueError("held-out signer entered transfer replay")
    transfer_rows = transfer["rows"]
    combined_rows = transfer_rows + rows
    rng.shuffle(combined_rows)
    signer_pool_indices = {
        signer: np.flatnonzero(signers == signer).astype(int).tolist()
        for signer in sorted(eligible)
    }
    pair_voice_counts = [len(values) for values in pair_voice_sets.values()]
    output = {
        "format": "slt_stage2_exhaustive_multivoice_bigram_plan_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "transfer_plan": args.transfer_plan.as_posix(),
        "transfer_plan_sha256": sha256(args.transfer_plan),
        "seed": args.seed,
        "sequence_count": len(combined_rows),
        "sequence_counts": {
            "synthetic_citizen_train": sum(
                row.get("source") == "synthetic_citizen_train" for row in transfer_rows
            ),
            "synthetic_multivoice_train": sum(
                row.get("source") == "synthetic_multivoice_train" for row in transfer_rows
            ),
            "synthetic_balanced_multivoice_train": len(rows),
        },
        "eligible_voice_count": len(eligible),
        "eligible_voice_counts_by_source": {
            "citizen": sum(signer.startswith("citizen:") for signer in eligible),
            "semlex": sum(signer.startswith("semlex:") for signer in eligible),
            "asllrp": sum(signer.startswith("asllrp:") for signer in eligible),
        },
        "ordered_pair_count": len(pair_signers),
        "repetitions_per_pair": args.repetitions_per_pair,
        "minimum_eligible_voices_per_pair": min(map(len, pair_signers.values())),
        "maximum_eligible_voices_per_pair": max(map(len, pair_signers.values())),
        "minimum_distinct_sampled_voices_per_pair": min(pair_voice_counts),
        "maximum_distinct_sampled_voices_per_pair": max(pair_voice_counts),
        "balanced_voice_sequence_counts": dict(sorted(voice_counts.items())),
        "signer_pool_indices": signer_pool_indices,
        "synthesis_contract": (
            "every ordered pair of distinct locked signs is composed twice using two "
            "different eligible train-only signer voices; the exhaustive direct replay is "
            "mixed with transition-preserving 63-voice and Citizen replay; no held-out "
            "identity or sealed split is used"
        ),
        "rows": combined_rows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    report = {key: value for key, value in output.items() if key != "rows"}
    report["output"] = args.output.as_posix()
    report["output_sha256"] = sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path,
        default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"),
    )
    parser.add_argument(
        "--transfer-plan", type=Path,
        default=Path("active/v17/stage2_multivoice_transfer_plan_v17.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("active/v17/stage2_balanced_multivoice_plan_v17.json"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("artifacts/reports/stage2_v17_balanced_multivoice/plan.json"),
    )
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--repetitions-per-pair", type=int, default=2)
    parser.add_argument("--minimum-classes-per-voice", type=int, default=2)
    parser.add_argument("--minimum-voices-per-pair", type=int, default=3)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--bridge-frames", type=int, default=2)
    parser.add_argument("--max-trim-frames", type=int, default=3)
    parser.add_argument("--minimum-keep-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8701)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
