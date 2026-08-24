#!/usr/bin/env python3
"""Build balanced signer-coherent compositions from 60+ train-only voices."""

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
        raise ValueError("multi-voice plan requires the locked v3 train-only pool")
    if not (len(targets) == len(sources) == len(signers) == len(durations)):
        raise ValueError("multi-voice pool arrays are not aligned")
    if any(value == "asllrp:JONATHAN" for value in signers):
        raise ValueError("held-out signer JONATHAN entered the training pool")

    by_signer_class: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    signer_source = {}
    for index, (target, source, signer) in enumerate(zip(targets, sources, signers)):
        by_signer_class[str(signer)][int(target)].append(index)
        previous = signer_source.setdefault(str(signer), int(source))
        if previous != int(source):
            raise ValueError(f"{signer}: identity spans multiple source datasets")
    eligible = {
        signer: classes for signer, classes in by_signer_class.items()
        if len(classes) >= args.minimum_classes_per_voice
    }
    asllrp_signers = sorted(signer for signer in eligible if signer.startswith("asllrp:"))
    additional_signers = sorted(signer for signer in eligible if not signer.startswith("asllrp:"))
    if len(asllrp_signers) != 3 or len(additional_signers) < 50:
        raise ValueError(
            f"unexpected voice coverage: ASLLRP={len(asllrp_signers)} additional={len(additional_signers)}"
        )

    rng = np.random.default_rng(args.seed)
    rows = []
    signer_counts = Counter()
    class_counts = Counter()

    def append_sequence(sequence_index: int, signer: str, prefix: str) -> None:
        available = sorted(eligible[signer])
        length = 3 if rng.random() < args.three_sign_probability else 2
        first_offset = signer_counts[signer] % len(available)
        labels = [available[first_offset]]
        while len(labels) < length:
            candidate = int(rng.choice(available))
            if candidate != labels[-1]:
                labels.append(candidate)
        pool_indices = [int(rng.choice(eligible[signer][label])) for label in labels]
        source_code = signer_source[signer]
        rows.append({
            "sequence_id": f"{prefix}_{sequence_index:05d}",
            "source": "synthetic_multivoice_train",
            "pool_source_code": source_code,
            "target_indices": labels,
            "pool_indices": pool_indices,
            "signer_voice_synthesis": {
                "signer_id": signer,
                "pool_source_code": source_code,
                "token_duration_frames": [int(durations[index]) for index in pool_indices],
                "context_frames": args.context_frames,
                "bridge_frames": args.bridge_frames,
                "max_trim_frames": args.max_trim_frames,
                "minimum_keep_frames": args.minimum_keep_frames,
            },
        })
        signer_counts[signer] += 1
        class_counts.update(labels)

    for index in range(args.asllrp_sequences):
        append_sequence(index, asllrp_signers[index % len(asllrp_signers)], "synthetic_asllrp_voice")
    for index in range(args.additional_voice_sequences):
        append_sequence(
            index, additional_signers[index % len(additional_signers)], "synthetic_additional_voice"
        )

    citizen_by_class = defaultdict(list)
    for index in np.flatnonzero(sources == 0):
        citizen_by_class[int(targets[index])].append(int(index))
    if sorted(citizen_by_class) != list(range(100)):
        raise ValueError("Citizen pool no longer covers the locked vocabulary")
    for sequence_index in range(args.citizen_replay_sequences):
        length = int(rng.integers(2, 6))
        labels = [sequence_index % 100]
        while len(labels) < length:
            candidate = int(rng.integers(100))
            if candidate != labels[-1]:
                labels.append(candidate)
        rows.append({
            "sequence_id": f"synthetic_citizen_train_{sequence_index:05d}",
            "source": "synthetic_citizen_train",
            "pool_source_code": 0,
            "target_indices": labels,
            "pool_indices": [int(rng.choice(citizen_by_class[label])) for label in labels],
        })
    rng.shuffle(rows)

    signer_pool_indices = {
        signer: np.flatnonzero(signers == signer).astype(int).tolist() for signer in sorted(eligible)
    }
    output = {
        "format": "slt_stage2_multivoice_composition_plan_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "seed": args.seed,
        "sequence_count": len(rows),
        "sequence_counts": {
            "synthetic_citizen_train": args.citizen_replay_sequences,
            "synthetic_multivoice_train": args.asllrp_sequences + args.additional_voice_sequences,
        },
        "eligible_voice_count": len(eligible),
        "eligible_voice_counts_by_source": {
            "asllrp": len(asllrp_signers),
            "citizen": sum(signer.startswith("citizen:") for signer in eligible),
            "semlex": sum(signer.startswith("semlex:") for signer in eligible),
        },
        "signer_pool_indices": signer_pool_indices,
        "signer_sequence_counts": dict(sorted(signer_counts.items())),
        "class_occurrences": {str(key): value for key, value in sorted(class_counts.items())},
        "synthesis_contract": (
            "half of multivoice rows preserve genuine contextual ASLLRP train-signer trajectories; "
            "half add signer-consistent official-train Citizen/SemLex timing and endpoint styles; "
            "all sequences use one dataset-local signer identity; no validation/test signer input"
        ),
        "rows": rows,
        "citizen_test_accessed": False,
        "semlex_test_accessed": False,
        "local_test_accessed": False,
        "held_out_validation_signer_accessed": False,
        "two_m_flores_devtest_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    report = {
        key: output[key] for key in (
            "pool", "pool_sha256", "sequence_count", "sequence_counts",
            "eligible_voice_count", "eligible_voice_counts_by_source", "signer_sequence_counts",
            "citizen_test_accessed", "semlex_test_accessed", "local_test_accessed",
            "held_out_validation_signer_accessed", "two_m_flores_devtest_accessed",
        )
    }
    report["output"] = args.output.as_posix()
    report["output_sha256"] = sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/local/stage2_v17_synthetic/train_only_multivoice_pool_v3.npz"))
    parser.add_argument("--output", type=Path, default=Path("active/v17/stage2_multivoice_plan_v17.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_multivoice/plan.json"))
    parser.add_argument("--asllrp-sequences", type=int, default=6000)
    parser.add_argument("--additional-voice-sequences", type=int, default=6000)
    parser.add_argument("--citizen-replay-sequences", type=int, default=6000)
    parser.add_argument("--minimum-classes-per-voice", type=int, default=2)
    parser.add_argument("--three-sign-probability", type=float, default=0.10)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--bridge-frames", type=int, default=2)
    parser.add_argument("--max-trim-frames", type=int, default=3)
    parser.add_argument("--minimum-keep-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5701)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
