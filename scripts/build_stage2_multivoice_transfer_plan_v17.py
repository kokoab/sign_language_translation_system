#!/usr/bin/env python3
"""Transfer 60+ train-signer timing styles onto genuine ASLLRP transition content."""

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
        raise ValueError("style transfer requires the locked v3 train-only pool")
    if any(signer == "asllrp:JONATHAN" for signer in signers):
        raise ValueError("held-out signer entered style transfer")

    by_signer_class: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    signer_source = {}
    for index, (target, source, signer) in enumerate(zip(targets, sources, signers)):
        by_signer_class[str(signer)][int(target)].append(index)
        previous = signer_source.setdefault(str(signer), int(source))
        if previous != int(source):
            raise ValueError(f"{signer}: identity spans multiple datasets")
    eligible = {
        signer: classes for signer, classes in by_signer_class.items()
        if len(classes) >= args.minimum_classes_per_voice
    }
    content_signers = sorted(signer for signer in eligible if signer.startswith("asllrp:"))
    style_signers = sorted(signer for signer in eligible if not signer.startswith("asllrp:"))
    if len(content_signers) != 3 or len(style_signers) < 50:
        raise ValueError("insufficient content/style voice coverage")
    all_indices_by_signer = {
        signer: [index for indices in classes.values() for index in indices]
        for signer, classes in eligible.items()
    }

    rng = np.random.default_rng(args.seed)
    rows = []
    style_counts = Counter()
    content_counts = Counter()
    class_counts = Counter()

    def choose_content(content_signer: str, sequence_index: int) -> tuple[list[int], list[int]]:
        available = sorted(eligible[content_signer])
        length = 3 if rng.random() < args.three_sign_probability else 2
        labels = [available[sequence_index % len(available)]]
        while len(labels) < length:
            candidate = int(rng.choice(available))
            if candidate != labels[-1]:
                labels.append(candidate)
        indices = [int(rng.choice(eligible[content_signer][label])) for label in labels]
        return labels, indices

    def style_durations(style_signer: str, labels: list[int]) -> list[int]:
        output = []
        for label in labels:
            candidates = eligible[style_signer].get(label, all_indices_by_signer[style_signer])
            output.append(int(durations[int(rng.choice(candidates))]))
        return output

    def append(sequence_index: int, content_signer: str, style_signer: str, prefix: str) -> None:
        labels, pool_indices = choose_content(content_signer, sequence_index)
        rows.append({
            "sequence_id": f"{prefix}_{sequence_index:05d}",
            "source": "synthetic_multivoice_train",
            "pool_source_code": 1,
            "target_indices": labels,
            "pool_indices": pool_indices,
            "signer_voice_synthesis": {
                "signer_id": style_signer,
                "style_source_code": signer_source[style_signer],
                "content_signer_id": content_signer,
                "content_pool_source_code": 1,
                "token_duration_frames": style_durations(style_signer, labels),
                "context_frames": (
                    args.context_frames if style_signer == content_signer
                    else args.transferred_context_frames
                ),
                "bridge_frames": args.bridge_frames,
                "max_trim_frames": args.max_trim_frames,
                "minimum_keep_frames": args.minimum_keep_frames,
                "style_transfer": style_signer != content_signer,
            },
        })
        style_counts[style_signer] += 1
        content_counts[content_signer] += 1
        class_counts.update(labels)

    for index in range(args.native_asllrp_sequences):
        signer = content_signers[index % len(content_signers)]
        append(index, signer, signer, "synthetic_native_asllrp_voice")
    for index in range(args.transferred_style_sequences):
        append(
            index,
            content_signers[index % len(content_signers)],
            style_signers[index % len(style_signers)],
            "synthetic_transferred_voice",
        )

    citizen_by_class = defaultdict(list)
    for index in np.flatnonzero(sources == 0):
        citizen_by_class[int(targets[index])].append(int(index))
    if sorted(citizen_by_class) != list(range(100)):
        raise ValueError("Citizen replay vocabulary changed")
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
        signer: sorted(all_indices_by_signer[signer]) for signer in sorted(eligible)
    }
    output = {
        "format": "slt_stage2_multivoice_style_transfer_plan_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pool": args.pool.as_posix(),
        "pool_sha256": sha256(args.pool),
        "seed": args.seed,
        "sequence_count": len(rows),
        "sequence_counts": {
            "synthetic_citizen_train": args.citizen_replay_sequences,
            "synthetic_multivoice_train": args.native_asllrp_sequences + args.transferred_style_sequences,
        },
        "content_voice_count": len(content_signers),
        "transferred_style_voice_count": len(style_signers),
        "total_style_voice_count": len(eligible),
        "style_voice_counts_by_source": {
            "asllrp": len(content_signers),
            "citizen": sum(signer.startswith("citizen:") for signer in eligible),
            "semlex": sum(signer.startswith("semlex:") for signer in eligible),
        },
        "signer_pool_indices": signer_pool_indices,
        "style_sequence_counts": dict(sorted(style_counts.items())),
        "content_sequence_counts": dict(sorted(content_counts.items())),
        "class_occurrences": {str(key): value for key, value in sorted(class_counts.items())},
        "synthesis_contract": (
            "all coarticulation content comes from one real ASLLRP training signer per sequence; "
            "additional official-train Citizen/SemLex voices transfer only observed duration "
            "distributions and neutral endpoint context; no held-out identity or test data"
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
            "pool", "pool_sha256", "sequence_count", "sequence_counts", "content_voice_count",
            "transferred_style_voice_count", "total_style_voice_count", "style_voice_counts_by_source",
            "style_sequence_counts", "content_sequence_counts", "citizen_test_accessed",
            "semlex_test_accessed", "local_test_accessed", "held_out_validation_signer_accessed",
            "two_m_flores_devtest_accessed",
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
    parser.add_argument("--output", type=Path, default=Path("active/v17/stage2_multivoice_transfer_plan_v17.json"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/reports/stage2_v17_multivoice/transfer_plan.json"))
    parser.add_argument("--native-asllrp-sequences", type=int, default=6000)
    parser.add_argument("--transferred-style-sequences", type=int, default=6000)
    parser.add_argument("--citizen-replay-sequences", type=int, default=6000)
    parser.add_argument("--minimum-classes-per-voice", type=int, default=2)
    parser.add_argument("--three-sign-probability", type=float, default=0.10)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--transferred-context-frames", type=int, default=3)
    parser.add_argument("--bridge-frames", type=int, default=2)
    parser.add_argument("--max-trim-frames", type=int, default=3)
    parser.add_argument("--minimum-keep-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6701)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2))


if __name__ == "__main__":
    main()
