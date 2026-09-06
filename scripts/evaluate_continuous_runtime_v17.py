#!/usr/bin/env python3
"""Replay development landmarks through the exact incremental runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.continuous_runtime_v17 import ContinuousRecognizer
from active.v17.continuous_evidence_v17 import load_continuous_samples
from active.v17.train_streaming_tcn_ctc_v17 import edit_distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/local/stage2_v17_grounded_signer_split"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.add_argument("--source", default="local_phrases")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    recognizer = ContinuousRecognizer(a.checkpoint, device=a.device)
    labels = {label: index - 1 for index, label in recognizer.labels.items() if label != "OTHER"}
    samples = [s for s in load_continuous_samples(a.root, "validation", labels) if s.source == a.source]
    if a.limit:
        samples = samples[:a.limit]
    details, times = [], []
    for sample in samples:
        recognizer.reset(); events = []
        for frame in sample.frames:
            result = recognizer.add(frame)
            if result:
                times.append(result["latency_ms"])
                events.append(result)
        final = recognizer.finish()
        expected = [recognizer.labels[t] for t in sample.targets]
        details.append({"identity": sample.identity, "source": sample.source, "expected": expected,
                        "predicted": final["glosses"], "edits": edit_distance(expected, final["glosses"]),
                        "events": events, "final": final})
    if not details:
        raise ValueError("no requested development samples")
    result = {
        "format": "continuous_runtime_replay_v17", "samples": len(details),
        "exact": sum(r["predicted"] == r["expected"] for r in details) / len(details),
        "wer": sum(r["edits"] for r in details) / max(1, sum(len(r["expected"]) for r in details)),
        "inference_ms_median": float(np.median(times)), "inference_ms_p95": float(np.percentile(times, 95)),
        "observation_ms": 1000 * recognizer.config.stride / recognizer.config.fps,
        "uses_expected_sequence_for_inference": False, "test_accessed": False, "details": details,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
