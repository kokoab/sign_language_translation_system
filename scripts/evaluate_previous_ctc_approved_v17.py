#!/usr/bin/env python3
"""Evaluate the earlier causal CTC checkpoint on today's approved phrases only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames
from active.v17.train_unified_streaming_ctc_v17 import rolling_windows
from scripts.build_combined_dataset_v17 import load_features
from scripts.train_youtube_motion_pilot_v17 import align_tokens

MANIFEST = ROOT / "data/local/combined_dataset_v17_20260922/manifest.json"
HEAD = ROOT / "artifacts/models/unified_streaming_aligned_grounded_v17_v1/best_model.pth"
REPORT = ROOT / "artifacts/reports/previous_ctc_approved_v17_20260922"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def collapse(path: np.ndarray) -> list[int]:
    out, previous = [], None
    for raw in path:
        value = int(raw)
        if value != previous and 1 <= value <= 101:
            out.append(value)
        previous = value
    return out


def phrase_rows(manifest: dict) -> list[dict]:
    rows = [
        row for row in manifest["records"]
        if row["role"] == "validation"
        and row["source"] in {"local_phrases", "asllrp_contiguous"}
        and row["supervision"] == "approved_phrase_or_subspan"
    ]
    counts = {source: sum(row["source"] == source for row in rows)
              for source in ("local_phrases", "asllrp_contiguous")}
    if len(rows) != 211 or counts != {"local_phrases": 199, "asllrp_contiguous": 12}:
        raise ValueError(f"expected approved validation 199 local + 12 ASLLRP, got {counts}")
    return rows


def old_training_ids() -> set[str]:
    """IDs used by the old checkpoint's local phrase training split, for disclosure."""
    root = ROOT / "data/local/stage2_v17_grounded_signer_split/train/local_phrases"
    ids = set()
    for path in root.glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
        ids.add(str(metadata["source_item_id"]))
    return ids


def metric_bucket() -> dict[str, int]:
    return dict(samples=0, exact=0, substitutions=0, deletions=0, insertions=0,
                target_tokens=0, predicted_tokens=0)


def finish(bucket: dict[str, int]) -> dict[str, float | int]:
    bucket["wer"] = (bucket["substitutions"] + bucket["deletions"] + bucket["insertions"]) / max(1, bucket["target_tokens"])
    bucket["exact_accuracy"] = bucket["exact"] / max(1, bucket["samples"])
    return bucket


def run() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    rows = phrase_rows(manifest)
    labels = manifest["label_to_index"]
    head_checkpoint = torch.load(HEAD, map_location="cpu", weights_only=False)
    base_path = ROOT / head_checkpoint["base_checkpoint"]
    if digest(base_path) != head_checkpoint["base_checkpoint_sha256"]:
        raise ValueError("old checkpoint's pinned Stage-1 hash does not match")
    if head_checkpoint["label_to_index"] != labels:
        raise ValueError("old CTC checkpoint vocabulary differs from approved manifest")
    if head_checkpoint.get("ctc_blank_index") != 0 or head_checkpoint.get("other_index") != 101:
        raise ValueError("unexpected CTC indexes")
    base_checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    base = SLTStage1V17(Stage1V17Config(**base_checkpoint["model_config"]))
    base.load_state_dict(base_checkpoint["model_state_dict"], strict=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this comparison")
    device = torch.device("mps")
    base.to(device).eval()
    head = load_unified_streaming_head(head_checkpoint, device=device)
    old_train = old_training_ids()
    results = defaultdict(metric_bucket)
    predictions = []
    for index, row in enumerate(rows, 1):
        features, targets = load_features(row, labels)  # hashes + archive contract
        with np.load(ROOT / row["feature_path"], allow_pickle=False) as payload:
            frames = restore_source_frames(payload["landmarks"], payload["window_source_ranges"])
        windows = rolling_windows(frames, stride=4, window_frames=8)
        with torch.inference_mode():
            batch = torch.from_numpy(windows.astype(np.float32)).to(device)
            logits, pooled = base(batch, return_embeddings=True)
            evidence = torch.cat((pooled, logits), dim=-1).unsqueeze(0)
            predicted = collapse(head(evidence)[0].argmax(-1).cpu().numpy())
        expected = list(targets)
        edits, _ = align_tokens(expected, predicted)
        bucket = results[row["source"]]
        bucket["samples"] += 1; bucket["exact"] += int(predicted == expected)
        bucket["target_tokens"] += len(expected); bucket["predicted_tokens"] += len(predicted)
        for name, value in edits.items():
            bucket[name] += value
        predictions.append(dict(source=row["source"], source_item_id=row["source_item_id"],
                                expected=expected, predicted=predicted,
                                old_training_identity_overlap=row["source_item_id"] in old_train,
                                edits=edits))
        if index % 32 == 0:
            print(f"evaluated {index}/{len(rows)}", flush=True)
    by_source = {name: finish(value) for name, value in results.items()}
    total = metric_bucket()
    for value in by_source.values():
        for key in ("samples", "exact", "substitutions", "deletions", "insertions", "target_tokens", "predicted_tokens"):
            total[key] += value[key]
    total = finish(total)
    return dict(
        format="previous_ctc_on_current_approved_validation_v17", device="mps",
        checkpoint=str(HEAD.relative_to(ROOT)), checkpoint_sha256=digest(HEAD),
        base_checkpoint=str(base_path.relative_to(ROOT)), base_checkpoint_sha256=digest(base_path),
        combined_manifest=str(MANIFEST.relative_to(ROOT)), combined_manifest_sha256=digest(MANIFEST),
        pipeline=dict(source_frames="restore_source_frames from approved archives", rolling_stride=4,
                      rolling_window_frames=8, stage1_evidence="pooled embedding + 100 logits",
                      decode="greedy causal CTC collapse; blank=0, OTHER=101"),
        validation_membership=dict(total=211, local_phrases=199, asllrp_contiguous=12,
                                   source_item_overlap_with_old_local_training=sum(p["old_training_identity_overlap"] for p in predictions)),
        by_source=by_source, total=total, predictions=predictions, test_accessed=False,
    )


def self_test() -> None:
    assert collapse(np.asarray([0, 1, 1, 0, 1, 101, 101])) == [1, 1, 101]
    assert finish(dict(samples=2, exact=1, substitutions=1, deletions=1, insertions=0,
                       target_tokens=4, predicted_tokens=3))["wer"] == 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    if REPORT.exists():
        raise FileExistsError(f"refusing to overwrite {REPORT}")
    output = run()
    REPORT.mkdir(parents=True)
    (REPORT / "result.json").write_text(json.dumps(output, indent=2) + "\n")
    (REPORT / "REPORT.md").write_text(
        "# Earlier causal CTC checkpoint on current approved validation\n\n"
        f"Evaluated {output['validation_membership']['total']} current approved phrase clips: "
        f"{output['validation_membership']['local_phrases']} local and "
        f"{output['validation_membership']['asllrp_contiguous']} ASLLRP.\n\n"
        "| Source | Clips | Exact | WER | S / D / I |\n|---|---:|---:|---:|---:|\n" +
        "\n".join(f"| {name} | {value['samples']} | {value['exact_accuracy']:.2%} | {value['wer']:.2%} | "
                  f"{value['substitutions']} / {value['deletions']} / {value['insertions']} |"
                  for name, value in list(output['by_source'].items()) + [('total', output['total'])]) +
        "\n\nThis uses the old checkpoint's source-frame restoration and 8-frame, stride-4 causal windows. "
        "It is not the newer cached-32-window joint-CTC pipeline. No protected test data was accessed.\n")
    print(json.dumps({"total": output["total"], "by_source": output["by_source"],
                      "membership": output["validation_membership"]}, indent=2))


if __name__ == "__main__":
    main()
