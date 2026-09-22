"""Matched contextual-core diagnostic for the completed combined CTC comparison."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_combined_dataset_v17 import load_features
from scripts.train_combined_frozen_joint_v17 import cached_chunks, make_model

MANIFEST = ROOT / "data/local/combined_dataset_v17_20260922/manifest.json"
CURATED = ROOT / "artifacts/reports/clean_boundary_subset_20260920/curated_manifest.json"
RESULTS = ROOT / "artifacts/reports/combined_frozen_joint_v17_20260922/results.json"
BASE = ROOT / "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"
OUT = ROOT / "artifacts/reports/combined_transition_diagnostic_v17_20260922"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_times(ranges, fps):
    """Map each normalized cached window back to its source-frame clock."""
    values = []
    for start, end in np.asarray(ranges, dtype=np.int64):
        if end <= start:
            raise ValueError("invalid source range")
        values.extend(np.linspace(start / fps, (end - 1) / fps, 32))
    return np.asarray(values)


def runs(path, times):
    output, start = [], 0
    for index in range(1, len(path) + 1):
        if index == len(path) or path[index] != path[start]:
            output.append(dict(index=int(path[start]), start=float(times[start]), end=float(times[index - 1])))
            start = index
    return output


def known_gap(left, right, all_events):
    """True only if the interval has no annotation, including explicit OTHER."""
    return not any(float(event["start"]) < right and float(event["end"]) > left for event in all_events)


def self_check():
    times = token_times([[0, 32], [32, 54]], 30.0)
    assert len(times) == 64 and times[0] == 0.0 and abs(times[31] - 31 / 30) < 1e-9
    assert abs(times[32] - 32 / 30) < 1e-9 and abs(times[-1] - 53 / 30) < 1e-9
    assert known_gap(.3, .5, [{"start": .1, "end": .2}])
    assert not known_gap(.3, .5, [{"start": .4, "end": .6}])


def labels_from(manifest):
    labels = manifest["label_to_index"]
    return labels, {value + 1: key for key, value in labels.items()} | {101: "__OTHER__", 0: "<blank>"}


def selected_rows(manifest, curated):
    all_events = defaultdict(list)
    eligible = defaultdict(list)
    for event in curated["events"]:
        if event["source"] == "asllrp_contiguous":
            all_events[event["item"]].append(event)
    for event in curated["eligible_cores"]:
        if event["source"] == "asllrp_contiguous" and event["kind"] == "known":
            eligible[event["item"]].append(event)
    rows = [row for row in manifest["records"] if row["source"] == "asllrp_contiguous" and row["source_item_id"] in all_events]
    if len(rows) != 56:
        raise ValueError(f"expected 56 annotated ASLLRP phrase rows, found {len(rows)}")
    return rows, all_events, eligible


def contextual_core_prediction(model, features, token_clock, start, end):
    """Apply the normal attention pool, restricted to an annotated core in whole context."""
    if model.base.config.static_hand_token != "none":
        raise ValueError("restricted contextual pooling requires no static-hand residual")
    value = torch.from_numpy(features.astype(np.float32, copy=False)).to(next(model.parameters()).device)
    encoded, active = model.base.encode(value)
    encoded, active = encoded.reshape(-1, encoded.shape[-1]), active.reshape(-1)
    core = torch.as_tensor((token_clock >= start) & (token_clock <= end), device=value.device)
    if not core.any():
        raise ValueError("eligible core has no mapped cached token")
    usable = core & active
    if not usable.any():
        usable = core
    scores = model.base.frame_attention(encoded).squeeze(-1).masked_fill(~usable, torch.finfo(encoded.dtype).min)
    pooled = (encoded * torch.softmax(scores, dim=0).unsqueeze(-1)).sum(dim=0)
    return int(model.base.classifier(pooled[None]).argmax().item()) + 1, int(core.sum().item())


def one_checkpoint(path, base_checkpoint, labels, names, rows, all_events, eligible, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(base_checkpoint, device, saved["arm"])
    model.base.load_state_dict(saved["base_state_dict"], strict=True)
    model.head.load_state_dict(saved["head_state_dict"], strict=True)
    model.eval()
    examples, roles = [], defaultdict(lambda: Counter())
    with torch.inference_mode():
        for row in rows:
            features, _ = load_features(row, labels)
            features = np.asarray(features, dtype=np.float32)
            with np.load(ROOT / row["feature_path"], allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                times = token_times(archive["window_source_ranges"], float(metadata["video_metadata"]["fps"]))
            logits, lengths = model.sequences([cached_chunks(features)])
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite CTC replay")
            length = lengths[0]
            if length != len(times):
                raise ValueError("token/source-clock length mismatch")
            path_values = logits[0, :length].argmax(-1).cpu().numpy().astype(int)
            core_rows = []
            for event in eligible.get(row["source_item_id"], []):
                prediction, mapped = contextual_core_prediction(model, features, times, float(event["start"]), float(event["end"]))
                target = int(event["target"]) + 1
                correct = prediction == target
                core_rows.append(dict(label=event["label"], target=target, contextual_top1=names[prediction], correct=bool(correct), mapped_tokens=mapped, start=float(event["start"]), end=float(event["end"])))
                roles[row["role"]]["cores"] += 1
                roles[row["role"]]["core_correct"] += int(correct)
            raw_runs = runs(path_values, times)
            gaps = []
            ordered = sorted(eligible.get(row["source_item_id"], []), key=lambda event: event["start"])
            for left, right in zip(ordered, ordered[1:]):
                if known_gap(float(left["end"]), float(right["start"]), all_events[row["source_item_id"]]):
                    gaps.append((float(left["end"]), float(right["start"])))
            gap_located = [run for run in raw_runs if 1 <= run["index"] <= 100 and any(run["start"] >= start and run["end"] <= end for start, end in gaps)]
            roles[row["role"]]["samples"] += 1
            roles[row["role"]]["ctc_exact"] += int([run["index"] for run in raw_runs if run["index"] != 0] == row["ctc_targets"])
            roles[row["role"]]["ctc_known_runs"] += sum(1 <= run["index"] <= 100 for run in raw_runs)
            roles[row["role"]]["gap_located_known_runs"] += len(gap_located)
            transcript = [names[run["index"]] for run in raw_runs if run["index"] != 0]
            examples.append(dict(source_item_id=row["source_item_id"], role=row["role"], reference=row["target_sequence"], whole_ctc_transcript=transcript, contextual_cores=core_rows, ctc_argmax_runs=[dict(label=names[run["index"]], start_seconds=round(run["start"], 4), end_seconds=round(run["end"], 4)) for run in raw_runs], eligible_annotation_free_gaps=[dict(start_seconds=round(start, 4), end_seconds=round(end, 4)) for start, end in gaps], gap_located_known_runs=[dict(label=names[run["index"]], start_seconds=round(run["start"], 4), end_seconds=round(run["end"], 4)) for run in gap_located]))
    summary = {role: dict(values, contextual_core_top1_accuracy=(values["core_correct"] / values["cores"] if values["cores"] else None)) for role, values in roles.items()}
    return dict(checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=digest(path), arm=saved["arm"], seed=saved["seed"], selected_epoch=saved["selected_epoch"], summary=summary, examples=examples)


def main():
    self_check()
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    manifest, curated, results = (json.loads(path.read_text()) for path in (MANIFEST, CURATED, RESULTS))
    labels, names = labels_from(manifest)
    rows, all_events, eligible = selected_rows(manifest, curated)
    checkpoints = [ROOT / results["results"][key]["checkpoint"] for key in sorted(results["results"])]
    if len(checkpoints) != 4 or len(set(checkpoints)) != 4:
        raise ValueError("expected exactly four selected checkpoints")
    torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.35)
    device = torch.device("mps")
    values = []
    for path in checkpoints:
        value = one_checkpoint(path, torch.load(BASE, map_location="cpu", weights_only=False), labels, names, rows, all_events, eligible, device)
        key = f"{value['seed']}:{value['arm']}"
        expected = results["results"][key]
        if value["checkpoint_sha256"] != expected["checkpoint_sha256"]:
            raise ValueError("selected checkpoint hash mismatch")
        for role, selected in (("train", expected["selected_train"]), ("validation", expected["selected_validation"])):
            if value["summary"][role]["ctc_exact"] != selected["asllrp_contiguous"]["exact"]:
                raise ValueError(f"matched replay exact count differs for {key} {role}")
        values.append(value)
    output = dict(format="combined_transition_diagnostic_v17", scope="all 56 annotated ASLLRP contiguous rows, with contextual core scores only where clean-boundary eligibility exists; no training", limitations=["CTC argmax run clock is an approximate interpolation of each cached normalized window", "A core top-1 is contextual token-classifier evidence, not a separately extracted isolated-sign result or a claim that the encoder independently recognizes the core", "Known runs located in annotation-free core gaps are timing diagnostics, not certified transition false positives: a CTC spike can be delayed from an earlier sign"], inputs={str(path.relative_to(ROOT)): digest(path) for path in (MANIFEST, CURATED, RESULTS, BASE, Path(__file__))}, matched_rows=dict(total=len(rows), by_role=dict(Counter(row["role"] for row in rows)), rows_with_eligible_known_cores_by_role=dict(Counter(row["role"] for row in rows if row["source_item_id"] in eligible)), eligible_known_cores_by_role=dict(Counter(event["role"] for values in eligible.values() for event in values))), checkpoints=values)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["# Matched contextual transition diagnostic", "", "This replays the four selected checkpoints on the same cached, whole ASLLRP phrase windows used in the paired run. It does not retrain or change live inference.", "", "## Coverage", "", f"- Rows: {len(rows)} ({dict(Counter(row['role'] for row in rows))})", f"- Rows with eligible known cores: {dict(Counter(row['role'] for row in rows if row['source_item_id'] in eligible))}", f"- Eligible known annotated cores: {dict(Counter(event['role'] for values in eligible.values() for event in values))}", "", "## Results", ""]
    for value in output["checkpoints"]:
        lines.append(f"### Seed {value['seed']} {value['arm']} (epoch {value['selected_epoch']})")
        for role, summary in value["summary"].items():
            lines.append(f"- {role}: contextual core top-1 {summary['core_correct']}/{summary['cores']} ({summary['contextual_core_top1_accuracy']:.1%}); replay CTC exact {summary['ctc_exact']}/{summary['samples']}; whole CTC known runs {summary['ctc_known_runs']}; gap-located known runs {summary['gap_located_known_runs']}.")
        friend = next(item for item in value["examples"] if item["source_item_id"] == "asllrp:7345233.mp4:span00")
        lines.append(f"- FRIEND MAYBE: transcript `{' '.join(friend['whole_ctc_transcript']) or '<blank>'}`; contextual cores `" + ", ".join(f"{core['label']}→{core['contextual_top1']}" for core in friend["contextual_cores"]) + "`.")
        lines.append("")
    lines.extend(["## Interpretation limits", "", *[f"- {value}" for value in output["limitations"]]])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
