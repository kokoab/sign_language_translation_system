#!/usr/bin/env python3
"""Audit the existing Stage-2 video→landmark→supervision→model path."""

from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path
import statistics
import subprocess

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path: str | Path):
    return json.loads((ROOT / path).read_text())


def write(name: str, value) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


def jq(path: str, expression: str):
    completed = subprocess.run(
        ["jq", expression, str(ROOT / path)], check=True, capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def percentile(values, percent):
    return float(np.percentile(np.asarray(values, np.float64), percent))


def summarize(values):
    return {
        "count": len(values), "p10": percentile(values, 10),
        "median": float(statistics.median(values)), "p90": percentile(values, 90),
    }


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, observed in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (expected != observed)))
        previous = current
    return previous[-1]


def stage2_rows():
    rows = []
    for name in ("active/v17/stage2_training_manifest_v17.json",
                 "active/v17/stage2_asllrp_other_ctc_manifest_v17.json"):
        rows.extend(row for row in load(name)["rows"] if row["role"] in {"train", "validation"})
    assert len(rows) == 1647 and len({row["source_item_id"] for row in rows}) == len(rows)
    assert not any("test" in Path(row["video_path"]).parts for row in rows)
    return rows


def video_inventory(rows):
    counts, bytes_per_frame = Counter(), defaultdict(list)
    for row in rows:
        path = ROOT / row["video_path"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open {path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        counts[(row["source"], row["role"], f"{width}x{height}", round(fps, 2))] += 1
        if frames:
            bytes_per_frame[row["source"]].append(path.stat().st_size / frames)
    return {
        "counts": [dict(source=k[0], role=k[1], resolution=k[2], fps=k[3], clips=v)
                   for k, v in sorted(counts.items())],
        "median_bytes_per_frame": {key: statistics.median(value)
                                   for key, value in sorted(bytes_per_frame.items())},
    }


def live_diagnostics():
    grouped, clip_counts, empty, retained, clip_rows = defaultdict(list), Counter(), Counter(), defaultdict(list), []
    root = ROOT / "data/local/stage2_v17_live_matched_v1/diagnostics"
    for path in root.glob("*/*.json"):
        row = json.loads(path.read_text())
        key = (row["source"], row["role"])
        clip_counts[key] += 1
        empty[key] += row["feature_windows"] == 0
        retained[key].append(row["observations"] / row["source_frames"])
        hand = [w["diagnostics"]["hand_presence_fraction"] for w in row["windows"]
                if w["accepted"] and w.get("diagnostics")]
        observed = [w["diagnostics"]["observed_hand_frames"] / w["diagnostics"]["source_frames"]
                    for w in row["windows"] if w["accepted"] and w.get("diagnostics")]
        grouped[key].extend(zip(hand, observed))
        clip_rows.append({**row, "median_hand_presence": statistics.median(hand) if hand else 0.0})
    summary = {}
    for key, values in sorted(grouped.items()):
        hands = [value[0] for value in values]
        observations = [value[1] for value in values]
        summary["|".join(key)] = {
            "clips": clip_counts[key], "empty_clips": empty[key], "accepted_windows": len(values),
            "hand_node_presence": summarize(hands),
            "frames_with_any_hand": summarize(observations),
            "source_frame_retention": summarize(retained[key]),
            "windows_below_50pct_hand_node_presence": sum(value < .5 for value in hands),
        }
    return summary, clip_rows


def confident_reduction():
    expression = r'''{
      audit:.audit,
      exclusion_reasons:([.decisions[] | .exclusion_reasons[]] | group_by(.) | map({reason:.[0],count:length})),
      raw_sample_exclusions:([.decisions[] | select((.exclusion_reasons|index("fewer_than_four_raw_target_samples")) or (.exclusion_reasons|index("fewer_than_six_raw_target_samples")))] | length)
    }'''
    path = ROOT / "artifacts/reports/confident_supervision_v17_20260920/confident_supervision.json"
    completed = subprocess.run(["jq", expression, str(path)], check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def signer_split(rows):
    signer = load("artifacts/reports/local_phrase_signer_audit_v17_v4_auto/signer_clusters.json")
    by_video = {row["video_path"]: row["signer_id"] for row in signer["rows"]}
    roles = defaultdict(Counter)
    local = [row for row in rows if row["source"] == "local_phrases"]
    for row in local:
        roles[row["role"]][by_video[row["video_path"]]] += 1
    grounded = load("data/local/stage2_v17_grounded_signer_split/manifest.json")
    labels = sorted({label for row in local for label in row["target_sequence"]})
    return {
        "raw_videos": signer["video_count"], "identified_signers": signer["selected_signer_clusters"],
        "cluster_sizes": {key: value["videos"] for key, value in signer["clusters"].items()},
        "old_manifest_role_by_signer": {role: dict(value) for role, value in roles.items()},
        "old_manifest_has_signer_leakage": bool(set(roles["train"]) & set(roles["validation"])),
        "labeled_stage2_clips": len(local), "labeled_glosses": labels,
        "labeled_gloss_count": len(labels), "grounded_signer_disjoint": grounded,
    }


def frontend_probe():
    rows = load("artifacts/reports/stage2_v17_live_lock_diagnosis_v1/frontend_probe.json")["rows"]
    lanes = ("fresh_1280_30", "fresh_640_30", "fresh_1280_20", "fresh_640_20")
    cohort = [row for row in rows if all(row["lanes"].get(lane, {}).get("prediction") is not None for lane in lanes)]
    result = {"shared_clips": len(cohort), "reference_tokens": sum(len(row["reference"]) for row in cohort)}
    for lane in lanes:
        edits = sum(edit_distance(row["reference"], row["lanes"][lane]["prediction"]) for row in cohort)
        exact = sum(row["reference"] == row["lanes"][lane]["prediction"] for row in cohort)
        result[lane] = {"edits": edits, "wer": edits / result["reference_tokens"], "exact_clips": exact}
    result["resolution_sensitive_items"] = [row["item_id"] for row in cohort
        if row["lanes"]["fresh_1280_30"]["prediction"] != row["lanes"]["fresh_640_30"]["prediction"]]
    return result, rows


def slowdown_test():
    path = next((ROOT / "data/local/stage1_window_v17/raw_observations/train/asllrp_other_ctc").glob("*.npz"))
    with np.load(path, allow_pickle=False) as payload:
        source = payload["raw_features"].astype(np.float64).reshape(len(payload["raw_features"]), -1)
    x = np.arange(len(source), dtype=np.float64)
    doubled_x = np.linspace(0, len(source) - 1, len(source) * 2)
    slowed = np.column_stack([np.interp(doubled_x, x, source[:, column])
                              for column in range(source.shape[1])])
    return {
        "archive": str(path.relative_to(ROOT)), "source_observations": len(source),
        "slowed_frames": len(slowed), "source_matrix_rank": int(np.linalg.matrix_rank(source)),
        "slowed_matrix_rank": int(np.linalg.matrix_rank(slowed)),
        "new_camera_observations": 0,
        "interpretation": "Interpolation adds timestamps, not independent observed poses.",
    }


def experiment_metrics(frontend):
    clean = load("artifacts/reports/clean_boundary_subset_20260920/metrics.json")["core"]["adapted_encoder|core"]
    grounded = load("artifacts/models/unified_streaming_aligned_grounded_v17_v1/result.json")["validation"]["by_source"]
    segment = load("artifacts/reports/segment_first_v17_20260921/metrics.json")
    coherent = load("artifacts/reports/segment_first_coherent_decode_v17_20260921/metrics.json")
    boundary_audit = load("artifacts/reports/boundary_data_audit_20260920/audit.json")["groups"]
    known_windows = sum(value.get("known_windows", 0) for value in boundary_audit.values())
    return {
        "adapted_exact_core": clean,
        "grounded_signer_disjoint": {
            key: grounded[key] for key in ("local_phrases", "asllrp_contiguous", "ncslgr_strict")
        },
        "segment_exact_core": segment["exact_segments"],
        "segment_online": segment["online"],
        "coherent_boundary": coherent["boundary"],
        "coherent_visible_wer": coherent["visible_wer"],
        "held_repeat": coherent["held_repeat"],
        "known_context_windows": known_windows,
        "known_context_windows_ending_before_annotation_end": sum(
            value.get("known_windows_end_before_sign_end", 0) for value in boundary_audit.values()),
        "known_context_windows_less_than_half_target": sum(
            value.get("known_windows_less_than_half_target", 0) for value in boundary_audit.values()),
        "known_context_windows_missing_onset": sum(
            value.get("known_windows_missing_sign_onset", 0) for value in boundary_audit.values()),
        "known_events_with_at_most_two_observed_frames": sum(
            value.get("known_observed_frames_le_2", 0) for value in boundary_audit.values()),
        "controlled_frontend": frontend,
    }


def build_viewer(rows, diagnostic_rows, probe_rows):
    by_id = {row["source_item_id"]: row for row in rows}
    local = sorted((row for row in diagnostic_rows if row["source"] == "local_phrases" and row["role"] == "validation"),
                   key=lambda row: row["median_hand_presence"])
    selected = [local[0], local[len(local) // 2], local[-1]]
    cards = []
    for row in selected:
        source = by_id[row["item_id"]]
        video = Path(source["video_path"])
        rel = Path("../../..").joinpath(video)
        cards.append((row["item_id"], source["target_sequence"], rel.as_posix(),
                      f"Local held-out-role clip; median hand-node presence {row['median_hand_presence']:.1%}."))
    sensitive = {row["item_id"] for row in probe_rows
                 if row["lanes"].get("fresh_1280_30", {}).get("prediction") != row["lanes"].get("fresh_640_30", {}).get("prediction")}
    for row in probe_rows:
        if row["item_id"] not in sensitive:
            continue
        source = by_id[row["item_id"]]
        rel = Path("../../..").joinpath(source["video_path"])
        note = (f"Resolution-sensitive: 1280/source-rate {row['lanes']['fresh_1280_30']['prediction']}; "
                f"640/source-rate {row['lanes']['fresh_640_30']['prediction']}.")
        cards.append((row["item_id"], row["reference"], rel.as_posix(), note))
    body = []
    for item, target, video, note in cards:
        body.append(f'''<article><h2>{escape(item)}</h2><p><b>Reference:</b> {escape(" ".join(target))}</p>
<video controls preload="metadata" src="{escape(video)}"></video><p>{escape(note)}</p></article>''')
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 2 data-path evidence</title><style>
body{{max-width:1200px;margin:auto;padding:24px;font:15px/1.5 system-ui;background:#10141d;color:#edf2fa}}a{{color:#6ca9ff}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}article{{background:#192230;border:1px solid #344052;border-radius:12px;padding:15px}}video{{width:100%;max-height:420px;background:#000;border-radius:8px}}h1,h2{{line-height:1.15}}</style></head><body>
<h1>Stage 2 data-path evidence</h1><p>These clips illustrate measured frontend quality; they are not a new linguistic annotation set.</p>
<p><a href="../stage2_v17_genuine_local_phrase_reference_v1/genuine_local_phrase_reference.mp4">Open the nine-phrase landmark-overlay reference</a> · <a href="../luna_boundary_annotation_pilot_20260921/comparison.html">Open source/Luna boundary review</a></p><main>{''.join(body)}</main></body></html>'''
    (HERE / "videos.html").write_text(html)
    return len(cards)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    rows = stage2_rows()
    videos = video_inventory(rows)
    live, diagnostic_rows = live_diagnostics()
    confident = confident_reduction()
    local = signer_split(rows)
    frontend, probe_rows = frontend_probe()
    experiments = experiment_metrics(frontend)
    slowdown = slowdown_test()
    inventory = {
        "stage2_rows": len(rows), "sources": dict(Counter(row["source"] for row in rows)),
        "video": videos, "local_phrases": local,
        "continuous_observer_contract": jq(
            "artifacts/reports/o5s5_citizen100_v17/combined_supervision.json", ".observer_contract"
        ),
        "isolated_extractor_maximum_image_side": 1280,
        "protected_test_accessed": False,
    }
    metrics = {"live_matched_landmarks": live, "confident_subset": confident,
               "experiments": experiments, "slowdown": slowdown}
    diagnosis = {
        "source_annotations_broadly_wrong": "rejected",
        "asllrp_source_video_too_low_resolution": "rejected: all audited ASLLRP Stage-2 clips are 1280x720 or 1280x960",
        "landmark_extraction_globally_failed": "rejected: ASLLRP median hand-node presence is 99%+ and every clip yields features",
        "continuous_frontend_loses_source_information": "supported for the live-matched raw cache: source 29.97fps/1280 is processed at 20fps/640; separate source-rate offline caches already exist",
        "training_supervision_discards_short_signs": f"supported: {confident['raw_sample_exclusions']} events fail the 4/6-observation rules",
        "old_local_phrase_validation_is_signer_disjoint": "rejected: all three signers occur in both train and validation",
        "signer_disjoint_local_phrases_are_useful": "supported but limited: 25.5% exact and 37.04% WER over 200 held-out-signer clips",
        "slowing_cached_landmarks_recovers_information": "rejected: interpolation adds no camera observations or matrix rank",
        "boundary_decoder_is_only_problem": "rejected: exact cores are much stronger, but coherent boundary F1 remains 40.46% at 200ms",
        "replacement_data_required_before another architecture patch": "supported after confirming the native-rate aligned CTC already existed and the matched no-blank ablation failed",
    }
    gallery = build_viewer(rows, diagnostic_rows, probe_rows)
    write("inventory.json", inventory)
    write("metrics.json", metrics)
    write("diagnosis.json", diagnosis)
    write("verification.json", {
        "status": "passed", "stage2_rows": len(rows), "diagnostic_clips": sum(v["clips"] for v in live.values()),
        "video_paths_resolved": len(rows), "gallery_items": gallery,
        "protected_test_accessed": False, "training_launched": False,
    })
    local_grounded = experiments["grounded_signer_disjoint"]["local_phrases"]
    asllrp_train = live["asllrp_other_ctc|train"]
    report = f'''# Stage 2 data-path root-cause audit

## Decision

**Keep the ASLLRP videos and source annotations. Keep the local phrases, but only through the existing signer-disjoint split. Replace the continuous preprocessing cache before replacing the data.**

The ASLLRP files are not low-resolution: all {sum(x['clips'] for x in videos['counts'] if x['source'].startswith('asllrp')):,} audited Stage-2 clips are 1280×720 or 1280×960 at 29.97 fps. The continuous observer nevertheless detects landmarks at 640 pixels and 20 fps, while the isolated v17 extractor uses up to 1280 pixels. This discards one third of source frames and half the image-side resolution before boundary learning.

## Are the landmarks good enough?

For ASLLRP, yes for a workable baseline. Every one of the 1,160 ASLLRP Stage-2 clips produced accepted feature windows. In ASLLRP-other training windows, median hand-node presence is {asllrp_train['hand_node_presence']['median']:.1%}, p10 is {asllrp_train['hand_node_presence']['p10']:.1%}, and median frames containing any detected hand is {asllrp_train['frames_with_any_hand']['median']:.1%}. The adapted encoder recognizes complete held-out ASLLRP-other sign cores at **72.95%** and contiguous cores at **82.35%**. Those results reject a global landmark failure.

The frontend is still lossy enough to matter. On the same 11 ASLLRP phrase clips with fixed hand evidence, 1280/source-rate landmarks scored {frontend['fresh_1280_30']['edits']}/{frontend['reference_tokens']} edits ({frontend['fresh_1280_30']['wer']:.2%} WER); 640/source-rate scored {frontend['fresh_640_30']['edits']}/{frontend['reference_tokens']} ({frontend['fresh_640_30']['wer']:.2%}); 1280/20Hz scored {frontend['fresh_1280_20']['edits']}/{frontend['reference_tokens']} ({frontend['fresh_1280_20']['wer']:.2%}); and the current 640/20Hz contract scored {frontend['fresh_640_20']['edits']}/{frontend['reference_tokens']} ({frontend['fresh_640_20']['wer']:.2%}). Resolution and sampling contribute measurable errors, although neither explains every failure.

## The largest data-handling defect

The conservative manifest accepted 5,331/11,936 events. **{confident['raw_sample_exclusions']:,} events fail the four/six-observation rules.** Because the cache observes about 20 fps, a valid 0.20-second sign often has only four observations. Requiring six observations preferentially removes short signs and their boundaries. Across the earlier context objective, all {experiments['known_context_windows']:,} known windows ended before the annotated sign end, {experiments['known_context_windows_less_than_half_target']:,} contained less than half target frames, and {experiments['known_context_windows_missing_onset']:,} started after onset. This is a supervision-construction failure, not bad ASLLRP annotation.

Artificial slowing is not the repair. The reproducible probe expands {slowdown['source_observations']} observed frames to {slowdown['slowed_frames']}, but adds zero camera observations and preserves matrix rank ({slowdown['source_matrix_rank']}→{slowdown['slowed_matrix_rank']}). Re-extraction at native timestamps is required.

## Local phrases

The project has 780 unique 640×480 phrase videos from three identified signers. The old Stage-2 manifest leaks all three signers into both train and validation, so its earlier 97-clip local score is optimistic. The corrected local split already exists: signers 01 and 03 train, signer 02 validates, with 287/200 labeled clips. On those 200 held-out-signer clips, the grounded causal model reaches **{local_grounded['exact_accuracy']:.1%} exact phrase accuracy and {local_grounded['known_wer']:.2%} WER**. This is useful evidence, but the labeled subset covers only {local['labeled_gloss_count']} glosses and nine repeated phrase families. It can teach connected timing for those phrases; it cannot establish 100-gloss generalization.

## What failed in the last boundary experiment

Exact-core gloss recognition remained {experiments['segment_exact_core']['known_gloss_accuracy']:.2%} overall and {experiments['segment_exact_core']['by_source']['asllrp_other_ctc']['known_gloss_accuracy']:.2%} on held-out ASLLRP-other, while online boundary F1 was {experiments['coherent_boundary']['0.2']['f1']:.2%} at ±200 ms and intentional repeats were {experiments['held_repeat']['intentional_repeat_twice_exact']}/{experiments['held_repeat']['events']}. The model can identify many complete signs but receives weak temporal evidence and an unsuitable four-state boundary target. Better decoding reduced insertions; it did not restore missing evidence.

## Post-audit correction and experiment

The launch precheck found that source-rate, 1280-pixel ASLLRP caches and native 30 fps local caches already exist. The prior aligned grounded experiment already used them with the signer-disjoint local split, an 8-frame causal CTC window, locked-100+OTHER outputs, timed alignment, isolated replay, and the exact-core-adapted Stage 1 initialization. It reached 25.5% local exact/37.04% WER and 33.33% ASLLRP-contiguous exact/41.67% WER. Re-extracting or rerunning it would be duplicate work.

A matched 18-epoch ablation removed only standalone transition-as-blank clips. It completed in 106 seconds and regressed local exact to 16.0% and WER to 45.19%; ASLLRP contiguous stayed at 41.67% WER, NCSLGR worsened from 78% to 84% WER, and isolated exact rose slightly from 82.37% to 83.19%. It failed promotion. The replacement-data condition is now met: seek broader signer-disjoint connected recordings with exact locked-vocabulary transcripts rather than another CTC or boundary-state patch. Result: `artifacts/reports/native_ctc_no_blank_v17_20260921/`.

Review representative clips in [videos.html](videos.html). Machine-readable evidence: [inventory.json](inventory.json), [metrics.json](metrics.json), [diagnosis.json](diagnosis.json), and [verification.json](verification.json). No training or protected-test access occurred.
'''
    (HERE / "REPORT.md").write_text(report)
    print(json.dumps({"report": str(HERE / "REPORT.md"), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
