#!/usr/bin/env python3
"""Matched source-adapter motion pretraining of the existing causal CTC temporal stack."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active.v17.approved_phrase_data_v17 import APPROVED_ROOT, require_training_manifest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from active.v17.model_unified_streaming_ctc_v17 import UnifiedStreamingCTCConfig, UnifiedStreamingCTCHeadV17
from active.v17.model_v17 import SLTStage1V17, Stage1V17Config
from active.v17.train_unified_streaming_aligned_grounded_v17 import parser as ctc_parser, run as ctc_run
import active.v17.train_unified_streaming_ctc_v17 as base

REPORT = ROOT / "artifacts/reports/youtube_motion_pretrain_v17_20260921"
MODELS = ROOT / "artifacts/models/youtube_motion_pretrain_v17_20260921"
RAW = ROOT / "data/local/youtube_asl_keypoints_20260921/pilot_2000_raw"
BASE = ROOT / "artifacts/generated/kaggle_stage1_orientation_robust_pull_v2/stage1_v17_orientation_robust_v1/best_model.pth"
SEEDS = (17321, 17322)
PHRASE_ROOT = APPROVED_ROOT / "phrases"
REUSE_INITIAL_HEADS = None


def save(name, value):
    path = REPORT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def motion_features(payload):
    """Independent 46-node XY+mask input; assign hands by pose wrists, never slot name."""
    frames = payload["keypoints"]
    result = np.zeros((len(frames), 46, 3), np.float32)
    widths = []
    for frame in frames:
        pose = np.asarray(frame["pose_landmarks"], dtype=np.float32)
        if pose.shape == (33, 2):
            widths.append(np.linalg.norm(pose[11] - pose[12]))
    widths = np.asarray(widths)
    widths = widths[widths > 1e-5]
    if len(widths) < len(frames) * .5:
        return result
    scale = float(np.median(widths))
    for index, frame in enumerate(frames):
        pose = np.asarray(frame["pose_landmarks"], dtype=np.float32)
        if pose.shape != (33, 2) or np.linalg.norm(pose[11] - pose[12]) < scale * .2:
            continue
        center = (pose[11] + pose[12]) / 2
        result[index, 42:, :2] = (pose[[11, 12, 13, 14]] - center) / scale
        result[index, 42:, 2] = 1
        hands = [np.asarray(frame[key], np.float32) for key in
                 ("left_hand_landmarks", "right_hand_landmarks") if len(frame[key]) == 21]
        if not hands:
            continue
        distances = np.array([[np.linalg.norm(hand[0] - pose[wrist]) / scale
                               for wrist in (15, 16)] for hand in hands])
        if len(hands) == 2:
            costs = [distances[0, 0] + distances[1, 1], distances[0, 1] + distances[1, 0]]
            if abs(costs[0] - costs[1]) < .10:
                continue  # ambiguous crossing: omit both rather than assign false chirality
            sides = [0, 1] if costs[0] < costs[1] else [1, 0]
        else:
            if abs(distances[0, 0] - distances[0, 1]) < .10:
                continue
            sides = [int(distances[0].argmin())]
        for hand, distances_row, side in zip(hands, distances, sides):
            if distances_row[side] > .75:
                continue
            valid = np.any(hand != 0, axis=1)
            nodes = slice(side * 21, (side + 1) * 21)
            result[index, nodes, :2] = ((hand - center) / scale) * valid[:, None]
            result[index, nodes, 2] = valid
    result[..., 1] *= -1
    return result


def prepare():
    audit = json.loads((REPORT / "compatibility.json").read_text())
    if not audit["audit"]["complete_manifest"] or audit["audit"]["errors"]:
        raise ValueError("complete structural audit required")
    windows, folds, members = [], [], []
    accepted = excluded = rejected_windows = 0
    source_metadata = RAW.parent / "metadata"
    source_train = json.loads((source_metadata / "YT.translations.train.json").read_text())
    source_dev = json.loads((source_metadata / "YT.translations.dev.json").read_text())
    with (REPORT / "compatibility_clips.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if not row["admitted_for_pretraining"]:
                excluded += 1
                continue
            if row["video_id"] not in source_train or row["video_id"] in source_dev:
                raise ValueError("source-video is not exclusive to official unlabeled train split")
            content = (RAW / row["member"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("raw source changed after audit")
            features = motion_features(json.loads(content))
            fold = int(hashlib.sha256(row["video_id"].encode()).hexdigest()[:8], 16) % 10 == 0
            used = False
            for start in range(0, len(features) - 31, 16):
                window = features[start:start + 32]
                if (window[:, :42, 2].max(-1) > 0).mean() < .75:
                    rejected_windows += 1
                    continue
                wrist = window[:, [0, 21]]
                valid_motion = (wrist[1:, :, 2] > .5) & (wrist[:-1, :, 2] > .5)
                jumps = np.linalg.norm(wrist[1:, :, :2] - wrist[:-1, :, :2], axis=-1)
                if np.any(jumps[valid_motion] > 1.5):
                    rejected_windows += 1
                    continue
                windows.append(window); folds.append(fold); members.append(row["member"]); used = True
            accepted += used
    if not windows or not any(folds) or all(folds):
        raise ValueError("insufficient train/held-out source-video windows")
    np.savez_compressed(REPORT / "motion_windows.npz", features=np.stack(windows),
                        validation=np.asarray(folds), members=np.asarray(members))
    details = {"source_clips_used": accepted, "count_mismatch_excluded": excluded,
               "windows": len(windows), "validation_windows": sum(folds),
               "windows_rejected_for_visibility_or_wrist_jumps": rejected_windows,
               "train_windows": len(folds) - sum(folds),
               "schema": "independent_mp_xy_bodyrelative_46x3_pose_wrist_assignment_v1",
               "clock": "frame index only; no seconds or FPS invented",
               "geometry_source": "https://github.com/JSALT2024/PoseEstimation/blob/main/predict_pose.py",
               "transfer": "CTC temporal blocks only; Apple Stage1 entirely frozen",
               "signer_overlap_between_unlabeled_and_supervised_sources": "unknown",
               "self_supervised_split": "source-video hash fold0 validation; never called signer-disjoint"}
    save("preparation.json", details)
    print(json.dumps(details), flush=True)


class MotionEncoder(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(46 * 3 + 1, head.config.hidden_dim), nn.GELU())
        self.blocks = copy.deepcopy(head.blocks)
        self.reconstruction = nn.Linear(head.config.hidden_dim, 46 * 2)

    def forward(self, features, artificial):
        inputs = features.flatten(2).masked_fill(artificial[..., None], 0)
        hidden = self.adapter(torch.cat((inputs, artificial[..., None].to(inputs.dtype)), -1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.reconstruction(hidden).reshape(*features.shape[:3], 2)


def masked_loss(model, batch, rng):
    starts = torch.randint(3, 23, (len(batch),), generator=rng)
    mask = (torch.arange(32)[None] >= starts[:, None]) & (torch.arange(32)[None] < starts[:, None] + 6)
    mask = mask.to(batch.device)
    valid = mask[..., None] & (batch[..., 2] > .5)
    if not valid.any():
        raise ValueError("no observed targets in masked spans")
    prediction = model(batch, mask)
    return F.smooth_l1_loss(prediction[valid], batch[..., :2][valid]), mask, valid


def pretrain(head, seed, device):
    with np.load(REPORT / "motion_windows.npz", allow_pickle=False) as data:
        train = torch.from_numpy(data["features"][~data["validation"]])
        validation = torch.from_numpy(data["features"][data["validation"]])
    model = MotionEncoder(head).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.adapter.parameters(), "lr": 1e-3},
        {"params": model.reconstruction.parameters(), "lr": 1e-3},
        {"params": model.blocks.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(8):
        model.train()
        order = torch.randperm(len(train), generator=rng)
        losses = []
        for indices in order.split(32):
            batch = train[indices].to(device)
            loss, _, _ = masked_loss(model, batch, rng)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(float(loss.detach()))
            ready_fd = os.environ.pop("SLT_TRAIN_READY_FD", None)
            if ready_fd is not None:
                save("status.json", {"state": "training", "phase": "masked_motion", "seed": seed,
                                     "training_started": True, "first_step_loss": losses[-1], "pid": os.getpid()})
                os.write(int(ready_fd), b"TRAINING_STARTED\n"); os.close(int(ready_fd))
        model.eval(); validation_rng = torch.Generator().manual_seed(seed + 100)
        errors = []; last_value = []; interpolation = []; comparable = []
        with torch.no_grad():
            for batch in validation.split(32):
                batch = batch.to(device)
                loss, mask, valid = masked_loss(model, batch, validation_rng)
                errors.append(float(loss))
                predicted = model(batch, mask)
                for row in range(len(batch)):
                    indices = torch.nonzero(mask[row]).flatten()
                    left, right = int(indices[0]) - 1, int(indices[-1]) + 1
                    eligible = valid[row, indices] & (batch[row, left, :, 2] > .5) & (batch[row, right, :, 2] > .5)
                    if not eligible.any():
                        continue
                    target = batch[row, indices, :, :2]
                    hold = batch[row, left, :, :2].expand_as(target)
                    alpha = torch.arange(1, 7, device=device)[:, None, None] / 7
                    linear = hold * (1 - alpha) + batch[row, right, :, :2] * alpha
                    last_value.append(float(F.smooth_l1_loss(hold[eligible], target[eligible])))
                    interpolation.append(float(F.smooth_l1_loss(linear[eligible], target[eligible])))
                    comparable.append(float(F.smooth_l1_loss(predicted[row, indices][eligible], target[eligible])))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_loss": float(np.mean(errors)),
                        "network_on_both_endpoint_observed": float(np.mean(comparable)),
                        "last_value_on_both_endpoint_observed": float(np.mean(last_value)),
                        "noncausal_interpolation_on_both_endpoint_observed": float(np.mean(interpolation))})
        print(json.dumps({"seed": seed, "pretraining": history[-1]}), flush=True)
    head.blocks.load_state_dict(model.blocks.cpu().state_dict(), strict=True)
    save(f"pretraining_{seed}.json", {"history": history, "selected": "fixed final epoch8",
         "baseline_caveat": "Interpolation uses future observations; baseline subset requires both endpoints present. Reconstruction loss alone is not a promotion criterion."})
    del model
    return head


def self_check():
    torch.set_num_threads(4)
    seed_all(42)
    config = UnifiedStreamingCTCConfig()
    head = UnifiedStreamingCTCHeadV17(config)
    model = MotionEncoder(head)
    batch = torch.randn(2, 32, 46, 3); batch[..., 2] = 1; batch[:, :, 0] = 0
    loss, mask, valid = masked_loss(model, batch, torch.Generator().manual_seed(42))
    assert not valid[..., 0].any() and int(mask.sum()) == 12
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.blocks.parameters())
    model.eval()
    with torch.no_grad():
        before = model(batch, mask)
        changed = batch.clone(); changed[:, 20:, :, :2] += 100
        after = model(changed, mask)
        assert torch.allclose(before[:, :20], after[:, :20], atol=1e-5), "temporal stack must be causal"
    # Slot-name reversal must not change pose-wrist assignment.
    pose = np.zeros((33, 2)); pose[11] = [-1, 1]; pose[12] = [1, 1]
    pose[15] = [-2, 0]; pose[16] = [2, 0]
    frame = {"pose_landmarks": pose.tolist(), "left_hand_landmarks": [[-2., 0.]] * 21,
             "right_hand_landmarks": [[2., 0.]] * 21}
    swapped = {**frame, "left_hand_landmarks": frame["right_hand_landmarks"], "right_hand_landmarks": frame["left_hand_landmarks"]}
    assert np.array_equal(motion_features({"keypoints": [frame]}), motion_features({"keypoints": [swapped]}))
    counts, matches = align_tokens([1, 2, 2, 3], [1, 2, 4, 3, 5])
    assert counts == {"substitutions": 1, "deletions": 0, "insertions": 1}
    assert matches == [(0, 0), (1, 1), (3, 3)]
    assert emission_steps([0, 1, 1, 0, 1, 101, 101, 2])[0] == [1, 1, 2]
    assert repeat_indices([1, 101, 1]) == []
    assert repeat_indices([101, 1, 1, 2, 2]) == [1, 3]
    print("motion loss, missing targets, gradient, causality and hand assignment checks passed")


def align_tokens(reference, prediction):
    costs = np.zeros((len(reference) + 1, len(prediction) + 1), np.int32)
    costs[:, 0] = np.arange(len(reference) + 1); costs[0] = np.arange(len(prediction) + 1)
    for i, expected in enumerate(reference, 1):
        for j, actual in enumerate(prediction, 1):
            costs[i, j] = min(costs[i - 1, j] + 1, costs[i, j - 1] + 1,
                              costs[i - 1, j - 1] + (expected != actual))
    i, j = len(reference), len(prediction)
    counts = dict(substitutions=0, deletions=0, insertions=0); matches = []
    while i or j:
        if i and j and costs[i, j] == costs[i - 1, j - 1] + (reference[i - 1] != prediction[j - 1]):
            if reference[i - 1] == prediction[j - 1]:
                matches.append((i - 1, j - 1))
            else:
                counts["substitutions"] += 1
            i -= 1; j -= 1
        elif i and costs[i, j] == costs[i - 1, j] + 1:
            counts["deletions"] += 1; i -= 1
        else:
            counts["insertions"] += 1; j -= 1
    return counts, list(reversed(matches))


def emission_steps(path):
    tokens, steps = [], []
    previous = None
    for index, token in enumerate(path):
        token = int(token)
        if token != previous and 1 <= token <= 100:
            tokens.append(token); steps.append(index)
        previous = token
    return tokens, steps


def repeat_indices(targets):
    positions = [index for index, token in enumerate(targets) if 1 <= token <= 100]
    return [index for index in range(1, len(positions))
            if positions[index] == positions[index - 1] + 1
            and targets[positions[index]] == targets[positions[index - 1]]]


@torch.inference_mode()
def behavior_inputs(device):
    """One shared frozen-Apple evaluation set, independent of all pretraining weights."""
    from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames
    from active.v17.train_stage_1_reel_emission_v17 import asllrp_annotations
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=False)
    stage1 = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    stage1.load_state_dict(checkpoint["model_state_dict"], strict=True)
    labels = checkpoint["label_to_index"]
    annotations = asllrp_annotations(Path("data/local/asllrp_contiguous_phrases_v17/manifest.json"),
                                    Path("data/local/asllrp_segmented_citizen100_v17/manifest.json"))
    root = PHRASE_ROOT / "validation"
    samples = base.phrase_sequences(root.parent, "validation", 4, 8)
    metadata, probes = {}, []
    for path in sorted(root.glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as data:
            m = json.loads(str(data["metadata_json"].item()))
            raw = restore_source_frames(data["landmarks"], data["window_source_ranges"])
        if m["role"] != "validation":
            raise ValueError("behavior split mismatch")
        ends = list(range(8, len(raw) + 1, 4))
        if not ends or ends[-1] != len(raw):
            ends.append(len(raw))
        video_metadata = m.get("video_metadata", {})
        fps = video_metadata.get("fps") if video_metadata.get("decoded_frame_count") == len(raw) else None
        info = {"ends": ends, "fps": fps, "events": []}
        if m["source"] == "asllrp_contiguous":
            span, signs = annotations[m["source_item_id"]]
            offset = int(span["utterance_start_frame_global"]) + int(span["crop_start_frame_local"])
            for sign in signs:
                start, stop = int(sign["sign_start_frame"]) - offset, int(sign["sign_end_frame"]) - offset + 1
                target = labels[sign["canonical_label"]] + 1
                info["events"].append((start, stop, target))
                if len(probes) >= 40 or start < 0 or stop > len(raw) or stop - start < 4:
                    continue
                core = raw[start:stop]; rest = raw[max(0, start - 1)]
                prefix = np.repeat(rest[None], 8, 0); suffix = np.repeat(rest[None], 8, 0)
                for repeated in (False, True):
                    middle = np.concatenate((core, suffix, core)) if repeated else np.concatenate((core, np.repeat(core[-1][None], 16, 0)))
                    frames = np.concatenate((prefix, middle, suffix))
                    probes.append(base.RawSequence(base.rolling_windows(frames, 4, 8),
                        (target, target) if repeated else (target,), "synthetic_repeat" if repeated else "synthetic_hold",
                        f"{m['source_item_id']}:{start}:{int(repeated)}"))
        metadata[m["source_item_id"]] = info
    encoded = base.encode(stage1, samples + probes, device, 64, "window")
    return encoded, metadata


@torch.inference_mode()
def behavior(checkpoint_path, inputs, device):
    from active.v17.model_unified_streaming_ctc_v17 import load_unified_streaming_head
    model = load_unified_streaming_head(torch.load(checkpoint_path, map_location="cpu", weights_only=False), device=device)
    samples, metadata = inputs
    totals = {}; rows = []; delays = []; repeated_refs = repeated_recovered = 0
    for sample in samples:
        path = model(torch.from_numpy(sample.evidence.astype(np.float32))[None].to(device))[0].argmax(-1).cpu().tolist()
        predicted, steps = emission_steps(path)
        expected = list(base.known(sample.targets, 101))
        counts, matches = align_tokens(expected, predicted)
        bucket = totals.setdefault(sample.source, dict(samples=0, references=0, exact=0,
            substitutions=0, deletions=0, insertions=0, adjacent_output_duplicates=0))
        bucket["samples"] += 1; bucket["references"] += len(expected); bucket["exact"] += expected == predicted
        for key, count in counts.items():
            bucket[key] += count
        bucket["adjacent_output_duplicates"] += sum(a == b for a, b in zip(predicted, predicted[1:]))
        if not sample.source.startswith("synthetic"):
            matched = dict(matches)
            for index in repeat_indices(sample.targets):
                repeated_refs += 1
                repeated_recovered += (index in matched and index - 1 in matched)
            info = metadata[sample.identity]
            if info["events"] and info["fps"]:
                if [event[2] for event in info["events"]] != expected:
                    raise ValueError("timed truth and reference sequence disagree")
                for ri, pi in matches:
                    # First emission relative to annotated sign END, not hardware inference time.
                    delay = ((info["ends"][steps[pi]] - 1) - (info["events"][ri][1] - 1)) / info["fps"]
                    delays.append(delay)
        rows.append({"source": sample.source, "id": sample.identity, "expected": expected,
                     "predicted": predicted, "emission_steps": steps, **counts})
    return {"by_source": totals, "real_adjacent_repeat_references": repeated_refs,
            "real_adjacent_repeat_recovered": repeated_recovered,
            "sign_end_relative_first_emission_seconds": {
                "matched_events": len(delays), "median": float(np.median(delays)) if delays else None,
                "p90": float(np.quantile(delays, .9)) if delays else None,
                "definition": "ASLLRP matched tokens only; negative means emission before annotated end; excludes missed/substituted signs; not device latency"},
            "held_repeat_limit": "Synthetic probes only; no independently annotated real held/repeat test supplied. Adjacent output duplicates are not automatically held-sign errors.",
            "predictions": rows}


def run():
    approved_args = ctc_parser().parse_args([])
    approved_args.phrase_root = PHRASE_ROOT
    require_training_manifest(approved_args)
    for name in ("preflight.json", "behavior_preflight.json"):
        if json.loads((REPORT / name).read_text()).get("status") != "passed":
            raise ValueError(f"required preflight did not pass: {name}")
    if MODELS.exists():
        raise FileExistsError("refusing to overwrite pilot checkpoints")
    MODELS.mkdir(parents=True)
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(.18)
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=False)
    config = UnifiedStreamingCTCConfig(stage1_dim=checkpoint["model_config"]["dim"])
    evaluation_inputs = None
    results = {}
    for seed in SEEDS:
        seed_all(seed)
        initial = UnifiedStreamingCTCHeadV17(config)
        if REUSE_INITIAL_HEADS is None:
            pretrained = pretrain(copy.deepcopy(initial), seed, device)
        else:
            pretrained = copy.deepcopy(initial)
            for arm, head in (("baseline", initial), ("pretrained", pretrained)):
                payload = torch.load(REUSE_INITIAL_HEADS / f"{arm}_{seed}_initial.pth", map_location="cpu", weights_only=False)
                if payload["head_config"] != config.to_dict():
                    raise ValueError("reused initialization config mismatch")
                head.load_state_dict(payload["head_state_dict"], strict=True)
        baseline_state = copy.deepcopy(initial.state_dict())
        for key, value in pretrained.state_dict().items():
            if not key.startswith("blocks."):
                assert torch.equal(value, baseline_state[key]), "transfer touched a non-temporal weight"
        for arm, head in (("baseline", initial), ("pretrained", pretrained)):
            initial_path = MODELS / f"{arm}_{seed}_initial.pth"
            torch.save({"head_config": config.to_dict(), "head_state_dict": head.state_dict()}, initial_path)
            args = ctc_parser().parse_args([])
            args.base = BASE; args.seed = seed; args.epochs = 18
            args.phrase_root = PHRASE_ROOT
            args.initial_head = initial_path
            args.output_dir = MODELS / f"{arm}_{seed}"
            args.embedding_batch_size = 64; args.device = str(device)
            save("status.json", {"state": "training", "phase": arm, "seed": seed, "pid": os.getpid(), "training_started": True})
            result = ctc_run(args)
            results[f"{arm}_{seed}"] = {key: value for key, value in result.items() if key != "history"}
            if evaluation_inputs is None:
                evaluation_inputs = behavior_inputs(device)
            details = behavior(result["output"], evaluation_inputs, device)
            save(f"behavior_{arm}_{seed}.json", details)
            results[f"{arm}_{seed}"]["behavior"] = {key: value for key, value in details.items() if key != "predictions"}
            save("comparison.json", results)
    lines = ["# Matched connected-motion pretraining pilot", "", "Two paired seeds; identical supervised CTC recipe and decoder. Apple Stage1 is frozen in all arms.", "",
             "| Seed | Arm | Local WER | ASLLRP WER | NCSLGR WER | Isolated CTC exact |", "|---|---|---:|---:|---:|---:|"]
    if REUSE_INITIAL_HEADS is not None:
        lines[2] += f" Reused exact initial heads from `{REUSE_INITIAL_HEADS.relative_to(ROOT)}`; motion pretraining was not rerun. Phrase root: `{PHRASE_ROOT.relative_to(ROOT)}`."
    passes = []
    for seed in SEEDS:
        for arm in ("baseline", "pretrained"):
            v = results[f"{arm}_{seed}"]["validation"]; s = v["by_source"]
            lines.append(f"| {seed} | {arm} | {s['local_phrases']['known_wer']:.2%} | {s['asllrp_contiguous']['known_wer']:.2%} | {s['ncslgr_strict']['known_wer']:.2%} | {v['isolated']['exact_accuracy']:.2%} |")
        a, b = [results[f"{arm}_{seed}"]["validation"] for arm in ("baseline", "pretrained")]
        passes.append(b["by_source"]["local_phrases"]["known_wer"] < a["by_source"]["local_phrases"]["known_wer"]
                      and all(b["by_source"][source]["known_wer"] <= a["by_source"][source]["known_wer"] for source in ("asllrp_contiguous", "ncslgr_strict"))
                      and b["isolated"]["exact_accuracy"] >= a["isolated"]["exact_accuracy"] - .01)
    lines += ["", f"Paired WER/retention gates passed: {sum(passes)}/{len(passes)}. No automatic promotion.", "",
              "Unlabeled signer overlap is unknown; source-video separation is not signer separation. Citizen test remained sealed. Frozen Stage1 isolated logits are unchanged; isolated CTC behavior is measured above.",
              "", "## Behavior", "", "| Seed/arm | Synthetic hold exact | Synthetic repeat exact | Real repeat recovery | Matched ASLLRP emission delay median |", "|---|---:|---:|---:|---:|"]
    for key, result in results.items():
        v = result["behavior"]; sources = v["by_source"]
        hold, repeat = sources.get("synthetic_hold", {}), sources.get("synthetic_repeat", {})
        lines.append(f"| {key} | {hold.get('exact', 0)}/{hold.get('samples', 0)} | {repeat.get('exact', 0)}/{repeat.get('samples', 0)} | {v['real_adjacent_repeat_recovered']}/{v['real_adjacent_repeat_references']} | {v['sign_end_relative_first_emission_seconds']['median']} s |")
    lines += ["", "Detailed substitution/deletion/insertion counts, emitted gloss IDs, and duplicate outputs are in `behavior_*.json`. Synthetic hold/repeat probes never select checkpoints. Real held/repeat coverage is not established. Delay is relative to annotated sign end and conditional on correct matches; it is not hardware latency."]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n")
    save("status.json", {"state": "complete", "training_started": True, "pid": os.getpid(), "paired_gates": passes})


def preflight():
    approved_args = ctc_parser().parse_args([])
    approved_args.phrase_root = PHRASE_ROOT
    require_training_manifest(approved_args)
    """Real-device backward passes, strict transfer and signer-split checks before detach."""
    torch.set_num_threads(4); seed_all(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(.18)
    self_check()
    with np.load(REPORT / "motion_windows.npz", allow_pickle=False) as data:
        features, folds, members = data["features"], data["validation"], data["members"]
    # Recheck every cached window, including caches prepared before a quality-rule edit.
    wrists = features[:, :, [0, 21]]
    valid = (wrists[:, 1:, :, 2] > .5) & (wrists[:, :-1, :, 2] > .5)
    jumps = np.linalg.norm(wrists[:, 1:, :, :2] - wrists[:, :-1, :, :2], axis=-1)
    keep = ~((jumps > 1.5) & valid).any(axis=(1, 2))
    features, folds, members = features[keep], folds[keep], members[keep]
    assert np.isfinite(features).all() and np.any(folds) and np.any(~folds)
    assert set(members[folds]).isdisjoint(set(members[~folds]))
    np.savez_compressed(REPORT / "motion_windows.npz", features=features, validation=folds, members=members)
    preparation = json.loads((REPORT / "preparation.json").read_text())
    preparation.update(windows=len(features), train_windows=int((~folds).sum()),
                       validation_windows=int(folds.sum()), postcache_jump_exclusions=int((~keep).sum()),
                       source_clips_used=len(set(members)))
    save("preparation.json", preparation)
    head = UnifiedStreamingCTCHeadV17(); original = copy.deepcopy(head.state_dict())
    motion = MotionEncoder(head).to(device)
    batch = torch.from_numpy(features[:2]).to(device)
    optimizer = torch.optim.AdamW(motion.parameters(), lr=1e-3)
    loss, _, _ = masked_loss(motion, batch, torch.Generator().manual_seed(42))
    loss.backward(); optimizer.step()
    assert torch.isfinite(loss)
    head.blocks.load_state_dict(motion.blocks.cpu().state_dict(), strict=True)
    assert any(not torch.equal(value, original[key]) for key, value in head.state_dict().items() if key.startswith("blocks."))
    assert all(torch.equal(value, original[key]) for key, value in head.state_dict().items() if not key.startswith("blocks."))
    from active.v17.train_streaming_tcn_ctc_v17 import restore_source_frames
    checkpoint = torch.load(BASE, map_location="cpu", weights_only=False)
    assert sorted(checkpoint["label_to_index"].values()) == list(range(100))
    stage1 = SLTStage1V17(Stage1V17Config(**checkpoint["model_config"]))
    stage1.load_state_dict(checkpoint["model_state_dict"], strict=True)
    root = PHRASE_ROOT
    path = next((root / "train/asllrp_contiguous").glob("*.npz"))
    with np.load(path, allow_pickle=False) as data:
        raw = restore_source_frames(data["landmarks"], data["window_source_ranges"])
        targets = tuple(int(x) + 1 for x in data["target_indices"])
    evidence = base.encode(stage1, [base.RawSequence(base.rolling_windows(raw, 4, 8), targets,
                                                    "asllrp_contiguous", str(path))], device, 32, "window")
    collated = base.collate(evidence)
    head.to(device)
    logits = head(collated["evidence"].to(device))
    ctc_loss = nn.CTCLoss(blank=0)(logits.log_softmax(-1).transpose(0, 1), collated["targets"].to(device),
                                collated["lengths"].to(device), collated["target_lengths"].to(device))
    assert torch.isfinite(ctc_loss)
    ctc_loss.backward()
    roles = {}
    for role in ("train", "validation"):
        grouped = {}
        for archive in (root / role).glob("*/*.npz"):
            with np.load(archive, allow_pickle=False) as data:
                m = json.loads(str(data["metadata_json"].item()))
            assert m["role"] == role
            signer = m.get("signer_id") or m.get("participant_id")
            if not signer:
                raise ValueError(f"missing signer identity: {archive}")
            grouped.setdefault(m["source"], set()).add(signer)
        roles[role] = grouped
    for source in roles["train"]:
        assert roles["train"][source].isdisjoint(roles["validation"].get(source, set())), source
    save("preflight.json", {"status": "passed", "device": str(device), "ssl_backward_loss": float(loss.detach()),
         "real_ctc_backward_loss": float(ctc_loss.detach()), "only_temporal_blocks_transferred": True,
         "source_video_split_disjoint": True, "phrase_signer_split_disjoint": True,
         "signers": {role: {key: sorted(value) for key, value in groups.items()} for role, groups in roles.items()},
         "base_sha256": base.sha256(BASE), "official_test_accessed": False})
    print("real motion backward, real CTC backward, strict temporal transfer and signer splits passed", flush=True)


def main():
    global REPORT, MODELS, PHRASE_ROOT, REUSE_INITIAL_HEADS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--model-dir", type=Path, default=MODELS)
    parser.add_argument("--phrase-root", type=Path, default=PHRASE_ROOT)
    parser.add_argument("--reuse-initial-heads", type=Path)
    args = parser.parse_args()
    REPORT, MODELS, PHRASE_ROOT = args.report_dir.resolve(), args.model_dir.resolve(), args.phrase_root.resolve()
    REUSE_INITIAL_HEADS = args.reuse_initial_heads.resolve() if args.reuse_initial_heads else None
    if args.self_check:
        self_check(); return
    if args.prepare:
        prepare(); return
    if args.preflight:
        preflight(); return
    state = "failed"
    try:
        run(); state = "complete"
    except Exception:
        save("status.json", {"state": "failed", "error": traceback.format_exc(), "pid": os.getpid()})
        (REPORT / "RUN_FAILURE.md").write_text("# Training failure\n\n```\n" + traceback.format_exc() + "```\n")
        raise
    finally:
        stamp = datetime.now(ZoneInfo("Asia/Manila")).isoformat(timespec="seconds")
        summary = f"Motion pilot training {state} at {stamp}; reports: {REPORT.relative_to(ROOT)}. No runtime promotion or protected test access."
        history = ROOT / "docs/ground_truth/data-sources/log.md"
        history.write_text(history.read_text().replace("\n---\n", "\n---\n\n## " + stamp + " — motion pilot completion\n\n" + summary + "\n", 1))
        ground = ROOT / "PROJECT_GROUND_TRUTH.md"
        lines = ground.read_text().splitlines()
        lines = ["**Motion pilot 2026-09-21:** " + summary if line.startswith("**Motion pilot 2026-09-21:**") else line for line in lines]
        ground.write_text("\n".join(lines) + "\n")
        subprocess.run([sys.executable, str(ROOT / "scripts/index_large_artifacts_v17.py")], check=False)
        notification = subprocess.run(["/usr/bin/osascript", "-e", "display notification " + json.dumps(f"Connected-motion training {state}. Reports ready.") + ' with title "SLT motion pilot"'], capture_output=True, text=True)
        save("training_notification.json", {"returncode": notification.returncode, "stderr": notification.stderr, "state": state})


if __name__ == "__main__":
    main()
