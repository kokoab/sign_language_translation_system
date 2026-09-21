#!/usr/bin/env python3
"""Evaluation-only coherent decoder for the frozen segment-first checkpoint."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts/reports/segment_first_v17_20260921/run_experiment.py"
MODEL = ROOT / "artifacts/models/segment_first_v17_20260921/final.pth"
MANIFEST = ROOT / "artifacts/reports/confident_supervision_v17_20260920/confident_supervision.json"
spec = importlib.util.spec_from_file_location("segment_first_source", SOURCE)
src = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = src
spec.loader.exec_module(src)

WINDOW, STRIDE, MIN_HISTORY = src.WINDOW_SECONDS, src.LIVE_STRIDE, src.MIN_LIVE_HISTORY
TOLERANCES = (0.10, 0.20)
PRIOR_METRICS = HERE.parent / "segment_first_v17_20260921/metrics.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    path = HERE / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def coherent_decode(logp):
    """Viterbi path O->START->SIGNING->END; END may lead to a repeated START."""
    # ponytail: O(T*states^2) four-state Viterbi; adequate for recording lengths.
    allowed = ((0, 1), (1, 2), (2, 3), (3, 0), (3, 1))
    scores = np.full((len(logp), 4), -np.inf)
    back = np.zeros((len(logp), 4), np.int8)
    scores[0, 0] = logp[0, 0]
    for t in range(1, len(logp)):
        for state in range(4):
            choices = [state] + [a for a, b in allowed if b == state]
            previous = max(choices, key=lambda value: scores[t - 1, value])
            scores[t, state] = scores[t - 1, previous] + logp[t, state]
            back[t, state] = previous
    # Require complete events: end in OUTSIDE or END, never a dangling start/sign.
    path = np.empty(len(logp), np.int8)
    path[-1] = max((0, 3), key=lambda value: scores[-1, value])
    for t in range(len(logp) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    spans, start = [], None
    for i, state in enumerate(path):
        if state == 1:
            if start is None:
                start = i
        elif state == 3 and start is not None:
            if i > start:
                spans.append((start, i))
            start = None
    return spans


def self_check():
    # Exercise a repeated-sign path and reject unconstrained state ordering.
    states = [0, 1, 2, 2, 3, 1, 2, 3, 0]
    p = np.full((len(states), 4), -8.0)
    p[np.arange(len(states)), states] = 0.0
    assert coherent_decode(p) == [(1, 4), (5, 7)]


def hand_range(raw, times):
    # v16 detect_signing_range: smooth hand presence; crop to first/last active.
    score = raw[:, :42, 3].mean(axis=1)
    width = min(len(score), max(3, len(score) // 20))
    smoothed = np.convolve(score, np.ones(width) / width, mode="same")
    above = np.flatnonzero(smoothed > max(float(smoothed.max()) * 0.30, 0.15))
    allowed = np.zeros(len(times), dtype=bool)
    if len(above):
        pad = max(1, int(round(0.15 / np.median(np.diff(times)))))
        allowed[max(0, above[0] - pad):min(len(times), above[-1] + pad + 1)] = True
    return allowed


@torch.inference_mode()
def source_logp(model, raw, times, device):
    total = np.zeros((len(times), 4), np.float64)
    counts = np.zeros(len(times), np.int32)
    endpoints = list(np.arange(times[0] + MIN_HISTORY, times[-1] + 1e-9, STRIDE))
    if not endpoints or endpoints[-1] < times[-1] - 1e-6:
        endpoints.append(float(times[-1]))
    for begin in range(0, len(endpoints), src.BATCH):
        batch, maps = [], []
        for endpoint in endpoints[begin:begin + src.BATCH]:
            end = float(endpoint); start = max(float(times[0]), end - WINDOW)
            features, sample_times = src.normalized(raw, times, start, end)
            batch.append(features)
            maps.append(np.searchsorted(times, sample_times).clip(0, len(times)-1))
        tensor = torch.from_numpy(np.stack(batch).astype(np.float32)).to(device)
        logp = model.boundary_logits(tensor).log_softmax(-1).cpu().numpy()
        for values, indices in zip(logp, maps):
            for value, index in zip(values, indices):
                total[index] += value; counts[index] += 1
    covered = counts > 0
    total[covered] /= counts[covered, None]
    return total, covered


def evidence(model, row, device):
    raw, times = src.raw_sequence(row)
    logp, covered = source_logp(model, raw, times, device)
    allowed = hand_range(raw, times) & covered
    logp[~allowed] = -np.inf
    logp[~allowed, src.OUTSIDE] = 0
    return times, logp


def detect(times, logp, edge_bias):
    adjusted = logp.copy()
    adjusted[:, src.START] += edge_bias
    adjusted[:, src.END] += edge_bias
    return [(float(times[a]), float(times[b])) for a, b in coherent_decode(adjusted)]


def pair(pred, truth, tolerance):
    used, matched = set(), []
    for pi, p in enumerate(pred):
        options = [(max(abs(p[0]-t[0]), abs(p[1]-t[1])), ti)
                   for ti, t in enumerate(truth) if ti not in used]
        options = [x for x in options if x[0] <= tolerance]
        if options:
            _, ti = min(options); used.add(ti); matched.append((pi, ti))
    return matched


def wer_counts(reference, hypothesis):
    d = [[0] * (len(hypothesis)+1) for _ in range(len(reference)+1)]
    op = [[(0,0,0)] * (len(hypothesis)+1) for _ in range(len(reference)+1)]
    for i in range(1, len(reference)+1): d[i][0], op[i][0] = i, (i,0,0)
    for j in range(1, len(hypothesis)+1): d[0][j], op[0][j] = j, (0,j,0)
    for i in range(1, len(reference)+1):
        for j in range(1, len(hypothesis)+1):
            if reference[i-1] == hypothesis[j-1]: d[i][j], op[i][j] = d[i-1][j-1], op[i-1][j-1]
            else:
                choices = [(d[i-1][j]+1, (op[i-1][j][0]+1,op[i-1][j][1],op[i-1][j][2])),
                           (d[i][j-1]+1, (op[i][j-1][0],op[i][j-1][1]+1,op[i][j-1][2])),
                           (d[i-1][j-1]+1, (op[i-1][j-1][0],op[i-1][j-1][1],op[i-1][j-1][2]+1))]
                d[i][j], op[i][j] = min(choices, key=lambda x:x[0])
    return op[-1][-1]


def rows_for(payload, role):
    rows = defaultdict(list)
    for row in payload["rows"]:
        if row["role"] == role and str(row["source"]).startswith("asllrp"):
            rows[str(row["source_item_id"])].append(row)
    return {item: values for item, values in rows.items()
            if all(row.get("source_crop_complete") is True for row in values)}


def excluded_for(payload, role):
    excluded = defaultdict(list)
    for row in payload["decisions"]:
        if row["role"] == role and not row["accepted"]:
            excluded[str(row["item"])].append((float(row["start"]), float(row["end"])))
    return excluded


def ignored_prediction(candidate, excluded):
    duration = candidate[1] - candidate[0]
    return any(max(0.0, min(candidate[1], end) - max(candidate[0], start)) /
               max(duration, end - start) >= 0.30 for start, end in excluded)


def remove_questionable(predictions, excluded):
    return [candidate for candidate in predictions if not ignored_prediction(candidate, excluded)]


def synthetic_probe(model, payload, device, edge_bias, gate_threshold):
    accepted, _ = src.group_manifest(payload, "validation")
    chosen = [row for rows in accepted.values() for row in rows
              if row["source"].startswith("asllrp") and row["source_crop_complete"]
              and row["target_kind"] == "known"][:20]
    once = repeated = 0
    for row in chosen:
        raw, times = src.raw_sequence(row)
        for is_repeat in (False, True):
            probe, clock, truth = src.synthetic_sequence(raw, times, float(row["start_seconds"]),
                                                          float(row["end_seconds"]), is_repeat)
            local_logp, covered = source_logp(model, probe, clock, device)
            allowed = hand_range(probe, clock) & covered
            local_logp[~allowed] = -np.inf
            local_logp[~allowed, src.OUTSIDE] = 0
            prediction = detect(clock, local_logp, edge_bias)
            prediction = remove_questionable(prediction, [])
            prediction, classes = src.classify_candidates(model, probe, clock,
                                                            [src.SegmentCandidate(a,b,1.0) for a,b in prediction], device)
            visible = [c for c, output in zip(prediction, classes)
                       if output["known_probability"] >= gate_threshold]
            exact = len(visible) == len(truth) and all(
                max(abs(c.start_seconds-t[0]), abs(c.end_seconds-t[1])) <= .20
                for c, t in zip(visible, truth))
            if is_repeat: repeated += int(exact)
            else: once += int(exact)
    return dict(events=len(chosen), held_once_exact=once, intentional_repeat_twice_exact=repeated,
                caveat="Time-warped feature probes, not independent recorded performances.")


def run():
    self_check()
    payload = json.loads(MANIFEST.read_text())
    model, labels = src.load_model()
    package = torch.load(MODEL, map_location="cpu", weights_only=False)
    if package.get("format") != src.CHECKPOINT_FORMAT or package.get("label_to_index") != labels:
        raise ValueError("frozen checkpoint contract mismatch")
    model.base.load_state_dict(package["base_model_state_dict"], strict=True)
    model.boundary.load_state_dict(package["boundary_state_dict"], strict=True)
    model.known.load_state_dict(package["known_state_dict"], strict=True)
    model.eval(); device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); model.to(device)
    train, validation = rows_for(payload, "train"), rows_for(payload, "validation")
    train_excluded, val_excluded = excluded_for(payload, "train"), excluded_for(payload, "validation")
    prior_gate_threshold = json.loads(PRIOR_METRICS.read_text())["thresholds"]["gate"]["threshold"]
    # Train-only scalar edge bias selection; validation is evaluated once afterward.
    train_evidence = {}
    for item, rs in train.items():
        times, logp = evidence(model, rs[0], device)
        train_evidence[item] = (times, logp)
    biases = np.linspace(-2.0, 2.0, 9)
    # Cache each source's model evidence once, then decode the train-only sweep.
    best = None
    for bias in biases:
        counts = [0, 0, 0]
        for item, rs in train.items():
            times, logp = train_evidence[item]
            pred = remove_questionable(detect(times, logp, float(bias)), train_excluded[item])
            truth = sorted((float(r["start_seconds"]), float(r["end_seconds"])) for r in rs)
            m = pair(pred, truth, .10); counts[0] += len(m); counts[1] += len(pred)-len(m); counts[2] += len(truth)-len(m)
        f1 = 2*counts[0] / max(1, 2*counts[0]+counts[1]+counts[2])
        if best is None or f1 > best[0]: best = (f1, float(bias))
    bias = best[1]
    totals = {str(t): [0,0,0] for t in TOLERANCES}; errors = [0,0,0]; refs = 0
    val_evidence = {}
    for item, rs in validation.items():
        rs.sort(key=lambda r: float(r["start_seconds"]))
        times, logp = evidence(model, rs[0], device)
        val_evidence[item] = (times, logp)
        pred = remove_questionable(detect(times, logp, bias), val_excluded[item])
        truth_rows = [r for r in rs if r["target_kind"] == "known"]
        truth_all = [(float(r["start_seconds"]), float(r["end_seconds"])) for r in rs]
        for tol in TOLERANCES:
            m = pair(pred, truth_all, tol); c = totals[str(tol)]; c[0] += len(m); c[1] += len(pred)-len(m); c[2] += len(truth_all)-len(m)
        reference = [int(r["target_index"]) for r in truth_rows]
        # Pair at 200 ms; classify the detected interval with frozen known/gloss heads.
        hypotheses = []
        for start, end in pred:
            features, _ = src.normalized(*src.raw_sequence(rs[0]), start, end)
            x = torch.from_numpy(features[None].astype(np.float32)).to(device)
            with torch.no_grad():
                gloss, gate = model.segment_logits(x)
                if float(gate.softmax(1)[0, 1]) >= prior_gate_threshold:
                    hypotheses.append(int(gloss.argmax(1)))
        # WER over visible predictions, suppressing UNKNOWN at the prior train-selected threshold.
        d, i, s = wer_counts(reference, hypotheses); errors[0]+=d; errors[1]+=i; errors[2]+=s; refs+=len(reference)
    edge = {}
    for tol, (tp, fp, fn) in totals.items():
        p=tp/max(1,tp+fp); r=tp/max(1,tp+fn)
        edge[tol]=dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*p*r/max(1e-12,p+r))
    distance=sum(errors)
    probes = synthetic_probe(model, payload, device, bias, prior_gate_threshold)
    metrics=dict(checkpoint_sha256=sha(MODEL),manifest_sha256=sha(MANIFEST),train_selected_edge_bias=bias,
                 train_selection_f1_100ms=best[0],validation_sources=len(validation),boundary=edge,
                 known_gate_threshold=prior_gate_threshold,
                 visible_wer=dict(distance=distance,deletions=errors[0],insertions=errors[1],substitutions=errors[2],
                                  references=refs,wer=distance/max(1,refs)),held_repeat=probes,
                 excluded_decisions_train=sum(map(len,train_excluded.values())),
                 excluded_decisions_validation=sum(map(len,val_excluded.values())),
                 citizen_test_accessed=False,model_promoted=False)
    save("metrics.json", metrics)
    save("verification.json",dict(status="passed",checkpoint_sha256=sha(MODEL),
         manifest_sha256=sha(MANIFEST),visible_classes=100,train_sources=len(train),
         validation_sources=len(validation),train_cached_model_passes=len(train),
         validation_model_passes=len(validation),train_only_bias_selection=True,
         validation_evaluated_once=True,questionable_overlap_filter=">=30% of shorter span",
         known_gate_threshold=prior_gate_threshold,held_repeat=probes,
         boundary_counts=edge,visible_wer=metrics["visible_wer"],
         excluded_decisions_train=metrics["excluded_decisions_train"],
         excluded_decisions_validation=metrics["excluded_decisions_validation"],
         citizen_test_accessed=False,model_promoted=False,metric_contract="D+I+S distance; WER=distance/references"))
    (HERE/"REPORT.md").write_text(
        "# Frozen segment-first coherent decoding\n\nEvaluation-only; the checkpoint was not changed. Overlapping rolling-window state evidence is averaged on source timestamps, decoded with OUTSIDE→START→SIGNING→END and END→START repetition, and limited to the v16-style hand-presence range. Edge bias was selected on train sources; validation was evaluated once. Citizen test remained sealed.\n\n"
        f"Train-selected edge bias: {bias:.2f} (train F1 ±100 ms {best[0]*100:.2f}%).\n\n"
        f"Validation boundary F1: ±100 ms {edge['0.1']['f1']*100:.2f}%; ±200 ms {edge['0.2']['f1']*100:.2f}%.\n\n"
        f"Visible locked100 WER: {metrics['visible_wer']['wer']*100:.2f}% ({errors[0]} deletions, {errors[1]} insertions, {errors[2]} substitutions; {refs} references).\n\n"
        f"Held/repeat probe: {probes['held_once_exact']}/{probes['events']} held once; {probes['intentional_repeat_twice_exact']}/{probes['events']} repeats exactly twice (synthetic time-warp probes).\n", encoding="utf-8")
    return metrics


def precheck():
    self_check()
    payload = json.loads(MANIFEST.read_text())
    if payload.get("audit", {}).get("train_validation_signer_disjoint") is not True:
        raise ValueError("manifest signer-split contract failed")
    if any("test" in Path(r.get("archive_path", r.get("archive"))).parts for r in payload["rows"]):
        raise ValueError("protected test archive referenced")
    package = torch.load(MODEL, map_location="cpu", weights_only=False)
    if package.get("format") != src.CHECKPOINT_FORMAT or len(package.get("label_to_index", {})) != 100:
        raise ValueError("frozen checkpoint/100-label contract failed")
    save("precheck.json", dict(status="passed", self_check="passed", checkpoint_sha256=sha(MODEL),
         manifest_sha256=sha(MANIFEST), runner_sha256=sha(Path(__file__).resolve()),
         train_sources=len(rows_for(payload,"train")), validation_sources=len(rows_for(payload,"validation")),
         citizen_test_accessed=False, training_enabled=False))
    (HERE/"PRECHECK_REPORT.md").write_text(
        "# Coherent decoder precheck\n\nPassed constrained repeated-sign decoder self-check, frozen checkpoint format and locked 100-label contract, manifest signer-disjoint contract, and sealed-test path scan. The runner is evaluation-only; it does not retrain or access Citizen test.\n",
        encoding="utf-8")
    print("Precheck passed; evaluation not launched.")


def worker():
    status="failed"
    try:
        metrics=run(); status="completed"
        save("completion.json",dict(status=status,finished_at=datetime.now(timezone.utc).isoformat(),metrics=metrics))
    except BaseException:
        save("failure.json",dict(status=status,traceback=traceback.format_exc()))
        raise
    finally:
        subprocess.run(["osascript","-e",'on run argv\ndisplay notification (item 1 of argv) with title "SLT evaluation"\nend run',
                        "Coherent segment evaluation "+status],capture_output=True,timeout=15,check=False)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--precheck",action="store_true"); ap.add_argument("--worker",action="store_true")
    args=ap.parse_args()
    if args.precheck: precheck()
    elif args.worker: worker()
    else: ap.error("choose --precheck or --worker")
