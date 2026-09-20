#!/usr/bin/env python3
"""Curate complete sign evidence and evaluate recognition versus localization."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

from active.v17.boundary_phase_v17 import KNOWN
from active.v17.live_transition_supervision_v17 import interior_gaps
from active.v17.stage1_window_v17 import normalize_time_window, window_end_times

SOURCE_PATH = ROOT / "artifacts/reports/boundary_phase_v17_20260920/run_experiment.py"
MANIFEST = ROOT / "artifacts/reports/o5s5_citizen100_v17/combined_supervision.json"
SUPERVISION = ROOT / "artifacts/reports/stage2_v17_revisable_v1/supervision.json"
BASE = ROOT / "artifacts/models/stage1_v17_phrase_adapt_reel_v2/best_model.pth"
PHASE_MODEL = ROOT / "artifacts/models/boundary_phase_v17_20260920/final.pth"
GUARD = 1.0 / 30.0
RAW_EDGE_TOLERANCE = 0.060
CONTEXT_SECONDS = 0.10
BATCH = 64


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(name: str, value) -> None:
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_module():
    return runpy.run_path(str(SOURCE_PATH), run_name="clean_boundary_source")


def load_models(source):
    original, labels = source["load_base"]()
    adapted, _ = source["load_base"]()
    package = torch.load(PHASE_MODEL, map_location="cpu", weights_only=False)
    adapted.base.load_state_dict(package["base_model_state_dict"], strict=True)
    adapted.phase_head.load_state_dict(package["phase_head_state_dict"], strict=True)
    return original.eval(), adapted.eval(), labels


def normalize(raw, timestamps, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError("non-positive interval")
    return normalize_time_window(raw, timestamps, end, end - start)[0].astype(np.float16)


def overlap(first, second) -> bool:
    return float(first["start_seconds"]) < float(second["end_seconds"]) and float(first["end_seconds"]) > float(second["start_seconds"])


def build_subset(source, payload, completeness, labels):
    cores, endpoints, events, rows_audit = [], [], [], []
    reasons = Counter()
    for row in payload["rows"]:
        raw, timestamps = source["raw_sequence"](row)
        intervals = sorted(row["intervals"], key=lambda value: (float(value["start_seconds"]), float(value["end_seconds"])))
        item = str(row["source_item_id"])
        with np.load(row["archive_path"], allow_pickle=False) as archive:
            video = json.loads(str(archive["metadata_json"].item()))["video_path"]
        complete_crop = completeness.get(item, {}).get("annotation_crop_complete") is True
        if row["source"] == "o5s5":
            complete_crop = None
        eligible_events = {}
        for position, event in enumerate(intervals):
            start, end = float(event["start_seconds"]), float(event["end_seconds"])
            failures = []
            if row["source"] != "o5s5" and not complete_crop:
                failures.append("source_crop_incomplete")
            if start < timestamps[0] - 1e-9 or end > timestamps[-1] + 1e-9:
                failures.append("outside_raw_clock")
            if any(overlap(event, other) for index, other in enumerate(intervals) if index != position):
                failures.append("overlapping_annotation")
            observed = timestamps[(timestamps >= start - 1e-9) & (timestamps <= end + 1e-9)]
            if len(observed) < 4:
                failures.append("fewer_than_four_raw_target_samples")
            elif observed[0] > start + RAW_EDGE_TOLERANCE or observed[-1] < end - RAW_EDGE_TOLERANCE:
                failures.append("raw_samples_do_not_cover_both_edges")
            label = str(event["label"])
            kind = "known" if label in labels else "unknown"
            identity = f"{item}:event:{position}"
            event_row = dict(identity=identity, item=item, role=row["role"], source=row["source"],
                             signer=row["signer_id"], video=video,
                             archive=row["archive_path"], label=label, kind=kind, position=position,
                             start=start, end=end, duration=end-start, raw_samples=len(observed),
                             complete_crop=complete_crop, eligible=not failures, exclusions=failures)
            events.append(event_row)
            if failures:
                reasons.update(failures)
                continue
            eligible_events[position] = event_row
            try:
                core = normalize(raw, timestamps, start, end)
            except ValueError as error:
                reasons[f"normalization:{error}"] += 1
                event_row["eligible"] = False
                event_row["exclusions"].append(f"normalization:{error}")
                eligible_events.pop(position, None)
                continue
            previous_end = float(intervals[position - 1]["end_seconds"]) if position else float(timestamps[0])
            next_start = float(intervals[position + 1]["start_seconds"]) if position + 1 < len(intervals) else float(timestamps[-1])
            context_start = max(float(timestamps[0]), previous_end, start - CONTEXT_SECONDS)
            context_end = min(float(timestamps[-1]), next_start, end + CONTEXT_SECONDS)
            context = normalize(raw, timestamps, context_start, context_end)
            cores.append(dict(**event_row, target=labels.get(label, -1), core=core, context=context,
                              context_start=context_start, context_end=context_end))
        # Boundary evidence requires a complete ASLLRP crop. O5S5 remains core-only.
        if row["source"] != "o5s5" and complete_crop:
            gaps = interior_gaps(intervals, GUARD)
            for endpoint in window_end_times(timestamps, 0.27, 0.067, include_final=True):
                matching = [index for index, event in enumerate(intervals)
                            if float(event["start_seconds"]) + GUARD <= endpoint <= float(event["end_seconds"]) - GUARD]
                if len(matching) == 1 and matching[0] in eligible_events:
                    event_row = eligible_events[matching[0]]
                    category = "known" if event_row["kind"] == "known" else "unknown"
                    target = labels.get(event_row["label"], -1)
                    event_identity = event_row["identity"]
                elif any(left <= endpoint <= right for left, right in gaps):
                    category, target, event_identity = "gap", -1, None
                else:
                    continue
                try:
                    features = normalize_time_window(raw, timestamps, endpoint, 0.27)[0].astype(np.float16)
                except ValueError:
                    reasons["boundary_window_normalization"] += 1
                    continue
                endpoints.append(dict(item=item, role=row["role"], source=row["source"], signer=row["signer_id"],
                                      video=video, endpoint=endpoint,
                                      category=category, target=target, event_identity=event_identity,
                                      features=features))
        rows_audit.append(dict(item=item, role=row["role"], source=row["source"], complete_crop=complete_crop,
                               annotated_events=len(intervals), eligible_events=len(eligible_events)))
    return cores, endpoints, events, rows_audit, reasons


@torch.inference_mode()
def predict(model, samples, feature_key: str, device):
    output = []
    model.eval()
    for begin in range(0, len(samples), BATCH):
        batch = samples[begin:begin+BATCH]
        values = torch.from_numpy(np.stack([row[feature_key] for row in batch]).astype(np.float32)).to(device)
        gloss, phase = model(values)
        gloss_probability = gloss.softmax(1)
        phase_probability = phase.softmax(1)
        for row, gl, ph, gc, gp in zip(batch, gloss.argmax(1).cpu(), phase.argmax(1).cpu(),
                                       gloss_probability.max(1).values.cpu(), phase_probability[:, KNOWN].cpu()):
            output.append(dict(identity=row.get("identity"), item=row["item"], prediction=int(gl),
                               phase_prediction=int(ph), gloss_confidence=float(gc), known_probability=float(gp)))
    return output


def group_metrics(samples, predictions, feature_key):
    grouped = {}
    for role in ("train", "validation"):
        for source_name in sorted({row["source"] for row in samples}):
            chosen = [(row, prediction) for row, prediction in zip(samples, predictions)
                      if row["role"] == role and row["source"] == source_name and row["kind"] == "known"]
            if feature_key == "context" and source_name == "o5s5":
                continue
            if not chosen:
                continue
            correct = sum(prediction["prediction"] == row["target"] for row, prediction in chosen)
            grouped[f"{role}|{source_name}"] = dict(correct=correct, total=len(chosen), accuracy=correct/len(chosen))
    return grouped


def threshold(scores, truth):
    values = sorted(zip(scores, truth), reverse=True)
    positives, negatives = sum(truth), len(truth)-sum(truth)
    tp = fp = 0
    best = (-1.0, 1.1)
    index = 0
    while index < len(values):
        score = values[index][0]
        while index < len(values) and values[index][0] == score:
            tp += values[index][1]
            fp += 1-values[index][1]
            index += 1
        balanced = ((tp/max(1, positives)) + ((negatives-fp)/max(1, negatives))) / 2
        if balanced > best[0]:
            best = (balanced, score)
    return best[1], best[0]


def binary_metrics(scores, truth, cutoff):
    predicted = [value >= cutoff for value in scores]
    tp = sum(a and b for a, b in zip(truth, predicted)); fn = sum(a and not b for a, b in zip(truth, predicted))
    fp = sum(not a and b for a, b in zip(truth, predicted)); tn = sum(not a and not b for a, b in zip(truth, predicted))
    return dict(threshold=cutoff, true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn,
                known_recall=tp/max(1,tp+fn), nonknown_recall=tn/max(1,tn+fp),
                balanced_accuracy=((tp/max(1,tp+fn))+(tn/max(1,tn+fp)))/2)


def evaluate(source, original, adapted, labels, cores, endpoints):
    device = torch.device("mps")
    original.to(device); adapted.to(device)
    core_results = {}
    stored_predictions = {}
    for name, model in (("starting_encoder", original), ("adapted_encoder", adapted)):
        for feature in ("core", "context"):
            prediction = predict(model, cores, feature, device)
            core_results[f"{name}|{feature}"] = group_metrics(cores, prediction, feature)
            stored_predictions[f"{name}|{feature}"] = prediction
    endpoint_prediction = predict(adapted, endpoints, "features", device)
    train_pairs = [(row, prediction) for row, prediction in zip(endpoints, endpoint_prediction) if row["role"] == "train"]
    validation_pairs = [(row, prediction) for row, prediction in zip(endpoints, endpoint_prediction) if row["role"] == "validation"]
    train_truth = [row["category"] == "known" for row, _ in train_pairs]
    phase_cutoff, phase_train = threshold([prediction["known_probability"] for _, prediction in train_pairs], train_truth)
    confidence_cutoff, confidence_train = threshold([prediction["gloss_confidence"] for _, prediction in train_pairs], train_truth)
    binary = dict(
        phase_known=dict(train_balanced_accuracy=phase_train,
                         validation=binary_metrics([p["known_probability"] for _,p in validation_pairs],
                                                   [r["category"]=="known" for r,_ in validation_pairs], phase_cutoff)),
        gloss_confidence=dict(train_balanced_accuracy=confidence_train,
                              validation=binary_metrics([p["gloss_confidence"] for _,p in validation_pairs],
                                                        [r["category"]=="known" for r,_ in validation_pairs], confidence_cutoff)),
    )
    phase_confusion = np.zeros((3,3), dtype=int)
    category_index = {"known":0, "unknown":1, "gap":2}
    for row, prediction in validation_pairs:
        phase_confusion[category_index[row["category"]], prediction["phase_prediction"]] += 1
    known_pairs = [(row,prediction) for row,prediction in validation_pairs if row["category"]=="known"]
    known_correct = sum(prediction["prediction"] == row["target"] for row,prediction in known_pairs)
    by_event = defaultdict(list)
    for row, prediction in validation_pairs:
        if row["event_identity"]:
            by_event[row["event_identity"]].append((row,prediction))
    event_rows = {row["identity"]:row for row in cores if row["role"]=="validation" and row["kind"]=="known" and row["source"]!="o5s5"}
    localized = []
    for identity, event in event_rows.items():
        detections = [(row,prediction) for row,prediction in by_event.get(identity,[])
                      if prediction["known_probability"] >= phase_cutoff and prediction["prediction"] == event["target"]]
        first = min((row["endpoint"] for row,_ in detections), default=None)
        localized.append(dict(identity=identity, detected=first is not None,
                              first_endpoint=first, start=event["start"], end=event["end"],
                              fraction=None if first is None else (first-event["start"])/(event["end"]-event["start"])))
    result = dict(core=core_results, binary_rejection=binary,
                  endpoint_validation=dict(samples=len(validation_pairs), phase_confusion=phase_confusion.tolist(),
                                           phase_accuracy=float(np.trace(phase_confusion)/max(1,phase_confusion.sum())),
                                           known_gloss_correct=known_correct, known_gloss_total=len(known_pairs),
                                           known_gloss_accuracy=known_correct/max(1,len(known_pairs))),
                  event_localization=dict(events=len(localized), detected=sum(row["detected"] for row in localized), rows=localized))
    return result, stored_predictions, endpoint_prediction


def make_viewer(cores, core_prediction, endpoints, endpoint_prediction, metrics):
    core_pairs = list(zip(cores, core_prediction))
    endpoint_pairs = list(zip(endpoints, endpoint_prediction))
    selected = []
    for role, source_name in (("validation","asllrp_other_ctc"),("validation","o5s5")):
        failures = [(r,p) for r,p in core_pairs if r["role"]==role and r["source"]==source_name and r["kind"]=="known" and p["prediction"]!=r["target"]]
        controls = [(r,p) for r,p in core_pairs if r["role"]==role and r["source"]==source_name and r["kind"]=="known" and p["prediction"]==r["target"]]
        selected.extend(("whole-sign failure",)+pair for pair in failures[:2])
        selected.extend(("whole-sign correct control",)+pair for pair in controls[:1])
    cutoff = metrics["binary_rejection"]["phase_known"]["validation"]["threshold"]
    false_accept = [(r,p) for r,p in endpoint_pairs if r["role"]=="validation" and r["category"]!="known" and p["known_probability"]>=cutoff]
    false_reject = [(r,p) for r,p in endpoint_pairs if r["role"]=="validation" and r["category"]=="known" and p["known_probability"]<cutoff]
    selected.extend(("non-known false accept",)+pair for pair in false_accept[:3])
    selected.extend(("known false reject",)+pair for pair in false_reject[:3])
    media = HERE / "media"; media.mkdir(exist_ok=True)
    cards = []; sheet = Image.new("RGB", (1200, max(1,len(selected))*225), "white"); draw = ImageDraw.Draw(sheet)
    for index, (reason, row, prediction) in enumerate(selected):
        if "start" in row:
            start, end = max(0,row["start"]-.35), row["end"]+.35
        else:
            start, end = max(0,row["endpoint"]-.45), row["endpoint"]+.20
        source_video = ROOT / row["video"]
        clip = media / f"example_{index+1:02d}.mp4"
        subprocess.run(["ffmpeg","-v","error","-y","-ss",str(start),"-i",str(source_video),"-t",str(end-start),
                        "-an","-vf","scale=640:-2","-c:v","libx264","-crf","24",str(clip)], check=True)
        title = f"{reason}: {row.get('label',row.get('category'))} → class {prediction['prediction']}"
        cards.append(f'<article><h2>{html.escape(title)}</h2><p>{html.escape(row["item"])}; known probability {prediction["known_probability"]:.3f}; gloss confidence {prediction["gloss_confidence"]:.3f}</p><video controls loop src="media/{clip.name}"></video><p><a href="{html.escape(os.path.relpath(source_video,HERE))}">Full source</a></p></article>')
        cap=cv2.VideoCapture(str(source_video)); draw.text((8,index*225+4),title,fill="black")
        for column, second in enumerate(np.linspace(start,end,6,endpoint=False)):
            cap.set(cv2.CAP_PROP_POS_MSEC,float(second)*1000); ok,frame=cap.read()
            if not ok: continue
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); image.thumbnail((195,170))
            sheet.paste(image,(column*200,index*225+25)); draw.text((column*200+4,index*225+198),f"{second:.3f}s",fill="black")
        cap.release()
    sheet.save(HERE/"contact_sheet.jpg")
    (HERE/"videos.html").write_text('<!doctype html><meta charset="utf-8"><title>Clean evidence review</title><style>body{font:16px system-ui;max-width:1000px;margin:30px auto}article{padding:18px;margin:20px 0;border:1px solid #bbb}video{width:700px;max-height:430px}</style><h1>Strict whole-sign and boundary diagnostic examples</h1><p>Selected failures and controls. Dataset annotations are shown through timing and labels in the accompanying manifest; this is a review pack, not an expert re-annotation.</p>'+''.join(cards),encoding="utf-8")
    return len(selected)


def report(metrics, audit):
    core = metrics["core"]
    lines = ["# Strict whole-sign recognition and boundary audit", "",
             "This report uses complete, non-overlapping sign intervals with at least four raw target observations covering both annotated edges. ASLLRP rows previously marked as crop-incomplete are excluded. O5S5 contributes exact sign cores only; its surrounding context is not treated as fully annotated.", "",
             "## Curated evidence", "",
             f"Eligible sign cores: **{audit['eligible_cores']:,}** of {audit['annotated_events']:,} annotated events. Known: **{audit['eligible_known']:,}**; unknown: **{audit['eligible_unknown']:,}**. Boundary windows: **{audit['boundary_windows']:,}** from complete ASLLRP crops.", "",
             "## Whole-sign recognition", "", "| Model/input | Split/source | Correct | Accuracy |", "| --- | --- | ---: | ---: |"]
    for family, groups in core.items():
        for group, value in groups.items():
            lines.append(f"| {family} | {group} | {value['correct']}/{value['total']} | {value['accuracy']*100:.2f}% |")
    phase = metrics["binary_rejection"]["phase_known"]["validation"]
    confidence = metrics["binary_rejection"]["gloss_confidence"]["validation"]
    endpoint = metrics["endpoint_validation"]; localization = metrics["event_localization"]
    lines += ["", "## Boundary and rejection evidence", "",
              f"A threshold selected only on training windows gives **{phase['balanced_accuracy']*100:.2f}%** held-out balanced accuracy for the phase head's known/not-known decision (known recall {phase['known_recall']*100:.2f}%, non-known recall {phase['nonknown_recall']*100:.2f}%). Gloss confidence alone gives **{confidence['balanced_accuracy']*100:.2f}%**.", "",
              f"The original three-way endpoint phase accuracy is **{endpoint['phase_accuracy']*100:.2f}%**. On endpoints inside eligible known cores, gloss top-1 is **{endpoint['known_gloss_correct']}/{endpoint['known_gloss_total']} = {endpoint['known_gloss_accuracy']*100:.2f}%**. Correct gloss plus the train-selected phase threshold localizes **{localization['detected']}/{localization['events']}** held-out ASLLRP known events.", "",
              "Annotation gaps remain named gaps. This audit does not certify them as physical transitions. See `metrics.json`, `curated_manifest.json`, and `videos.html` for full evidence.", "",
              "No model was trained or promoted. Citizen test remained sealed.", ""]
    return "\n".join(lines)


def precheck():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    source=source_module(); original,adapted,labels=load_models(source)
    payload=json.loads(MANIFEST.read_text()); completeness=json.loads(SUPERVISION.read_text())["items"]
    if payload.get("citizen_test_accessed") is not False or len(labels)!=100:
        raise ValueError("input contract changed")
    row=next(row for row in payload["rows"] if row["role"]=="train" and row["source"]=="asllrp_other_ctc")
    raw,timestamps=source["raw_sequence"](row); event=next(event for event in row["intervals"] if event["label"] in labels)
    features=normalize(raw,timestamps,float(event["start_seconds"]),float(event["end_seconds"]))
    original.to("mps")
    with torch.inference_mode():
        outputs=original(torch.from_numpy(features.astype(np.float32))[None].to("mps"))
    if outputs[0].shape!=(1,100) or outputs[1].shape!=(1,3): raise ValueError("output mismatch")
    code=Path(__file__).resolve()
    result=dict(status="passed",rows=len(payload["rows"]),labels=len(labels),output_shapes=[[1,100],[1,3]],
                base_sha256=sha(BASE),phase_model_sha256=sha(PHASE_MODEL),manifest_sha256=sha(MANIFEST),
                supervision_sha256=sha(SUPERVISION),code_sha256=sha(code),citizen_test_accessed=False)
    save("precheck.json",result)
    (HERE/"PRECHECK_REPORT.md").write_text("# Strict-subset precheck\n\nPassed input hashes, 100-label contract, real raw-core normalization and MPS model outputs. The run is evaluation-only and cannot access Citizen test.\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


def worker():
    started=datetime.now(timezone.utc).isoformat(); status="failed"
    try:
        pre=json.loads((HERE/"precheck.json").read_text())
        for path,key in ((BASE,"base_sha256"),(PHASE_MODEL,"phase_model_sha256"),(MANIFEST,"manifest_sha256"),(SUPERVISION,"supervision_sha256"),(Path(__file__).resolve(),"code_sha256")):
            if sha(path)!=pre[key]: raise RuntimeError(f"prechecked input changed: {path}")
        source=source_module(); original,adapted,labels=load_models(source)
        payload=json.loads(MANIFEST.read_text()); completeness=json.loads(SUPERVISION.read_text())["items"]
        cores,endpoints,events,rows_audit,reasons=build_subset(source,payload,completeness,labels)
        metrics,stored,endpoint_prediction=evaluate(source,original,adapted,labels,cores,endpoints)
        audit=dict(annotated_events=len(events),eligible_cores=len(cores),eligible_known=sum(row["kind"]=="known" for row in cores),
                   eligible_unknown=sum(row["kind"]=="unknown" for row in cores),boundary_windows=len(endpoints),
                   exclusions=dict(reasons),rows=rows_audit,citizen_test_accessed=False)
        serial_cores=[{key:value for key,value in row.items() if key not in {"core","context"}} for row in cores]
        serial_endpoints=[{key:value for key,value in row.items() if key!="features"} for row in endpoints]
        save("curated_manifest.json",dict(events=events,eligible_cores=serial_cores,boundary_windows=serial_endpoints,audit=audit))
        save("metrics.json",metrics)
        examples=make_viewer(cores,stored["adapted_encoder|core"],endpoints,endpoint_prediction,metrics)
        (HERE/"REPORT.md").write_text(report(metrics,audit),encoding="utf-8")
        save("verification.json",dict(status="passed",annotated_events=len(events),eligible_cores=len(cores),
                                      boundary_windows=len(endpoints),examples=examples,new_training=False,
                                      citizen_test_accessed=False,model_promoted=False))
        subprocess.run([sys.executable,str(ROOT/"scripts/index_large_artifacts_v17.py")],cwd=ROOT,check=True)
        status="completed"
    except BaseException:
        (HERE/"FAILURE.md").write_text("# Strict-subset audit failed\n\n```text\n"+traceback.format_exc()+"```\n",encoding="utf-8")
        traceback.print_exc()
    finally:
        save("completion.json",dict(status=status,started_at=started,finished_at=datetime.now(timezone.utc).isoformat()))
        subprocess.run(["osascript","-e",'on run argv\ndisplay notification (item 1 of argv) with title "SLT experiment"\nend run',
                        "Strict whole-sign and boundary audit "+status+". See "+str(HERE)],capture_output=True,timeout=15,check=False)
    if status!="completed": raise SystemExit(1)


def launch():
    if (HERE/"launch.json").exists(): raise RuntimeError("experiment already launched")
    if json.loads((HERE/"precheck.json").read_text()).get("status")!="passed": raise RuntimeError("precheck did not pass")
    with (HERE/"process.log").open("ab") as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--worker"],cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save("launch.json",dict(pid=child.pid,launched_at=datetime.now(timezone.utc).isoformat(),notification_on_exit=True,polling=False))
    print(f"Launched {child.pid} with one exit notification; no polling.")


if __name__=="__main__":
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(); parser.add_argument("--precheck",action="store_true"); parser.add_argument("--worker",action="store_true"); parser.add_argument("--launch",action="store_true"); args=parser.parse_args()
    if args.precheck: precheck()
    elif args.worker: worker()
    elif args.launch: launch()
    else: parser.error("choose --precheck or --launch")
