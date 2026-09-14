#!/usr/bin/env python3
"""Audit the public O5S5 and RWTH-BOSTON-104 continuous-ASL downloads."""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET


ROOT = Path("data/local/open_asl_alternatives_20260913")
REPORT = Path("artifacts/reports/open_asl_alternatives_20260913/corpus_quality_audit.json")
O5S5_CSV = REPORT.with_name("o5s5_locked100_candidate_events.csv")
RWTH_CSV = REPORT.with_name("rwth_boston104_sentences.csv")
MANIFEST = Path("active/v17/citizen100_manifest.json")
O5S5_PAIRS = {
    "002": "211016_O5S5_002_N_LG.mov",
    "007": "211020_O5S5_007_N_JAH.mp4",
    "025": "211118_O5S5_025_N_CK.mov.mp4",
    "026": "211119_O5S5_026_N_DR.mov",
    "027": "211119_O5S5_027_N_LR.mov",
    "029": "211201_O5S5_029_N_RD.MOV",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def probe(path):
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of", "json", str(path),
    ]))["streams"][0]
    assert int(stream["nb_read_frames"]) > 0
    return stream


def eaf_events(path):
    root = ET.parse(path).getroot()
    times = {node.get("TIME_SLOT_ID"): int(node.get("TIME_VALUE"))
             for node in root.findall("./TIME_ORDER/TIME_SLOT") if node.get("TIME_VALUE")}
    events = []
    tier_counts = {}
    participant = ""
    for tier in root.findall("TIER"):
        tier_id = tier.get("TIER_ID", "")
        if tier_id not in {"RightHand_IDg", "LeftHand_IDg"}:
            continue
        participant = tier.get("PARTICIPANT", participant)
        count = 0
        for annotation in tier.findall("./ANNOTATION/ALIGNABLE_ANNOTATION"):
            value = (annotation.findtext("ANNOTATION_VALUE") or "").strip()
            if not value:
                continue
            events.append({"tier": tier_id, "value": value,
                           "start_ms": times[annotation.get("TIME_SLOT_REF1")],
                           "end_ms": times[annotation.get("TIME_SLOT_REF2")]})
            count += 1
        tier_counts[tier_id] = count
    assert events and participant
    return participant, tier_counts, events


def corpus_rows(path):
    rows = []
    for recording in ET.parse(path).getroot().findall("recording"):
        words = [word for word in " ".join((recording.findtext(".//orth") or "").split()).split()
                 if word != "[SILENCE]"]
        rows.append({"recording": recording.get("name"),
                     "speaker": recording.find(".//speaker").get("name"), "words": words})
    return rows


def main():
    classes = {item["citizen_raw_gloss"]: item
               for item in json.loads(MANIFEST.read_text())["classes"]}
    locked = set(classes)
    o5s5_files = []
    o5s5_candidates = []
    coverage = defaultdict(lambda: {"events": 0, "signers": set()})
    for session, video_name in O5S5_PAIRS.items():
        eaf = next((ROOT / "o5s5/annotations").glob(f"O5S5_{session}_*.eaf"))
        video = ROOT / "o5s5/videos" / video_name
        assert video.exists(), f"missing {video}"
        participant, tier_counts, events = eaf_events(eaf)
        stream = probe(video)
        duration_ms = float(stream["duration"]) * 1000
        assert max(event["end_ms"] for event in events) <= duration_ms + 250
        for event in events:
            if event["value"] in locked:
                coverage[event["value"]]["events"] += 1
                coverage[event["value"]]["signers"].add(participant)
                cls = classes[event["value"]]
                o5s5_candidates.append({
                    "session": session, "participant": participant, "tier": event["tier"],
                    "video": str(video), "start_ms": event["start_ms"], "end_ms": event["end_ms"],
                    "citizen_raw_gloss": event["value"], "canonical_label": cls["canonical_label"],
                    "citizen_asl_lex_code": cls["citizen_asl_lex_code"],
                    "class_index": cls["class_index"], "variant_review_status": "PENDING",
                })
        o5s5_files.append({
            "session": session, "participant": participant,
            "video": str(video), "video_bytes": video.stat().st_size,
            "video_sha256": digest(video), "eaf": str(eaf), "eaf_sha256": digest(eaf),
            "tier_counts": tier_counts, "hand_annotations": len(events),
            "last_annotation_ms": max(event["end_ms"] for event in events),
            "stream": stream,
        })

    rwth_root = ROOT / "rwth_boston_104/corpus"
    rwth_rows = corpus_rows(rwth_root / "mpg.all.sentences.corpus")
    train_rows = corpus_rows(rwth_root / "mpg.train.sentences.pronunciations.corpus")
    test_rows = corpus_rows(rwth_root / "mpg.test.sentences.corpus")
    rwth_words = Counter(word for row in rwth_rows for word in row["words"])
    rwth_coverage = {word: rwth_words[word] for word in sorted(locked & rwth_words.keys())}
    rwth_download = json.loads((REPORT.parent / "rwth_boston104_audit.json").read_text())

    result = {
        "locked_manifest": str(MANIFEST),
        "o5s5": {
            "source": "https://osf.io/769sw/",
            "access": "Public OSF files; no account or access request required.",
            "license": "CC BY-NC-SA 4.0",
            "license_source": "https://ida.gallaudet.edu/o5s5/index.html",
            "files": o5s5_files,
            "signers": sorted({item["participant"] for item in o5s5_files}),
            "video_bytes": sum(item["video_bytes"] for item in o5s5_files),
            "hand_annotations": sum(item["hand_annotations"] for item in o5s5_files),
            "right_hand_annotations": sum(item["tier_counts"].get("RightHand_IDg", 0) for item in o5s5_files),
            "left_hand_annotations": sum(item["tier_counts"].get("LeftHand_IDg", 0) for item in o5s5_files),
            "exact_locked_label_overlap": len(coverage),
            "exact_locked_events": sum(item["events"] for item in coverage.values()),
            "coverage": {label: {"events": item["events"], "signers": sorted(item["signers"])}
                         for label, item in sorted(coverage.items())},
            "training_eligible": False,
            "auxiliary_temporal_training_candidate": True,
            "limitation": "Research/noncommercial auxiliary temporal training candidate; direct locked-100 use still requires Citizen ASL-LEX visual-variant review.",
        },
        "rwth_boston_104": {
            "source": "https://www-i6.informatik.rwth-aachen.de/aslr/database-rwth-boston-104.php",
            "access": "Official page says freely available for download; no account or access request required.",
            "license": "Research-use availability with required publication citation; no explicit Creative Commons license found.",
            "videos": len(rwth_rows), "tokens": sum(rwth_words.values()), "vocabulary": len(rwth_words),
            "speakers": dict(Counter(row["speaker"] for row in rwth_rows)),
            "train_videos": len(train_rows), "test_videos": len(test_rows),
            "train_speakers": sorted({row["speaker"] for row in train_rows}),
            "test_speakers": sorted({row["speaker"] for row in test_rows}),
            "signer_disjoint": not ({row["speaker"] for row in train_rows} &
                                    {row["speaker"] for row in test_rows}),
            "exact_locked_label_overlap": len(rwth_coverage), "coverage": rwth_coverage,
            "video_bytes": rwth_download["video_bytes"],
            "decoded_videos": len(rwth_download["decoded"]),
            "frames": sum(int(item["nb_read_frames"]) for item in rwth_download["decoded"]),
            "duration_seconds": sum(float(item["duration"]) for item in rwth_download["decoded"]),
            "training_eligible": False,
            "auxiliary_temporal_training_candidate": True,
            "limitation": "Low-resolution supplemental pretraining only; official train/test reuse all three signers and exact Citizen variants are unverified.",
        },
        "some_asl": {
            "status": "excluded",
            "reason": "User visual review found the material predominantly one-handed and unsuitable for this project.",
        },
    }
    exact_path = REPORT.parent.parent / "o5s5_citizen100_v17/audit.json"
    if exact_path.exists():
        exact = json.loads(exact_path.read_text())
        o5s5 = result["o5s5"]
        o5s5["legacy_raw_string_overlap"] = {
            "classes": o5s5.pop("exact_locked_label_overlap"),
            "tier_events": o5s5.pop("exact_locked_events"),
        }
        o5s5.update({
            "exact_signbank_locked_classes": exact["exact_locked_classes"],
            "exact_signbank_locked_occurrences": exact["exact_locked_occurrences"],
            "exact_signbank_manifest": str(exact_path.parent / "exact_occurrences.csv"),
            "training_eligible": True,
            "background_training_eligible": False,
            "limitation": "Exact-ID positive contextual supervision only; unannotated gaps are not background.",
        })
    assert len(o5s5_files) == 6 and len(rwth_rows) == 201
    REPORT.write_text(json.dumps(result, indent=2) + "\n")
    with O5S5_CSV.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(o5s5_candidates[0]))
        writer.writeheader()
        writer.writerows(o5s5_candidates)
    train_ids = {row["recording"] for row in train_rows}
    test_ids = {row["recording"] for row in test_rows}
    rwth_manifest = [{"recording": row["recording"], "speaker": row["speaker"],
                      "split": "train" if row["recording"] in train_ids else
                               "test" if row["recording"] in test_ids else "unknown",
                      "video": str(ROOT / "rwth_boston_104/videoBank/camera0" /
                                   f"{row['recording']}_0.mpg"),
                      "gloss_sequence": " ".join(row["words"])} for row in rwth_rows]
    with RWTH_CSV.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rwth_manifest[0]))
        writer.writeheader()
        writer.writerows(rwth_manifest)
    print(f"O5S5: {len(o5s5_files)} videos, {result['o5s5']['hand_annotations']} hand annotations, "
          f"{len(coverage)} locked-label strings; RWTH: {len(rwth_rows)} videos, "
          f"{len(rwth_coverage)} locked-label strings")


if __name__ == "__main__":
    main()
