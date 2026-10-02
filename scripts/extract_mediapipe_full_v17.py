#!/usr/bin/env python3
"""Re-extract the Apple v17 training inputs with the MediaPipe-only detector (Android family).

Commands
  isolated  landmark archive + hand-crop archive for every isolated clip the Apple chain trained on
            (Citizen train/val, SemLex train/val, local deep-clean train/val)
  embed     MobileCLIP2-S0 embeddings of the MediaPipe hand crops (same encoder as Apple)

Outputs mirror each Apple path under ``--output`` (default a new dated root). Nothing existing is
overwritten: finished outputs are verified and skipped, and a root that holds a different schema is
refused. Sealed test splits are never listed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.mediapipe_full_v17 import (  # noqa: E402
    SCHEMA_NAME,
    MediaPipeFullV17Config,
    derived_fingerprint,
    schema_fingerprint,
    schema_payload,
)

DEFAULT_OUTPUT = Path("data/local/mediapipe_full_v17_20261003")
RECYCLE = 3  # shard exit code: stopped early by design, relaunch
DATA_LOCAL = Path("data/local")
CITIZEN = Path("data/local/citizen100_v17")
SOURCES = ("citizen_train", "citizen_val", "semlex_train", "semlex_val", "local_train", "local_val",
           "local_tier_a")
TIER_A_MANIFEST = Path("artifacts/reports/local_citizen100_quality_audit/cap14_exact_consensus/"
                       "consensus_review_manifest.json")  # hand-branch base (2026-08) local source


# ----------------------------------------------------------------------------- jobs

def mirrored(output: Path, kind: str, apple_path: Path, suffix: str) -> Path:
    relative = Path(apple_path).relative_to(DATA_LOCAL)
    name = relative.name.removesuffix(".v17.npz") + suffix
    return output / kind / relative.parent / name


def isolated_jobs(output: Path, sources=SOURCES) -> list[dict[str, str]]:
    from active.v17.extract_hand_rgb_semlex_val_v17 import validation_items
    from active.v17.extract_hand_rgb_supplement_v17 import selection_items
    from active.v17.train_unified_multimodal_student_v17 import build_parser as student_parser

    student = student_parser().parse_args([])
    jobs = []

    def add(source, label, item_id, raw, landmark):
        raw, landmark = Path(raw), Path(landmark)
        jobs.append(dict(
            source=source, label=label, item_id=item_id, raw=str(raw), apple_landmark=str(landmark),
            landmark=str(mirrored(output, "landmarks", landmark, ".v17.npz")),
            crops=str(mirrored(output, "hand_rgb", landmark, ".hand_rgb_v17.npz")),
            embedding=str(mirrored(output, "hand_mobileclip2_s0", landmark, ".hand_mobileclip2_v17.npz")),
        ))

    for split in ("train", "val"):
        if f"citizen_{split}" not in sources:
            continue
        for landmark in sorted((CITIZEN / "landmarks" / split).glob("*/*.v17.npz")):
            label, stem = landmark.parent.name, landmark.name.removesuffix(".v17.npz")
            add(f"citizen_{split}", label, stem, CITIZEN / "raw" / split / label / f"{stem}.mp4", landmark)
    if "semlex_train" in sources:
        for item in selection_items(student.semlex_train_manifest, "semlex")[0]:
            add("semlex_train", item.label, item.item_id, item.raw_path, item.landmark_path)
    if "semlex_val" in sources:
        for item in validation_items(student.semlex_val_manifest)[0]:
            add("semlex_val", item.label, item.item_id, item.raw_path, item.landmark_path)
    for source, manifest, kind in (("local_train", student.local_train_manifest, "local_deep_clean"),
                                   ("local_val", student.local_val_manifest, "local_deep_clean_val")):
        if source in sources:
            for item in selection_items(manifest, kind)[0]:
                add(source, item.label, item.item_id, item.raw_path, item.landmark_path)
    if "local_tier_a" in sources:
        for item in selection_items(TIER_A_MANIFEST, "local_tier_a")[0]:
            add("local_tier_a", item.label, item.item_id, item.raw_path, item.landmark_path)
    for job in jobs:
        if "test" in Path(job["raw"]).parts:
            raise RuntimeError(f"sealed test path listed: {job['raw']}")
    return jobs


def guard_root(output: Path, config: MediaPipeFullV17Config) -> None:
    """A root may only ever hold this exact schema; refuse anything else."""
    marker = output / "_schema_mediapipe_full_v17.json"
    payload = schema_payload(config)
    if marker.exists():
        existing = json.loads(marker.read_text())
        if existing.get("fingerprint") != schema_fingerprint(config):
            raise SystemExit(f"{output} holds a different schema; choose a new --output")
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} exists without a MediaPipe schema marker; refusing to write into it")
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"fingerprint": schema_fingerprint(config), "schema": payload},
                                 indent=2, sort_keys=True) + "\n")


# ----------------------------------------------------------------------------- isolated worker

_DETECTOR = None
_CONFIG = None


def _init_worker(config_json: str) -> None:
    global _DETECTOR, _CONFIG
    os.environ.setdefault("GLOG_minloglevel", "2")
    _CONFIG = MediaPipeFullV17Config(**json.loads(config_json))
    from active.v17.mediapipe_full_v17 import MediaPipeFullDetector
    _DETECTOR = MediaPipeFullDetector(_CONFIG)


def _valid_archive(path: Path, fingerprint: str, verify: bool = False) -> bool:
    """Existence means complete: writes are atomic and ``guard_root`` keeps foreign schemas out."""
    if not path.exists():
        return False
    if verify:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
        if metadata.get("schema_fingerprint") != fingerprint:
            raise RuntimeError(f"{path} exists with a foreign schema; refusing to overwrite")
    return True


def _isolated_one(job: dict[str, str]) -> dict[str, object]:
    from active.v17.extract_hand_rgb_v17 import extract_clip, save_archive as save_crops
    from active.v17.extract_v17 import extract_frames_v17, read_video_frames, rotate_frame_clockwise
    from active.v17.mediapipe_full_v17 import save_result, stamp_result
    from active.v17.schema_hand_rgb_v17 import HandRGBV17Config, schema_fingerprint as crop_fp

    from active.v17.mediapipe_full_v17 import CROP_DETECTION, DetectionMemo

    config = _CONFIG
    detector = DetectionMemo(_DETECTOR)
    landmark, crops = Path(job["landmark"]), Path(job["crops"])
    crop_config = HandRGBV17Config()
    crop_fingerprint = derived_fingerprint(crop_fp(crop_config), config)
    status = dict(job=job["landmark"], source=job["source"])
    started = time.perf_counter()
    def oriented_frames():
        with np.load(job["apple_landmark"], allow_pickle=False) as payload:
            apple = json.loads(str(payload["metadata_json"]))
        frames, metadata = read_video_frames(job["raw"], config.maximum_source_frames,
                                             config.maximum_image_side, rotation="auto")
        # Reuse the orientation decision recorded for this video, never a new probe.
        coarse = float(apple.get("vision_coarse_rotation_clockwise") or 0.0)
        if coarse:
            frames = [rotate_frame_clockwise(frame, coarse) for frame in frames]
        metadata.update(vision_auto_orientation_enabled=False, vision_coarse_rotation_clockwise=coarse)
        return frames, metadata

    try:
        if not _valid_archive(landmark, schema_fingerprint(config)):
            frames, metadata = oriented_frames()
            detector.reset_sequence()
            result = extract_frames_v17(frames, config, detector=detector, metadata=metadata)
            if result is None:
                return dict(status, outcome="no_hands", seconds=time.perf_counter() - started)
            stamp_result(result, config, apple_landmark=job["apple_landmark"], source=job["source"],
                         canonical_label=job["label"], source_item_id=job["item_id"])
            save_result(landmark, result, config)
        if not _valid_archive(crops, crop_fingerprint):
            if not detector.memo:  # resumed clip: replay the same dense landmark pass first
                frames, _ = oriented_frames()
                detector.reset_sequence()
                for index, frame in enumerate(frames):
                    detector.detect(frame, include_body=index % config.body_interval == 0,
                                    include_face=config.include_face and index % config.face_interval == 0)
            detector.reset_sequence()  # identical state before the crop pass on both paths
            arrays, metadata, diagnostics = extract_clip(Path(job["raw"]), landmark, detector, crop_config)
            diagnostics["crop_detection_replayed_frames"] = detector.hits
            metadata.update(schema_fingerprint=crop_fingerprint, base_schema_fingerprint=crop_fp(crop_config),
                            landmark_schema=SCHEMA_NAME, crop_detection=CROP_DETECTION,
                            source=job["source"], canonical_label=job["label"],
                            source_item_id=job["item_id"], split="val" if job["source"].endswith("_val") else "train",
                            training_eligible=not job["source"].endswith("_val"))
            save_crops(crops, arrays, metadata, diagnostics, crop_config)  # absent: checked above
        return dict(status, outcome="ok", seconds=time.perf_counter() - started)
    except Exception as error:  # one bad video must not stop the run
        return dict(status, outcome="failed", error=repr(error)[:300], seconds=time.perf_counter() - started)


def outcome_marker(job: dict[str, str]) -> Path:
    return Path(job["landmark"] + ".outcome.json")


def finished(job: dict[str, str]) -> bool:
    return (Path(job["landmark"]).exists() and Path(job["crops"]).exists()) or outcome_marker(job).exists()


def run_isolated(args) -> None:
    """Shards run as independent OS processes: a multiprocessing pool deadlocks MediaPipe's GPU."""
    config = MediaPipeFullV17Config(pose_model=args.pose_model)
    if args.shard is not None:
        jobs = json.loads(Path(args.jobs_file).read_text())
    else:
        guard_root(args.output, config)
        jobs = isolated_jobs(args.output, tuple(args.sources))
        if args.limit:
            jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    if args.shard is None:
        stamp = time.strftime('%Y%m%dT%H%M%S')
        jobs_file = args.output / f"_jobs_{stamp}.json"
        jobs_file.write_text(json.dumps(jobs))
        print(json.dumps(dict(jobs=len(jobs), shards=args.workers, fingerprint=schema_fingerprint(config))),
              flush=True)
        def launch(shard, attempt):
            command = [sys.executable, str(Path(__file__).resolve()), "isolated", "--output", str(args.output),
                       "--pose-model", args.pose_model, "--jobs-file", str(jobs_file),
                       "--shard", str(shard), "--shards", str(args.workers), "--stamp", f"{stamp}_try{attempt}"]
            return subprocess.Popen(command, cwd=ROOT)

        # MediaPipe's GPU runtime occasionally aborts a whole process. Each clip resets tracking and
        # finished outputs are verified and skipped, so a relaunched shard reproduces the same files.
        running = {shard: (launch(shard, 0), 0) for shard in range(args.workers)}
        codes = {}
        started, last_report = time.perf_counter(), time.perf_counter()
        initial = sum(finished(job) for job in jobs)
        while running:
            time.sleep(5)
            if time.perf_counter() - last_report > 120:
                last_report = time.perf_counter()
                done = sum(finished(job) for job in jobs)
                rate = (done - initial) / (last_report - started)
                print(json.dumps(dict(progress=done, of=len(jobs), clips_per_s=round(rate, 2),
                                      eta_min=round((len(jobs) - done) / max(rate, 1e-6) / 60, 1))), flush=True)
            for shard, (child, attempt) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                if code == RECYCLE:
                    running[shard] = (launch(shard, attempt), attempt)
                elif code != 0 and attempt < args.max_restarts:
                    print(json.dumps(dict(shard=shard, exit_code=code, relaunch=attempt + 1)), flush=True)
                    running[shard] = (launch(shard, attempt + 1), attempt + 1)
                else:
                    codes[shard] = code
                    del running[shard]
        codes = [codes[shard] for shard in sorted(codes)]
        outcomes = {}
        for log in sorted(args.output.glob(f"isolated_{stamp}_try*_shard*.jsonl")):
            for line in log.read_text().splitlines():
                status = json.loads(line)
                outcomes[status["job"]] = status["outcome"]  # a relaunch re-reports finished clips
        counts = {}
        for outcome in outcomes.values():
            counts[outcome] = counts.get(outcome, 0) + 1
        print(json.dumps(dict(finished=True, exit_codes=codes, **counts)), flush=True)
        return
    mine = [job for job in jobs[args.shard::args.shards] if not finished(job)]
    if not mine:
        return
    _init_worker(json.dumps(dict(pose_model=args.pose_model)))
    recycle = False
    log_path = args.output / f"isolated_{args.stamp}_shard{args.shard}.jsonl"
    counts = {}
    with log_path.open("a") as log:
        for index, job in enumerate(mine, start=1):
            if _DETECTOR.calls >= args.recycle_calls:
                recycle = True
                break
            status = _isolated_one(job)
            counts[status["outcome"]] = counts.get(status["outcome"], 0) + 1
            log.write(json.dumps(status) + "\n")
            log.flush()
            if status["outcome"] != "ok":
                # Recorded once so relaunched shards do not retry it forever; delete to retry.
                outcome_marker(job).parent.mkdir(parents=True, exist_ok=True)
                outcome_marker(job).write_text(json.dumps(status) + "\n")
            if status["outcome"] == "failed":
                print("FAILED", status["job"], status["error"], flush=True)
    _DETECTOR.close()
    if recycle:
        sys.exit(RECYCLE)


# ----------------------------------------------------------------------------- embeddings

def run_embed(args) -> None:
    import torch
    from PIL import Image
    from active.v17.extract_hand_rgb_v17 import decode_packed_crops
    from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
    from active.v17.schema_hand_mobileclip2_v17 import (
        HandMobileCLIP2V17Config, schema_fingerprint as hand_fp, schema_payload as hand_payload)
    from active.v17.schema_hand_rgb_v17 import CROP_SIZE, HandRGBV17Config, schema_fingerprint as crop_fp

    config = MediaPipeFullV17Config(pose_model=args.pose_model)
    guard_root(args.output, config)
    hand_config = HandMobileCLIP2V17Config()
    crop_fingerprint = derived_fingerprint(crop_fp(HandRGBV17Config()), config)
    embed_fingerprint = derived_fingerprint(hand_fp(hand_config), config)
    import gc
    device = select_device(args.device)
    if device.type == "mps":  # bound the MPS cache, as the Apple encoders do
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    model, preprocess = build_encoder(device)
    jobs = [job for job in isolated_jobs(args.output, tuple(args.sources)) if Path(job["crops"]).exists()]
    todo = [job for job in jobs if not _valid_archive(Path(job["embedding"]), embed_fingerprint)]
    print(json.dumps(dict(crops=len(jobs), todo=len(todo), device=str(device))), flush=True)
    started = time.perf_counter()
    for start in range(0, len(todo), args.archives_per_batch):
        batch = todo[start:start + args.archives_per_batch]
        records = []
        for job in batch:
            with np.load(job["crops"], allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"]))
                if metadata.get("schema_fingerprint") != crop_fingerprint:
                    raise RuntimeError(f"{job['crops']}: crop schema mismatch")
                records.append((decode_packed_crops(payload["jpeg_blob"], payload["jpeg_offsets"], CROP_SIZE),
                                payload["valid"].astype(bool), payload["boxes_normalized"], metadata))
        images, places = [], []
        for a, (crops, valid, _, _) in enumerate(records):
            for f, v in np.argwhere(valid):
                images.append(crops[f, v])
                places.append((a, int(f), int(v)))
        embeddings = [np.zeros((16, 3, 512), np.float32) for _ in records]
        with torch.inference_mode():
            step = args.image_batch_size
            for i in range(0, len(images), step):
                tensor = torch.stack([preprocess(Image.fromarray(x)) for x in images[i:i + step]]).to(
                    device=device, dtype=next(model.visual.parameters()).dtype)
                encoded = model.encode_image(tensor, normalize=True).float().cpu().numpy()
                for value, (a, f, v) in zip(encoded, places[i:i + step]):
                    embeddings[a][f, v] = value
        for job, embedding, (_, valid, boxes, crop_metadata) in zip(batch, embeddings, records):
            if not np.isfinite(embedding).all():
                raise RuntimeError(f"{job['crops']}: non-finite embeddings")
            target = Path(job["embedding"])
            target.parent.mkdir(parents=True, exist_ok=True)
            metadata = dict(schema_fingerprint=embed_fingerprint, base_schema_fingerprint=hand_fp(hand_config),
                            crop_schema_fingerprint=crop_fingerprint, crop_archive=job["crops"],
                            landmark_schema=SCHEMA_NAME, video_path=crop_metadata.get("video_path"),
                            source=job["source"], source_item_id=job["item_id"], canonical_label=job["label"],
                            split=crop_metadata.get("split"), training_eligible=crop_metadata.get("training_eligible"),
                            test_accessed=False)
            temporary = target.with_name(target.name + ".partial.npz")
            np.savez_compressed(temporary, embeddings=embedding.astype(np.float16), valid=valid,
                                boxes_normalized=boxes.astype(np.float16),
                                metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
                                schema_json=np.array(json.dumps(hand_payload(hand_config), sort_keys=True)))
            if target.exists():
                raise FileExistsError(target)
            temporary.rename(target)
        del records, images, places, embeddings
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        done = min(start + args.archives_per_batch, len(todo))
        rate = done / (time.perf_counter() - started)
        if done % (args.archives_per_batch * 10) == 0 or done == len(todo):
            print(json.dumps(dict(done=done, of=len(todo), archives_per_s=round(rate, 2),
                                  eta_min=round((len(todo) - done) / rate / 60, 1))), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("isolated", "embed", "jobs"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES[:6]), choices=SOURCES)
    parser.add_argument("--pose-model", default="pose_landmarker_lite",
                        choices=("pose_landmarker_lite", "pose_landmarker_full"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--archives-per-batch", type=int, default=32)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--max-restarts", type=int, default=50)
    parser.add_argument("--recycle-calls", type=int, default=4000,
                        help="GPU calls per shard process; MediaPipe's macOS GPU path leaks pixel buffers")
    parser.add_argument("--jobs-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--stamp", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "isolated":
        run_isolated(args)
    elif args.command == "embed":
        run_embed(args)
    else:
        jobs = isolated_jobs(args.output, tuple(args.sources))
        counts = {}
        for job in jobs:
            counts[job["source"]] = counts.get(job["source"], 0) + 1
        missing = sum(not Path(job["raw"]).exists() for job in jobs)
        print(json.dumps(dict(total=len(jobs), by_source=counts, missing_raw=missing), indent=1))


if __name__ == "__main__":
    main()
