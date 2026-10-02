#!/usr/bin/env python3
"""MediaPipe-detector analog of the Stage-2 windowed phrase archives (Android model family).

Commands
  windows  re-extract every Apple ``data/local/stage2_v17_multimodal`` archive (landmark windows +
           hand crops) with ``MediaPipeFullDetector``, reusing each archive's row metadata, targets and
           recorded orientation decision; crop boxes replay the dense landmark-pass detections
  embed    MobileCLIP2-S0 embeddings of those crops with the Apple Stage-2 encoder path

Outputs mirror the Apple paths under ``--output``; nothing existing is overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_mediapipe_full_v17 import DEFAULT_OUTPUT, RECYCLE, guard_root  # noqa: E402
from active.v17.mediapipe_full_v17 import (  # noqa: E402
    CROP_DETECTION, SCHEMA_NAME, MediaPipeFullV17Config, derived_fingerprint, schema_fingerprint)

APPLE_RGB = Path("data/local/stage2_v17_multimodal")
APPLE_HAND = Path("data/local/stage2_v17_hand_mobileclip2")


def jobs(output: Path) -> list[dict[str, str]]:
    result = []
    for path in sorted(APPLE_RGB.glob("*/*/*.stage2_rgb_v17.npz")):
        relative = path.relative_to(APPLE_RGB)
        stem = path.name.removesuffix(".stage2_rgb_v17.npz")
        result.append(dict(
            apple=str(path),
            rgb=str(output / "stage2_v17_multimodal" / relative),
            embedding=str(output / "stage2_v17_hand_mobileclip2" / relative.parent /
                          f"{stem}.stage2_hand_mobileclip2_v17.npz")))
    return result


def marker(job) -> Path:
    return Path(job["rgb"] + ".outcome.json")


def finished(job) -> bool:
    return Path(job["rgb"]).exists() or marker(job).exists()


def stage2_config(apple_metadata):
    from active.v17.schema_stage2_features_v17 import Stage2FeatureV17Config, schema_fingerprint as fp
    config = apple_metadata["schema"]["config"]
    candidate = Stage2FeatureV17Config(**{k: v for k, v in config.items()
                                          if k in Stage2FeatureV17Config.__dataclass_fields__})
    if fp(candidate) != apple_metadata["schema_fingerprint"]:
        raise ValueError("cannot reproduce the Apple Stage-2 feature config")
    return candidate


def extract_one(job, detector, config: MediaPipeFullV17Config) -> dict[str, object]:
    import scripts.extract_stage2_multimodal_v17 as stage2
    from active.v17.mediapipe_full_v17 import DetectionMemo
    from active.v17.schema_hand_rgb_v17 import HandRGBV17Config
    from active.v17.schema_stage2_features_v17 import schema_fingerprint as stage2_fp

    with np.load(job["apple"], allow_pickle=False) as payload:
        apple = json.loads(str(payload["metadata_json"]))
        targets = payload["target_indices"].tolist()
    row = {key: apple.get(key) for key in ("video_path", "source_item_id", "source", "role", "video_sha256",
                                           "source_group", "signer_id", "target_sequence", "zero_lip_nodes",
                                           "lip_supervision")}
    row["target_indices"] = targets
    features = stage2_config(apple)
    correction = float(apple.get("vision_coarse_rotation_clockwise") or 0.0)
    # Reuse the orientation decision recorded for this video instead of probing again.
    stage2.choose_coarse_orientation_v17 = lambda frames, _detector: (
        correction, {"reused_apple_orientation_decision": correction})
    memo = DetectionMemo(detector)
    arrays, metadata = stage2.extract_row(row, memo, memo, features, HandRGBV17Config(),
                                          apple["training_manifest_sha256"], features.window_stride)
    metadata.update(schema_fingerprint=derived_fingerprint(stage2_fp(features), config),
                    base_schema_fingerprint=stage2_fp(features), landmark_schema=SCHEMA_NAME,
                    landmark_schema_fingerprint=schema_fingerprint(config), crop_detection=CROP_DETECTION,
                    crop_detection_replayed_frames=memo.hits, apple_archive=job["apple"])
    target = Path(job["rgb"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    temporary.rename(target)
    return dict(job=job["rgb"], outcome="ok", windows=int(metadata["window_count"]), replayed=memo.hits)


def run_windows(args) -> None:
    config = MediaPipeFullV17Config()
    all_jobs = jobs(args.output)
    if args.shard is not None:
        from active.v17.mediapipe_full_v17 import MediaPipeFullDetector
        mine = [job for job in all_jobs[args.shard::args.workers] if not finished(job)]
        if not mine:
            return
        detector = MediaPipeFullDetector(config)
        log = args.output / f"phrases_shard{args.shard}.jsonl"
        with log.open("a") as handle:
            for job in mine:
                if detector.calls >= args.recycle_calls:
                    detector.close()
                    sys.exit(RECYCLE)
                try:
                    status = extract_one(job, detector, config)
                except Exception as error:  # one bad video must not stop the run
                    status = dict(job=job["rgb"], outcome="failed", error=repr(error)[:300])
                    print("FAILED", job["apple"], status["error"], flush=True)
                handle.write(json.dumps(status) + "\n")
                handle.flush()
                if status["outcome"] != "ok":
                    marker(job).parent.mkdir(parents=True, exist_ok=True)
                    marker(job).write_text(json.dumps(status) + "\n")
        detector.close()
        return

    guard_root(args.output, config)
    print(json.dumps(dict(archives=len(all_jobs), done=sum(map(finished, all_jobs)))), flush=True)

    def launch(shard):
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "windows", "--output",
                                 str(args.output), "--workers", str(args.workers), "--recycle-calls",
                                 str(args.recycle_calls), "--shard", str(shard)], cwd=ROOT)

    running = {shard: (launch(shard), 0) for shard in range(args.workers)}
    while running:
        time.sleep(5)
        for shard, (child, attempt) in list(running.items()):
            code = child.poll()
            if code is None:
                continue
            if code == RECYCLE:
                running[shard] = (launch(shard), attempt)
            elif code != 0 and attempt < args.max_restarts:
                print(json.dumps(dict(shard=shard, exit_code=code, relaunch=attempt + 1)), flush=True)
                running[shard] = (launch(shard), attempt + 1)
            else:
                del running[shard]
    print(json.dumps(dict(finished=True, done=sum(map(finished, all_jobs)), of=len(all_jobs))), flush=True)


def run_embed(args) -> None:
    import scripts.encode_stage2_hand_mobileclip2_v17 as encoder
    from active.v17.extract_mobileclip2_v17 import build_encoder, select_device
    from active.v17.schema_stage2_hand_mobileclip2_v17 import (
        Stage2HandMobileCLIP2V17Config, schema_fingerprint as hand_fp, schema_payload as hand_payload)

    config = MediaPipeFullV17Config()
    guard_root(args.output, config)
    import gc
    import torch
    device = select_device(args.device)
    if device.type == "mps":  # as the Apple encoder: bound the MPS cache (it otherwise grew past 30 GB)
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    model, preprocess = build_encoder(device, "fp32")
    hand_config = Stage2HandMobileCLIP2V17Config()
    embed_fingerprint = derived_fingerprint(hand_fp(hand_config), config)
    todo = [job for job in jobs(args.output) if Path(job["rgb"]).exists() and not Path(job["embedding"]).exists()]
    print(json.dumps(dict(todo=len(todo), device=str(device))), flush=True)
    for index, job in enumerate(todo, start=1):
        with np.load(job["rgb"], allow_pickle=False) as payload:
            crop_fingerprint = json.loads(str(payload["metadata_json"]))["schema_fingerprint"]
        embeddings, valid, boxes, metadata = encoder.encode_one(
            Path(job["rgb"]), model, preprocess, device, args.image_batch_size, crop_fingerprint)
        metadata.update(schema_fingerprint=embed_fingerprint, base_schema_fingerprint=hand_fp(hand_config),
                        schema=hand_payload(hand_config), landmark_schema=SCHEMA_NAME,
                        crop_detection=CROP_DETECTION)
        target = Path(job["embedding"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial.npz")
        np.savez_compressed(temporary, embeddings=embeddings, valid=valid, boxes_normalized=boxes,
                            metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
        temporary.rename(target)
        del embeddings, valid, boxes, metadata
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        if index % 50 == 0 or index == len(todo):
            print(json.dumps(dict(embedded=index, of=len(todo))), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("windows", "embed", "jobs"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--recycle-calls", type=int, default=3500)
    parser.add_argument("--max-restarts", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.12)
    parser.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "windows":
        run_windows(args)
    elif args.command == "embed":
        run_embed(args)
    else:
        all_jobs = jobs(args.output)
        roles = {}
        for job in all_jobs:
            role = Path(job["apple"]).relative_to(APPLE_RGB).parts[0]
            roles[role] = roles.get(role, 0) + 1
        print(json.dumps(dict(archives=len(all_jobs), by_role=roles)))


if __name__ == "__main__":
    main()
