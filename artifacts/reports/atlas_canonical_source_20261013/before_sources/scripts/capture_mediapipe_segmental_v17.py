#!/usr/bin/env python3
"""MediaPipe-detector analog of the segmental-decoder caches (Android model family).

For every continuous video the Apple segmental chain used, run the live 20 Hz observation path
with ``MediaPipeFullDetector`` and write
  continuous/av_raw/<key>.npz   raw [T,61,5] + times  (boundary-student inputs, as Apple av_raw)
  continuous/spans/<key>.pkl    {span: verifier inputs | None} for exactly the span keys in the
                                Apple memo and train_labels (DGS-teacher spans, extractor-independent)
Span inputs are built like ``SpanScorer.score`` -> ``full.classify`` (no motion trim):
``landmarks_from_observations`` + ``hand_inputs`` + the same Core ML MobileCLIP2 image encoder.
Apple model scores are not computed. Nothing is written into the Apple cache, and finished
outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_mediapipe_full_v17 import DEFAULT_OUTPUT, RECYCLE, guard_root  # noqa: E402
from active.v17.mediapipe_full_v17 import MediaPipeFullV17Config, schema_fingerprint  # noqa: E402

SPAN_SETS = ("local_train", "tune", "test", "asllrp_val")   # need span inputs
RAW_SETS = ("asllrp_train", "youtube")                       # boundary-student inputs only
CONTEXT = .1
FPS = 20


def rows(output: Path) -> list[dict[str, object]]:
    from scripts import segmental_lab_v17 as lab
    result = []
    for name in SPAN_SETS + ("asllrp_train",):
        for row in lab.rows_for(name):
            key = lab.key(row["source_item_id"])
            spans = set()
            if name in SPAN_SETS:
                spans |= set(lab.load_memo(row))
                labels = lab.CACHE / "train_labels" / f"{key}.json"
                if labels.exists():
                    spans |= {tuple(item["span"]) for item in json.loads(labels.read_text())["spans"]}
            result.append(dict(set=name, key=key, source_item_id=row["source_item_id"],
                               video_path=str(row["video_path"]), spans=sorted(spans)))
    clips = ROOT / "data/local/youtube_asl_boundary_distill_v17/clips"
    for clip in sorted(clips.glob("*.mp4")):
        item = "yt:" + clip.stem
        apple = lab.CACHE / "av_raw" / f"{lab.key(item)}.npz"
        if apple.exists() and apple.stat().st_size > 0:  # exactly the clips the Apple student used
            result.append(dict(set="youtube", key=lab.key(item), source_item_id=item,
                               video_path=str(clip.relative_to(ROOT)), spans=[]))
    for row in result:
        row["av_raw"] = str(output / "continuous/av_raw" / f"{row['key']}.npz")
        row["memo"] = str(output / "continuous/spans" / f"{row['key']}.pkl") if row["set"] in SPAN_SETS else ""
    return result


def outcome_marker(row) -> Path:
    return Path(row["av_raw"] + ".outcome.json")


def finished(row) -> bool:
    done = Path(row["av_raw"]).exists() and (not row["memo"] or Path(row["memo"]).exists())
    return done or outcome_marker(row).exists()


class Capture:
    def __init__(self):
        import coremltools as ct
        from scripts.evaluate_temporal_boundary_v17 import arguments
        from active.v17.mediapipe_full_v17 import MediaPipeFullDetector
        from scripts import segmental_lab_v17 as lab
        self.args = arguments("data", lab.REPORT / "sessions")
        self.args.no_motion_trim = True
        self.detector = MediaPipeFullDetector(MediaPipeFullV17Config())
        self.encoder = ct.models.MLModel(str(self.args.image_encoder), compute_units=ct.ComputeUnit.ALL)
        self.cache: dict[bytes, np.ndarray] = {}

    def encode(self, crops, valid) -> np.ndarray:
        import cv2
        from PIL import Image
        embeddings = np.zeros((16, 3, 512), np.float32)
        for frame in range(16):
            for view in range(3):
                crop = crops[frame][view]
                if crop is None or not valid[frame, view]:
                    continue
                key = hashlib.blake2b(crop.tobytes(), digest_size=16).digest() + str(crop.shape).encode()
                if key not in self.cache:
                    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    self.cache[key] = np.asarray(self.encoder.predict({"image": image})["embedding"]).reshape(512)
                embeddings[frame, view] = self.cache[key]
        return embeddings

    def span_inputs(self, obs, start, end):
        from scripts.live_isolated_v17 import hand_inputs, landmarks_from_observations
        from active.v17.schema_v17 import V17Config
        selected = [o for o in obs if start - CONTEXT <= o.seconds <= end + CONTEXT]
        if len(selected) < 4:
            return None
        try:
            features, diagnostics = landmarks_from_observations(selected, V17Config())
        except ValueError as error:
            if str(error) != "not enough detected hand frames":
                raise
            return None
        crops, valid, boxes = hand_inputs(selected, int(diagnostics["trim_start"]),
                                          int(diagnostics["trim_end_exclusive"]))
        return dict(landmarks=features.astype(np.float16),
                    hand_embeddings=self.encode(crops, valid).astype(np.float16),
                    hand_valid=valid.astype(np.bool_), hand_boxes=boxes.astype(np.float16))

    def run(self, row) -> dict[str, object]:
        from scripts.evaluate_temporal_boundary_v17 import observations
        from active.v17.stage1_window_v17 import raw_observation_features
        started = time.perf_counter()
        self.cache = {}
        self.detector.reset_sequence()
        obs = observations(row, self.args, self.detector)
        raw_path, memo_path = Path(row["av_raw"]), Path(row["memo"]) if row["memo"] else None
        if not obs:
            return dict(key=row["key"], outcome="no_frames")
        if not raw_path.exists():
            raw, times = raw_observation_features(obs)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = raw_path.with_name(raw_path.name + ".partial.npz")
            np.savez_compressed(temporary, raw=raw.astype(np.float16), times=times)
            temporary.rename(raw_path)
        if memo_path is not None and not memo_path.exists():
            memo = {tuple(span): self.span_inputs(obs, span[0] / FPS, span[1] / FPS) for span in row["spans"]}
            memo = {span: (None if value is None else dict(inputs=value)) for span, value in memo.items()}
            memo_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = memo_path.with_name(memo_path.name + ".partial")
            temporary.write_bytes(pickle.dumps(memo))
            temporary.rename(memo_path)
        return dict(key=row["key"], set=row["set"], outcome="ok", frames=len(obs), spans=len(row["spans"]),
                    seconds=round(time.perf_counter() - started, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sets", nargs="+", default=list(SPAN_SETS + RAW_SETS), choices=SPAN_SETS + RAW_SETS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recycle-calls", type=int, default=3500,
                        help="stop a shard before the next video once this many GPU calls were made")
    parser.add_argument("--max-restarts", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    parser.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--jobs-file", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = MediaPipeFullV17Config()
    if args.shard is not None:
        jobs = json.loads(Path(args.jobs_file).read_text())
        mine = [row for row in jobs[args.shard::args.workers] if not finished(row)]
        if not mine:
            return
        capture = Capture()
        log = args.output / "continuous" / f"capture_shard{args.shard}.jsonl"
        with log.open("a") as handle:
            for row in mine:
                if capture.detector.calls >= args.recycle_calls:
                    capture.detector.close()
                    sys.exit(RECYCLE)
                try:
                    status = capture.run(row)
                except Exception as error:  # one bad video must not stop the run
                    status = dict(key=row["key"], set=row["set"], outcome="failed", error=repr(error)[:300])
                    print("FAILED", row["video_path"], status["error"], flush=True)
                handle.write(json.dumps(status) + "\n")
                handle.flush()
                if status["outcome"] != "ok":  # recorded once; delete the marker to retry
                    outcome_marker(row).parent.mkdir(parents=True, exist_ok=True)
                    outcome_marker(row).write_text(json.dumps(status) + "\n")
        capture.detector.close()
        return

    jobs = [row for row in rows(args.output) if row["set"] in args.sets]
    if args.dry_run:
        counts = {}
        for row in jobs:
            counts[row["set"]] = counts.get(row["set"], 0) + 1
        missing = sum(not (ROOT / row["video_path"]).exists() for row in jobs)
        print(json.dumps(dict(videos=len(jobs), by_set=counts, missing_video=missing,
                              spans=sum(len(r["spans"]) for r in jobs))))
        return
    guard_root(args.output, config)
    if args.limit:
        jobs = jobs[::max(1, len(jobs) // args.limit)][:args.limit]
    (args.output / "continuous").mkdir(parents=True, exist_ok=True)
    jobs_file = args.output / "continuous" / f"_jobs_{time.strftime('%Y%m%dT%H%M%S')}.json"
    jobs_file.write_text(json.dumps(jobs))
    counts = {}
    for row in jobs:
        counts[row["set"]] = counts.get(row["set"], 0) + 1
    print(json.dumps(dict(videos=len(jobs), by_set=counts, spans=sum(len(r["spans"]) for r in jobs),
                          fingerprint=schema_fingerprint(config))), flush=True)

    def launch(shard):
        return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--output", str(args.output),
                                 "--workers", str(args.workers), "--recycle-calls", str(args.recycle_calls),
                                 "--shard", str(shard), "--jobs-file", str(jobs_file)], cwd=ROOT)

    running = {shard: (launch(shard), 0) for shard in range(args.workers)}
    started = last = time.perf_counter()
    initial = sum(finished(row) for row in jobs)
    while running:
        time.sleep(5)
        if time.perf_counter() - last > 120:
            last = time.perf_counter()
            done = sum(finished(row) for row in jobs)
            rate = (done - initial) / (last - started)
            print(json.dumps(dict(progress=done, of=len(jobs), videos_per_min=round(60 * rate, 1),
                                  eta_min=round((len(jobs) - done) / max(rate, 1e-6) / 60, 1))), flush=True)
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
    done = sum(finished(row) for row in jobs)
    print(json.dumps(dict(finished=True, done=done, of=len(jobs))), flush=True)


if __name__ == "__main__":
    main()
