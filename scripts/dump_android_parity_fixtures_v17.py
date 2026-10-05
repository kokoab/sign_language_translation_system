"""Parity fixtures for the Android Kotlin Live Reel port (words only, landmark-only mode).

Runs the same path as ``replay_segmental_v17.py --detector mediapipe_full --backend tflite
--no-fingerspelling`` with the landmark-only config and records, per video, every processed
observation (assigned hands, body, face, seconds, frame size) and the words the Python runtime
commits. The Kotlin harness (debug build, ``live.LiveParityActivity``) feeds the same observations
to its runtime and must commit the same words, which isolates the port from MediaPipe
desktop-vs-Android differences.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIG = "artifacts/reports/mediapipe_rebuild_v17_20261004/stream_config_mediapipe_v2_landmark_only_tflite.json"


def hand_json(hand):
    if hand is None:
        return None
    return {
        "xy": [[round(float(v), 7) for v in p] for p in hand.xy],
        "c": [round(float(v), 7) for v in hand.confidence],
        "chirality": hand.chirality,
        "score": float(hand.score),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="tune")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    from scripts import segmental_lab_v17 as lab
    from scripts.evaluate_temporal_boundary_v17 import arguments
    from scripts.live_stage2_ctc_v17 import observe_stage2_frame
    from active.v17.mediapipe_full_v17 import MediaPipeFullDetector, MediaPipeFullV17Config
    from active.v17.segmental_runtime_v17 import build_runtime

    rt = build_runtime(config=CONFIG, device="cpu", backend="tflite", fingerspelling=False)
    args = arguments("data", lab.REPORT / "sessions")
    detector = MediaPipeFullDetector(MediaPipeFullV17Config())
    rows = lab.rows_for(a.set)
    if a.limit:
        rows = rows[: a.limit]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for row in rows:
        rt.reset()
        if detector.calls > 3000:
            detector.renew()
        detector.reset_sequence()
        cap = cv2.VideoCapture(str(ROOT / row["video_path"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        wrists = {"left": None, "right": None}
        index = processed = 0
        deadline = 0.0
        frames, words = [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seconds = index / fps
            index += 1
            if seconds + 1e-6 < deadline:
                continue
            deadline = max(deadline + 1 / 20, seconds)
            obs = observe_stage2_frame(frame, seconds, processed, detector, wrists, args)
            processed += 1
            new = rt.observe(obs)
            h, w = obs.frame.shape[:2]
            d = obs.detection
            frames.append({
                "seconds": seconds, "width": w, "height": h,
                "left": hand_json(obs.assigned["left"]), "right": hand_json(obs.assigned["right"]),
                "body_xy": d.body_xy.round(7).tolist(), "body_c": d.body_confidence.round(7).tolist(),
                "face_xy": d.face_xy.round(7).tolist(), "face_c": d.face_confidence.round(7).tolist(),
                "face_for_features": bool(obs.face_for_features),
                "committed": [x["gloss"] for x in new],
            })
            words += [dict(x, eof=False) for x in new]
        cap.release()
        words += [dict(x, eof=True) for x in rt.finish()]
        keep = ("gloss", "start_frame", "end_frame", "commit_frame", "score", "early", "eof")
        out = {
            "id": row["source_item_id"], "reference": row["target_sequence"], "config": CONFIG,
            "frames": frames, "words": [{k: x.get(k) for k in keep} for x in words],
        }
        name = f"{len(index_rows):03d}.json"
        (a.output_dir / name).write_text(json.dumps(out))
        index_rows.append({"file": name, "id": row["source_item_id"], "frames": len(frames),
                           "words": [x["gloss"] for x in words]})
        print(row["source_item_id"], [x["gloss"] for x in words], flush=True)
    (a.output_dir / "index.json").write_text(json.dumps(index_rows, indent=1))


if __name__ == "__main__":
    main()
