"""Per-frame cost breakdown of the Android (TFLite) live chain on a few tuning videos (Mac CPU)."""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import cv2
ROOT = Path('/Volumes/secret/SLT/SLT'); sys.path.insert(0, str(ROOT))
from scripts import segmental_lab_v17 as lab
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from active.v17.mediapipe_full_v17 import MediaPipeFullDetector, MediaPipeFullV17Config
from active.v17 import segmental_runtime_v17 as srt

cfg = str(ROOT / 'artifacts/reports/mediapipe_rebuild_v17_20261004/stream_config_mediapipe_v2_tflite.json')
rt = srt.build_runtime(config=cfg, backend='tflite', fingerspelling=False)
T = defaultdict(float); N = defaultdict(int)


def timed(obj, name, key, count=None):
    fn = getattr(obj, name)

    def wrapper(*a, **k):
        t = time.perf_counter(); r = fn(*a, **k); T[key] += time.perf_counter() - t
        if count:
            N[key] += count(a, r)
        return r
    setattr(obj, name, wrapper)


det = MediaPipeFullDetector(MediaPipeFullV17Config())
timed(det, 'detect', 'mediapipe_detect_ms')
timed(rt.recognizer, 'frame_hand', 'hand_crops_and_mobileclip_ms')
timed(rt.recognizer.encoder, '_one', 'mobileclip_only_ms', count=lambda a, r: 1)
timed(rt.recognizer, 'logits', 'span_recognizer_ms', count=lambda a, r: len(a[0]))
timed(rt.boundary, 'update', 'boundary_ms')
args = arguments('data', lab.REPORT / 'sessions')
frames, total, per_frame = 0, 0., []
for row in lab.rows_for('tune')[:int(sys.argv[1]) if len(sys.argv) > 1 else 6]:
    rt.reset(); det.reset_sequence()
    cap = cv2.VideoCapture(str(ROOT / row['video_path'])); fps = cap.get(cv2.CAP_PROP_FPS)
    wrists = {'left': None, 'right': None}; index = processed = 0; deadline = 0.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        seconds = index / fps; index += 1
        if seconds + 1e-6 < deadline:
            continue
        deadline = max(deadline + 1 / 20, seconds)
        t = time.perf_counter()
        obs = observe_stage2_frame(frame, seconds, processed, det, wrists, args); processed += 1
        rt.observe(obs)
        spent = time.perf_counter() - t
        total += spent; frames += 1; per_frame.append(1000 * spent)
    rt.finish(); cap.release()
out = {k: round(1000 * v / frames, 2) for k, v in T.items()}
per_frame.sort()
out.update(all_per_frame_ms_mean=round(1000 * total / frames, 2), all_per_frame_ms_median=round(per_frame[len(per_frame) // 2], 2),
           frames=frames, mobileclip_crops_per_frame=round(N['mobileclip_only_ms'] / frames, 2),
           spans_scored_per_frame=round(N['span_recognizer_ms'] / frames, 2),
           note='means per processed frame; Mac M4, TFLite CPU (4 threads) + MediaPipe GPU; not phone timing')
print(json.dumps(out, indent=1))
