"""Extract FSboard fingerspelling clips through the live Apple Vision contract (20 Hz).

Each clip -> one .npz with per-frame raw landmarks [T,61,5] (raw_observation_features), the live
hand-crop embeddings [T,3,512] / valid [T,3] / boxes [T,3,4] (SpanRecognizer.frame_hand), frame
times, and the phrase with its annotated start/end relative to the clip. Frames are sampled like
the live app (deadline every 0.05 s) and capped to the live image sides. FSboard's own MediaPipe
landmarks are never read (the Apple Vision lock).

--decoder ffmpeg decodes, rotates and downscales to the live 1280 long side in one ffmpeg pass
(FSboard clips are 3264x1836); --decoder opencv decodes full frames and lets observe_stage2_frame
downscale them. --letters also runs the live letter decoder (phone spell mode) and stores its output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LONG_SIDE = 1280                                             # live maximum_image_side


FOLDER_FPS = 30.                                         # image-folder sequences (ChicagoFSWild): YouTube rate, not stored


def folder_frames(path):
    for f in sorted(Path(path).glob('*.jpg')):
        frame = cv2.imread(str(f))
        if frame is not None:
            yield frame


def video_fps(path):
    if Path(path).is_dir():
        return FOLDER_FPS
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return fps


def container_frames(path):
    if Path(path).is_dir():
        return len(list(Path(path).glob('*.jpg')))
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames',
                          '-of', 'csv=p=0', str(path)], capture_output=True, text=True).stdout.strip().strip(',')
    return int(out) if out.isdigit() else -1


def opencv_frames(path):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)                # clips are rotation-tagged portrait
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def ffmpeg_frames(path, long_side=LONG_SIDE):
    probe = json.loads(subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'stream=width,height:stream_side_data=rotation', '-of', 'json', str(path)],
        capture_output=True, check=True, text=True).stdout)['streams'][0]
    width, height = probe['width'], probe['height']
    rotation = int(next((s.get('rotation', 0) for s in probe.get('side_data_list', [])), 0))
    if rotation % 180:
        width, height = height, width                        # ffmpeg autorotates before the filter
    scale = min(1., long_side / max(width, height))          # same rounding as limit_image_side
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    # passthrough: never duplicate/drop to a declared rate (one clip declares 120 fps). The rawvideo muxer
    # may warn about repeated dts; on 100 pilot clips (6 warn) every frame count still equals OpenCV's.
    proc = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-fps_mode', 'passthrough',
                             '-vf', f'scale={w}:{h}:flags=area',
                             '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def shrink_frame(frame, factor):
    """Simulate a more distant camera: the frame scaled by `factor`, centred on a blurred full-size copy."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, round(w * factor)), max(1, round(h * factor))), interpolation=cv2.INTER_AREA)
    canvas = cv2.GaussianBlur(cv2.resize(frame, (w // 8, h // 8), interpolation=cv2.INTER_AREA), (0, 0), 3)
    canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


class Extractor:
    shrink = 1.0                                             # <1: simulate a more distant camera

    def __init__(self, decoder='ffmpeg', letters=False):
        from active.v17.segmental_runtime_v17 import build_runtime
        from scripts.evaluate_temporal_boundary_v17 import arguments
        self.runtime = build_runtime()
        self.recognizer = self.runtime.words.recognizer
        self.args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')
        self.frames = ffmpeg_frames if decoder == 'ffmpeg' else opencv_frames
        self.letters = letters

    def clip(self, video, clip, target):
        from active.v17.extract_v17 import AppleVisionDetector
        from active.v17.stage1_window_v17 import raw_observation_features
        from scripts.live_stage2_ctc_v17 import observe_stage2_frame
        det = AppleVisionDetector(self.args.minimum_point_confidence)
        fps = video_fps(video)
        letter_rt = self.runtime.letter_rt
        if self.letters:
            self.runtime.shared.clear(); letter_rt.reset()
        wrists, obs, hands, out, j, deadline = {'left': None, 'right': None}, [], [], [], 0, 0.
        cost = dict(decode_s=0., vision_s=0., hands_s=0.)
        shape = None
        frames = folder_frames(video) if Path(video).is_dir() else self.frames(video)
        while True:
            t0 = time.perf_counter()
            frame = next(frames, None)
            cost['decode_s'] += time.perf_counter() - t0
            if frame is None:
                break
            seconds = j / fps; j += 1
            if seconds + 1e-6 < deadline:
                continue
            deadline = max(deadline + .05, seconds)
            if self.shrink < 1:
                frame = shrink_frame(frame, self.shrink)
            shape = frame.shape[:2]
            t0 = time.perf_counter()
            o = observe_stage2_frame(frame, seconds, len(obs), det, wrists, self.args)
            cost['vision_s'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            hands.append(self.recognizer.frame_hand(o))
            cost['hands_s'] += time.perf_counter() - t0
            obs.append(o)
            if self.letters:
                out += letter_rt.observe(o, hands=hands[-1])
        if self.letters:
            out += letter_rt.finish()
        expected = container_frames(video)
        if expected != j:
            print('frame count mismatch', video.name, 'decoded', j, 'container', expected, flush=True)
        raw, times = raw_observation_features(obs)
        offset = clip['clipStartTimeS']
        spelled = ''.join(w['gloss'][3:] for w in out if w['gloss'].startswith('FS_'))
        np.savez_compressed(
            target, raw=raw.astype(np.float16), times=np.asarray(times, np.float64),
            hand_embeddings=np.stack([h[0] for h in hands]).astype(np.float16),
            hand_valid=np.stack([h[1] for h in hands]) > .5,
            hand_boxes=np.stack([h[2] for h in hands]).astype(np.float16),
            phrase=clip['phrase'], signer=clip['signerId'], letter_output=spelled,
            letter_times=np.array([[w['start_seconds'], w['end_seconds']] for w in out if w['gloss'].startswith('FS_')],
                                  np.float64).reshape(-1, 2),
            decoded_frames=j, container_frames=expected,
            annotation=np.array([clip['annotationStartTimeS'] - offset, clip['annotationEndTimeS'] - offset]))
        present = (raw[:, :42, 4] > 0).reshape(len(raw), 2, 21).sum(2) >= 15
        return dict(clip=clip['clipFilename'], signer=clip['signerId'], phrase=clip['phrase'], frames=len(obs),
                    decoded=j, source_shape=list(shape or ()), any_hand=float(present.any(1).mean()),
                    letter_output=spelled, **cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=ROOT / 'data/local/fsboard_v17/pilot/manifest.json')
    ap.add_argument('--videos', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--decoder', choices=('ffmpeg', 'opencv'), default='ffmpeg')
    ap.add_argument('--letters', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    videos = a.videos or a.manifest.parent / 'videos'
    out = a.out or a.manifest.parent / 'features'
    out.mkdir(parents=True, exist_ok=True)
    ex = Extractor(a.decoder, a.letters)
    clips = json.loads(a.manifest.read_text())['clips']
    clips = clips[:a.limit] if a.limit else clips
    summary = []
    for n, clip in enumerate(clips):
        target = out / (Path(clip['clipFilename']).stem + '.npz')
        if target.exists():
            continue
        row = ex.clip(videos / clip['clipFilename'], clip, target)
        summary.append(row)
        print(n, json.dumps(row), flush=True)
    (out / 'extract_summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
