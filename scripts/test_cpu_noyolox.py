"""Test CPU multiprocessing without YOLOX."""
import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import multiprocessing as mp
import sys, glob, time
import numpy as np
import cv2
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import types
_f = types.ModuleType('mediapipe')
_f.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _f
sys.modules['mediapipe.solutions'] = _f.solutions

from extract_batch_fast_v2 import preprocess_frame, batch_inference

RTMW = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx')

_sess = None

def worker_init():
    global _sess
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    _sess = ort.InferenceSession(RTMW, sess_options=opts, providers=['CPUExecutionProvider'])

def process_video(vp):
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(f)
    cap.release()
    if len(frames) < 8: return None

    h, w = frames[0].shape[:2]
    bbox = np.array([0, 0, w, h], dtype=np.float32)  # No YOLOX

    preprocessed, centers, scales = [], [], []
    for f in frames:
        img, c, s = preprocess_frame(f, bbox=bbox)
        preprocessed.append(img)
        centers.append(c)
        scales.append(s)

    batch_inference(_sess, np.stack(preprocessed).astype(np.float32),
                    np.stack(centers), np.stack(scales))
    return 1


if __name__ == '__main__':
    videos = sorted(glob.glob('data/raw_videos/ASL VIDEOS/HELLO/*.mp4'))[:30]

    for n in [4, 6, 8]:
        print(f"\n=== {n} CPU workers, no YOLOX ===", flush=True)
        t0 = time.time()
        with mp.Pool(n, initializer=worker_init) as pool:
            list(pool.map(process_video, videos))
        elapsed = time.time() - t0
        avg = elapsed / len(videos)
        print(f"  {len(videos)} videos in {elapsed:.1f}s = {avg:.2f}s/video")
        print(f"  Estimated 67k: {67000 * avg / 3600:.1f} hours")
