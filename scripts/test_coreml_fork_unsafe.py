"""Test CoreML with fork safety disabled."""
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

from extract_batch_fast_v2 import preprocess_frame, decode_simcc_batch, postprocess_keypoints
from rtmlib import YOLOX

RTMW = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx')
DET = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx')

_sess = None
_det = None

def worker_init():
    global _sess, _det
    _sess = ort.InferenceSession(RTMW, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    _det = YOLOX(DET, model_input_size=(640, 640))
    # Warmup
    _sess.run(None, {'input': np.random.randn(1, 3, 384, 288).astype(np.float32)})
    print(f"Worker {os.getpid()} ready: {_sess.get_providers()}", flush=True)

def process_video(vp):
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(f)
    cap.release()
    if len(frames) < 8: return None

    h0, w0 = frames[0].shape[:2]
    # No YOLOX — full frame
    bbox = np.array([0, 0, w0, h0], dtype=np.float32)

    preprocessed, centers, scales = [], [], []
    for f in frames:
        img, c, s = preprocess_frame(f, bbox=bbox)
        preprocessed.append(img); centers.append(c); scales.append(s)

    for i in range(len(preprocessed)):
        _sess.run(None, {'input': np.stack([preprocessed[i]]).astype(np.float32)})

    return os.path.basename(vp)


if __name__ == '__main__':
    videos = sorted(glob.glob('data/raw_videos/ASL VIDEOS/HELLO/*.mp4'))[:30]

    for n_workers in [1, 2, 4]:
        print(f"\n=== {n_workers} CoreML worker(s), fork safety DISABLED ===", flush=True)
        t0 = time.time()
        if n_workers == 1:
            worker_init()
            for vp in videos:
                process_video(vp)
        else:
            with mp.Pool(n_workers, initializer=worker_init) as pool:
                list(pool.map(process_video, videos))
        elapsed = time.time() - t0
        avg = elapsed / len(videos)
        print(f"  {len(videos)} videos in {elapsed:.1f}s = {avg:.2f}s/video")
        print(f"  Estimated 67k: {67000 * avg / 3600:.1f} hours")
