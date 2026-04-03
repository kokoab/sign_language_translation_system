"""Test CoreML multiprocessing with fork + initializer."""
import multiprocessing as mp
import os, sys, glob, time
import numpy as np
import cv2

# Force fork on macOS (CoreML needs this)
mp.set_start_method('fork', force=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import types
_f = types.ModuleType('mediapipe')
_f.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _f
sys.modules['mediapipe.solutions'] = _f.solutions

from extract_batch_fast_v2 import preprocess_frame, decode_simcc_batch, postprocess_keypoints
from rtmlib import YOLOX
import onnxruntime as ort

RTMW = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx')
DET = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx')

_sess = None
_det = None

def worker_init():
    global _sess, _det
    _sess = ort.InferenceSession(RTMW, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    _det = YOLOX(DET, model_input_size=(640, 640))
    _sess.run(None, {'input': np.random.randn(1, 3, 384, 288).astype(np.float32)})

def process_video(vp):
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(f)
    cap.release()
    if len(frames) < 8:
        return None

    h0, w0 = frames[0].shape[:2]
    bboxes = _det(frames[0])
    bbox = bboxes[0][:4].astype(np.float32) if bboxes is not None and len(bboxes) > 0 else np.array([0, 0, w0, h0], dtype=np.float32)

    preprocessed, centers, scales = [], [], []
    for f in frames:
        img, c, s = preprocess_frame(f, bbox=bbox)
        preprocessed.append(img)
        centers.append(c)
        scales.append(s)

    for i in range(len(preprocessed)):
        _sess.run(None, {'input': np.stack([preprocessed[i]]).astype(np.float32)})

    return os.path.basename(vp)

if __name__ == '__main__':
    videos = sorted(glob.glob('data/raw_videos/ASL VIDEOS/HELLO/*.mp4'))[:20]

    for n in [1, 2, 4]:
        print(f"\n=== {n} worker(s) ===", flush=True)
        t0 = time.time()
        if n == 1:
            worker_init()
            for vp in videos:
                process_video(vp)
        else:
            with mp.Pool(n, initializer=worker_init) as pool:
                list(pool.map(process_video, videos))
        elapsed = time.time() - t0
        avg = elapsed / len(videos)
        print(f"  {len(videos)} videos in {elapsed:.1f}s = {avg:.2f}s/video")
        print(f"  Estimated 67k: {67000 * avg / 3600:.1f} hours")
