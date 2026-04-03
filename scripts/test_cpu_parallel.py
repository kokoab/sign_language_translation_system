"""Test CPU multiprocessing extraction on Mac M4."""
import multiprocessing as mp
import os, sys, glob, time
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
from rtmlib import YOLOX

RTMW = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx')
DET = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx')
N_THREADS = 1  # Will be set per test

_sess = None
_det = None

def worker_init():
    global _sess, _det
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = N_THREADS
    opts.inter_op_num_threads = 1
    _sess = ort.InferenceSession(RTMW, sess_options=opts, providers=['CPUExecutionProvider'])
    _det = YOLOX(DET, model_input_size=(640, 640))

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

    total = len(frames)
    indices = list(range(total))
    if total > 128:
        step = total / 128
        indices = [int(i * step) for i in range(128)]
        frames = [frames[i] for i in indices]

    h0, w0 = frames[0].shape[:2]
    bboxes = _det(frames[0])
    bbox = bboxes[0][:4].astype(np.float32) if bboxes is not None and len(bboxes) > 0 else np.array([0, 0, w0, h0], dtype=np.float32)

    preprocessed, centers, scales = [], [], []
    for f in frames:
        img, c, s = preprocess_frame(f, bbox=bbox)
        preprocessed.append(img)
        centers.append(c)
        scales.append(s)

    # Use batch inference (batch=32)
    all_kps, all_scs = batch_inference(
        _sess,
        np.stack(preprocessed).astype(np.float32),
        np.stack(centers),
        np.stack(scales))

    return os.path.basename(vp)


if __name__ == '__main__':
    videos = sorted(glob.glob('data/raw_videos/ASL VIDEOS/HELLO/*.mp4'))
    test_vids = videos[:30]

    configs = [
        (1, 1),   # 1 worker, 1 thread (baseline)
        (2, 4),   # 2 workers, 4 threads each = 8 threads total
        (4, 2),   # 4 workers, 2 threads each = 8 threads total
        (6, 1),   # 6 workers, 1 thread each = 6 threads total
        (8, 1),   # 8 workers, 1 thread each = 8 threads total
    ]

    for n_workers, n_threads in configs:
        N_THREADS = n_threads
        print(f"\n=== {n_workers} workers × {n_threads} threads = {n_workers*n_threads} total ===", flush=True)
        t0 = time.time()
        if n_workers == 1:
            worker_init()
            for vp in test_vids:
                process_video(vp)
        else:
            with mp.Pool(n_workers, initializer=worker_init) as pool:
                list(pool.map(process_video, test_vids))
        elapsed = time.time() - t0
        avg = elapsed / len(test_vids)
        print(f"  {len(test_vids)} videos in {elapsed:.1f}s = {avg:.2f}s/video")
        print(f"  Estimated 67k: {67000 * avg / 3600:.1f} hours")
