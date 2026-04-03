"""Test CoreML with async video reading (overlap I/O with inference)."""
import os, sys, glob, time, queue, threading
import numpy as np
import cv2

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
DET_PATH = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx')

# Single CoreML session
sess = ort.InferenceSession(RTMW, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
det = YOLOX(DET_PATH, model_input_size=(640, 640))

# Warmup
sess.run(None, {'input': np.random.randn(1, 3, 384, 288).astype(np.float32)})

print(f"Provider: {sess.get_providers()}")

def read_and_preprocess(vp):
    """Read video and preprocess all frames (CPU-bound)."""
    cap = cv2.VideoCapture(vp)
    frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(f)
    cap.release()
    if len(frames) < 8:
        return None

    total_frames = len(frames)
    indices = list(range(total_frames))
    if total_frames > 128:
        step = total_frames / 128
        indices = [int(i * step) for i in range(128)]
        frames = [frames[i] for i in indices]

    h0, w0 = frames[0].shape[:2]
    bboxes = det(frames[0])
    bbox = bboxes[0][:4].astype(np.float32) if bboxes is not None and len(bboxes) > 0 else np.array([0, 0, w0, h0], dtype=np.float32)

    preprocessed, centers, scales = [], [], []
    for f in frames:
        img, c, s = preprocess_frame(f, bbox=bbox)
        preprocessed.append(img)
        centers.append(c)
        scales.append(s)

    return {
        'prep': np.stack(preprocessed).astype(np.float32),
        'centers': np.stack(centers),
        'scales': np.stack(scales),
        'name': os.path.basename(vp),
        'n_frames': len(frames),
    }


def run_inference(data):
    """Run CoreML inference on preprocessed frames."""
    if data is None:
        return None
    all_kps, all_scs = [], []
    for i in range(len(data['prep'])):
        out = sess.run(None, {'input': data['prep'][i:i+1]})
        locs, scores = decode_simcc_batch(out[0], out[1])
        kps, scs = postprocess_keypoints(locs, scores, data['centers'][i:i+1], data['scales'][i:i+1])
        all_kps.append(kps[0])
        all_scs.append(scs[0])
    return data['name']


# Pipeline: read thread → queue → inference thread
videos = sorted(glob.glob('data/raw_videos/ASL VIDEOS/HELLO/*.mp4'))[:30]

# === Test 1: Sequential (baseline) ===
print(f"\n=== Sequential (30 videos) ===", flush=True)
t0 = time.time()
for vp in videos:
    data = read_and_preprocess(vp)
    run_inference(data)
elapsed_seq = time.time() - t0
print(f"  {elapsed_seq:.1f}s = {elapsed_seq/30:.2f}s/video")
print(f"  ETA 67k: {67000 * elapsed_seq / 30 / 3600:.1f} hours")

# === Test 2: Async pipeline (read ahead while inferring) ===
print(f"\n=== Async pipeline (read ahead) ===", flush=True)
data_queue = queue.Queue(maxsize=4)  # Buffer 4 videos ahead
done = threading.Event()

def reader_thread(vids):
    for vp in vids:
        data = read_and_preprocess(vp)
        data_queue.put(data)
    data_queue.put(None)  # Sentinel

t0 = time.time()
reader = threading.Thread(target=reader_thread, args=(videos,))
reader.start()

count = 0
while True:
    data = data_queue.get()
    if data is None:
        break
    run_inference(data)
    count += 1

reader.join()
elapsed_async = time.time() - t0
print(f"  {elapsed_async:.1f}s = {elapsed_async/30:.2f}s/video")
print(f"  ETA 67k: {67000 * elapsed_async / 30 / 3600:.1f} hours")
print(f"  Speedup: {elapsed_seq/elapsed_async:.2f}x")
