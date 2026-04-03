"""Launch multiple independent CoreML extraction processes."""
import subprocess, os, sys, glob, time, math

videos_dir = "data/raw_videos/ASL VIDEOS"
all_videos = []
for class_dir in sorted(os.listdir(videos_dir)):
    class_path = os.path.join(videos_dir, class_dir)
    if not os.path.isdir(class_path): continue
    for vid in sorted(os.listdir(class_path)):
        if vid.endswith('.mp4'):
            all_videos.append(os.path.join(class_path, vid))

# Test with 30 videos
test_videos = all_videos[:30]
N_PROCS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# Split videos into chunks
chunks = [[] for _ in range(N_PROCS)]
for i, v in enumerate(test_videos):
    chunks[i % N_PROCS].append(v)

# Write chunk files
for i, chunk in enumerate(chunks):
    with open(f'/tmp/coreml_chunk_{i}.txt', 'w') as f:
        f.write('\n'.join(chunk))

# Worker script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
worker_script = '''
import os, sys, time
import numpy as np
import cv2
sys.path.insert(0, "PROJECT_ROOT")
sys.path.insert(0, "PROJECT_ROOT/src")'''.replace('PROJECT_ROOT', project_root) + '''
import types
_f = types.ModuleType('mediapipe')
_f.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _f
sys.modules['mediapipe.solutions'] = _f.solutions
import onnxruntime as ort
from extract_batch_fast_v2 import preprocess_frame, decode_simcc_batch, postprocess_keypoints
from rtmlib import YOLOX

rtmw = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx')
det_path = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx')

sess = ort.InferenceSession(rtmw, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
det = YOLOX(det_path, model_input_size=(640, 640))
sess.run(None, {'input': np.random.randn(1, 3, 384, 288).astype(np.float32)})  # warmup

chunk_file = sys.argv[1]
worker_id = sys.argv[2]
with open(chunk_file) as f:
    videos = [l.strip() for l in f if l.strip()]

done = 0
t0 = time.time()
for vp in videos:
    cap = cv2.VideoCapture(vp); frames = []
    while True:
        r, f = cap.read()
        if not r: break
        frames.append(f)
    cap.release()
    if len(frames) < 8: continue

    h0, w0 = frames[0].shape[:2]
    bboxes = det(frames[0])
    bbox = bboxes[0][:4].astype(np.float32) if bboxes is not None and len(bboxes) > 0 else np.array([0,0,w0,h0], dtype=np.float32)
    preprocessed, centers, scales = [], [], []
    for f in frames:
        img, c, s = preprocess_frame(f, bbox=bbox)
        preprocessed.append(img); centers.append(c); scales.append(s)

    for i in range(len(preprocessed)):
        sess.run(None, {'input': np.stack([preprocessed[i]]).astype(np.float32)})

    done += 1

elapsed = time.time() - t0
print(f"Worker {worker_id}: {done} videos in {elapsed:.1f}s = {elapsed/max(done,1):.2f}s/video", flush=True)
'''

# Save worker script
worker_path = '/tmp/coreml_worker.py'
with open(worker_path, 'w') as f:
    f.write(worker_script)

print(f"Launching {N_PROCS} independent CoreML processes on {len(test_videos)} videos...")
t0 = time.time()

procs = []
for i in range(N_PROCS):
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    p = subprocess.Popen(
        [sys.executable, worker_path, f'/tmp/coreml_chunk_{i}.txt', str(i)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd()
    )
    procs.append(p)

# Wait for all
for p in procs:
    stdout, stderr = p.communicate()
    out = stdout.decode().strip()
    if out:
        print(f"  {out}")

elapsed = time.time() - t0
avg = elapsed / len(test_videos)
print(f"\nTotal: {elapsed:.1f}s for {len(test_videos)} videos = {avg:.2f}s/video")
print(f"Estimated 67k: {67000 * avg / 3600:.1f} hours ({N_PROCS} processes)")
