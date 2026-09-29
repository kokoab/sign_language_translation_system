"""Score a whole video with the FINGERSPELL detector: prints the per-window probability over time."""
import pickle, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from scripts.evaluate_temporal_boundary_v17 import arguments
from scripts.live_stage2_ctc_v17 import observe_stage2_frame
from active.v17.extract_v17 import AppleVisionDetector
from active.v17.stage1_window_v17 import raw_observation_features
from train_detector import window_features, WINDOW, STEP
args = arguments('data', ROOT / 'artifacts/generated/tmp_sessions')
m = pickle.load(open(HERE / 'model_boosted.pkl', 'rb'))
for video in sys.argv[1:]:
    det = AppleVisionDetector(args.minimum_point_confidence)
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    wr, j, deadline, obs = {'left': None, 'right': None}, 0, 0., []
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = j / fps; j += 1
        if t + 1e-6 < deadline: continue
        deadline = max(deadline + .05, t)
        obs.append(observe_stage2_frame(frame, t, len(obs), det, wr, args))
    raw, times = raw_observation_features(obs)
    s = m['model'].predict_proba(window_features(raw))[:, 1]
    s = np.convolve(s, np.ones(3) / 3, mode='same') if len(s) >= 3 else s
    print(video, 'threshold', round(m['threshold'], 3), 'max', round(float(s.max()), 3))
    print('  ', ' '.join(f'{times[min(i * STEP + WINDOW - 1, len(times) - 1)]:.1f}:{v:.2f}' for i, v in enumerate(s)))
