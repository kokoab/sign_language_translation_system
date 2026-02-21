"""
SLT Stage 1 — Real-Time Inference
Uses the same DSGCNEncoder + ClassifierHead as train_kaggle.py.
Normalization matches extract_augment.py exactly.
"""

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from scipy.interpolate import CubicSpline

from train_kaggle import SLTStage1

# ── Device ────────────────────────────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)

# ── Load Checkpoint ───────────────────────────────────────────────
CKPT_PATH = "weights/best_model.pth"
try:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    idx_to_label = ckpt['idx_to_label']
    labels = [idx_to_label[str(i)] for i in range(ckpt['num_classes'])]

    model = SLTStage1(
        num_classes=ckpt['num_classes'],
        in_channels=ckpt['in_channels'],
        d_model=ckpt['d_model'],
        nhead=ckpt['nhead'],
        num_transformer_layers=ckpt['num_transformer_layers'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded: {labels}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ── MediaPipe ─────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
    model_complexity=1,
    max_num_hands=1,
)

WINDOW_SIZE = 32
CONF_THRESHOLD = 0.85
STABILITY_THRESHOLD = 4


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Must match extract_augment.py exactly."""
    seq = seq.copy().astype(np.float64)
    seq -= seq[:, 0:1, :]
    dists = np.linalg.norm(seq, axis=-1)
    max_val = np.max(dists) + 1e-8
    seq /= max_val
    return seq.astype(np.float32)


def compute_kinematics(seq: np.ndarray) -> np.ndarray:
    """Must match extract_augment.py exactly."""
    vel = np.zeros_like(seq)
    vel[1:-1] = (seq[2:] - seq[:-2]) / 2.0
    vel[0]  = seq[1]  - seq[0]
    vel[-1] = seq[-1] - seq[-2]

    acc = np.zeros_like(seq)
    acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
    acc[0]  = vel[1]  - vel[0]
    acc[-1] = vel[-1] - vel[-2]

    return np.concatenate([seq, vel, acc], axis=-1)


def interpolate_to_target(raw: np.ndarray, target: int = 32) -> np.ndarray:
    """Cubic interpolation to fixed frame count."""
    N = len(raw)
    if N == target:
        return raw
    x_src = np.linspace(0, 1, N)
    x_dst = np.linspace(0, 1, target)
    if N >= 4:
        flat = raw.reshape(N, -1)
        cs = CubicSpline(x_src, flat, bc_type='not-a-knot')
        return cs(x_dst).reshape(target, 21, 3).astype(np.float32)
    else:
        from scipy.interpolate import interp1d
        flat = raw.reshape(N, -1)
        f = interp1d(x_src, flat, axis=0, kind='linear')
        return f(x_dst).reshape(target, 21, 3).astype(np.float32)


# ── Main Loop ─────────────────────────────────────────────────────
raw_buffer = deque(maxlen=64)
sentence = []
stability_buffer = []

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    current_char, current_conf = "...", 0.0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks and res.multi_handedness:
        hand_lms = res.multi_hand_landmarks[0]
        conf = res.multi_handedness[0].classification[0].score
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        if conf >= 0.7:
            pts = [[l.x, l.y, l.z] for l in hand_lms.landmark]
            raw_buffer.append(pts)

        if len(raw_buffer) >= 10:
            raw_np = np.array(list(raw_buffer), dtype=np.float32)
            interp = interpolate_to_target(raw_np, WINDOW_SIZE)
            normed = normalize_sequence(interp)
            features = compute_kinematics(normed)

            with torch.no_grad():
                tensor = torch.from_numpy(features).unsqueeze(0).to(device)
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)
                conf_val, idx = torch.max(prob, 1)
                current_char = labels[idx.item()]
                current_conf = conf_val.item()

                if current_conf > CONF_THRESHOLD:
                    stability_buffer.append(current_char)
                else:
                    stability_buffer.append(None)
                stability_buffer = stability_buffer[-STABILITY_THRESHOLD:]

                if (len(stability_buffer) == STABILITY_THRESHOLD and
                        stability_buffer[0] is not None and
                        all(x == stability_buffer[0] for x in stability_buffer)):
                    if not sentence or sentence[-1] != current_char:
                        sentence.append(current_char)
    else:
        raw_buffer.clear()

    # ── HUD ───────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 110), (30, 30, 30), -1)
    bar_w = int(current_conf * 200)
    cv2.rectangle(frame, (20, 55), (220, 65), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 55), (20 + bar_w, 65), (0, 255, 100), -1)

    color = (0, 255, 0) if current_conf > CONF_THRESHOLD else (0, 150, 255)
    cv2.putText(frame, f"{current_char}", (20, 40), 1, 2, color, 2)
    cv2.putText(frame, f"{current_conf*100:.0f}%", (230, 65), 1, 1, (200, 200, 200), 1)
    cv2.putText(frame, " ".join(sentence), (20, 95), 1, 1.8, (255, 255, 255), 2)

    cv2.imshow('SLT Stage 1 - Real-Time', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        sentence = []

cap.release()
cv2.destroyAllWindows()
