"""
SLT Stage 1 — Real-Time Inference
DS-GCN Encoder + Attention Pooling + Classifier Head
Normalization and kinematics match extract_augment.py exactly.

Key design choices for real-time accuracy:
  - Movement gating: only classifies when the hand is holding steady,
    preventing transition frames between signs from being misclassified.
  - Buffer reset: after accepting a sign the frame buffer clears so
    the next prediction starts fresh (no bleed from previous sign).
  - Cooldown: short pause after acceptance prevents double-triggers
    and lets the user transition to the next sign.
"""

import sys
import time
import argparse
import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
from scipy.interpolate import CubicSpline
from pathlib import Path

# _SRC_DIR is SLT/scripts/src/
_SRC_DIR = Path(__file__).resolve().parent
# _ROOT_DIR is SLT/
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Note: You MUST have train_kaggle.py in the same folder for this to work!
from train_stage_1 import SLTStage1


# ── Feature pipeline (must match extract_augment.py) ──────────

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    seq = seq.copy().astype(np.float64)
    seq -= seq[:, 0:1, :]
    max_val = np.max(np.linalg.norm(seq, axis=-1)) + 1e-8
    seq /= max_val
    return seq.astype(np.float32)


def compute_kinematics(seq: np.ndarray) -> np.ndarray:
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
    N = len(raw)
    if N == target:
        return raw
    x_src = np.linspace(0, 1, N)
    x_dst = np.linspace(0, 1, target)
    flat = raw.reshape(N, -1)
    if N >= 4:
        cs = CubicSpline(x_src, flat, bc_type='not-a-knot')
        return cs(x_dst).reshape(target, 21, 3).astype(np.float32)
    from scipy.interpolate import interp1d
    f = interp1d(x_src, flat, axis=0, kind='linear')
    return f(x_dst).reshape(target, 21, 3).astype(np.float32)


# ── Hand stability detection ─────────────────────────────────

def hand_velocity(buffer, n_frames=5):
    """Mean per-landmark displacement over the last n frames.
    Returns a scalar: low = steady, high = moving."""
    if len(buffer) < 2:
        return float('inf')
    recent = list(buffer)[-min(n_frames + 1, len(buffer)):]
    disps = []
    for i in range(1, len(recent)):
        diff = np.array(recent[i]) - np.array(recent[i - 1])
        disps.append(np.mean(np.abs(diff)))
    return float(np.mean(disps))


# ── CLI ───────────────────────────────────────────────────────

DYNAMIC_SIGNS = {'J', 'Z'}

def parse_args():
    p = argparse.ArgumentParser(description='SLT Stage 1 — Real-Time Inference')
    
    # UPDATED PATH: Uses _ROOT_DIR to correctly step back to the main SLT folder
    p.add_argument('--model', '-m', type=str,
                   default=str(_ROOT_DIR / 'weights' / 'SLT_Stage1_Results' / 'best_model.pth'),
                   help='Path to your trained weights file')
    
    p.add_argument('--camera', '-c', type=int, default=0)
    p.add_argument('--threshold', '-t', type=float, default=0.80,
                   help='Min confidence to consider a prediction (default: 0.80)')
    p.add_argument('--stability', '-s', type=int, default=5,
                   help='Max consecutive agreeing frames to accept (default: 5)')
    p.add_argument('--cooldown', type=int, default=8,
                   help='Frames to skip after accepting a sign (default: 8)')
    p.add_argument('--velocity', type=float, default=0.008,
                   help='Max hand velocity to allow classification (default: 0.008)')
    p.add_argument('--margin', type=float, default=0.10,
                   help='Min gap between top-1 and top-2 confidence (default: 0.10)')
    p.add_argument('--conf-dynamic', type=float, default=0.90,
                   help='Min confidence for J/Z (default: 0.90)')
    p.add_argument('--margin-dynamic', type=float, default=0.18,
                   help='Min margin for J/Z (default: 0.18)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    WINDOW       = 32
    CONF_THRESH  = args.threshold
    STAB_THRESH  = args.stability
    COOLDOWN_MAX = args.cooldown
    VEL_THRESH   = args.velocity
    MARGIN_THRESH = args.margin
    MIN_FRAMES   = 16

    FAST_THRESH  = 0.95
    FAST_FRAMES  = 3
    MED_THRESH   = 0.90
    MED_FRAMES   = 4

    CONF_DYNAMIC   = args.conf_dynamic
    MARGIN_DYNAMIC = args.margin_dynamic

    # M4 Air will automatically select 'mps' here!
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load checkpoint ───────────────────────────────────────
    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f'Checkpoint not found: {ckpt_path}')
        print('Please make sure your weights are inside SLT/weights/SLT_Stage1_Results/')
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    idx_to_label = ckpt['idx_to_label']
    num_classes  = ckpt['num_classes']
    labels = [idx_to_label[str(i)] for i in range(num_classes)]

    model = SLTStage1(
        num_classes=num_classes,
        in_channels=ckpt.get('in_channels', 9),
        d_model=ckpt.get('d_model', 256),
        nhead=ckpt.get('nhead', 8),
        num_transformer_layers=ckpt.get('num_transformer_layers', 4),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dynamic_indices = [i for i, l in enumerate(labels) if l in DYNAMIC_SIGNS]

    val_acc = ckpt.get('val_acc', ckpt.get('best_acc', 0))
    print(f'Loaded: {num_classes} classes | val acc {val_acc:.2f}%')
    print(f'Labels: {labels}')
    print(f'Dynamic signs (motion-only): {[labels[i] for i in dynamic_indices]}')
    print(f'Conf threshold : {CONF_THRESH*100:.0f}%')
    print(f'Stability      : {STAB_THRESH} frames')
    print(f'Velocity gate  : {VEL_THRESH}')
    print(f'Margin         : {MARGIN_THRESH*100:.0f}%')
    print(f'J/Z conf       : {CONF_DYNAMIC*100:.0f}%')
    print(f'J/Z margin     : {MARGIN_DYNAMIC*100:.0f}%')
    print(f'Cooldown       : {COOLDOWN_MAX} frames')
    print(f'Min frames     : {MIN_FRAMES}')
    print()
    print('Controls:  q = quit | c = clear | BKSP = undo | SPACE = separator')
    print('Tip: hold each sign steady until the lock bar fills.\n')

    # ── MediaPipe ─────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        model_complexity=1,
        max_num_hands=1,
    )

    # ── State ─────────────────────────────────────────────────
    raw_buffer: deque    = deque(maxlen=48)
    sentence: list       = []
    stability_buf: list  = []
    cooldown             = 0
    fps_q                = deque(maxlen=30)

    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    FONT_S = cv2.FONT_HERSHEY_PLAIN

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Cannot open camera {args.camera}')
        sys.exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        pred_label = '...'
        pred_conf  = 0.0
        top_preds  = []
        hand_on    = False
        vel        = float('inf')
        is_stable  = False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if cooldown > 0:
            cooldown -= 1

        if res.multi_hand_landmarks and res.multi_handedness:
            hand_lms  = res.multi_hand_landmarks[0]
            hand_conf = res.multi_handedness[0].classification[0].score
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            hand_on = True

            if hand_conf >= 0.7:
                raw_buffer.append([[l.x, l.y, l.z] for l in hand_lms.landmark])

            vel = hand_velocity(raw_buffer)
            is_stable = vel < VEL_THRESH

            if len(raw_buffer) >= MIN_FRAMES and is_stable and cooldown == 0:
                n_use  = min(len(raw_buffer), 40)
                raw_frames = list(raw_buffer)[-n_use:]
                raw_np = np.array(raw_frames, dtype=np.float32)
                interp = interpolate_to_target(raw_np, WINDOW)
                normed = normalize_sequence(interp)
                features = compute_kinematics(normed)

                motion_count = 0
                for fi in range(1, len(raw_frames)):
                    diff = np.array(raw_frames[fi]) - np.array(raw_frames[fi - 1])
                    if float(np.mean(np.abs(diff))) > VEL_THRESH:
                        motion_count += 1
                had_motion = motion_count >= 5

                with torch.no_grad():
                    t = torch.from_numpy(features).unsqueeze(0).to(device)
                    logits = model(t)
                    prob   = torch.softmax(logits, dim=1)[0]

                    if not had_motion and dynamic_indices:
                        for di in dynamic_indices:
                            prob[di] = 0.0
                        prob = prob / (prob.sum() + 1e-8)

                    vals, idxs = torch.topk(prob, min(5, num_classes))
                    top_preds  = [(labels[i.item()], v.item())
                                  for v, i in zip(vals, idxs)]
                    pred_label = top_preds[0][0]
                    pred_conf  = top_preds[0][1]

                margin_ok = (len(top_preds) < 2
                             or top_preds[0][1] - top_preds[1][1] >= MARGIN_THRESH)

                if pred_label in DYNAMIC_SIGNS:
                    dynamic_ok = (pred_conf >= CONF_DYNAMIC
                                  and (len(top_preds) < 2
                                       or top_preds[0][1] - top_preds[1][1] >= MARGIN_DYNAMIC))
                else:
                    dynamic_ok = True

                if pred_conf > CONF_THRESH and margin_ok and dynamic_ok:
                    stability_buf.append(pred_label)
                else:
                    stability_buf.append(None)
                stability_buf = stability_buf[-STAB_THRESH:]

                if pred_conf >= FAST_THRESH:
                    needed = FAST_FRAMES
                elif pred_conf >= MED_THRESH:
                    needed = MED_FRAMES
                else:
                    needed = STAB_THRESH

                if (len(stability_buf) >= needed
                        and all(x == stability_buf[-1]
                                for x in stability_buf[-needed:])
                        and stability_buf[-1] is not None):
                    accepted = stability_buf[-1]
                    if not sentence or sentence[-1] != accepted:
                        sentence.append(accepted)
                    raw_buffer.clear()
                    stability_buf.clear()
                    cooldown = COOLDOWN_MAX

            elif not is_stable:
                stability_buf.clear()
        else:
            raw_buffer.clear()
            stability_buf.clear()

        # ── FPS ───────────────────────────────────────────────
        dt = time.perf_counter() - t0
        fps_q.append(dt)
        fps = len(fps_q) / sum(fps_q)

        # ── HUD ───────────────────────────────────────────────
        hud_h = 155
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (fw, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if cooldown > 0:
            clr = (200, 200, 50)
        elif pred_conf > CONF_THRESH:
            clr = (0, 255, 100)
        elif pred_conf > 0.5:
            clr = (0, 200, 255)
        else:
            clr = (80, 80, 80)
        cv2.putText(frame, pred_label, (20, 50), FONT, 1.6, clr, 3)

        # Confidence bar
        bx, by, bw = 200, 18, 200
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 16), (50, 50, 50), -1)
        fill = int(pred_conf * bw)
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 16), clr, -1)
        cv2.putText(frame, f'{pred_conf*100:.0f}%',
                    (bx + bw + 8, by + 14), FONT, 0.5, (200, 200, 200), 1)

        # Top-3 predictions
        for i, (lbl, c) in enumerate(top_preds[:3]):
            yp = 46 + i * 20
            fw_bar = int(c * 130)
            cv2.rectangle(frame, (bx, yp), (bx + 130, yp + 16), (40, 40, 40), -1)
            cv2.rectangle(frame, (bx, yp), (bx + fw_bar, yp + 16),
                          (90, 90, 180), -1)
            cv2.putText(frame, f'{lbl}: {c*100:.0f}%', (bx + 4, yp + 12),
                        FONT_S, 1.0, (220, 220, 220), 1)

        # Status badges
        sx = bx + bw + 60

        hand_clr = (0, 255, 0) if hand_on else (0, 0, 200)
        cv2.putText(frame, 'HAND' if hand_on else 'NO HAND',
                    (sx, 28), FONT, 0.45, hand_clr, 1)

        if hand_on:
            if cooldown > 0:
                stab_txt, stab_clr = f'WAIT {cooldown}', (200, 200, 50)
            elif is_stable:
                stab_txt, stab_clr = 'STEADY', (0, 255, 0)
            else:
                stab_txt, stab_clr = 'MOVING', (0, 100, 255)
            cv2.putText(frame, stab_txt, (sx, 48), FONT, 0.45, stab_clr, 1)

        cv2.putText(frame, f'BUF {len(raw_buffer)}/{MIN_FRAMES}+',
                    (sx, 66), FONT, 0.4, (130, 130, 130), 1)

        vel_display = f'{vel:.4f}' if vel < 1.0 else 'N/A'
        cv2.putText(frame, f'VEL {vel_display}',
                    (sx, 82), FONT, 0.35, (130, 130, 130), 1)

        cv2.putText(frame, f'FPS {fps:.0f}',
                    (fw - 90, 25), FONT, 0.5, (130, 130, 130), 1)
        cv2.putText(frame, str(device).upper(),
                    (fw - 90, 45), FONT, 0.4, (90, 90, 90), 1)

        # Stability lock progress bar (adaptive)
        if pred_conf >= FAST_THRESH:
            cur_needed = FAST_FRAMES
        elif pred_conf >= MED_THRESH:
            cur_needed = MED_FRAMES
        else:
            cur_needed = STAB_THRESH
        n_locked = sum(1 for x in stability_buf if x is not None)
        lock_pct = min(n_locked / cur_needed, 1.0)
        cv2.rectangle(frame, (20, 112), (180, 122), (50, 50, 50), -1)
        if n_locked > 0:
            bar_clr = (0, 255, 100) if pred_conf >= FAST_THRESH else (0, 200, 255)
            cv2.rectangle(frame, (20, 112), (20 + int(160 * lock_pct), 122),
                          bar_clr, -1)
        cv2.putText(frame, f'Lock: {n_locked}/{cur_needed}',
                    (185, 121), FONT, 0.35, (150, 150, 150), 1)

        # Sentence
        sent_str = ' '.join(sentence) if sentence else '(hold a sign steady)'
        sent_clr = (255, 255, 255) if sentence else (90, 90, 90)
        cv2.putText(frame, sent_str, (20, 147), FONT, 0.6, sent_clr, 2)

        cv2.line(frame, (0, hud_h), (fw, hud_h), (50, 50, 50), 1)
        cv2.imshow('SLT Stage 1 — Real-Time Inference', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence.clear()
            stability_buf.clear()
            raw_buffer.clear()
            cooldown = 0
        elif key in (8, 127):
            if sentence:
                sentence.pop()
        elif key == ord(' '):
            sentence.append('|')

    cap.release()
    cv2.destroyAllWindows()
    print(f'\nFinal: {" ".join(sentence)}')


if __name__ == '__main__':
    main()