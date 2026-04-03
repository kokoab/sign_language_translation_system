"""
Debug tool: compare extracted keypoints from a webcam recording vs training .npy
Shows exactly where the numerical differences are.

Usage: python src/debug_extraction.py
Records from webcam, extracts, compares stats with a training sample.
"""
import os, sys, cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract_batch_fast_v2 import preprocess_frame, batch_inference, normalize_sequence_v2, compute_kinematics_v2, interpolate_generic
from extract import interpolate_hand, temporal_resample, one_euro_filter, stabilize_bones, reject_temporal_outliers
from train_stage_1 import compute_bone_features_np
import onnxruntime as ort


def extract_from_frames(frames, session, det_model=None):
    _LHAND, _RHAND = 91, 112
    _FACE_INDICES = [53,31,50,23,39, 71,77,74,80, 40,44,45,49, 59,68]
    _BODY_INDICES = [5,6,7,8]

    T = len(frames)
    h0, w0 = frames[0].shape[:2]
    bbox = np.array([0,0,w0,h0], dtype=np.float32)
    if det_model is not None:
        try:
            bboxes = det_model(frames[0])
            if bboxes is not None and len(bboxes) > 0:
                bbox = bboxes[0][:4].astype(np.float32)
        except: pass

    prep, ctrs, scls = [], [], []
    for f in frames:
        img, c, s = preprocess_frame(f, bbox=bbox)
        prep.append(img); ctrs.append(c); scls.append(s)
    ctrs = np.stack(ctrs); scls = np.stack(scls)

    all_kps, all_scs = [], []
    for st in range(0, len(prep), 32):
        en = min(st+32, len(prep))
        b = np.stack(prep[st:en]).astype(np.float32)
        k, s = batch_inference(session, b, ctrs[st:en], scls[st:en])
        all_kps.append(k); all_scs.append(s)
    all_kps = np.concatenate(all_kps); all_scs = np.concatenate(all_scs)

    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []
    for i in range(len(frames)):
        kps, scs = all_kps[i], all_scs[i]
        h, w = frames[i].shape[:2]
        if scs[_LHAND:_LHAND+21].mean() >= 0.25:
            l_seq.append([[kps[_LHAND+j][0]/w, kps[_LHAND+j][1]/h, 0.0] for j in range(21)]); l_valid.append(i)
        if scs[_RHAND:_RHAND+21].mean() >= 0.25:
            r_seq.append([[kps[_RHAND+j][0]/w, kps[_RHAND+j][1]/h, 0.0] for j in range(21)]); r_valid.append(i)
        fc = []
        for fi in _FACE_INDICES:
            if scs[fi] >= 0.25: fc.append([kps[fi][0]/w, kps[fi][1]/h, 0.0])
            else: break
        if len(fc) == 15: face_seq.append(fc); face_valid.append(i)
        bc = []
        for bi in _BODY_INDICES:
            if scs[bi] >= 0.30: bc.append([kps[bi][0]/w, kps[bi][1]/h, 0.0])
            else: break
        if len(bc) == 4: body_seq.append(bc); body_valid.append(i)

    if not l_valid and not r_valid: return None
    if l_valid: l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid: r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid: return None

    le, re = bool(l_valid), bool(r_valid)
    fe, be = bool(face_valid), bool(body_valid)
    lf = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0,21,3)), l_valid, T)
    rf = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0,21,3)), r_valid, T)
    ff = interpolate_generic(np.array(face_seq) if face_seq else np.zeros((0,15,3)), face_valid, T, 15)
    bf = interpolate_generic(np.array(body_seq) if body_seq else np.zeros((0,4,3)), body_valid, T, 4)
    combined = np.concatenate([lf, rf, ff, bf], axis=1)
    seg = temporal_resample(combined, 32)
    seg[:,:,:3] = one_euro_filter(seg[:,:,:3])
    if le: seg = stabilize_bones(seg, 0, 21)
    if re: seg = stabilize_bones(seg, 21, 42)
    normed = normalize_sequence_v2(seg, le, re)
    result = compute_kinematics_v2(normed, le, re, fe, be)
    return result


def print_stats(arr, name):
    print(f"\n  {name}: shape={arr.shape}")
    xyz = arr[:, :, :3]
    vel = arr[:, :, 3:6]
    acc = arr[:, :, 6:9]
    mask = arr[:, :, 9:10]

    print(f"    XYZ:  range=[{xyz.min():.3f}, {xyz.max():.3f}]  mean={xyz.mean():.4f}  std={xyz.std():.4f}")
    print(f"    Vel:  range=[{vel.min():.3f}, {vel.max():.3f}]  mean={vel.mean():.4f}  std={vel.std():.4f}")
    print(f"    Acc:  range=[{acc.min():.3f}, {acc.max():.3f}]  mean={acc.mean():.4f}  std={acc.std():.4f}")

    # Per-group stats
    for gname, start, end in [("L_Hand", 0, 21), ("R_Hand", 21, 42), ("Face", 42, 57), ("Body", 57, 61)]:
        g_xyz = xyz[:, start:end, :]
        g_mask = mask[:, start:end, :]
        active = g_mask.mean()
        if active > 0.01:
            print(f"    {gname:8s}: xyz_range=[{g_xyz.min():.3f}, {g_xyz.max():.3f}]  mask={active:.2f}")
        else:
            print(f"    {gname:8s}: INACTIVE (mask={active:.2f})")


def main():
    cache_dir = os.path.expanduser("~/.cache/rtmlib/hub/checkpoints")
    rtmw_path = os.path.join(cache_dir, "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx")
    if not os.path.exists(rtmw_path):
        print(f"RTMW model not found: {rtmw_path}")
        return

    print("Loading RTMW-xl...")
    session = ort.InferenceSession(rtmw_path, providers=['CPUExecutionProvider'])

    det_model = None
    yolox_path = os.path.join(cache_dir, "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx")
    if os.path.exists(yolox_path):
        from rtmlib import YOLOX
        det_model = YOLOX(yolox_path, backend='onnxruntime', device='cpu')

    # Load a training sample for comparison
    train_path = "ASL_landmarks_v2"
    if os.path.isdir(train_path):
        import json
        manifest = json.load(open(f"{train_path}/manifest.json"))
        # Find HELLO sample
        hello_files = [f for f, l in manifest.items() if l == 'HELLO'][:1]
        if hello_files:
            train_arr = np.load(f"{train_path}/{hello_files[0]}").astype(np.float32)
            print_stats(train_arr, f"Training HELLO ({hello_files[0]})")

    print("\n" + "="*50)
    print("WEBCAM RECORDING")
    print("Sign HELLO, then press Q to stop")
    print("="*50)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frames = []
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if recording:
            frames.append(frame.copy())
            cv2.rectangle(display, (2,2), (637,477), (0,0,255), 3)
            cv2.putText(display, f"REC {len(frames)/30:.1f}s", (500,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        else:
            cv2.putText(display, "Press S to start, Q to quit", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Debug", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
            frames = []
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) < 8:
        print("Not enough frames")
        return

    print(f"\nRecorded {len(frames)} frames")
    print("Extracting...")
    webcam_arr = extract_from_frames(frames, session, det_model)
    if webcam_arr is None:
        print("Extraction failed")
        return

    print_stats(webcam_arr, "Webcam HELLO")

    # Compare
    if os.path.isdir(train_path) and hello_files:
        print("\n" + "="*50)
        print("COMPARISON")
        print("="*50)
        train_arr_16 = compute_bone_features_np(train_arr)
        webcam_arr_16 = compute_bone_features_np(webcam_arr)

        for ch_name, ch_start, ch_end in [("XYZ", 0, 3), ("Vel", 3, 6), ("Acc", 6, 9)]:
            t_ch = train_arr_16[:, :, ch_start:ch_end]
            w_ch = webcam_arr_16[:, :, ch_start:ch_end]
            print(f"\n  {ch_name}:")
            print(f"    Train:  mean={t_ch.mean():.4f}  std={t_ch.std():.4f}  range=[{t_ch.min():.3f}, {t_ch.max():.3f}]")
            print(f"    Webcam: mean={w_ch.mean():.4f}  std={w_ch.std():.4f}  range=[{w_ch.min():.3f}, {w_ch.max():.3f}]")
            print(f"    Ratio:  std_ratio={w_ch.std()/max(t_ch.std(), 1e-6):.2f}  range_ratio={w_ch.ptp()/max(t_ch.ptp(), 1e-6):.2f}")


if __name__ == "__main__":
    main()
