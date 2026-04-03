"""
CoreML phrase extraction for SLT Stage 2.
Uses CoreMLExecutionProvider with batch_size=1.
Output: [N*32, 61, 10] per video.

Usage:
    python scripts/extract_phrases_coreml.py
"""
import os, sys, glob, hashlib, warnings, json, numpy as np, cv2
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

import onnxruntime as ort
from extract_batch_fast_v2 import (
    preprocess_frame, decode_simcc_batch, postprocess_keypoints,
    interpolate_generic, normalize_sequence_v2, compute_kinematics_v2,
    _FACE_INDICES, _BODY_INDICES, _LHAND_START, _RHAND_START,
    NUM_NODES_V2, NUM_FACE_V2, NUM_BODY_V2,
    HAND_CONF, FACE_CONF, BODY_CONF,
)
from extract import (
    interpolate_hand, temporal_resample,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
)
from rtmlib import YOLOX

PHRASE_GLOSSES = {
    "GOOD_MORNING": "GOOD MORNING",
    "HELLO_HOW_YOU": "HELLO HOW YOU",
    "PLEASE_HELP_I": "PLEASE HELP I",
    "PLEASE_HELP_ME": "PLEASE HELP I",
    "SORRY_I_LATE": "SORRY I LATE",
    "MY_NAME": "MY NAME",
    "YESTERDAY_TEACHER_MEET": "YESTERDAY TEACHER MEET",
    "THANKYOU_FRIEND": "THANKYOU FRIEND",
    "TOMORROW_SCHOOL_GO": "TOMORROW SCHOOL GO",
    "I_WANT_FOOD": "I WANT FOOD",
}


def batch_inference_single(session, frames, centers, scales):
    all_kps, all_scs = [], []
    for i in range(len(frames)):
        out = session.run(None, {'input': frames[i:i+1]})
        locs, scores = decode_simcc_batch(out[0], out[1])
        kps, scs = postprocess_keypoints(locs, scores, centers[i:i+1], scales[i:i+1])
        all_kps.append(kps[0])
        all_scs.append(scs[0])
    return np.stack(all_kps), np.stack(all_scs)


if __name__ == "__main__":
    rtmw_path = glob.glob(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_*.onnx"))[0]
    det_path = glob.glob(os.path.expanduser("~/.cache/rtmlib/hub/checkpoints/yolox_m_*.onnx"))[0]

    session = ort.InferenceSession(rtmw_path, providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    det = YOLOX(det_path, model_input_size=(640, 640))
    print(f"Provider: {session.get_providers()}")

    phrase_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw_videos', 'phrases')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ASL_phrases_coreml')

    if len(sys.argv) > 1:
        phrase_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)
    print(f"Phrase dir: {phrase_dir}")
    print(f"Output dir: {output_dir}")

    count = 0
    manifest = {}

    for phrase_name, gloss_str in PHRASE_GLOSSES.items():
        pdir = os.path.join(phrase_dir, phrase_name)
        if not os.path.isdir(pdir):
            print(f"  Skipping {phrase_name} (not found)")
            continue

        videos = sorted(glob.glob(os.path.join(pdir, "*.mp4")))
        print(f"  {phrase_name}: {len(videos)} videos")

        for vid in videos:
            cap = cv2.VideoCapture(vid)
            frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(f)
            cap.release()
            if len(frames) < 8:
                continue

            total_frames = len(frames)
            max_process = 256
            indices = list(range(total_frames))
            if total_frames > max_process:
                step = total_frames / max_process
                indices = [int(i * step) for i in range(max_process)]
                frames = [frames[i] for i in indices]

            h0, w0 = frames[0].shape[:2]
            bboxes = det(frames[0])
            bbox = bboxes[0][:4].astype(np.float32) if bboxes is not None and len(bboxes) > 0 else np.array([0, 0, w0, h0], dtype=np.float32)

            preprocessed, centers, scales = [], [], []
            for bgr in frames:
                img, c, s = preprocess_frame(bgr, bbox=bbox)
                preprocessed.append(img)
                centers.append(c)
                scales.append(s)
            prep = np.stack(preprocessed).astype(np.float32)
            centers = np.stack(centers)
            scales = np.stack(scales)

            all_kps, all_scores = batch_inference_single(session, prep, centers, scales)

            l_seq, r_seq, face_seq, body_seq = [], [], [], []
            l_valid, r_valid, face_valid, body_valid = [], [], [], []
            for i in range(len(frames)):
                fi = indices[i] if total_frames > max_process else i
                kps = all_kps[i]
                scs = all_scores[i]
                h, w = frames[i].shape[:2]
                if scs[_LHAND_START:_LHAND_START + 21].mean() >= HAND_CONF:
                    coords = [[kps[_LHAND_START + j][0] / w, kps[_LHAND_START + j][1] / h, 0.0] for j in range(21)]
                    if not l_valid or l_valid[-1] != fi:
                        l_seq.append(coords); l_valid.append(fi)
                if scs[_RHAND_START:_RHAND_START + 21].mean() >= HAND_CONF:
                    coords = [[kps[_RHAND_START + j][0] / w, kps[_RHAND_START + j][1] / h, 0.0] for j in range(21)]
                    if not r_valid or r_valid[-1] != fi:
                        r_seq.append(coords); r_valid.append(fi)
                fc = []
                for fi_idx in _FACE_INDICES:
                    if fi_idx < len(kps) and scs[fi_idx] >= FACE_CONF:
                        fc.append([kps[fi_idx][0] / w, kps[fi_idx][1] / h, 0.0])
                    else:
                        break
                if len(fc) == NUM_FACE_V2:
                    if not face_valid or face_valid[-1] != fi:
                        face_seq.append(fc); face_valid.append(fi)
                bc = []
                for bi in _BODY_INDICES:
                    if bi < len(kps) and scs[bi] >= BODY_CONF:
                        bc.append([kps[bi][0] / w, kps[bi][1] / h, 0.0])
                    else:
                        break
                if len(bc) == NUM_BODY_V2:
                    if not body_valid or body_valid[-1] != fi:
                        body_seq.append(bc); body_valid.append(fi)

            if not l_valid and not r_valid:
                continue
            if l_valid:
                l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
            if r_valid:
                r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
            if not l_valid and not r_valid:
                continue

            l_ever, r_ever = bool(l_valid), bool(r_valid)
            face_ever, body_ever = bool(face_valid), bool(body_valid)
            l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)), l_valid, total_frames)
            r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)), r_valid, total_frames)
            face_full = interpolate_generic(
                np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)),
                face_valid, total_frames, NUM_FACE_V2)
            body_full = interpolate_generic(
                np.array(body_seq) if body_seq else np.zeros((0, NUM_BODY_V2, 3)),
                body_valid, total_frames, NUM_BODY_V2)
            combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)

            # Split into 32-frame clips
            T = combined.shape[0]
            num_clips = T // 32
            remainder = T % 32
            if num_clips == 0:
                combined = np.concatenate([combined, np.zeros((32 - T, 61, 3), dtype=np.float32)])
                num_clips = 1
            elif remainder > 0:
                combined = np.concatenate([combined, np.zeros((32 - remainder, 61, 3), dtype=np.float32)])
                num_clips += 1

            clips = []
            for ci in range(num_clips):
                seg = combined[ci * 32:(ci + 1) * 32].copy()
                seg[:, :, :3] = one_euro_filter(seg[:, :, :3])
                if l_ever:
                    seg = stabilize_bones(seg, 0, 21)
                if r_ever:
                    seg = stabilize_bones(seg, 21, 42)
                normed = normalize_sequence_v2(seg, l_ever, r_ever)
                clips.append(compute_kinematics_v2(normed, l_ever, r_ever, face_ever, body_ever))

            result = np.concatenate(clips, axis=0).astype(np.float16)
            vid_hash = os.path.splitext(os.path.basename(vid))[0]
            idx = len([f for f in os.listdir(output_dir) if f.startswith(phrase_name)])
            out_name = f"{phrase_name}_{idx:04d}_{vid_hash[:8]}.npy"
            np.save(os.path.join(output_dir, out_name), result)
            manifest[out_name] = gloss_str
            count += 1

    # Save manifest
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\nDone: {count} phrase files")
    print(f"Manifest: {len(manifest)} files, {len(set(manifest.values()))} phrases")
