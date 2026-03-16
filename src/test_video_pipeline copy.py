"""
End-to-End SLT Video Pipeline (Terminal Only)
Stage 0 (Accurate MediaPipe) -> Stage 2 (Recognition) -> Stage 3 (Translation)

Stage 0 extracts landmarks and tries multiple sign-count hypotheses
  (N=1,2,3). Each hypothesis segments the video into N parts, resamples
  each to 32 frames (matching extract.py), then the model's own confidence
  picks the best hypothesis.
"""

import os
import sys
import subprocess
import tempfile
import cv2
import torch
import numpy as np
import warnings

# Suppress verbose warnings from transformers/mediapipe
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import your Stage 2 architecture
try:
    from train_stage_2 import SLTStage2CTC
except ImportError:
    print("Error: Could not import SLTStage2CTC. Ensure train_stage_2.py is in the same directory.")
    sys.exit(1)

# =====================================================================
# CONFIGURATION
# =====================================================================
VIDEO_PATH = "sample_videos/how you.mp4"  
STAGE2_CKPT = "weights/stage2_best_model.pth"
STAGE3_DIR = "weights/slt_final_t5_model"

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Constants (matching extract.py)
L_WRIST = 0; R_WRIST = 21
L_MIDDLE_MCP = 9; R_MIDDLE_MCP = 30
TARGET_FRAMES = 32
MIN_RAW_FRAMES = 5
MIN_DETECTION_CONF = 0.65
MIN_TRACKING_CONF = 0.65
MODEL_COMPLEXITY = 1

# =====================================================================
# VFR FIX: Re-encode to constant frame rate with ffmpeg
# =====================================================================
def reencode_to_cfr(video_path):
    """Re-encode VFR video to 30fps CFR via ffmpeg so OpenCV reads all frames."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [FFMPEG_PATH, "-y", "-i", video_path, "-r", "30",
             "-vsync", "cfr", "-an", "-c:v", "libx264",
             "-preset", "ultrafast", "-crf", "18", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        return tmp.name
    except (subprocess.CalledProcessError, FileNotFoundError):
        os.unlink(tmp.name)
        return None

# =====================================================================
# HELPERS (copied verbatim from extract.py)
# =====================================================================
def interpolate_hand(hand_seq, valid_indices, total_frames):
    if not valid_indices:
        return np.zeros((total_frames, 21, 3), dtype=np.float32)
    flat = hand_seq.reshape(len(hand_seq), -1)
    if len(valid_indices) == 1:
        return np.tile(flat[0], (total_frames, 1)).reshape(total_frames, 21, 3).astype(np.float32)
    x_new = np.arange(total_frames, dtype=np.float64)
    xp = np.array(valid_indices, dtype=np.float64)
    result = np.column_stack([
        np.interp(x_new, xp, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(total_frames, 21, 3).astype(np.float32)

def temporal_resample(seq, target_frames):
    N, P, C = seq.shape
    if N == target_frames:
        return seq
    flat = seq.reshape(N, -1)
    x_old = np.linspace(0, 1, N)
    x_new = np.linspace(0, 1, target_frames)
    result = np.column_stack([
        np.interp(x_new, x_old, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(target_frames, P, C).astype(np.float32)

def normalize_sequence(seq, l_ever, r_ever):
    norm_seq = seq.copy().astype(np.float64)
    valid_wrists = []
    if l_ever:
        valid_wrists.append(norm_seq[:, L_WRIST, :])
    if r_ever:
        valid_wrists.append(norm_seq[:, R_WRIST, :])
    if valid_wrists:
        all_wrists = np.concatenate(valid_wrists, axis=0)
        nonzero = all_wrists[np.linalg.norm(all_wrists, axis=-1) > 1e-6]
        center = np.median(nonzero, axis=0) if len(nonzero) > 0 else np.zeros(3)
    else:
        center = np.zeros(3)
    if l_ever:
        norm_seq[:, 0:21] -= center
    if r_ever:
        norm_seq[:, 21:42] -= center
    bone_lengths = []
    if l_ever:
        bone_lengths.extend(np.linalg.norm(norm_seq[:, L_MIDDLE_MCP] - norm_seq[:, L_WRIST], axis=-1))
    if r_ever:
        bone_lengths.extend(np.linalg.norm(norm_seq[:, R_MIDDLE_MCP] - norm_seq[:, R_WRIST], axis=-1))
    if bone_lengths:
        filtered = [b for b in bone_lengths if b > 1e-6]
        if filtered:
            norm_seq /= (np.median(filtered) + 1e-8)
    return norm_seq.astype(np.float32)

def compute_kinematics(seq, l_ever, r_ever):
    """Central-difference velocity & acceleration on a single [F,42,3] sequence."""
    F, P, _ = seq.shape
    vel = np.zeros_like(seq)
    vel[1:-1] = (seq[2:] - seq[:-2]) / 2.0
    vel[0] = vel[1] if F > 1 else vel[0]
    vel[-1] = vel[-2] if F > 1 else vel[-1]
    acc = np.zeros_like(seq)
    acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
    acc[0] = acc[1] if F > 1 else acc[0]
    acc[-1] = acc[-2] if F > 1 else acc[-1]
    mask = np.zeros((F, P, 1), dtype=np.float32)
    if l_ever:
        mask[:, 0:21, 0] = 1.0
    if r_ever:
        mask[:, 21:42, 0] = 1.0
    return np.concatenate([seq, vel, acc, mask], axis=-1).astype(np.float32)

# =====================================================================
# SIGN SEGMENTATION HELPERS
# =====================================================================
def find_best_split_points(xyz_seq, n_splits):
    """Find the n_splits lowest-energy frames to split the sequence.
    Ignores the first/last 15% of frames (lead-in/lead-out) to avoid
    splitting on idle hands at the start or end of the video.
    Returns sorted list of split indices."""
    T = xyz_seq.shape[0]
    min_seg = max(8, T // (n_splits + 2))

    energy = np.zeros(T)
    diff = xyz_seq[1:] - xyz_seq[:-1]
    energy[1:] = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=1)

    k = min(7, max(3, T // 6))
    smoothed = np.convolve(energy, np.ones(k) / k, mode='same')

    # Skip the first/last 15% — those are typically lead-in/lead-out idle frames
    margin = max(min_seg, int(T * 0.15))
    search_start = margin
    search_end = T - margin
    if search_start >= search_end:
        # Fallback: uniform split
        return [int(T * i / (n_splits + 1)) for i in range(1, n_splits + 1)]

    # Rank interior frames by energy (lowest = best split candidates)
    frame_energies = [(i, smoothed[i]) for i in range(search_start, search_end)]
    frame_energies.sort(key=lambda x: x[1])

    # Greedily pick n_splits points, maintaining minimum segment length
    selected = []
    for pos, _ in frame_energies:
        if len(selected) >= n_splits:
            break
        if all(abs(pos - s) >= min_seg for s in selected):
            selected.append(pos)

    # If we couldn't find enough points, fill with uniform splits
    if len(selected) < n_splits:
        selected = [int(T * i / (n_splits + 1)) for i in range(1, n_splits + 1)]

    selected.sort()
    return selected

def build_hypothesis(xyz_seq, n_signs, l_ever, r_ever):
    """Build a feature tensor for the hypothesis that the video contains n_signs signs.
    Segments the xyz sequence, resamples each segment to 32 frames, normalizes, and
    computes kinematics independently per segment (matching extract.py training pipeline).
    Returns [n_signs * 32, 42, 10] float32 array.
    """
    if n_signs == 1:
        segments = [xyz_seq]
    else:
        splits = find_best_split_points(xyz_seq, n_signs - 1)
        if len(splits) < n_signs - 1:
            # Not enough split points found, fall back to uniform splitting
            boundaries = [int(xyz_seq.shape[0] * i / n_signs) for i in range(1, n_signs)]
        else:
            boundaries = splits
        segments = []
        prev = 0
        for b in boundaries:
            segments.append(xyz_seq[prev:b])
            prev = b
        segments.append(xyz_seq[prev:])

    processed = []
    for seg in segments:
        resampled = temporal_resample(seg, TARGET_FRAMES)
        normalized = normalize_sequence(resampled, l_ever, r_ever)
        features = compute_kinematics(normalized, l_ever, r_ever)
        processed.append(features)

    return np.concatenate(processed, axis=0)

# =====================================================================
# STAGE 0: EXTRACTION (reads video, returns raw xyz + metadata)
# =====================================================================
def extract_landmarks_from_video(video_path):
    """Extract raw hand landmarks from video. Returns (combined_xyz, l_ever, r_ever)."""
    print(f"\n[STAGE 0] Extracting Landmarks from: {os.path.basename(video_path)}")

    cfr_path = reencode_to_cfr(video_path)
    read_path = cfr_path if cfr_path else video_path
    if cfr_path:
        print("   -> VFR fix: re-encoded to 30fps CFR via ffmpeg")
    else:
        print("   -> ffmpeg not available, reading original file directly")

    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        if cfr_path:
            os.unlink(cfr_path)
        raise ValueError(f"Could not open video: {video_path}")

    print(f"   -> MediaPipe Hands | complexity={MODEL_COMPLEXITY} | "
          f"static=False | reading ALL frames")

    l_seq, r_seq, l_valid, r_valid = [], [], [], []
    frame_idx = 0
    processed_count = 0

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
        model_complexity=MODEL_COMPLEXITY,
    ) as detector:
        while cap.isOpened():
            ret = cap.grab()
            if not ret:
                break
            ret, frame = cap.retrieve()
            if not ret:
                break

            h, w = frame.shape[:2]
            if max(h, w) > 384:
                scale = 384 / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = detector.process(rgb)

            if res.multi_hand_landmarks:
                for hand_lm, handedness in zip(res.multi_hand_landmarks,
                                               res.multi_handedness):
                    h_label = handedness.classification[0].label
                    if handedness.classification[0].score >= MIN_DETECTION_CONF:
                        coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                        if h_label == "Left" and (not l_valid or l_valid[-1] != frame_idx):
                            l_seq.append(coords)
                            l_valid.append(frame_idx)
                        elif h_label == "Right" and (not r_valid or r_valid[-1] != frame_idx):
                            r_seq.append(coords)
                            r_valid.append(frame_idx)

            processed_count += 1
            if processed_count % 15 == 0:
                print(f"   -> Processed {processed_count} frames...")
            frame_idx += 1

    cap.release()
    if cfr_path:
        os.unlink(cfr_path)

    print(f"   -> Raw frames read: {processed_count} | "
          f"Left detections: {len(l_valid)} | Right detections: {len(r_valid)}")

    if processed_count < MIN_RAW_FRAMES or (not l_valid and not r_valid):
        raise ValueError(f"Too few detections ({processed_count} frames, "
                         f"L={len(l_valid)}, R={len(r_valid)})")

    total_raw = frame_idx
    l_ever, r_ever = bool(l_valid), bool(r_valid)

    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)),
                              l_valid, total_raw)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)),
                              r_valid, total_raw)
    combined = np.concatenate([l_full, r_full], axis=1)

    return combined, l_ever, r_ever

# =====================================================================
# STAGE 2: RECOGNITION (Multi-Hypothesis + CTC DECODING)
# =====================================================================
def _score_hypothesis(model, features_np, idx_to_gloss):
    """Run Stage 2 on a feature array, return (decoded_glosses, score).
    Score = minimum confidence across unique decoded tokens.
    CTC duplicates are merged before scoring."""
    t = torch.from_numpy(features_np).unsqueeze(0).float().to(DEVICE)
    lens = torch.tensor([t.shape[1]], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits, out_lens = model(t, lens)
        probs = torch.softmax(logits[0], dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        n_tokens = out_lens[0].item()

        decoded = []
        confidences = []
        last_id = -1
        for i in range(n_tokens):
            idx = int(pred_ids[i])
            if idx != last_id and idx != 0:
                word = idx_to_gloss.get(str(idx), idx_to_gloss.get(int(idx), f"UNKNOWN_{idx}"))
                decoded.append(word)
                confidences.append(probs[i, idx].item())
            last_id = idx

    # Merge CTC duplicates: keep highest confidence per unique gloss
    seen = {}
    for word, conf in zip(decoded, confidences):
        if word not in seen or conf > seen[word]:
            seen[word] = conf
    unique_glosses = list(dict.fromkeys(decoded))  # preserve order, deduplicate
    unique_confs = [seen[w] for w in unique_glosses]

    score = float(min(unique_confs)) if unique_confs else 0.0
    return unique_glosses, score

def run_stage2_recognition(model, xyz_seq, l_ever, r_ever, idx_to_gloss):
    """Try N=1,2,3 sign hypotheses, pick the one the model is most confident about.
    Uses minimum non-blank confidence per hypothesis to avoid rewarding over-segmentation."""
    print("\n[STAGE 2] Running BiLSTM + CTC Decoder (multi-hypothesis)...")
    model.eval()

    max_signs = min(4, max(1, xyz_seq.shape[0] // 10))
    best_glosses = []
    best_conf = -1.0
    best_n = 0

    for n in range(1, max_signs + 1):
        features = build_hypothesis(xyz_seq, n, l_ever, r_ever)
        glosses, conf = _score_hypothesis(model, features, idx_to_gloss)
        # Penalize: if model decodes more glosses than hypothesized signs,
        # it's hallucinating extra signs from noise
        if len(glosses) > n:
            conf *= (n / len(glosses))
        label = " ".join(glosses) if glosses else "(empty)"
        print(f"   -> N={n} signs: [{label}] (confidence: {conf:.3f})")
        if conf > best_conf:
            best_conf = conf
            best_glosses = glosses
            best_n = n

    print(f"   -> Best: N={best_n} ({best_conf:.3f})")
    return best_glosses

# =====================================================================
# STAGE 3: TRANSLATION (T5)
# =====================================================================
def run_stage3_translation(model, tokenizer, gloss_list):
    print("\n🗣️  [STAGE 3] Translating Gloss to Natural English...")
    if not gloss_list:
        return "[No signs detected]"
        
    gloss_string = " ".join(gloss_list)
    prompt = f"translate ASL gloss to English: {gloss_string}"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    print("=" * 60)
    print("🚀 INITIALIZING END-TO-END SLT PIPELINE")
    print("=" * 60)
    
    print(f"📦 Loading Models onto {DEVICE.type.upper()}...")
    
    if not os.path.exists(STAGE2_CKPT):
        print(f"❌ Error: Cannot find Stage 2 checkpoint at {STAGE2_CKPT}")
        return
        
    ckpt = torch.load(STAGE2_CKPT, map_location=DEVICE, weights_only=False)
    idx_to_gloss = ckpt["idx_to_gloss"]
    vocab_size = ckpt["vocab_size"]
    
    s2_model = SLTStage2CTC(vocab_size=vocab_size).to(DEVICE)
    s2_model.load_state_dict(ckpt["model_state_dict"])
    
    s3_tokenizer = AutoTokenizer.from_pretrained(STAGE3_DIR)
    s3_model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3_DIR).to(DEVICE)
    
    try:
        # A. Extract landmarks
        xyz_seq, l_ever, r_ever = extract_landmarks_from_video(VIDEO_PATH)

        # B. Recognize (multi-hypothesis)
        glosses = run_stage2_recognition(s2_model, xyz_seq, l_ever, r_ever, idx_to_gloss)
        gloss_str = " ".join(glosses)
        print(f"   -> Detected ASL Gloss: [{gloss_str}]")
        
        # C. Translate
        english_sentence = run_stage3_translation(s3_model, s3_tokenizer, glosses)
        
        print("\n" + "=" * 60)
        print("🎯 FINAL OUTPUT")
        print("=" * 60)
        print(f"Raw ASL Gloss : {gloss_str}")
        print(f"English       : {english_sentence}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {e}")

if __name__ == "__main__":
    main()