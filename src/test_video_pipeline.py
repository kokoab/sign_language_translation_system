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

import json

# Import your Stage 2 architecture
try:
    from train_stage_2 import SLTStage2CTC
except ImportError:
    print("Error: Could not import SLTStage2CTC. Ensure train_stage_2.py is in the same directory.")
    sys.exit(1)

# =====================================================================
# CONFIGURATION
# =====================================================================
VIDEO_PATH = "sample_videos/HOW_YOU_training.mp4"  
STAGE2_CKPT = "weights/stage2_best_model.pth"
STAGE3_DIR = "weights/slt_final_t5_model"

FFMPEG_PATH = "/opt/homebrew/bin/ffmpeg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Constants (matching extract.py)
L_WRIST = 0; R_WRIST = 21
L_MIDDLE_MCP = 9; R_MIDDLE_MCP = 30
TARGET_FRAMES = 32
MIN_RAW_FRAMES = 5
MIN_DETECTION_CONF = 0.80
MIN_TRACKING_CONF = 0.80
MODEL_COMPLEXITY = 0

# =====================================================================
# HAND-COUNT PRIOR: Load from training data to bias scoring
# =====================================================================
GLOSS_HAND_COUNT = {}  # gloss -> 1 or 2 (number of hands used)

def _build_hand_count_lookup():
    """Build a lookup of how many hands each gloss uses, from training .npy files."""
    manifest_path = os.path.join("ASL_landmarks_float16", "manifest.json")
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Group files by gloss and sample a few
    gloss_files = {}
    for fname, gloss in manifest.items():
        if gloss not in gloss_files:
            gloss_files[gloss] = []
        gloss_files[gloss].append(fname)
    for gloss, files in gloss_files.items():
        counts = []
        for fname in files[:3]:
            try:
                data = np.load(os.path.join("ASL_landmarks_float16", fname))
                l_on = data[:, :21, 9].max() > 0.5
                r_on = data[:, 21:, 9].max() > 0.5
                counts.append(int(l_on) + int(r_on))
            except Exception:
                pass
        if counts:
            GLOSS_HAND_COUNT[gloss] = max(counts)

# =====================================================================
# VFR FIX: Re-encode to constant frame rate with ffmpeg
# =====================================================================
def reencode_to_cfr(video_path, fps=30):
    """Re-encode VFR video to CFR via ffmpeg so OpenCV reads all frames."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        subprocess.run(
            [FFMPEG_PATH, "-y", "-i", video_path, "-r", str(fps),
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

    IMPORTANT: For each segment, detects which hand(s) are actually ACTIVE (moving).
    In training, isolated sign videos only show relevant hand(s), so the mask for
    unused hands is 0. In continuous user video, both hands are always visible but
    may not both be active for every sign. We detect activity per segment and zero
    out inactive hands to match the training distribution.

    Returns (features_array [n_signs * 32, 42, 10], seg_hand_counts [list of int per segment]).
    """
    if n_signs == 1:
        segments = [xyz_seq]
    else:
        splits = find_best_split_points(xyz_seq, n_signs - 1)
        if len(splits) < n_signs - 1:
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
    seg_hand_counts = []
    for seg_i, seg in enumerate(segments):
        # Detect per-segment hand activity via motion energy
        l_motion = np.sqrt(np.diff(seg[:, 0:21, :], axis=0) ** 2).sum() / max(seg.shape[0], 1)
        r_motion = np.sqrt(np.diff(seg[:, 21:42, :], axis=0) ** 2).sum() / max(seg.shape[0], 1)

        max_motion = max(l_motion, r_motion, 1e-8)
        l_ratio = l_motion / max_motion if l_ever else 0.0
        r_ratio = r_motion / max_motion if r_ever else 0.0
        seg_l_active = l_ever and (l_ratio > 0.45)
        seg_r_active = r_ever and (r_ratio > 0.45)

        if not seg_l_active and not seg_r_active:
            seg_l_active = l_ever
            seg_r_active = r_ever

        seg_hand_counts.append(int(seg_l_active) + int(seg_r_active))

        seg_copy = seg.copy()
        if not seg_l_active:
            seg_copy[:, 0:21, :] = 0.0
        if not seg_r_active:
            seg_copy[:, 21:42, :] = 0.0

        resampled = temporal_resample(seg_copy, TARGET_FRAMES)
        normalized = normalize_sequence(resampled, seg_l_active, seg_r_active)
        features = compute_kinematics(normalized, seg_l_active, seg_r_active)
        processed.append(features)

    return np.concatenate(processed, axis=0), seg_hand_counts

# =====================================================================
# STAGE 0: EXTRACTION (reads video, returns raw xyz + metadata)
# =====================================================================
def extract_landmarks_from_video(video_path, override_fps=None, mirror=False, swap_hands=False):
    """Extract raw hand landmarks from video, matching extract.py EXACTLY.

    Key match points with extract.py:
    - skip = max(1, total_est // 48)  (1.5x oversampling)
    - static_image_mode = True if skip > 1 else False
    - frame_idx tracks ALL frames (including skipped) for correct interpolation

    If mirror=True, flips the video horizontally.
    If swap_hands=True, swaps left/right hand data channels after extraction.
    """
    tag_parts = []
    if mirror: tag_parts.append("MIRROR")
    if swap_hands: tag_parts.append("SWAP")
    tag = f"[{'+'.join(tag_parts)}] " if tag_parts else ""
    print(f"\n[STAGE 0] Extracting Landmarks from: {tag}{os.path.basename(video_path)}")

    cfr_path = reencode_to_cfr(video_path, fps=override_fps or 30)
    read_path = cfr_path if cfr_path else video_path

    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        if cfr_path:
            os.unlink(cfr_path)
        raise ValueError(f"Could not open video: {video_path}")

    total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # MATCH extract.py exactly: skip = max(1, total_est // int(32 * 1.5))
    skip = max(1, total_est // int(TARGET_FRAMES * 1.5))
    use_static = skip > 1

    l_seq, r_seq, l_valid, r_valid = [], [], [], []
    frame_idx = 0
    processed_count = 0

    with mp.solutions.hands.Hands(
        static_image_mode=use_static,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
        model_complexity=MODEL_COMPLEXITY,
    ) as detector:
        while cap.isOpened():
            ret = cap.grab()
            if not ret:
                break

            if frame_idx % skip != 0:
                frame_idx += 1
                continue

            ret, frame = cap.retrieve()
            if not ret:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

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
            frame_idx += 1

    cap.release()
    if cfr_path:
        os.unlink(cfr_path)

    print(f"   -> Frames: {processed_count}/{frame_idx} | skip={skip} | "
          f"L={len(l_valid)} R={len(r_valid)}")

    if processed_count < MIN_RAW_FRAMES or (not l_valid and not r_valid):
        raise ValueError(f"Too few detections ({processed_count} frames, "
                         f"L={len(l_valid)}, R={len(r_valid)})")

    total_raw = frame_idx
    l_ever, r_ever = bool(l_valid), bool(r_valid)

    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)),
                              l_valid, total_raw)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)),
                              r_valid, total_raw)

    if swap_hands:
        # Swap left and right hand data channels
        combined = np.concatenate([r_full, l_full], axis=1)
        l_ever, r_ever = r_ever, l_ever
    else:
        combined = np.concatenate([l_full, r_full], axis=1)

    return combined, l_ever, r_ever

# =====================================================================
# STAGE 2: RECOGNITION (Multi-Hypothesis + CTC DECODING)
# =====================================================================
def _ctc_beam_search(log_probs, beam_width=25, blank=0):
    """CTC prefix beam search. Returns list of (log_prob, token_tuple)."""
    T, V = log_probs.shape
    beams = {(): (0.0, float('-inf'))}

    for t in range(T):
        new_beams = {}
        for prefix, (pb, pnb) in beams.items():
            p = np.logaddexp(pb, pnb)
            for c in range(V):
                lp = log_probs[t, c]
                if c == blank:
                    key = prefix
                    new_pb = p + lp
                    if key in new_beams:
                        new_beams[key] = (np.logaddexp(new_beams[key][0], new_pb),
                                          new_beams[key][1])
                    else:
                        new_beams[key] = (new_pb, float('-inf'))
                else:
                    if len(prefix) > 0 and prefix[-1] == c:
                        new_pnb = pb + lp
                        key = prefix
                    else:
                        new_pnb = p + lp
                        key = prefix + (c,)
                    if key in new_beams:
                        new_beams[key] = (new_beams[key][0],
                                          np.logaddexp(new_beams[key][1], new_pnb))
                    else:
                        new_beams[key] = (float('-inf'), new_pnb)
                    if len(prefix) > 0 and prefix[-1] == c:
                        key2 = prefix + (c,)
                        new_pnb2 = p + lp
                        if key2 in new_beams:
                            new_beams[key2] = (new_beams[key2][0],
                                               np.logaddexp(new_beams[key2][1], new_pnb2))
                        else:
                            new_beams[key2] = (float('-inf'), new_pnb2)

        scored = [(np.logaddexp(pb, pnb), pf)
                  for pf, (pb, pnb) in new_beams.items()]
        scored.sort(key=lambda x: -x[0])
        beams = {pf: new_beams[pf] for _, pf in scored[:beam_width]}

    results = []
    for prefix, (pb, pnb) in beams.items():
        results.append((np.logaddexp(pb, pnb), prefix))
    results.sort(key=lambda x: -x[0])
    return results


def _score_hypothesis(model, features_np, idx_to_gloss):
    """Run Stage 2 on a feature array using CTC beam search.
    Returns list of (glosses_list, log_probability) for top beams."""
    t = torch.from_numpy(features_np).unsqueeze(0).float().to(DEVICE)
    lens = torch.tensor([t.shape[1]], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits, out_lens = model(t, lens)
        log_probs = torch.log_softmax(logits[0], dim=-1).cpu().numpy()
        n_tokens = out_lens[0].item()

    beam_results = _ctc_beam_search(log_probs[:n_tokens], beam_width=25)

    decoded_beams = []
    for log_prob, token_ids in beam_results[:15]:
        if len(token_ids) == 0:
            continue  # Skip empty predictions
        glosses = [idx_to_gloss.get(int(i), idx_to_gloss.get(str(i), f"UNK_{i}"))
                   for i in token_ids]
        decoded_beams.append((glosses, log_prob))

    return decoded_beams


def run_stage2_recognition(model, xyz_seq, l_ever, r_ever, idx_to_gloss):
    """Try N=1,2,3,4 sign hypotheses with beam search decoding.
    Applies hand-count prior: if a segment has 2 active hands, predictions
    requiring 2 hands get a scoring boost (and vice versa).
    Collects top beams from all N values and picks the best."""
    model.eval()

    max_signs = min(4, max(1, xyz_seq.shape[0] // 10))
    all_candidates = []

    for n in range(1, max_signs + 1):
        features, seg_hand_counts = build_hypothesis(xyz_seq, n, l_ever, r_ever)
        beams = _score_hypothesis(model, features, idx_to_gloss)

        n_candidates = []
        for bi, (glosses, log_prob) in enumerate(beams[:8]):
            raw_prob = float(np.exp(log_prob))
            adj_prob = raw_prob
            if len(glosses) > n:
                adj_prob *= (n / len(glosses))
                raw_prob *= (n / len(glosses))

            if GLOSS_HAND_COUNT and len(glosses) == len(seg_hand_counts):
                match_bonus = 1.0
                for gloss, observed_hands in zip(glosses, seg_hand_counts):
                    expected_hands = GLOSS_HAND_COUNT.get(gloss, 0)
                    if expected_hands == observed_hands:
                        match_bonus *= 1.5
                    elif expected_hands == 2 and observed_hands < 2:
                        match_bonus *= 0.7
                    elif expected_hands == 1 and observed_hands == 2:
                        match_bonus *= 1.0
                adj_prob *= match_bonus

            n_candidates.append((glosses, raw_prob, adj_prob))

        if n_candidates:
            n_candidates.sort(key=lambda x: -x[2])
            best_glosses_n, best_raw_n, best_adj_n = n_candidates[0]
            label = " ".join(best_glosses_n) if best_glosses_n else "(empty)"
            print(f"      N={n}: [{label}] (raw={best_raw_n:.4f}, adj={best_adj_n:.4f})")
            all_candidates.append((best_glosses_n, best_raw_n, n))

    if all_candidates:
        all_candidates.sort(key=lambda x: -x[1])
        best_glosses, best_prob, best_n = all_candidates[0]
        return best_glosses, best_prob, best_n
    return [], 0.0, 0

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
    print("INITIALIZING END-TO-END SLT PIPELINE")
    print("=" * 60)

    print(f"Loading Models onto {DEVICE.type.upper()}...")

    # Build hand-count prior from training data
    _build_hand_count_lookup()
    if GLOSS_HAND_COUNT:
        print(f"   Loaded hand-count prior for {len(GLOSS_HAND_COUNT)} glosses")

    if not os.path.exists(STAGE2_CKPT):
        print(f"Error: Cannot find Stage 2 checkpoint at {STAGE2_CKPT}")
        return

    ckpt = torch.load(STAGE2_CKPT, map_location=DEVICE, weights_only=False)
    idx_to_gloss = ckpt["idx_to_gloss"]
    vocab_size = ckpt["vocab_size"]

    s2_model = SLTStage2CTC(vocab_size=vocab_size).to(DEVICE)
    s2_model.load_state_dict(ckpt["model_state_dict"])

    s3_tokenizer = AutoTokenizer.from_pretrained(STAGE3_DIR)
    s3_model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3_DIR).to(DEVICE)

    try:
        xyz_seq, l_ever, r_ever = extract_landmarks_from_video(
            VIDEO_PATH, override_fps=15)

        overall_best_glosses, overall_best_conf, best_n = run_stage2_recognition(
            s2_model, xyz_seq, l_ever, r_ever, idx_to_gloss)

        gloss_str = " ".join(overall_best_glosses) if overall_best_glosses else "(empty)"
        print(f"\n[STAGE 2] Best: N={best_n} [{gloss_str}] "
              f"(confidence={overall_best_conf:.4f})")

        # C. Translate
        english_sentence = run_stage3_translation(s3_model, s3_tokenizer,
                                                   overall_best_glosses)

        print("\n" + "=" * 60)
        print("FINAL OUTPUT")
        print("=" * 60)
        print(f"Raw ASL Gloss : {gloss_str}")
        print(f"English       : {english_sentence}")
        print(f"Confidence    : {overall_best_conf:.3f}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nPipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()