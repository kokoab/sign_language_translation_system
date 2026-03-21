"""
Camera-based SLT Inference — Same pipeline as test_video_pipeline.py
Stage 0 (MediaPipe landmarks from live frames) -> Stage 2 (Recognition) -> Stage 3 (Translation)

Opens the camera, buffers hand landmarks per frame. Press SPACE to run the full
pipeline on the current buffer (same algorithms: multi-hypothesis, CTC beam search,
hand-count prior, T5 translation). No functions or algorithms are skipped.
"""

import os
import sys
import cv2
import torch
import numpy as np
import warnings
from collections import deque

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import pickle
import math
from collections import defaultdict

try:
    from train_stage_2 import SLTStage2CTC
except ImportError:
    print("Error: Could not import SLTStage2CTC. Ensure train_stage_2.py is in the same directory.")
    sys.exit(1)

# =====================================================================
# CONFIGURATION
# =====================================================================
STAGE2_CKPT = "weights/stage2_best_model.pth"
STAGE3_DIR = "weights/slt_conversational_t5_model"  # Updated to conversational model
GLOSS_LM_PATH = "weights/gloss_bigram_lm.pkl"  # N-gram language model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Constants (ALIGNED with extract.py - this is critical for train/inference consistency!)
L_WRIST = 0
R_WRIST = 21
L_MIDDLE_MCP = 9
R_MIDDLE_MCP = 30
NUM_NODES = 47  # 42 hand + 5 face
TARGET_FRAMES = 32
MIN_RAW_FRAMES = 8  # Aligned with extract.py v7.0

# Face landmark indices in MediaPipe FaceMesh
FACE_LANDMARK_INDICES = [1, 152, 10, 234, 454]  # nose, chin, forehead, left ear, right ear

# MediaPipe configuration (ALIGNED with extract.py for consistency)
MIN_DETECTION_CONF = 0.80  # Upgraded to 0.80 per Master Plan
MIN_TRACKING_CONF = 0.80   # Upgraded to 0.80 per Master Plan
MODEL_COMPLEXITY = 1       # Was 0 - now matches extract.py

# Confidence threshold for rejecting low-confidence predictions
MIN_CONFIDENCE_THRESHOLD = 0.15

# Language model weight for beam search rescoring
LM_WEIGHT = 0.3

# Camera buffer: max frames to keep (e.g. ~3 sec at 30 fps)
CAMERA_BUFFER_MAX = 120

# Display: max dimension for window (larger = bigger window; 0 = no resize, use camera native)
DISPLAY_MAX_SIZE = 960

# Conversation history for context-aware translation
CONVERSATION_HISTORY = []
MAX_HISTORY_TURNS = 5

# =====================================================================
# N-GRAM LANGUAGE MODEL FOR BEAM SEARCH RESCORING
# =====================================================================
# =====================================================================
# HAND-COUNT PRIOR: Load from training data to bias scoring
# =====================================================================
GLOSS_HAND_COUNT = {}

def _build_hand_count_lookup():
    """Build a lookup of how many hands each gloss uses, from training .npy files."""
    manifest_path = os.path.join("ASL_landmarks_float16", "manifest.json")
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
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
                r_on = data[:, 21:42, 9].max() > 0.5
                counts.append(int(l_on) + int(r_on))
            except Exception:
                pass
        if counts:
            GLOSS_HAND_COUNT[gloss] = max(counts)

# =====================================================================
# N-GRAM LANGUAGE MODEL: For CTC beam search rescoring
# =====================================================================
class GlossNGramLM:
    """Simple bigram language model for CTC beam rescoring."""

    def __init__(self, smoothing=0.1):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.smoothing = smoothing
        self.vocab = set()
        self.total_unigrams = 0
        self.vocab_size = 0

    def log_prob(self, gloss, prev_gloss='<s>'):
        """Get log probability of gloss given previous gloss."""
        bigram_count = self.bigram_counts[prev_gloss][gloss]
        prev_count = self.unigram_counts[prev_gloss]

        if prev_count == 0:
            # Fallback to unigram
            prob = (self.unigram_counts[gloss] + self.smoothing) / \
                   (self.total_unigrams + self.smoothing * self.vocab_size)
        else:
            prob = (bigram_count + self.smoothing) / \
                   (prev_count + self.smoothing * self.vocab_size)

        return math.log(prob + 1e-10)

    def score_sequence(self, gloss_list):
        """Score a full gloss sequence."""
        score = 0.0
        prev = '<s>'
        for gloss in gloss_list:
            score += self.log_prob(gloss, prev)
            prev = gloss
        score += self.log_prob('</s>', prev)
        return score

    @classmethod
    def load(cls, path):
        """Load a trained language model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lm = cls(smoothing=data.get('smoothing', 0.1))
        lm.unigram_counts = defaultdict(int, data.get('unigram_counts', {}))
        lm.bigram_counts = defaultdict(lambda: defaultdict(int))
        for k, v in data.get('bigram_counts', {}).items():
            lm.bigram_counts[k] = defaultdict(int, v)
        lm.vocab = data.get('vocab', set())
        lm.total_unigrams = data.get('total_unigrams', 0)
        lm.vocab_size = len(lm.vocab)
        return lm


# Global LM instance (loaded lazily)
GLOSS_LM = None

def _load_gloss_lm():
    """Load the N-gram language model if available."""
    global GLOSS_LM
    if os.path.exists(GLOSS_LM_PATH):
        try:
            GLOSS_LM = GlossNGramLM.load(GLOSS_LM_PATH)
            print(f"   Loaded gloss LM (vocab={GLOSS_LM.vocab_size})")
        except Exception as e:
            print(f"   Warning: Could not load LM: {e}")
            GLOSS_LM = None
    else:
        print(f"   No LM found at {GLOSS_LM_PATH} - using acoustic-only decoding")

# =====================================================================
# HELPERS (same as test_video_pipeline.py / extract.py)
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

def interpolate_face(face_seq, valid_indices, total_frames):
    if not valid_indices:
        return np.zeros((total_frames, 5, 3), dtype=np.float32)
    flat = face_seq.reshape(len(face_seq), -1)
    if len(valid_indices) == 1:
        return np.tile(flat[0], (total_frames, 1)).reshape(total_frames, 5, 3).astype(np.float32)
    x_new = np.arange(total_frames, dtype=np.float64)
    xp = np.array(valid_indices, dtype=np.float64)
    result = np.column_stack([
        np.interp(x_new, xp, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(total_frames, 5, 3).astype(np.float32)

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
    """Normalize 47-point sequence (hands + face landmarks)."""
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
    # Face landmarks normalized to same center
    norm_seq[:, 42:47] -= center
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

def compute_kinematics(seq, l_ever, r_ever, face_ever=False):
    """Central-difference velocity & acceleration on a single [F,47,3] sequence."""
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
    if face_ever:
        mask[:, 42:47, 0] = 1.0
    return np.concatenate([seq, vel, acc, mask], axis=-1).astype(np.float32)

# =====================================================================
# SIGN SEGMENTATION HELPERS
# =====================================================================
def find_best_split_points(xyz_seq, n_splits):
    """Find the n_splits lowest-energy frames to split the sequence."""
    T = xyz_seq.shape[0]
    min_seg = max(8, T // (n_splits + 2))
    energy = np.zeros(T)
    diff = xyz_seq[1:] - xyz_seq[:-1]
    energy[1:] = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=1)
    k = min(7, max(3, T // 6))
    smoothed = np.convolve(energy, np.ones(k) / k, mode='same')
    margin = max(min_seg, int(T * 0.15))
    search_start = margin
    search_end = T - margin
    if search_start >= search_end:
        return [int(T * i / (n_splits + 1)) for i in range(1, n_splits + 1)]
    frame_energies = [(i, smoothed[i]) for i in range(search_start, search_end)]
    frame_energies.sort(key=lambda x: x[1])
    selected = []
    for pos, _ in frame_energies:
        if len(selected) >= n_splits:
            break
        if all(abs(pos - s) >= min_seg for s in selected):
            selected.append(pos)
    if len(selected) < n_splits:
        selected = [int(T * i / (n_splits + 1)) for i in range(1, n_splits + 1)]
    selected.sort()
    return selected

def build_hypothesis(xyz_seq, n_signs, l_ever, r_ever, face_ever=False):
    """Build a feature tensor for the hypothesis that the video contains n_signs signs."""
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
        features = compute_kinematics(normalized, seg_l_active, seg_r_active, face_ever)
        processed.append(features)
    return np.concatenate(processed, axis=0), seg_hand_counts

# =====================================================================
# STAGE 0: EXTRACTION FROM CAMERA FRAME BUFFER
# =====================================================================
def extract_landmarks_from_frames(frames_landmarks, mirror=False, swap_hands=False):
    """Extract and process hand+face landmarks from a list of per-frame data.

    frames_landmarks: list of (l_coords, r_coords, face_coords) per frame.
      - l_coords / r_coords: list of 21 [x,y,z] or None
      - face_coords: list of 5 [x,y,z] or None
    Returns [T, 47, 3] combined sequence, l_ever, r_ever, face_ever.
    """
    total_raw = len(frames_landmarks)
    if total_raw < MIN_RAW_FRAMES:
        raise ValueError(f"Too few frames ({total_raw}, need at least {MIN_RAW_FRAMES})")

    l_valid = [i for i in range(total_raw) if frames_landmarks[i][0] is not None]
    r_valid = [i for i in range(total_raw) if frames_landmarks[i][1] is not None]
    face_valid = [i for i in range(total_raw) if len(frames_landmarks[i]) > 2 and frames_landmarks[i][2] is not None]

    if not l_valid and not r_valid:
        raise ValueError("No hand detections in buffer")

    l_seq = np.array([frames_landmarks[i][0] for i in l_valid], dtype=np.float32) if l_valid else np.zeros((0, 21, 3), dtype=np.float32)
    r_seq = np.array([frames_landmarks[i][1] for i in r_valid], dtype=np.float32) if r_valid else np.zeros((0, 21, 3), dtype=np.float32)
    face_seq = np.array([frames_landmarks[i][2] for i in face_valid], dtype=np.float32) if face_valid else np.zeros((0, 5, 3), dtype=np.float32)

    l_ever, r_ever, face_ever = bool(l_valid), bool(r_valid), bool(face_valid)
    l_full = interpolate_hand(l_seq, l_valid, total_raw)
    r_full = interpolate_hand(r_seq, r_valid, total_raw)
    face_full = interpolate_face(face_seq, face_valid, total_raw)

    if swap_hands:
        combined = np.concatenate([r_full, l_full, face_full], axis=1)
        l_ever, r_ever = r_ever, l_ever
    else:
        combined = np.concatenate([l_full, r_full, face_full], axis=1)

    return combined, l_ever, r_ever, face_ever

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
                        new_beams[key] = (np.logaddexp(new_beams[key][0], new_pb), new_beams[key][1])
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
                        new_beams[key] = (new_beams[key][0], np.logaddexp(new_beams[key][1], new_pnb))
                    else:
                        new_beams[key] = (float('-inf'), new_pnb)
                    if len(prefix) > 0 and prefix[-1] == c:
                        key2 = prefix + (c,)
                        new_pnb2 = p + lp
                        if key2 in new_beams:
                            new_beams[key2] = (new_beams[key2][0], np.logaddexp(new_beams[key2][1], new_pnb2))
                        else:
                            new_beams[key2] = (float('-inf'), new_pnb2)
        scored = [(np.logaddexp(pb, pnb), pf) for pf, (pb, pnb) in new_beams.items()]
        scored.sort(key=lambda x: -x[0])
        beams = {pf: new_beams[pf] for _, pf in scored[:beam_width]}
    results = []
    for prefix, (pb, pnb) in beams.items():
        results.append((np.logaddexp(pb, pnb), prefix))
    results.sort(key=lambda x: -x[0])
    return results

def _score_hypothesis(model, features_np, idx_to_gloss):
    """Run Stage 2 on a feature array using CTC beam search with optional LM rescoring."""
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
            continue
        glosses = [idx_to_gloss.get(int(i), idx_to_gloss.get(str(i), f"UNK_{i}")) for i in token_ids]

        # Apply language model rescoring if available
        lm_score = 0.0
        if GLOSS_LM is not None:
            prev = '<s>'
            for gloss in glosses:
                lm_score += GLOSS_LM.log_prob(gloss, prev)
                prev = gloss
            lm_score += GLOSS_LM.log_prob('</s>', prev)

        # Combine acoustic and language model scores
        combined_score = log_prob + LM_WEIGHT * lm_score
        decoded_beams.append((glosses, combined_score))

    # Re-sort by combined score
    decoded_beams.sort(key=lambda x: -x[1])
    return decoded_beams

def run_stage2_recognition(model, xyz_seq, l_ever, r_ever, idx_to_gloss, face_ever=False):
    """Try N=1,2,3,4 sign hypotheses with beam search decoding and hand-count prior."""
    model.eval()
    max_signs = min(4, max(1, xyz_seq.shape[0] // 10))
    all_candidates = []
    for n in range(1, max_signs + 1):
        features, seg_hand_counts = build_hypothesis(xyz_seq, n, l_ever, r_ever, face_ever)
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
            all_candidates.append((best_glosses_n, best_raw_n, n))
    if all_candidates:
        all_candidates.sort(key=lambda x: -x[1])
        best_glosses, best_prob, best_n = all_candidates[0]
        return best_glosses, best_prob, best_n
    return [], 0.0, 0

# =====================================================================
# STAGE 3: TRANSLATION (T5)
# =====================================================================
def run_stage3_translation(model, tokenizer, gloss_list, context_history=None):
    if not gloss_list:
        return "[No signs detected]"
    gloss_string = " ".join(gloss_list)
    
    if context_history:
        # Limit to last 4 turns to match training context window
        context_str = " | ".join(context_history[-4:])
        prompt = f"[Previous: {context_str}] Translate this ASL gloss to natural conversational English: {gloss_string}"
    else:
        prompt = f"Translate this ASL gloss to natural conversational English: {gloss_string}"
        
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
# MAIN: CAMERA LOOP
# =====================================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="SLT camera inference — same pipeline as test_video_pipeline")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera index")
    parser.add_argument("--mirror", action="store_true", help="Flip video horizontally")
    parser.add_argument("--swap-hands", action="store_true", help="Swap left/right hand channels")
    parser.add_argument("--size", "-s", type=int, default=DISPLAY_MAX_SIZE,
                        help=f"Max window dimension (default {DISPLAY_MAX_SIZE}, 0 = camera native)")
    args = parser.parse_args()
    display_max_size = args.size

    print("=" * 60)
    print("SLT CAMERA INFERENCE (Stage 0 → 2 → 3)")
    print("=" * 60)
    print(f"Device: {DEVICE.type.upper()}")

    _build_hand_count_lookup()
    if GLOSS_HAND_COUNT:
        print(f"   Loaded hand-count prior for {len(GLOSS_HAND_COUNT)} glosses")

    # Load N-gram language model for beam search rescoring
    _load_gloss_lm()
    if GLOSS_LM is not None:
        print(f"   LM vocab size: {GLOSS_LM.vocab_size}")

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

    # Buffer: list of (l_coords, r_coords) per frame
    frame_buffer = deque(maxlen=CAMERA_BUFFER_MAX)
    last_gloss_str = ""
    last_english = ""
    last_conf = 0.0

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
        model_complexity=MODEL_COMPLEXITY,
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
        refine_landmarks=False,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    cv2.namedWindow("SLT Camera Inference", cv2.WINDOW_NORMAL)
    if display_max_size > 0:
        cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max(cap_w, cap_h) > display_max_size:
            scale = display_max_size / max(cap_w, cap_h)
            cv2.resizeWindow("SLT Camera Inference", int(cap_w * scale), int(cap_h * scale))

    print("\nControls: SPACE = run recognition on buffer | q = quit | c = clear buffer")
    print(f"Buffer: need at least {MIN_RAW_FRAMES} frames. Max {CAMERA_BUFFER_MAX} frames.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if display_max_size > 0 and max(h, w) > display_max_size:
            scale = display_max_size / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands.process(rgb)

        l_coords = None
        r_coords = None
        face_coords = None
        if res.multi_hand_landmarks and res.multi_handedness:
            for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                if handedness.classification[0].score >= MIN_DETECTION_CONF:
                    coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                    if handedness.classification[0].label == "Left":
                        l_coords = coords
                    else:
                        r_coords = coords
            for hand_lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        # Face detection for 5 reference landmarks
        face_res = face_mesh.process(rgb)
        if face_res.multi_face_landmarks:
            fl = face_res.multi_face_landmarks[0]
            face_coords = [[fl.landmark[idx].x, fl.landmark[idx].y, fl.landmark[idx].z]
                           for idx in FACE_LANDMARK_INDICES]

        frame_buffer.append((l_coords, r_coords, face_coords))

        # HUD
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (fw, 90), (30, 30, 30), -1)
        cv2.putText(frame, f"Buffer: {len(frame_buffer)} / {MIN_RAW_FRAMES}+ (SPACE = recognize)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if last_gloss_str:
            cv2.putText(frame, f"Gloss: {last_gloss_str}", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
            cv2.putText(frame, f"English: {last_english}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
            cv2.putText(frame, f"Conf: {last_conf:.3f}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 255), 1)
        cv2.imshow("SLT Camera Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            frame_buffer.clear()
            last_gloss_str = ""
            last_english = ""
            last_conf = 0.0
            CONVERSATION_HISTORY.clear()
            continue
        if key == ord(" "):
            if len(frame_buffer) < MIN_RAW_FRAMES:
                print(f"   Need at least {MIN_RAW_FRAMES} frames, have {len(frame_buffer)}")
                continue
            try:
                xyz_seq, l_ever, r_ever, face_ever = extract_landmarks_from_frames(
                    list(frame_buffer), mirror=False, swap_hands=args.swap_hands
                )
                overall_best_glosses, overall_best_conf, best_n = run_stage2_recognition(
                    s2_model, xyz_seq, l_ever, r_ever, idx_to_gloss, face_ever
                )
                gloss_str = " ".join(overall_best_glosses) if overall_best_glosses else "(empty)"
                
                # Pass conversation history to translation
                english_sentence = run_stage3_translation(s3_model, s3_tokenizer, overall_best_glosses, CONVERSATION_HISTORY)
                
                if overall_best_glosses and overall_best_conf >= MIN_CONFIDENCE_THRESHOLD:
                    CONVERSATION_HISTORY.append(english_sentence)
                    if len(CONVERSATION_HISTORY) > MAX_HISTORY_TURNS:
                        CONVERSATION_HISTORY.pop(0)
                        
                last_gloss_str = gloss_str
                last_english = english_sentence
                last_conf = overall_best_conf
                print(f"\n[Result] N={best_n} | {gloss_str} | Conf={overall_best_conf:.4f}")
                print(f"         English: {english_sentence}\n")
            except Exception as e:
                print(f"   Pipeline error: {e}")
                import traceback
                traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()
    print("Done.")

if __name__ == "__main__":
    main()
