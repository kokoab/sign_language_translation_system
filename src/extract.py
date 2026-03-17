"""
SLT 42-Point Extraction v5.5 (The Final Cloud-Ready Edition)
Optimized for Apple M4: 1.5x Oversampling, 384px Resize, Static Mode,
No-SciPy, Fast-Fail Checks, Longest-First Sorting, and Fixed Chunking (50).
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import logging
import hashlib
import concurrent.futures
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-Fast")

# ─────────────────────────────────────────────
# Node Index Reference (42 Points Total)
# ─────────────────────────────────────────────
L_WRIST = 0; R_WRIST = 21
L_MIDDLE_MCP = 9; R_MIDDLE_MCP = 30

@dataclass
class PipelineConfig:
    target_frames: int = 32
    min_raw_frames: int = 5           
    max_missing_ratio: float = 0.40   
    min_detection_conf: float = 0.65
    min_tracking_conf: float = 0.65
    model_complexity: int = 1         
    max_num_hands: int = 2
    mirror_handedness: bool = False    

    raw_video_dir: str = "data/raw_videos/ASL VIDEOS"        
    output_dir: str = "ASL_landmarks_float16" 
    seed: int = 42

CFG = PipelineConfig()

# ─────────────────────────────────────────────
# Core Math & Normalization Helpers (NumPy Native)
# ─────────────────────────────────────────────
def interpolate_hand(hand_seq: np.ndarray, valid_indices: list, total_frames: int) -> np.ndarray:
    if not valid_indices: return np.zeros((total_frames, 21, 3), dtype=np.float32)
    flat = hand_seq.reshape(len(hand_seq), -1)
    if len(valid_indices) == 1:
        return np.tile(flat[0], (total_frames, 1)).reshape(total_frames, 21, 3).astype(np.float32)
    
    x_new = np.arange(total_frames, dtype=np.float64)
    xp = np.array(valid_indices, dtype=np.float64)
    result = np.column_stack([
        np.interp(x_new, xp, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(total_frames, 21, 3).astype(np.float32)

def temporal_resample(seq: np.ndarray, target_frames: int) -> np.ndarray:
    N, P, C = seq.shape
    if N == target_frames: return seq
    flat = seq.reshape(N, -1)
    x_old = np.linspace(0, 1, N)
    x_new = np.linspace(0, 1, target_frames)
    result = np.column_stack([
        np.interp(x_new, x_old, flat[:, c]) for c in range(flat.shape[1])
    ])
    return result.reshape(target_frames, P, C).astype(np.float32)

def normalize_sequence(seq: np.ndarray, l_ever: bool, r_ever: bool) -> np.ndarray:
    norm_seq = seq.copy().astype(np.float64)
    valid_wrists = []
    if l_ever: valid_wrists.append(norm_seq[:, L_WRIST, :])
    if r_ever: valid_wrists.append(norm_seq[:, R_WRIST, :])
    
    if valid_wrists:
        all_wrists = np.concatenate(valid_wrists, axis=0)
        nonzero = all_wrists[np.linalg.norm(all_wrists, axis=-1) > 1e-6]
        center = np.median(nonzero, axis=0) if len(nonzero) > 0 else np.zeros(3)
    else: center = np.zeros(3)

    if l_ever: norm_seq[:, 0:21] -= center
    if r_ever: norm_seq[:, 21:42] -= center

    bone_lengths = []
    if l_ever: bone_lengths.extend(np.linalg.norm(norm_seq[:, L_MIDDLE_MCP] - norm_seq[:, L_WRIST], axis=-1))
    if r_ever: bone_lengths.extend(np.linalg.norm(norm_seq[:, R_MIDDLE_MCP] - norm_seq[:, R_WRIST], axis=-1))

    if bone_lengths:
        filtered = [b for b in bone_lengths if b > 1e-6]
        if filtered: norm_seq /= (np.median(filtered) + 1e-8)

    return norm_seq.astype(np.float32)

def compute_kinematics_batch(seqs: np.ndarray, l_ever: bool, r_ever: bool) -> np.ndarray:
    B, F, P, _ = seqs.shape
    
    vel = np.zeros_like(seqs)
    vel[:, 1:-1] = (seqs[:, 2:] - seqs[:, :-2]) / 2.0
    vel[:, 0] = vel[:, 1]; vel[:, -1] = vel[:, -2]
    
    acc = np.zeros_like(seqs)
    acc[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / 2.0
    acc[:, 0] = acc[:, 1]; acc[:, -1] = acc[:, -2]
    
    mask = np.zeros((B, F, P, 1), dtype=np.float32)
    if l_ever: mask[:, :, 0:21, 0] = 1.0
    if r_ever: mask[:, :, 21:42, 0] = 1.0
    
    return np.concatenate([seqs, vel, acc, mask], axis=-1).astype(np.float32)

# ─────────────────────────────────────────────
# Worker Process (Hardened & Ultra-Fast)
# ─────────────────────────────────────────────
def process_single_video(task_info):
    try:
        root, video_name, label, cfg, _ = task_info
        out_path = Path(cfg.output_dir)
        video_path = os.path.join(root, video_name)
        
        cap = cv2.VideoCapture(video_path)
        total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FAST-FAIL CHECK (Safe guard with 0 < included)
        if 0 < total_est < cfg.min_raw_frames:
            cap.release()
            return 0
            
        skip = max(1, total_est // int(cfg.target_frames * 1.5)) 
        use_static = True if skip > 1 else False
        
        with mp.solutions.hands.Hands(
            static_image_mode=use_static, 
            max_num_hands=cfg.max_num_hands,
            min_detection_confidence=cfg.min_detection_conf, 
            min_tracking_confidence=cfg.min_tracking_conf,
            model_complexity=cfg.model_complexity
        ) as detector:
            
            l_seq, r_seq, l_valid, r_valid = [], [], [], []
            frame_idx = 0
            processed_count = 0  

            while cap.isOpened():
                ret = cap.grab()
                if not ret: break
                
                if frame_idx % skip != 0:
                    frame_idx += 1
                    continue
                
                ret, frame = cap.retrieve()
                if not ret: break
                
                h, w = frame.shape[:2]
                if max(h, w) > 384:
                    scale = 384 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = detector.process(rgb)
                
                if res.multi_hand_landmarks:
                    for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                        h_label = handedness.classification[0].label
                        if cfg.mirror_handedness: h_label = "Right" if h_label == "Left" else "Left"
                        if handedness.classification[0].score >= cfg.min_detection_conf:
                            coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                            if h_label == "Left" and (not l_valid or l_valid[-1] != frame_idx):
                                l_seq.append(coords); l_valid.append(frame_idx)
                            elif h_label == "Right" and (not r_valid or r_valid[-1] != frame_idx):
                                r_seq.append(coords); r_valid.append(frame_idx)
                
                processed_count += 1 
                frame_idx += 1
                
            cap.release()

        if processed_count < cfg.min_raw_frames or (not l_valid and not r_valid): return 0

        max_detected = max(len(l_valid), len(r_valid))
        if 1.0 - (max_detected / processed_count) > cfg.max_missing_ratio: return 0

        l_full = interpolate_hand(np.array(l_seq), l_valid, frame_idx)
        r_full = interpolate_hand(np.array(r_seq), r_valid, frame_idx)
        combined = np.concatenate([l_full, r_full], axis=1) 
        l_ever, r_ever = bool(l_valid), bool(r_valid)
        
        resampled = temporal_resample(combined, cfg.target_frames)
        normalized = normalize_sequence(resampled, l_ever, r_ever)
        
        final_data = compute_kinematics_batch(normalized[np.newaxis, ...], l_ever, r_ever)
        final_data = final_data.squeeze(0).astype(np.float16) 

        file_hash = hashlib.md5(video_name.encode()).hexdigest()[:6]
        save_name = f"{label}_{Path(video_name).stem}_{file_hash}.npy"
        np.save(out_path / save_name, final_data)
        return 1

    except Exception as e:
        log.error(f"Failed processing {video_name}: {e}")
        return 0

def run_pipeline():
    out_path = Path(CFG.output_dir); out_path.mkdir(parents=True, exist_ok=True)
    
    done_files = set()
    for f in os.listdir(out_path):
        if f.endswith('.npy'):
            done_files.add(f.replace('.npy', ''))
            
    all_videos = []
    
    for root, _, files in os.walk(CFG.raw_video_dir):
        label = Path(root).name
        for f in files:
            if f.lower().endswith(('.mp4', '.mov')):
                file_hash = hashlib.md5(f.encode()).hexdigest()[:6]
                stem = Path(f).stem
                expected_save_name = f"{label}_{stem}_{file_hash}"
                
                if expected_save_name not in done_files:
                    file_path = os.path.join(root, f)
                    file_size = os.path.getsize(file_path)
                    all_videos.append((root, f, label, CFG, file_size))

    total_skipped = len(done_files)
    
    if total_skipped > 0:
        log.info(f"⏭️  Skipping {total_skipped} already-processed videos.")
        
    if len(all_videos) == 0:
        log.info("✅ All videos are already processed!")
        return

    # Sort largest files first to balance the multiprocessing load perfectly
    all_videos.sort(key=lambda x: x[4], reverse=True)

    safe_workers = max(1, min(multiprocessing.cpu_count() - 2, 9))
    chunk_size = 50 # Claude's recommended fixed chunk size
    
    log.info(f"🚀 Found {len(all_videos)} remaining videos. Processing with {safe_workers} workers (Chunksize: {chunk_size})...")
    
    saved_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=safe_workers) as executor:
        for result in executor.map(process_single_video, all_videos, chunksize=chunk_size):
            saved_count += result
            if saved_count > 0 and saved_count % 100 == 0:
                print(f"  → Saved {saved_count} ultra-light sequences...", end="\r")
                
    log.info(f"\n✅ DATASET COMPLETE. Successfully processed {saved_count} new videos.")

if __name__ == "__main__":
    run_pipeline()