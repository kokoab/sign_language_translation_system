"""
SLT Sign Language Classifier — GUI App (Modern UI)
Browse a video OR record from camera -> classify -> skeleton animation.
Uses Apple Vision Framework for extraction (same pipeline as training).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python src/demo_classify.py
"""
import os, sys, time, threading, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from train_stage_1 import compute_bone_features_np
from train_stage_2 import SLTStage2CTC
from model_v14 import SLTStage1V14
from extract_apple_vision import extract_one_video, extract_phrase_video, extract_frames_isolated, extract_frames_continuous
from camera_inference import _ctc_beam_search, _mirror_tta

CKPT_PATH = "models/output_v15_clean/best_model.pth"
STAGE2_CKPT_PATH = "models/output_stage2_v15_reextracted/stage2_best_model.pth"
for p in [CKPT_PATH, "models/output_v14/best_model.pth"]:
    if os.path.exists(p):
        CKPT_PATH = p
        break
for p in [STAGE2_CKPT_PATH, "models/output_stage2_v13/stage2_best_model.pth"]:
    if os.path.exists(p):
        STAGE2_CKPT_PATH = p
        break

_HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

# ── Theme colors ────────────────────────────────────────
COLORS = {
    "bg_dark":       "#0f0f1a",
    "bg_card":       "#1a1a2e",
    "bg_card_alt":   "#16213e",
    "bg_input":      "#0a0a15",
    "accent":        "#e94560",
    "accent_hover":  "#c73e54",
    "accent_blue":   "#3b82f6",
    "accent_green":  "#10b981",
    "accent_green_h":"#059669",
    "text_primary":  "#e0e0e0",
    "text_secondary":"#a0a0b0",
    "text_muted":    "#606070",
    "border":        "#2a2a3e",
    "bar_1":         "#3b82f6",
    "bar_2":         "#8b5cf6",
    "bar_3":         "#06b6d4",
    "bar_4":         "#10b981",
    "bar_5":         "#f59e0b",
    "progress_bg":   "#1a1a2e",
    "skeleton_bg":   "#0f0f1a",
    "recording":     "#e94560",
}

LEGEND_COLORS = {
    "Left Hand":  "#f59e0b",
    "Right Hand": "#8b5cf6",
    "Face":       "#f59e0b",
    "Body":       "#6b7280",
}


def extract_frames_61_legacy(raw_frames, ort_session=None, det_model=None, multi_clip=False):
    """LEGACY: RTMW extraction. No longer used — kept for reference only."""
    if len(raw_frames) < 8:
        return None, None, f"Too short ({len(raw_frames)} frames)"

    total_frames = len(raw_frames)
    max_process = 256 if multi_clip else 128  # Match extract_phrases.py for continuous
    indices = list(range(total_frames))
    if total_frames > max_process:
        step = total_frames / max_process
        indices = [int(i * step) for i in range(max_process)]
        raw_frames = [raw_frames[i] for i in indices]

    h0, w0 = raw_frames[0].shape[:2]
    bbox = np.array([0, 0, w0, h0], dtype=np.float32)
    if det_model is not None:
        try:
            bboxes = det_model(raw_frames[0])
            if bboxes is not None and len(bboxes) > 0:
                bbox = bboxes[0][:4].astype(np.float32)
        except Exception:
            pass

    preprocessed, centers, scales_list = [], [], []
    for bgr in raw_frames:
        img, center, scale = preprocess_frame(bgr, bbox=bbox)
        preprocessed.append(img)
        centers.append(center)
        scales_list.append(scale)
    centers = np.stack(centers)
    scales_arr = np.stack(scales_list)

    all_kps, all_scores = [], []
    bs = 32
    for start in range(0, len(preprocessed), bs):
        end = min(start + bs, len(preprocessed))
        batch_np = np.stack(preprocessed[start:end]).astype(np.float32)
        kps, scs = batch_inference(ort_session, batch_np, centers[start:end], scales_arr[start:end])
        all_kps.append(kps)
        all_scores.append(scs)
    all_kps = np.concatenate(all_kps, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)

    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []
    raw_kps_per_frame = []

    for i in range(len(raw_frames)):
        fi = indices[i] if total_frames > max_process else i
        kps = all_kps[i]
        scs = all_scores[i]
        h, w = raw_frames[i].shape[:2]

        raw_kps_per_frame.append((kps.copy(), scs.copy(), h, w))

        # Duplicate frame prevention (matches training extraction exactly)
        l_scs = scs[_LHAND_START:_LHAND_START+21]
        if l_scs.mean() >= HAND_CONF:
            coords = [[kps[_LHAND_START+j][0]/w, kps[_LHAND_START+j][1]/h, 0.0] for j in range(21)]
            if not l_valid or l_valid[-1] != fi:
                l_seq.append(coords); l_valid.append(fi)
        r_scs = scs[_RHAND_START:_RHAND_START+21]
        if r_scs.mean() >= HAND_CONF:
            coords = [[kps[_RHAND_START+j][0]/w, kps[_RHAND_START+j][1]/h, 0.0] for j in range(21)]
            if not r_valid or r_valid[-1] != fi:
                r_seq.append(coords); r_valid.append(fi)
        face_coords = []
        for fi_idx in _FACE_INDICES:
            if fi_idx < len(kps) and scs[fi_idx] >= FACE_CONF:
                face_coords.append([kps[fi_idx][0]/w, kps[fi_idx][1]/h, 0.0])
            else:
                break
        if len(face_coords) == NUM_FACE_V2:
            if not face_valid or face_valid[-1] != fi:
                face_seq.append(face_coords); face_valid.append(fi)
        body_coords = []
        for bi in _BODY_INDICES:
            if bi < len(kps) and scs[bi] >= BODY_CONF:
                body_coords.append([kps[bi][0]/w, kps[bi][1]/h, 0.0])
            else:
                break
        if len(body_coords) == NUM_BODY_V2:
            if not body_valid or body_valid[-1] != fi:
                body_seq.append(body_coords); body_valid.append(fi)

    if not l_valid and not r_valid:
        return None, None, "No hands detected"
    if l_valid: l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid: r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        return None, None, "All detections rejected"

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever, body_ever = bool(face_valid), bool(body_valid)

    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0,21,3)), l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0,21,3)), r_valid, total_frames)
    face_full = interpolate_generic(np.array(face_seq) if face_seq else np.zeros((0,NUM_FACE_V2,3)),
                                    face_valid, total_frames, NUM_FACE_V2)
    body_full = interpolate_generic(np.array(body_seq) if body_seq else np.zeros((0,NUM_BODY_V2,3)),
                                    body_valid, total_frames, NUM_BODY_V2)

    combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)

    if multi_clip and combined.shape[0] > 40:
        # Multi-clip mode: split raw frames into 32-frame clips (matching extract_phrases.py)
        # NO resampling — use raw interpolated frames directly
        T = combined.shape[0]
        num_clips = T // 32
        remainder = T % 32
        if num_clips == 0:
            pad = np.zeros((32 - T, 61, 3), dtype=np.float32)
            combined = np.concatenate([combined, pad], axis=0)
            num_clips = 1
        elif remainder > 0:
            pad = np.zeros((32 - remainder, 61, 3), dtype=np.float32)
            combined = np.concatenate([combined, pad], axis=0)
            num_clips += 1

        all_clips = []
        for ci in range(num_clips):
            seg = combined[ci*32:(ci+1)*32].copy()
            seg[:, :, :3] = one_euro_filter(seg[:, :, :3])
            if l_ever: seg = stabilize_bones(seg, 0, 21)
            if r_ever: seg = stabilize_bones(seg, 21, 42)
            normed = normalize_sequence_v2(seg, l_ever, r_ever)
            data = compute_kinematics_v2(normed, l_ever, r_ever, face_ever, body_ever)
            all_clips.append(data)
        result = np.concatenate(all_clips, axis=0)  # [N*32, 61, 10]
    else:
        # Single clip mode (isolated sign)
        seg = temporal_resample(combined, 32)
        seg[:, :, :3] = one_euro_filter(seg[:, :, :3])
        if l_ever: seg = stabilize_bones(seg, 0, 21)
        if r_ever: seg = stabilize_bones(seg, 21, 42)
        normed = normalize_sequence_v2(seg, l_ever, r_ever)
        result = compute_kinematics_v2(normed, l_ever, r_ever, face_ever, body_ever)

    return result, raw_kps_per_frame, None


def extract_frames_mediapipe(raw_frames, multi_clip=False):
    """Extract 61-node landmarks using MediaPipe Holistic (hands + face + pose).
    If multi_clip=True, returns [N*32, 61, 10] split into 32-frame clips.
    Otherwise returns [32, 61, 10] (single clip, resampled)."""
    from mediapipe.python.solutions import holistic as mp_holistic

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    if len(raw_frames) < 8:
        return None, None, f"Too short ({len(raw_frames)} frames)"

    total_frames = len(raw_frames)
    max_process = 256 if multi_clip else 128
    if total_frames > max_process:
        step = total_frames / max_process
        indices = [int(i * step) for i in range(max_process)]
        raw_frames = [raw_frames[i] for i in indices]
        total_frames = len(raw_frames)

    l_seq, r_seq, face_seq, body_seq = [], [], [], []
    l_valid, r_valid, face_valid, body_valid = [], [], [], []
    raw_kps_per_frame = []

    _MP_FACE = [1, 152, 10, 234, 454, 61, 291, 13, 14, 107, 70, 336, 300, 33, 263]
    _MP_BODY = [11, 12, 13, 14]

    for fidx, frame in enumerate(raw_frames):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        kps_133 = np.zeros((133, 2))
        scs_133 = np.zeros(133)

        l_detected = False
        r_detected = False

        if res.left_hand_landmarks:
            pts = [[lm.x, lm.y, 0.0] for lm in res.left_hand_landmarks.landmark]
            px = [[lm.x * w, lm.y * h] for lm in res.left_hand_landmarks.landmark]
            kps_133[91:112] = px
            scs_133[91:112] = 0.9
            l_seq.append(pts)
            l_valid.append(fidx)
            l_detected = True

        if res.right_hand_landmarks:
            pts = [[lm.x, lm.y, 0.0] for lm in res.right_hand_landmarks.landmark]
            px = [[lm.x * w, lm.y * h] for lm in res.right_hand_landmarks.landmark]
            kps_133[112:133] = px
            scs_133[112:133] = 0.9
            r_seq.append(pts)
            r_valid.append(fidx)
            r_detected = True

        if l_detected and not r_detected:
            r_seq.append(l_seq[-1])
            r_valid.append(fidx)
            kps_133[112:133] = kps_133[91:112]
            scs_133[112:133] = 0.9
        elif r_detected and not l_detected:
            l_seq.append(r_seq[-1])
            l_valid.append(fidx)
            kps_133[91:112] = kps_133[112:133]
            scs_133[91:112] = 0.9

        if res.face_landmarks:
            face_coords = []
            for i, fi in enumerate(_MP_FACE):
                lm = res.face_landmarks.landmark[fi]
                face_coords.append([lm.x, lm.y, 0.0])
                coco_fi = _FACE_INDICES[i] if i < len(_FACE_INDICES) else None
                if coco_fi is not None:
                    kps_133[coco_fi] = [lm.x * w, lm.y * h]
                    scs_133[coco_fi] = 0.9
            face_seq.append(face_coords)
            face_valid.append(fidx)

        if res.pose_landmarks:
            body_coords = []
            body_ok = True
            for bi in _MP_BODY:
                lm = res.pose_landmarks.landmark[bi]
                if lm.visibility > 0.5:
                    body_coords.append([lm.x, lm.y, 0.0])
                else:
                    body_ok = False
                    break
            if body_ok and len(body_coords) == 4:
                body_seq.append(body_coords)
                body_valid.append(fidx)

            for bi in [11, 12, 13, 14, 15, 16]:
                if bi < len(res.pose_landmarks.landmark):
                    lm = res.pose_landmarks.landmark[bi]
                    mp_to_coco = {11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10}
                    if bi in mp_to_coco:
                        ci = mp_to_coco[bi]
                        kps_133[ci] = [lm.x * w, lm.y * h]
                        scs_133[ci] = lm.visibility

        raw_kps_per_frame.append((kps_133, scs_133, h, w))

    holistic.close()

    if not l_valid and not r_valid:
        return None, None, "No hands detected"

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever, body_ever = bool(face_valid), bool(body_valid)

    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)),
                              l_valid, total_frames)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)),
                              r_valid, total_frames)
    face_full = interpolate_generic(np.array(face_seq) if face_seq else np.zeros((0, NUM_FACE_V2, 3)),
                                    face_valid, total_frames, NUM_FACE_V2)
    body_full = interpolate_generic(np.array(body_seq) if body_seq else np.zeros((0, NUM_BODY_V2, 3)),
                                    body_valid, total_frames, NUM_BODY_V2)

    combined = np.concatenate([l_full, r_full, face_full, body_full], axis=1)

    if multi_clip and combined.shape[0] > 40:
        T = combined.shape[0]
        num_clips = T // 32
        remainder = T % 32
        if num_clips == 0:
            pad = np.zeros((32 - T, 61, 3), dtype=np.float32)
            combined = np.concatenate([combined, pad], axis=0)
            num_clips = 1
        elif remainder > 0:
            pad = np.zeros((32 - remainder, 61, 3), dtype=np.float32)
            combined = np.concatenate([combined, pad], axis=0)
            num_clips += 1

        all_clips = []
        for ci in range(num_clips):
            seg = combined[ci*32:(ci+1)*32].copy()
            seg[:, :, :3] = one_euro_filter(seg[:, :, :3])
            if l_ever: seg = stabilize_bones(seg, 0, 21)
            if r_ever: seg = stabilize_bones(seg, 21, 42)
            normed = normalize_sequence_v2(seg, l_ever, r_ever)
            data = compute_kinematics_v2(normed, l_ever, r_ever, face_ever, body_ever)
            all_clips.append(data)
        result_arr = np.concatenate(all_clips, axis=0)
    else:
        seg = temporal_resample(combined, 32)
        seg[:, :, :3] = one_euro_filter(seg[:, :, :3])
        if l_ever: seg = stabilize_bones(seg, 0, 21)
        if r_ever: seg = stabilize_bones(seg, 21, 42)
        normed = normalize_sequence_v2(seg, l_ever, r_ever)
        result_arr = compute_kinematics_v2(normed, l_ever, r_ever, face_ever, body_ever)

    return result_arr, raw_kps_per_frame, None


def render_skeleton_frame(kp_data, size=480):
    """Render a single skeleton frame on a dark canvas."""
    bg = np.full((size, size, 3), [15, 15, 26], dtype=np.uint8)
    if kp_data is None:
        return bg
    kps, scs, orig_h, orig_w = kp_data
    scale = size / max(orig_h, orig_w) * 0.85
    ox = (size - orig_w * scale) / 2
    oy = (size - orig_h * scale) / 2
    def pt(idx): return (int(kps[idx][0] * scale + ox), int(kps[idx][1] * scale + oy))
    def ok(idx): return scs[idx] > 0.3

    # Body — slate gray
    for a, b in [(5,6),(5,7),(6,8),(7,9),(8,10)]:
        if ok(a) and ok(b): cv2.line(bg, pt(a), pt(b), (100,116,139), 2, cv2.LINE_AA)
    for idx in [5,6,7,8,9,10]:
        if ok(idx): cv2.circle(bg, pt(idx), 5, (71,85,105), -1, cv2.LINE_AA)

    # Left hand — amber/orange
    for a, b in _HAND_EDGES:
        ia, ib = _LHAND_START+a, _LHAND_START+b
        if ok(ia) and ok(ib): cv2.line(bg, pt(ia), pt(ib), (0,140,217), 2, cv2.LINE_AA)
    for i in range(21):
        if ok(_LHAND_START+i): cv2.circle(bg, pt(_LHAND_START+i), 3, (0,120,200), -1, cv2.LINE_AA)

    # Right hand — indigo/purple
    for a, b in _HAND_EDGES:
        ia, ib = _RHAND_START+a, _RHAND_START+b
        if ok(ia) and ok(ib): cv2.line(bg, pt(ia), pt(ib), (235,67,99), 2, cv2.LINE_AA)
    for i in range(21):
        if ok(_RHAND_START+i): cv2.circle(bg, pt(_RHAND_START+i), 3, (220,50,80), -1, cv2.LINE_AA)

    # Face — warm brown
    for fi in _FACE_INDICES:
        if ok(fi): cv2.circle(bg, pt(fi), 4, (30,100,180), -1, cv2.LINE_AA)
    return bg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Modern UI App
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SignClassifierApp:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("SLT  —  Sign Language Translator")
        self.root.geometry("1400x800")
        self.root.minsize(1100, 650)
        self.root.configure(fg_color=COLORS["bg_dark"])

        self.model = None
        self.idx_to_label = None
        self.device = None
        self.skeleton_frames = []

        self.camera_open = False
        self.recording = False
        self.recorded_frames = []
        self.cap = None
        self.camera_timer = None
        self._playback_running = False
        self._playback_source_frames = []
        self._extracted_result = None   # [T, 61, C] array stored after extraction

        self._build_ui()
        self._load_models()

    # ── Layout ──────────────────────────────────────────────
    def _build_ui(self):
        # Main two-column: fixed narrow left sidebar, expanding right panel
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ── LEFT SIDEBAR ── (fixed 380px)
        sidebar = ctk.CTkFrame(self.root, fg_color=COLORS["bg_card"],
                               corner_radius=20, border_width=1,
                               border_color=COLORS["border"], width=380)
        sidebar.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="ns")
        sidebar.pack_propagate(False)

        # Header + version badge
        hdr = ctk.CTkFrame(sidebar, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(20, 0))

        title_row = ctk.CTkFrame(hdr, fg_color="transparent")
        title_row.pack(fill="x")
        ctk.CTkLabel(title_row, text="Sign Language",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")
        ver_pill = ctk.CTkFrame(title_row, fg_color="transparent",
                               corner_radius=6, border_width=1,
                               border_color=COLORS["border"], height=22)
        ver_pill.pack(side="right")
        ver_pill.pack_propagate(False)
        self.version_label = ctk.CTkLabel(ver_pill, text="DS-GCN-TCN v15",
                     font=ctk.CTkFont(size=10),
                     text_color=COLORS["text_muted"],
                     fg_color="transparent")
        self.version_label.pack(padx=8, pady=2)

        ctk.CTkLabel(hdr, text="Classifier",
                     font=ctk.CTkFont(size=14),
                     text_color=COLORS["text_secondary"]).pack(anchor="w")

        # LIVE POSE ESTIMATION panel
        pose_outer = ctk.CTkFrame(sidebar, fg_color="transparent")
        pose_outer.pack(fill="x", padx=20, pady=(12, 0))
        ctk.CTkLabel(pose_outer, text="LIVE POSE ESTIMATION",
                     font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_muted"]).pack(anchor="w", pady=(0, 4))

        pose_box = ctk.CTkFrame(pose_outer, fg_color=COLORS["skeleton_bg"],
                                corner_radius=12, border_width=1,
                                border_color=COLORS["border"], height=160)
        pose_box.pack(fill="x")
        pose_box.pack_propagate(False)

        self.skeleton_label = ctk.CTkLabel(
            pose_box, text="",
            fg_color="transparent")
        self.skeleton_label.pack(expand=True, fill="both", padx=8, pady=8)

        pose_indicator = ctk.CTkFrame(pose_box, fg_color="transparent")
        pose_indicator.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-8)
        ctk.CTkFrame(pose_indicator, fg_color=COLORS["accent_green"],
                      corner_radius=4, width=8, height=8).pack(side="left")
        ctk.CTkLabel(pose_indicator, text="LIVE POSE",
                     font=ctk.CTkFont(size=9),
                     text_color=COLORS["text_muted"]).pack(side="left", padx=(4, 0))

        # Buttons
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(12, 0))
        btn_frame.grid_columnconfigure((0, 1), weight=1)

        self.browse_btn = ctk.CTkButton(
            btn_frame, text="Browse Video",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["accent_blue"], hover_color="#2563eb",
            height=38, corner_radius=10,
            command=self._browse)
        self.browse_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.camera_btn = ctk.CTkButton(
            btn_frame, text="Open Camera",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=COLORS["accent_green"], hover_color=COLORS["accent_green_h"],
            height=38, corner_radius=10,
            command=self._toggle_camera)
        self.camera_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        self.record_btn = ctk.CTkButton(
            sidebar, text="●  Record",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            text_color="white",
            height=44, corner_radius=12,
            command=self._toggle_record, state="disabled")
        self.record_btn.pack(fill="x", padx=20, pady=(8, 0))

        # Mode toggle
        self.mode_seg = ctk.CTkSegmentedButton(
            sidebar,
            values=["Isolated Sign", "Continuous"],
            font=ctk.CTkFont(size=11),
            selected_color=COLORS["accent_blue"],
            selected_hover_color="#1d4ed8",
            unselected_color="#6b7280",
            unselected_hover_color="#4b5563",
            corner_radius=8, height=30,
            command=self._on_mode_change)
        self.mode_seg.set("Isolated Sign")
        self.mode_seg.pack(fill="x", padx=20, pady=(12, 0))
        self.use_continuous = False

        # Status + progress
        status_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        status_frame.pack(fill="x", padx=20, pady=(14, 0))

        sf_top = ctk.CTkFrame(status_frame, fg_color="transparent")
        sf_top.pack(fill="x")
        ctk.CTkLabel(sf_top, text="Processing Status",
                     font=ctk.CTkFont(size=11),
                     text_color=COLORS["text_muted"]).pack(side="left")
        self.status_label = ctk.CTkLabel(
            sf_top, text="Loading...",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["accent_green"], anchor="e")
        self.status_label.pack(side="right")

        self.progress = ctk.CTkProgressBar(
            status_frame, height=5, corner_radius=3,
            fg_color=COLORS["progress_bg"],
            progress_color=COLORS["accent_green"])
        self.progress.pack(fill="x", pady=(4, 0))
        self.progress.set(0)
        self._progress_animating = False

        # Predictions
        pred_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        pred_frame.pack(fill="x", padx=20, pady=(16, 0))

        pred_hdr = ctk.CTkFrame(pred_frame, fg_color="transparent")
        pred_hdr.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(pred_hdr, text="PREDICTIONS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["text_primary"]).pack(side="left")
        ctk.CTkLabel(pred_hdr, text="Top 5",
                     font=ctk.CTkFont(size=10),
                     text_color=COLORS["text_muted"],
                     fg_color=COLORS["bg_card_alt"],
                     corner_radius=6, width=40, height=20).pack(side="right")

        bar_colors = [COLORS["bar_1"], COLORS["bar_2"], COLORS["bar_3"],
                      COLORS["bar_4"], COLORS["bar_5"]]

        self.pred_rows = []
        for i in range(5):
            row = ctk.CTkFrame(pred_frame, fg_color="transparent", height=30)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)

            rank = ctk.CTkLabel(row, text=f"#{i+1}",
                                font=ctk.CTkFont(size=10),
                                text_color=COLORS["text_muted"], width=24)
            rank.pack(side="left")

            name_lbl = ctk.CTkLabel(row, text="---",
                                    font=ctk.CTkFont(size=13 if i == 0 else 12,
                                                     weight="bold" if i == 0 else "normal"),
                                    text_color=COLORS["text_primary"] if i == 0 else COLORS["text_secondary"],
                                    anchor="w", width=80)
            name_lbl.pack(side="left", padx=(0, 6))

            bar_bg = ctk.CTkFrame(row, fg_color=COLORS["progress_bg"],
                                  corner_radius=3, height=12)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(0, 6))
            bar_bg.pack_propagate(False)

            bar_fill = ctk.CTkFrame(bar_bg, fg_color=bar_colors[i],
                                    corner_radius=3, height=12, width=0)
            bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0)

            conf_lbl = ctk.CTkLabel(row, text="",
                                    font=ctk.CTkFont(size=11),
                                    text_color=COLORS["text_muted"], width=45, anchor="e")
            conf_lbl.pack(side="right")

            self.pred_rows.append((name_lbl, bar_fill, bar_bg, conf_lbl))

        # Gloss display
        self.gloss_label = ctk.CTkLabel(
            sidebar, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"], anchor="w",
            wraplength=300, justify="left")
        self.gloss_label.pack(fill="x", padx=20, pady=(10, 0), anchor="w")

        # Legend at bottom of sidebar
        legend_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        legend_frame.pack(side="bottom", fill="x", padx=20, pady=(0, 16))
        for name, color in [("Left Hand", "#f59e0b"), ("Right Hand", "#8b5cf6")]:
            pill = ctk.CTkFrame(legend_frame, fg_color=COLORS["bg_card_alt"],
                                corner_radius=10, height=24)
            pill.pack(side="left", padx=2)
            pill.pack_propagate(False)
            ctk.CTkFrame(pill, fg_color=color, corner_radius=4,
                          width=8, height=8).pack(side="left", padx=(8, 4), pady=8)
            ctk.CTkLabel(pill, text=name, font=ctk.CTkFont(size=9),
                         text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 8))

        # ── RIGHT PANEL ──
        right = ctk.CTkFrame(self.root, fg_color=COLORS["bg_card"],
                             corner_radius=20, border_width=1,
                             border_color=COLORS["border"])
        right.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Camera status header
        self.cam_header = ctk.CTkLabel(
            right, text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"], anchor="w")
        self.cam_header.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="w")

        # Video display (fills most of right panel)
        self.video_label = ctk.CTkLabel(
            right, text="Waiting for video input...",
            fg_color=COLORS["skeleton_bg"],
            text_color=COLORS["text_muted"],
            font=ctk.CTkFont(size=14),
            corner_radius=16)
        self.video_label.grid(row=1, column=0, padx=12, pady=(6, 6), sticky="nsew")

        # Bottom area: prediction + glosses + frame info
        bottom = ctk.CTkFrame(right, fg_color="transparent")
        bottom.grid(row=2, column=0, padx=16, pady=(0, 12), sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(bottom, text="CURRENT PREDICTION",
                     font=ctk.CTkFont(size=10),
                     text_color=COLORS["text_muted"]).grid(row=0, column=0, sticky="w")

        self.translation_label = ctk.CTkLabel(
            bottom, text="",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS["accent_blue"],
            wraplength=700, justify="left", anchor="w")
        self.translation_label.grid(row=1, column=0, sticky="w")

        self.skel_title = ctk.CTkLabel(
            bottom, text="",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"], anchor="w")
        self.skel_title.grid(row=2, column=0, sticky="w")

        self.frame_info = ctk.CTkLabel(
            bottom, text="",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_primary"],
            fg_color=COLORS["accent"],
            corner_radius=6, width=90, height=22)
        self.frame_info.grid(row=1, column=0, sticky="e")

        # Legend at bottom of right panel
        r_legend = ctk.CTkFrame(right, fg_color=COLORS["bg_card_alt"],
                                corner_radius=12, height=36)
        r_legend.grid(row=3, column=0, padx=16, pady=(0, 12), sticky="ew")
        r_legend.pack_propagate(False)
        r_legend_inner = ctk.CTkFrame(r_legend, fg_color="transparent")
        r_legend_inner.pack(expand=True)
        for name, color in LEGEND_COLORS.items():
            ctk.CTkFrame(r_legend_inner, fg_color=color, corner_radius=5,
                          width=10, height=10).pack(side="left", padx=(16, 4), pady=13)
            ctk.CTkLabel(r_legend_inner, text=name, font=ctk.CTkFont(size=11),
                         text_color=COLORS["text_secondary"]).pack(side="left", padx=(0, 8))

        # Store right panel reference for border color changes during recording
        self._right_panel = right

    # ── Model loading ──────────────────────────────────────
    def _load_models(self):
        self._start_progress()

        def _load():
            try:
                self.device = torch.device('cpu')
                ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
                model_type = ckpt.get('model_type', '')
                self.root.after(0, lambda: self.version_label.configure(text="DS-GCN-TCN v15"))
                self.model = SLTStage1V14(
                    num_classes=ckpt['num_classes'],
                    d_model=ckpt.get('d_model', 384),
                    use_arcface=True,
                ).to(self.device)
                sd = {k.replace('_orig_mod.', ''): v
                      for k, v in (ckpt.get('ema_shadow') or ckpt['model_state_dict']).items()}
                self.model.load_state_dict(sd, strict=False)
                self.model.eval()
                self.model.set_epoch(200)
                self.idx_to_label = ckpt.get('idx_to_label', {str(v): k for k, v in ckpt['label_to_idx'].items()})

                # Load Stage 2 if available
                self.stage2_model = None
                self.idx_to_gloss = None
                if os.path.exists(STAGE2_CKPT_PATH):
                    ckpt2 = torch.load(STAGE2_CKPT_PATH, map_location=self.device, weights_only=False)
                    s2_d_model = ckpt2.get('d_model', 384)
                    s2_vocab = ckpt2['vocab_size']
                    self.stage2_model = SLTStage2CTC(
                        vocab_size=s2_vocab,
                        stage1_ckpt=None,
                        d_model=s2_d_model,
                    ).to(self.device)
                    s2_sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt2['model_state_dict'].items()}
                    self.stage2_model.load_state_dict(s2_sd, strict=False)
                    self.stage2_model.eval()
                    self.idx_to_gloss = ckpt2['idx_to_gloss']

                # Load Stage 3 (T5 translator)
                self.t5_model = None
                self.t5_tokenizer = None
                # Try multiple paths for T5 weights
                t5_candidates = [
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "weights", "slt_final_t5_model"),
                    "weights/slt_final_t5_model",
                    os.path.join(os.getcwd(), "weights", "slt_final_t5_model"),
                ]
                t5_dir = None
                for cand in t5_candidates:
                    if os.path.isdir(cand):
                        t5_dir = cand
                        break
                if t5_dir:
                    try:
                        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                        self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_dir)
                        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_dir).to(self.device)
                        self.t5_model.eval()
                    except Exception as e:
                        print(f"T5 load error: {e}")

                n_classes = len(self.idx_to_label)
                s2_status = " + CTC" if self.stage2_model else ""
                t5_status = " + T5" if self.t5_model else ""
                self.root.after(0, lambda: self._set_status(f"Ready  --  {n_classes} classes{s2_status}{t5_status}"))
                self.root.after(0, self._stop_progress)
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error: {e}"))
                self.root.after(0, self._stop_progress)

        threading.Thread(target=_load, daemon=True).start()

    # ── Mode toggle ────────────────────────────────────────
    def _on_mode_change(self, value):
        self.use_continuous = (value == "Continuous")

    # ── Progress helpers ───────────────────────────────────
    def _start_progress(self):
        self._progress_animating = True
        self._progress_value = 0.0
        self._animate_progress()

    def _animate_progress(self):
        if not self._progress_animating:
            return
        self._progress_value = (self._progress_value + 0.015) % 1.0
        self.progress.set(self._progress_value)
        self.root.after(30, self._animate_progress)

    def _stop_progress(self):
        self._progress_animating = False
        self.progress.set(0)

    def _set_status(self, text):
        self.status_label.configure(text=text)

    # ── Extractor toggle ───────────────────────────────────
    def _on_extractor_change(self, value):
        pass  # No longer needed — always dual ensemble

    # ── Camera ─────────────────────────────────────────────
    def _toggle_camera(self):
        if not self.camera_open:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)  # Force 30fps to match training
            if not self.cap.isOpened():
                self._set_status("Cannot open camera")
                return
            self.camera_open = True
            self._latest_frame = None
            self._start_capture_thread()
            self.camera_btn.configure(text="Close Camera",
                                      fg_color="#6b7280",
                                      hover_color="#4b5563")
            self.record_btn.configure(state="normal")
            self._set_status("Camera open  --  press Record to start")
            self._update_camera()
        else:
            self._close_camera()

    def _close_camera(self):
        self.camera_open = False
        self.recording = False
        self._stop_capture_thread()
        if self.camera_timer:
            self.root.after_cancel(self.camera_timer)
            self.camera_timer = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_btn.configure(text="Open Camera",
                                  fg_color=COLORS["accent_green"],
                                  hover_color=COLORS["accent_green_h"])
        self.record_btn.configure(text="●  Record",
                                  fg_color=COLORS["accent"],
                                  text_color="white",
                                  state="disabled")
        self.video_label.configure(image=None, text="Waiting for video input...")
        self.cam_header.configure(text="", text_color=COLORS["text_muted"])

    def _start_capture_thread(self):
        """Background thread that reads frames from camera as fast as possible."""
        self._latest_frame = None
        self._capture_running = True
        def capture_loop():
            while self._capture_running and self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if self.recording:
                    self.recorded_frames.append(frame.copy())
                self._latest_frame = frame
        self._capture_thread = threading.Thread(target=capture_loop, daemon=True)
        self._capture_thread.start()

    def _stop_capture_thread(self):
        self._capture_running = False
        if hasattr(self, '_capture_thread') and self._capture_thread:
            self._capture_thread.join(timeout=1.0)

    def _update_camera(self):
        if not self.camera_open or not self.cap:
            return

        frame = self._latest_frame
        if frame is None:
            self.camera_timer = self.root.after(33, self._update_camera)
            return

        fh, fw = frame.shape[:2]
        display_frame = cv2.flip(frame, 1)  # Flip for display (mirror effect)

        if self.recording:
            self.cam_header.configure(
                text="●  RECORDING...",
                text_color=COLORS["accent"])
            self._right_panel.configure(border_color=COLORS["accent"])
            sec = time.time() - self._record_start_time
            n_frames = len(self.recorded_frames)
            cv2.putText(display_frame, f"REC {sec:.1f}s  {n_frames}f", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.cam_header.configure(
                text="●  CAMERA READY",
                text_color=COLORS["text_muted"])
            self._right_panel.configure(border_color=COLORS["border"])

        # Display in right panel (scale to fit)
        img = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        # Get actual widget size for responsive scaling
        vw = max(self.video_label.winfo_width(), 640)
        vh = max(self.video_label.winfo_height(), 480)
        # Maintain aspect ratio
        scale = min(vw / fw, vh / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=(dw, dh))
        self.video_label.configure(image=photo, text="")
        self.video_label._photo = photo

        self.camera_timer = self.root.after(33, self._update_camera)

    def _toggle_record(self):
        if not self.recording:
            self.recording = True
            self.recorded_frames = []
            self._record_start_time = time.time()
            self.record_btn.configure(text="■  Stop Recording",
                                      fg_color=COLORS["recording"],
                                      hover_color="#c73e54",
                                      text_color="white")
            self.browse_btn.configure(state="disabled")
            self._set_status("Recording...  sign at normal speed (1-2 sec per sign)")
            self._clear_predictions()
        else:
            self.recording = False
            record_elapsed = time.time() - self._record_start_time
            self.record_btn.configure(text="●  Record", fg_color=COLORS["accent"], text_color="white")
            self._close_camera()

            if len(self.recorded_frames) < 30:
                self._set_status("Too short  --  record at least 1-2 seconds")
                return

            frames = self.recorded_frames.copy()
            self.recorded_frames = []

            # Compute actual capture FPS
            actual_fps = len(frames) / record_elapsed if record_elapsed > 0 else 30.0

            # Save debug copy with actual FPS
            tmp_path = '/tmp/slt_camera_recording.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_path, fourcc, actual_fps, (frames[0].shape[1], frames[0].shape[0]))
            for f in frames:
                out.write(f)
            out.release()

            self._set_status(f"Processing {len(frames)} frames...")
            self._start_progress()
            self.browse_btn.configure(state="disabled")

            self._playback_source_frames = frames.copy()
            threading.Thread(target=lambda: self._process_frames(frames),
                           daemon=True).start()

    # ── Browse ─────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select a sign language video",
            filetypes=[("Video files", "*.mp4 *.mov *.webm *.avi"), ("All files", "*.*")])
        if not path:
            return

        if self.model is None:
            self._set_status("Still loading...")
            return

        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 8:
            self._set_status("Video too short")
            return

        self._clear_predictions()
        self._set_status(f"Processing {os.path.basename(path)}...")
        self._start_progress()
        self.browse_btn.configure(state="disabled")

        self._playback_source_frames = frames.copy()
        threading.Thread(target=lambda: self._process_frames(frames),
                       daemon=True).start()

    def _process_npy(self, path):
        """Process a pre-extracted .npy file directly (bypasses extraction)."""
        t0 = time.time()
        arr = np.load(path).astype(np.float32)
        result_16ch = compute_bone_features_np(arr)
        x = torch.from_numpy(result_16ch).unsqueeze(0).to(self.device)

        predictions = []
        stage2_glosses = None

        if self.use_continuous and arr.shape[0] > 32:
            # Multi-clip: per-clip classification
            n_clips = x.shape[1] // 32
            with torch.no_grad():
                for ci in range(min(n_clips, 5)):
                    clip = x[:, ci*32:(ci+1)*32, :, :]
                    logits = self.model(clip)
                    probs = torch.softmax(logits, dim=-1)[0]
                    top1_prob, top1_idx = probs.topk(1)
                    label = self.idx_to_label[str(top1_idx[0].item())]
                    predictions.append((f"Clip {ci+1}: {label}", top1_prob[0].item() * 100))
            while len(predictions) < 5:
                predictions.append(("---", 0))

            if self.stage2_model is not None:
                try:
                    n_frames = x.shape[1]
                    with torch.no_grad():
                        x_lens = torch.tensor([n_frames], dtype=torch.long, device=self.device)
                        s2_logits, s2_lens = self.stage2_model(x, x_lens)
                        s2_log_probs = torch.log_softmax(s2_logits[0], dim=-1).cpu().numpy()
                        preds = s2_log_probs.argmax(axis=-1)
                        decoded = []
                        prev = 0
                        for p in preds:
                            if p != 0 and p != prev:
                                gloss = self.idx_to_gloss.get(int(p), self.idx_to_gloss.get(str(p), f"UNK_{p}"))
                                decoded.append(gloss)
                            prev = p
                        if decoded:
                            stage2_glosses = " ".join(decoded)
                except Exception as e:
                    stage2_glosses = f"(error: {e})"
        else:
            # Single clip
            with torch.no_grad():
                logits = self.model(x[:, :32])
                probs = torch.softmax(logits, dim=-1)[0]
            top5_probs, top5_idx = probs.topk(5)
            predictions = [(self.idx_to_label[str(i.item())], p.item() * 100)
                          for p, i in zip(top5_probs, top5_idx)]

        # T5 translation
        translation = None
        gloss_input = stage2_glosses if stage2_glosses else predictions[0][0]
        # Strip "Clip N: " prefix if present
        if gloss_input.startswith("Clip"):
            gloss_input = gloss_input.split(": ", 1)[-1]
        if self.t5_model is not None and self.t5_tokenizer is not None:
            try:
                prompt = f"Translate this ASL gloss to natural conversational English: {gloss_input}"
                inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.t5_model.generate(**inputs, max_new_tokens=64)
                translation = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception:
                pass

        elapsed = time.time() - t0

        def _update():
            self._stop_progress()
            self.browse_btn.configure(state="normal")
            self._set_status(f"Done in {elapsed:.1f}s (.npy direct)")
            for i, (label, conf) in enumerate(predictions):
                name_lbl, bar_fill, bar_bg, conf_lbl = self.pred_rows[i]
                name_lbl.configure(text=label)
                bar_fill.place(x=0, y=0, relheight=1.0, relwidth=conf / 100.0)
                conf_lbl.configure(text=f"{conf:.1f}%" if conf > 0 else "")
            title = stage2_glosses if stage2_glosses else predictions[0][0]
            self.skel_title.configure(text=title)
            self.gloss_label.configure(text=f"Gloss: {gloss_input}")
            if translation:
                self.translation_label.configure(text=translation)
            else:
                self.translation_label.configure(text="")

        self.root.after(0, _update)

    # ── Shared processing ──────────────────────────────────
    def _clear_predictions(self):
        for name_lbl, bar_fill, _, conf_lbl in self.pred_rows:
            name_lbl.configure(text="---")
            bar_fill.place(x=0, y=0, relheight=1.0, relwidth=0)
            conf_lbl.configure(text="")
        self.skel_title.configure(text="")
        self.frame_info.configure(text="")
        self.gloss_label.configure(text="")
        self.translation_label.configure(text="")
        self._playback_running = False

    def _process_frames(self, frames):
        t0 = time.time()

        # Extract directly from frames (no temp file — avoids codec re-encoding artifacts)
        predictions = []
        stage2_glosses = None

        if self.use_continuous:
            result = extract_frames_continuous(frames)
        else:
            result = extract_frames_isolated(frames)

        if result is None:
            self.root.after(0, lambda: self._set_status("Extraction failed — no hands detected"))
            self.root.after(0, self._stop_progress)
            self.root.after(0, lambda: self.browse_btn.configure(state="normal"))
            return

        # Store raw XYZ result for skeleton playback [T, 61, 10]
        self._extracted_result = result.astype(np.float32)

        result_16ch = compute_bone_features_np(result.astype(np.float32))
        x = torch.from_numpy(result_16ch).unsqueeze(0).to(self.device)

        if self.use_continuous:
            # Stage 2 CTC decode — primary method for continuous recognition
            if self.stage2_model is not None:
                try:
                    n_frames = x.shape[1]
                    with torch.no_grad():
                        x_lens = torch.tensor([n_frames], dtype=torch.long, device=self.device)
                        s2_logits, s2_lens = self.stage2_model(x, x_lens)
                        log_probs = torch.log_softmax(s2_logits[0], dim=-1).cpu().numpy()
                        preds = log_probs[:s2_lens[0].item()].argmax(axis=-1)
                        decoded = []
                        prev = 0
                        for p in preds:
                            if p != 0 and p != prev:
                                gloss = self.idx_to_gloss.get(int(p), self.idx_to_gloss.get(str(p), f"UNK_{p}"))
                                decoded.append(gloss)
                            prev = p
                        if decoded:
                            stage2_glosses = " ".join(decoded)
                except Exception:
                    stage2_glosses = None

            # Show CTC glosses as predictions (one per detected sign)
            if stage2_glosses:
                for i, g in enumerate(stage2_glosses.split()):
                    predictions.append((g, 100.0))
            else:
                # Fallback: per-clip Stage 1
                n_clips = x.shape[1] // 32
                with torch.no_grad():
                    for ci in range(min(n_clips, 8)):
                        clip = x[:, ci*32:(ci+1)*32, :, :]
                        logits = self.model(clip)
                        probs = torch.softmax(logits, dim=-1)[0]
                        top1_prob, top1_idx = probs.topk(1)
                        label = self.idx_to_label[str(top1_idx[0].item())]
                        conf = top1_prob[0].item() * 100
                        predictions.append((label, conf))
                deduped = []
                for label, _ in predictions:
                    if not deduped or deduped[-1] != label:
                        deduped.append(label)
                stage2_glosses = " ".join(deduped)

            while len(predictions) < 5:
                predictions.append(("---", 0))
        else:
            # Isolated mode: single Apple Vision extraction
            with torch.no_grad():
                logits = self.model(x[:, :32])
                probs = torch.softmax(logits, dim=-1)[0]
            top5_probs, top5_idx = probs.topk(5)
            predictions = [(self.idx_to_label[str(i.item())], p.item() * 100)
                          for p, i in zip(top5_probs, top5_idx)]

        # Disambiguate confused signs using motion features when confidence is low
        if predictions and not self.use_continuous:
            top_label = predictions[0][0]
            top_conf = predictions[0][1]
            top5_labels = set(p[0] for p in predictions[:5])
            data_f32 = result.astype(np.float32) if hasattr(result, 'astype') else result

            if top_conf < 90 and data_f32.shape[0] >= 16:
                lh_mask = data_f32[:, 0, 9].mean()
                rh_mask = data_f32[:, 21, 9].mean()
                wrist = 0 if lh_mask > rh_mask else 21
                nose = 42
                T = data_f32.shape[0]
                mid = T // 2
                end = min(T - 2, 28)

                y_traj = data_f32[end, wrist, 1] - data_f32[4, wrist, 1]
                face_dist_mid = np.linalg.norm(data_f32[mid, wrist, :3] - data_f32[mid, nose, :3])
                face_dist_end = np.linalg.norm(data_f32[end, wrist, :3] - data_f32[end, nose, :3])
                motion = np.abs(np.diff(data_f32[:, wrist, :3], axis=0)).sum()
                hand_height = data_f32[:, wrist, 1].mean() - data_f32[:, nose, 1].mean()
                face_ratio = face_dist_end / (face_dist_mid + 1e-6)

                new_label = top_label

                # GOOD vs THANKYOU: y_trajectory separates them
                if top_label in ('GOOD', 'THANKYOU', 'HAPPY', 'FEEL') or \
                   top5_labels & {'GOOD', 'THANKYOU'}:
                    if y_traj < -0.3:
                        new_label = 'GOOD'       # hand moves down
                    elif face_ratio > 1.10:
                        new_label = 'THANKYOU'    # hand moves away from face

                # SIX vs W: motion and height
                elif top_label in ('SIX', 'W') or top5_labels & {'SIX', 'W'}:
                    if motion > 0.4 and hand_height > 1.4:
                        new_label = 'SIX'
                    elif motion < 0.3:
                        new_label = 'W'

                # O vs ZERO: hand height
                elif top_label in ('O', 'ZERO') or top5_labels & {'O', 'ZERO'}:
                    if hand_height > 1.0:
                        new_label = 'ZERO'
                    else:
                        new_label = 'O'

                # SCHOOL vs COOK: y_trajectory
                elif top_label in ('SCHOOL', 'COOK') or top5_labels & {'SCHOOL', 'COOK'}:
                    if y_traj > 0.1:
                        new_label = 'COOK'
                    else:
                        new_label = 'SCHOOL'

                if new_label != top_label:
                    predictions[0] = (new_label, top_conf)

        # Stage 3: T5 translation
        translation = None
        gloss_input = stage2_glosses if stage2_glosses else predictions[0][0]
        if gloss_input.startswith("Clip"):
            gloss_input = gloss_input.split(": ", 1)[-1]
        if self.t5_model is not None and self.t5_tokenizer is not None:
            try:
                prompt = f"Translate this ASL gloss to natural conversational English: {gloss_input}"
                inputs = self.t5_tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.t5_model.generate(**inputs, max_new_tokens=64)
                translation = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception:
                pass

        elapsed = time.time() - t0

        def _update():
            self._stop_progress()
            self.browse_btn.configure(state="normal")
            self._set_status(f"Done in {elapsed:.1f}s (Apple Vision)")

            for i, (label, conf) in enumerate(predictions[:len(self.pred_rows)]):
                name_lbl, bar_fill, _, conf_lbl = self.pred_rows[i]
                name_lbl.configure(text=label)
                pct = conf / 100.0
                bar_fill.place(x=0, y=0, relheight=1.0, relwidth=pct)
                conf_lbl.configure(text=f"{conf:.1f}%" if conf > 0 else "")

            self.gloss_label.configure(text=f"Gloss: {gloss_input}")

            # Start video playback with glosses
            if hasattr(self, '_playback_source_frames') and self._playback_source_frames:
                clip_gloss_list = []
                if self.use_continuous and stage2_glosses:
                    for g in stage2_glosses.split():
                        clip_gloss_list.append((g, 100.0))
                elif not self.use_continuous and predictions:
                    clip_gloss_list.append((predictions[0][0], predictions[0][1]))
                self._playback_translation = translation
                self._start_playback(self._playback_source_frames, clip_gloss_list)
            else:
                # No playback — show results directly
                title = stage2_glosses if stage2_glosses else predictions[0][0]
                self.skel_title.configure(text=title)
                if translation:
                    self.translation_label.configure(text=translation)
                else:
                    self.translation_label.configure(text="")

        self.root.after(0, _update)

    # ── Video playback with glosses ─────────────────────────
    def _start_playback(self, frames, clip_glosses, fps=30):
        """Play back recorded video in right panel with per-clip glosses appearing."""
        self._playback_frames = frames
        self._playback_glosses = clip_glosses
        self._playback_fps = fps
        self._playback_idx = 0
        self._playback_running = True
        self._shown_glosses = []
        self._all_glosses_shown = False
        # Clear display — glosses will appear one by one in the big label
        self.translation_label.configure(text="")
        self.skel_title.configure(text="")
        self.cam_header.configure(text="PLAYBACK", text_color=COLORS["accent_blue"])
        self._animate_playback()

    def _animate_playback(self):
        if not self._playback_running or not self._playback_frames:
            return
        idx = self._playback_idx
        if idx >= len(self._playback_frames):
            self._playback_running = False
            self.frame_info.configure(text="")
            return

        frame = self._playback_frames[idx]
        display = cv2.flip(frame, 1)  # Mirror for display
        n_play = len(self._playback_frames)

        # Show glosses one by one in big text, then switch to translation
        if self._playback_glosses and not self._all_glosses_shown:
            n_glosses = len(self._playback_glosses)
            # Each gloss appears AFTER its portion of the video (end of each sign)
            # Use 75% of video for signing, last 25% for translation
            gloss_portion = 0.75
            progress = idx / max(n_play - 1, 1)

            if progress < gloss_portion:
                # Gloss appears at the END of each sign's portion
                gloss_idx = min(int(progress / gloss_portion * n_glosses), n_glosses - 1)

                # Only add gloss when we cross into the next sign's territory
                while len(self._shown_glosses) <= gloss_idx:
                    self._shown_glosses.append(self._playback_glosses[len(self._shown_glosses)][0])

                # Show accumulated glosses as big text
                self.translation_label.configure(text=" ".join(self._shown_glosses))
                self.skel_title.configure(text="")
            else:
                # All glosses done — show translation, move glosses below
                self._all_glosses_shown = True
                all_glosses = " ".join(g for g, _ in self._playback_glosses)
                if hasattr(self, '_playback_translation') and self._playback_translation:
                    self.translation_label.configure(text=self._playback_translation)
                    self.skel_title.configure(text=all_glosses)
                else:
                    self.translation_label.configure(text=all_glosses)

        self.frame_info.configure(text=f"Frame {idx+1} / {n_play}")

        # Draw skeleton on darkened video frame in the pose panel
        if self._extracted_result is not None:
            n_ext = self._extracted_result.shape[0]
            seg_idx = idx // 28
            seg_pos = (idx % 28) / 27.0
            ext_seg_start = seg_idx * 32
            ext_frame = int(ext_seg_start + seg_pos * 31)
            ext_frame = max(0, min(n_ext - 1, ext_frame))
            self._draw_skeleton_on_frame(frame, self._extracted_result[ext_frame])

        img = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        vw = max(self.video_label.winfo_width(), 640)
        vh = max(self.video_label.winfo_height(), 480)
        fh, fw = display.shape[:2]
        scale = min(vw / fw, vh / fh)
        dw, dh = int(fw * scale), int(fh * scale)
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=(dw, dh))
        self.video_label.configure(image=photo, text="")
        self.video_label._photo = photo

        self._playback_idx += 1
        delay = int(1000 / self._playback_fps)
        self.root.after(delay, self._animate_playback)


    # ── Skeleton drawing ────────────────────────────────────
    def _draw_skeleton_on_frame(self, frame_bgr, data_frame):
        """Run Apple Vision on the frame and draw raw landmarks as a stick figure
        on a dark canvas — shows the full signer pose like sign.mt."""
        from extract_apple_vision import frame_to_ciimage, detect_all, assign_hand_slots
        from Foundation import NSAutoreleasePool

        W, H = 340, 152
        canvas = np.full((H, W, 3), [15, 15, 26], dtype=np.uint8)
        fh, fw = frame_bgr.shape[:2]

        try:
            pool = NSAutoreleasePool.alloc().init()
            ci = frame_to_ciimage(frame_bgr)
            hands, face_pts, body_pts = detect_all(ci, fw, fh)
            del ci

            assigned = assign_hand_slots(hands, face_pts)

            AMBER  = (0, 158, 245)
            PURPLE = (219, 118, 139)
            ORANGE = (30, 140, 230)
            GRAY   = (100, 110, 120)

            # Maintain video aspect ratio inside the canvas
            vid_aspect = fw / fh
            can_aspect = W / H
            if vid_aspect > can_aspect:
                draw_w = W
                draw_h = int(W / vid_aspect)
            else:
                draw_h = H
                draw_w = int(H * vid_aspect)
            ox = (W - draw_w) // 2
            oy = (H - draw_h) // 2

            def to_px(nx, ny):
                # Coords already in screen space (Y flipped in detect_all)
                # Mirror X to match cv2.flip(frame, 1) display
                px = int((1.0 - nx) * draw_w + ox)
                py = int(ny * draw_h + oy)
                return (max(0, min(W - 1, px)), max(0, min(H - 1, py)))

            # Draw body (estimated from face)
            if body_pts is not None and len(body_pts) == 4:
                pts_body = [to_px(body_pts[i][0], body_pts[i][1]) for i in range(4)]
                # shoulders
                cv2.line(canvas, pts_body[0], pts_body[1], GRAY, 1, cv2.LINE_AA)
                cv2.circle(canvas, pts_body[0], 3, GRAY, -1, cv2.LINE_AA)
                cv2.circle(canvas, pts_body[1], 3, GRAY, -1, cv2.LINE_AA)
                # elbows
                cv2.line(canvas, pts_body[0], pts_body[2], GRAY, 1, cv2.LINE_AA)
                cv2.line(canvas, pts_body[1], pts_body[3], GRAY, 1, cv2.LINE_AA)
                cv2.circle(canvas, pts_body[2], 2, GRAY, -1, cv2.LINE_AA)
                cv2.circle(canvas, pts_body[3], 2, GRAY, -1, cv2.LINE_AA)

            # Draw face
            if face_pts is not None:
                # Nose (index 0), chin (1), forehead (2), ears (3,4)
                for i in range(min(5, len(face_pts))):
                    color = ORANGE
                    r = 4 if i == 0 else 2
                    cv2.circle(canvas, to_px(face_pts[i][0], face_pts[i][1]), r, color, -1, cv2.LINE_AA)

            # Draw hands
            for slot, coords, conf in assigned:
                color = AMBER if slot == 'left' else PURPLE
                hand_px = [to_px(coords[j][0], coords[j][1]) for j in range(21)]
                for a, b in _HAND_EDGES:
                    cv2.line(canvas, hand_px[a], hand_px[b], color, 1, cv2.LINE_AA)
                for j in range(21):
                    cv2.circle(canvas, hand_px[j], 1, color, -1, cv2.LINE_AA)

                # Connect wrist to nearest elbow
                if body_pts is not None and len(body_pts) == 4:
                    elbow_idx = 2 if slot == 'left' else 3
                    cv2.line(canvas, hand_px[0], to_px(body_pts[elbow_idx][0], body_pts[elbow_idx][1]),
                            GRAY, 1, cv2.LINE_AA)

            del pool
        except Exception:
            pass

        self._push_skeleton_canvas(canvas)

    def _push_skeleton_canvas(self, canvas_bgr):
        """Convert a BGR numpy canvas and display it in skeleton_label."""
        img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
        # Query actual widget width; fall back to 324 (pose_box fill minus padx=8 each side)
        w = max(self.skeleton_label.winfo_width(), 324)
        h = max(self.skeleton_label.winfo_height(), 144)
        photo = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.skeleton_label.configure(image=photo)
        self.skeleton_label._skel_photo = photo  # prevent GC

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if not os.path.exists(CKPT_PATH):
        print(f"Checkpoint not found: {CKPT_PATH}")
        print(f"Download: scp -P 56060 root@47.186.29.91:/workspace/output_v12/best_model.pth {CKPT_PATH}")
        sys.exit(1)
    app = SignClassifierApp()
    app.run()
