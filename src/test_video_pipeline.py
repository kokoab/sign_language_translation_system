"""
End-to-End SLT Video Pipeline Test
Video -> Extract (using extract.py functions) -> Stage 2 (CTC) -> Stage 3 (Translation)

Usage:
    python src/test_video_pipeline.py sample_videos/how_you.mp4
    python src/test_video_pipeline.py video1.mp4 video2.mp4
"""

import os, sys, subprocess, tempfile, json, warnings, argparse
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import extraction functions from extract.py (exact same pipeline as training)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import (
    interpolate_hand, interpolate_face, temporal_resample, normalize_sequence,
    compute_kinematics_batch, one_euro_filter, stabilize_bones,
    reject_temporal_outliers, PipelineConfig, NUM_NODES,
    FACE_LANDMARK_INDICES,
)
import mediapipe as mp

# =====================================================================
# PATHS
# =====================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1_CKPT  = os.path.join(PROJECT_ROOT, "models", "output_joint", "best_model.pth")
STAGE2_CKPT  = os.path.join(PROJECT_ROOT, "models", "output", "stage2_best_model.pth")
STAGE3_DIR  = os.path.join(PROJECT_ROOT, "weights", "slt_final_t5_model")
MANIFEST    = os.path.join(PROJECT_ROOT, "models", "manifest.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")
CFG = PipelineConfig()

# =====================================================================
# MODEL ARCHITECTURE (must match training exactly)
# =====================================================================

class DSGCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, A, temporal_kernel=3, dropout=0.1, num_vertices=47):
        super().__init__()
        K = 3
        self.adj_residual = nn.Parameter(torch.zeros(K, num_vertices, num_vertices))
        self.dw_weights = nn.Parameter(torch.ones(K, in_ch))
        self.pointwise = nn.Linear(K * in_ch, out_ch, bias=False)
        self.temporal_conv = nn.Conv1d(out_ch, out_ch, kernel_size=temporal_kernel,
                                        padding=temporal_kernel // 2, groups=out_ch, bias=False)
        self.temporal_norm = nn.GroupNorm(8, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Linear(in_ch, out_ch, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x, A):
        B, T, N, C = x.shape
        residual = self.residual(x)
        A_eff = A + torch.tanh(self.adj_residual) * 0.3
        A_eff = A_eff / A_eff.abs().sum(dim=-1, keepdim=True).clamp(min=1)
        agg = torch.einsum('knm,btnc->kbtnc', A_eff, x)
        agg = agg * self.dw_weights.view(3, 1, 1, 1, C)
        h = agg.permute(1, 2, 3, 0, 4).reshape(B, T, N, 3 * C)
        h = self.drop(self.pointwise(h))
        C_out = h.shape[-1]
        h_t = h.permute(0, 2, 3, 1).reshape(B * N, C_out, T)
        h_t = self.temporal_norm(self.temporal_conv(h_t))
        h = h_t.reshape(B, N, C_out, T).permute(0, 3, 1, 2)
        return self.act(self.norm(h + residual))


# Geometric feature indices (hand landmarks)
_THUMB_MCP=2; _THUMB_IP=3; _THUMB_TIP=4
_INDEX_MCP=5; _INDEX_PIP=6; _INDEX_TIP=8
_MIDDLE_MCP=9; _MIDDLE_PIP=10; _MIDDLE_TIP=12
_RING_MCP=13; _RING_PIP=14; _RING_TIP=16
_PINKY_MCP=17; _PINKY_PIP=18; _PINKY_TIP=20
NOSE_NODE=42; CHIN_NODE=43; FOREHEAD_NODE=44
N_GEO_FEATURES = 76


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0: return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * (torch.rand(shape, device=x.device) < keep) / keep


class DSGCNEncoder(nn.Module):
    def __init__(self, in_channels=16, d_model=384, nhead=8, num_transformer_layers=6,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.register_buffer('A', self._build_adjacency(47))
        self.input_norm = nn.LayerNorm(in_channels)
        ch1 = d_model // 4
        ch2 = d_model // 2
        self.input_proj = nn.Sequential(nn.Linear(in_channels, ch1), nn.LayerNorm(ch1), nn.GELU())
        self.gcn1 = DSGCNLayer(ch1, ch2, None, temporal_kernel=3, dropout=dropout)
        self.gcn2 = DSGCNLayer(ch2, ch2, None, temporal_kernel=3, dropout=dropout)
        self.gcn3 = DSGCNLayer(ch2, d_model, None, temporal_kernel=5, dropout=dropout)
        self.node_attn = nn.Sequential(nn.Linear(d_model, d_model//4), nn.GELU(), nn.Linear(d_model//4, 1))
        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        dp_rates = [drop_path_rate * i / max(num_transformer_layers-1, 1) for i in range(num_transformer_layers)]
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))
        self.transformer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _build_adjacency(V):
        edges_single = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                         (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
        edges = edges_single + [(u+21,v+21) for u,v in edges_single]
        edges += [(42,43),(42,44),(42,45),(42,46),(43,44)]
        edges += [(0,42),(0,43),(0,44),(21,42),(21,43),(21,44),
                  (4,42),(4,44),(4,43),(8,42),(8,44),(8,43),
                  (25,42),(25,44),(25,43),(29,42),(29,44),(29,43)]
        A_self = np.eye(V, dtype=np.float32)
        A_out = np.zeros((V,V), dtype=np.float32)
        A_in = np.zeros((V,V), dtype=np.float32)
        for s,d in edges:
            if s < V and d < V:
                A_out[s,d] = 1.0; A_in[d,s] = 1.0
        def norm(M):
            deg = M.sum(axis=1, keepdims=True).clip(min=1); return M/deg
        return torch.from_numpy(np.stack([norm(A_self), norm(A_out), norm(A_in)]))

    @staticmethod
    def _geo_dist(a, b): return torch.sqrt(((a-b)**2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz, face_mask=None):
        d = self._geo_dist
        def hand_feats(base):
            tips = [d(xyz[:,:,base+_THUMB_TIP],xyz[:,:,base+_INDEX_TIP]),
                    d(xyz[:,:,base+_INDEX_TIP],xyz[:,:,base+_MIDDLE_TIP]),
                    d(xyz[:,:,base+_MIDDLE_TIP],xyz[:,:,base+_RING_TIP]),
                    d(xyz[:,:,base+_RING_TIP],xyz[:,:,base+_PINKY_TIP]),
                    d(xyz[:,:,base+_THUMB_TIP],xyz[:,:,base+_PINKY_TIP])]
            curls = [d(xyz[:,:,base+_THUMB_MCP],xyz[:,:,base+_THUMB_TIP])/(d(xyz[:,:,base+_THUMB_MCP],xyz[:,:,base+_THUMB_IP])+1e-4),
                     d(xyz[:,:,base+_INDEX_MCP],xyz[:,:,base+_INDEX_TIP])/(d(xyz[:,:,base+_INDEX_MCP],xyz[:,:,base+_INDEX_PIP])+1e-4),
                     d(xyz[:,:,base+_MIDDLE_MCP],xyz[:,:,base+_MIDDLE_TIP])/(d(xyz[:,:,base+_MIDDLE_MCP],xyz[:,:,base+_MIDDLE_PIP])+1e-4),
                     d(xyz[:,:,base+_RING_MCP],xyz[:,:,base+_RING_TIP])/(d(xyz[:,:,base+_RING_MCP],xyz[:,:,base+_RING_PIP])+1e-4),
                     d(xyz[:,:,base+_PINKY_MCP],xyz[:,:,base+_PINKY_TIP])/(d(xyz[:,:,base+_PINKY_MCP],xyz[:,:,base+_PINKY_PIP])+1e-4)]
            cross = xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0]
            d_ti = d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_MCP])
            return tips + curls + [cross, d_ti]
        f1 = hand_feats(0); f2 = hand_feats(21)
        face_f = [d(xyz[:,:,0],xyz[:,:,NOSE_NODE]), d(xyz[:,:,0],xyz[:,:,CHIN_NODE]),
                  d(xyz[:,:,0],xyz[:,:,FOREHEAD_NODE]), d(xyz[:,:,21],xyz[:,:,NOSE_NODE]),
                  d(xyz[:,:,21],xyz[:,:,CHIN_NODE]), d(xyz[:,:,21],xyz[:,:,FOREHEAD_NODE]),
                  d(xyz[:,:,_INDEX_TIP],xyz[:,:,NOSE_NODE]), d(xyz[:,:,_INDEX_TIP],xyz[:,:,FOREHEAD_NODE]),
                  d(xyz[:,:,21+_INDEX_TIP],xyz[:,:,NOSE_NODE]), d(xyz[:,:,21+_INDEX_TIP],xyz[:,:,FOREHEAD_NODE])]
        if face_mask is not None:
            fg = face_mask[:,:,0,0]
            face_f = [f*fg for f in face_f]
        _TRIPLETS = [(0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),
                     (0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),
                     (0,17,18),(17,18,19),(18,19,20)]
        angle_feats = []
        for base in [0, 21]:
            for p, j, c in _TRIPLETS:
                v1 = xyz[:,:,base+p,:] - xyz[:,:,base+j,:]
                v2 = xyz[:,:,base+c,:] - xyz[:,:,base+j,:]
                cos_a = (v1*v2).sum(-1) / (v1.norm(dim=-1)*v2.norm(dim=-1) + 1e-6)
                angle_feats.append(torch.acos(cos_a.clamp(-1+1e-6, 1-1e-6)))
        palm_feats = []
        for base in [0, 21]:
            wrist = xyz[:,:,base,:]
            v1 = xyz[:,:,base+5,:] - wrist
            v2 = xyz[:,:,base+17,:] - wrist
            normal = torch.cross(v1, v2, dim=-1)
            normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
            for i in range(3): palm_feats.append(normal[...,i])
        _SPREAD = [5, 9, 13, 17]
        spread_feats = []
        for base in [0, 21]:
            wrist = xyz[:,:,base,:]
            for i in range(len(_SPREAD)-1):
                v1 = xyz[:,:,base+_SPREAD[i],:] - wrist
                v2 = xyz[:,:,base+_SPREAD[i+1],:] - wrist
                cos_s = (v1*v2).sum(-1) / (v1.norm(dim=-1)*v2.norm(dim=-1) + 1e-6)
                spread_feats.append(torch.acos(cos_s.clamp(-1+1e-6, 1-1e-6)))
        return torch.stack(f1 + f2 + face_f + angle_feats + palm_feats + spread_feats, dim=-1)

    def forward(self, x):
        xyz = x[:,:,:,:3]
        face_mask = x[:,:,42:47,9:10]
        h = self.input_proj(self.input_norm(x))
        h[:,:,42:47,:] = h[:,:,42:47,:] * face_mask
        h = self.gcn3(self.gcn2(self.gcn1(h, self.A), self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)
        geo = self._compute_geo_features(xyz, face_mask)
        h = self.geo_proj(torch.cat([h, self.geo_norm(geo)], dim=-1)) + self.pos_enc
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            h = h + dp(layer(h) - h)
        return self.transformer_norm(h)


class MultiScaleTCN(nn.Module):
    def __init__(self, d_model, out_tokens=4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=k//2, groups=d_model),
                nn.GroupNorm(8, d_model), nn.GELU()
            ) for k in [3, 5, 9]
        ])
        self.fuse = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)

    def forward(self, x):  # x: [clips, d_model, 32]
        outs = [b(x) for b in self.branches]
        merged = torch.cat(outs, dim=1).permute(0, 2, 1)  # [clips, 32, 3*d]
        fused = self.fuse(merged).permute(0, 2, 1)         # [clips, d, 32]
        return self.pool(fused)                              # [clips, d, 4]


class SequenceTransformer(nn.Module):
    def __init__(self, d_model=384, nhead=8, num_layers=4, dropout=0.3, max_len=512):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        x = x + self.pos_enc[:, :x.size(1)]
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


class SLTStage2CTC(nn.Module):
    def __init__(self, vocab_size=311, d_model=384, seq_layers=4, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.encoder = DSGCNEncoder(in_channels=16, d_model=d_model, num_transformer_layers=4)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.temporal_pool = MultiScaleTCN(d_model, out_tokens=4)
        self.seq_transformer = SequenceTransformer(
            d_model=d_model, nhead=8, num_layers=seq_layers, dropout=dropout)
        self.classifier = nn.Linear(d_model, vocab_size)
        self.inter_ctc_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, x_lens):
        B = x.size(0)
        V, C = x.shape[2], x.shape[3]
        all_clips = []
        clip_counts = []
        for b in range(B):
            valid = x[b, :x_lens[b]]
            nc = valid.size(0) // 32
            rem = valid.size(0) % 32
            if rem > 0:
                pad = torch.zeros(32 - rem, V, C, device=valid.device)
                valid = torch.cat([valid, pad], dim=0)
                nc += 1
            clips = valid.view(nc, 32, V, C)
            all_clips.append(clips)
            clip_counts.append(nc)
        all_clips_cat = torch.cat(all_clips, dim=0)
        with torch.no_grad():
            enc_out = self.encoder(all_clips_cat)       # [total_clips, 32, d_model]
        enc_out = enc_out.permute(0, 2, 1)              # [total_clips, d_model, 32]
        pooled = self.temporal_pool(enc_out)              # [total_clips, d_model, 4]
        pooled = pooled.permute(0, 2, 1)                 # [total_clips, 4, d_model]
        out_seqs = []
        out_lens = []
        idx = 0
        for nc in clip_counts:
            seq = pooled[idx:idx+nc].reshape(nc * 4, -1)  # [nc*4, d_model]
            out_seqs.append(seq)
            out_lens.append(nc * 4)
            idx += nc
        padded = pad_sequence(out_seqs, batch_first=True)
        max_len = padded.size(1)
        mask = torch.arange(max_len, device=padded.device).unsqueeze(0) >= torch.tensor(out_lens, device=padded.device).unsqueeze(1)
        transformer_out = self.seq_transformer(padded, padding_mask=mask)
        logits = self.classifier(transformer_out)
        return logits, torch.tensor(out_lens, device=x.device)


# =====================================================================
# VIDEO EXTRACTION — uses extract.py's exact pipeline
# =====================================================================

_BONE_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

def compute_bone_features(xyz):
    """Compute bone direction + bone motion from XYZ. Returns [T,47,6]."""
    bone = np.zeros_like(xyz)
    for p, c in _BONE_PAIRS:
        bone[..., c, :] = xyz[..., c, :] - xyz[..., p, :]
        bone[..., c+21, :] = xyz[..., c+21, :] - xyz[..., p+21, :]
    for fn in [43, 44, 45, 46]:
        bone[..., fn, :] = xyz[..., fn, :] - xyz[..., 42, :]
    bone_motion = np.zeros_like(bone)
    T = xyz.shape[0]
    if T > 2:
        bone_motion[1:-1] = (bone[2:] - bone[:-2]) / 2.0
        bone_motion[0] = bone_motion[1]
        bone_motion[-1] = bone_motion[-2]
    return np.concatenate([bone, bone_motion], axis=-1)


def reencode_to_cfr(video_path, fps=30):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-r", str(fps),
                        "-vsync", "cfr", "-an", "-c:v", "libx264",
                        "-preset", "ultrafast", "-crf", "18", tmp.name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return tmp.name
    except (subprocess.CalledProcessError, FileNotFoundError):
        os.unlink(tmp.name)
        return None


def extract_from_video(video_path):
    """Extract landmarks using the EXACT same pipeline as extract.py (CPU/MediaPipe path).
    Returns [32, 47, 16] numpy array (float32) matching training data format."""
    print(f"\n[EXTRACT] Processing: {os.path.basename(video_path)}")

    cfr_path = reencode_to_cfr(video_path)
    read_path = cfr_path or video_path

    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        if cfr_path: os.unlink(cfr_path)
        raise ValueError(f"Cannot open: {video_path}")

    # Read all frames
    frames_rgb = []
    frame_indices = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        frame_indices.append(idx)
        idx += 1
    cap.release()
    if cfr_path: os.unlink(cfr_path)

    total_frames = len(frames_rgb)
    if total_frames < 8:
        raise ValueError(f"Too few frames: {total_frames}")
    print(f"   Total frames: {total_frames}")

    # Subsample if too many frames (same logic as extract.py)
    max_process = CFG.target_frames * 3  # ~96 frames max
    if total_frames > max_process:
        step = total_frames / max_process
        selected = [int(i * step) for i in range(max_process)]
        frames_rgb = [frames_rgb[i] for i in selected]
        frame_indices = [frame_indices[i] for i in selected]

    processed_count = len(frames_rgb)

    # Run MediaPipe detection (video mode then static mode, like extract.py CPU path)
    l_seq, r_seq, l_valid, r_valid = [], [], [], []
    face_seq, face_valid = [], []

    # Pass 1: Video mode (tracking)
    hands_v = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=CFG.min_detection_conf,
        min_tracking_confidence=CFG.min_tracking_conf,
        model_complexity=CFG.model_complexity
    )
    face_v = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        min_detection_confidence=CFG.min_detection_conf,
        min_tracking_confidence=CFG.min_tracking_conf,
        refine_landmarks=False,
    )

    for i, rgb in enumerate(frames_rgb):
        fi = frame_indices[i]
        # Hands
        res = hands_v.process(rgb)
        if res.multi_hand_landmarks:
            for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                if score >= CFG.min_detection_conf:
                    coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                    if label == "Left" and (not l_valid or l_valid[-1] != fi):
                        l_seq.append(coords); l_valid.append(fi)
                    elif label == "Right" and (not r_valid or r_valid[-1] != fi):
                        r_seq.append(coords); r_valid.append(fi)
        # Face
        face_res = face_v.process(rgb)
        if face_res.multi_face_landmarks:
            fl = face_res.multi_face_landmarks[0]
            face_pts = [[fl.landmark[idx].x, fl.landmark[idx].y, fl.landmark[idx].z]
                        for idx in FACE_LANDMARK_INDICES]
            face_seq.append(face_pts)
            face_valid.append(fi)

    hands_v.close(); face_v.close()

    # Pass 2: Static mode (if coverage < 80%)
    dom_coverage = max(len(l_valid), len(r_valid)) / max(processed_count, 1)
    if dom_coverage < 0.80:
        print(f"   Video pass coverage: {dom_coverage:.0%}, running static pass...")
        s_l_seq, s_r_seq, s_l_valid, s_r_valid = [], [], [], []
        s_face_seq, s_face_valid = [], []

        hands_s = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=CFG.min_detection_conf,
            min_tracking_confidence=CFG.min_tracking_conf,
            model_complexity=CFG.model_complexity
        )
        face_s = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            min_detection_confidence=CFG.min_detection_conf,
            min_tracking_confidence=CFG.min_tracking_conf,
            refine_landmarks=False,
        )

        for i, rgb in enumerate(frames_rgb):
            fi = frame_indices[i]
            res = hands_s.process(rgb)
            if res.multi_hand_landmarks:
                for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness.classification[0].label
                    score = handedness.classification[0].score
                    if score >= CFG.min_detection_conf:
                        coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                        if label == "Left" and (not s_l_valid or s_l_valid[-1] != fi):
                            s_l_seq.append(coords); s_l_valid.append(fi)
                        elif label == "Right" and (not s_r_valid or s_r_valid[-1] != fi):
                            s_r_seq.append(coords); s_r_valid.append(fi)
            face_res = face_s.process(rgb)
            if face_res.multi_face_landmarks:
                fl = face_res.multi_face_landmarks[0]
                face_pts = [[fl.landmark[idx].x, fl.landmark[idx].y, fl.landmark[idx].z]
                            for idx in FACE_LANDMARK_INDICES]
                s_face_seq.append(face_pts)
                s_face_valid.append(fi)

        hands_s.close(); face_s.close()

        # Merge: pick pass with more detections
        if len(s_l_valid) > len(l_valid):
            l_seq, l_valid = s_l_seq, s_l_valid
        if len(s_r_valid) > len(r_valid):
            r_seq, r_valid = s_r_seq, s_r_valid
        if len(s_face_valid) > len(face_valid):
            face_seq, face_valid = s_face_seq, s_face_valid

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever = bool(face_valid)
    print(f"   Detections: L={len(l_valid)} R={len(r_valid)} Face={len(face_valid)}")

    if not l_valid and not r_valid:
        raise ValueError("No hands detected in video")

    # Temporal coherence rejection (from extract.py)
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)

    if not l_valid and not r_valid:
        raise ValueError("All hand detections rejected as outliers")

    # Interpolation (from extract.py)
    total_idx = total_frames
    l_full = interpolate_hand(np.array(l_seq) if l_seq else np.zeros((0, 21, 3)),
                              l_valid, total_idx)
    r_full = interpolate_hand(np.array(r_seq) if r_seq else np.zeros((0, 21, 3)),
                              r_valid, total_idx)
    face_full = interpolate_face(np.array(face_seq) if face_seq else np.zeros((0, 5, 3)),
                                 face_valid, total_idx)

    # Combine [T, 47, 3]
    combined = np.concatenate([l_full, r_full, face_full], axis=1)

    # Temporal resample to 32 frames (from extract.py)
    resampled = temporal_resample(combined, CFG.target_frames)

    # 1-Euro adaptive filter (from extract.py — removes jitter, preserves fast motion)
    smoothed_xyz = one_euro_filter(resampled[:, :, :3])
    resampled[:, :, :3] = smoothed_xyz

    # Bone length stabilization (from extract.py)
    if l_ever:
        resampled = stabilize_bones(resampled, 0, 21)
    if r_ever:
        resampled = stabilize_bones(resampled, 21, 42)

    # Normalize (from extract.py)
    normalized = normalize_sequence(resampled, l_ever, r_ever)

    # Per-frame confidence mask (from extract.py)
    T = CFG.target_frames
    per_frame_mask = np.zeros((1, T, NUM_NODES, 1), dtype=np.float32)
    if l_valid:
        l_coverage = np.interp(
            np.linspace(0, total_idx - 1, T),
            sorted(l_valid), np.ones(len(l_valid))
        )
        for t in range(T):
            per_frame_mask[0, t, 0:21, 0] = l_coverage[t]
    if r_valid:
        r_coverage = np.interp(
            np.linspace(0, total_idx - 1, T),
            sorted(r_valid), np.ones(len(r_valid))
        )
        for t in range(T):
            per_frame_mask[0, t, 21:42, 0] = r_coverage[t]
    if face_valid:
        per_frame_mask[0, :, 42:47, 0] = 1.0

    # Kinematics with Savitzky-Golay (from extract.py)
    features_10ch = compute_kinematics_batch(
        normalized[np.newaxis, ...], l_ever, r_ever, face_ever,
        per_frame_mask=per_frame_mask
    ).squeeze(0)  # [32, 47, 10]

    # Bone features [32, 47, 6]
    bone = compute_bone_features(normalized)

    # Full 16-channel [32, 47, 16]
    features_16ch = np.concatenate([features_10ch, bone], axis=-1).astype(np.float32)

    print(f"   Output shape: {features_16ch.shape} (matches training format)")
    return features_16ch


# =====================================================================
# CTC DECODING
# =====================================================================

def ctc_greedy_decode(log_probs, idx_to_gloss, blank=0):
    preds = log_probs.argmax(axis=-1)
    decoded = []
    prev = blank
    for p in preds:
        if p != blank and p != prev:
            gloss = idx_to_gloss.get(int(p), idx_to_gloss.get(str(p), f"UNK_{p}"))
            decoded.append(gloss)
        prev = p
    return decoded


# =====================================================================
# STAGE 1 MODEL + ISOLATED CLASSIFICATION
# =====================================================================

class ClassifierHead(nn.Module):
    def __init__(self, d_model=256, num_classes=29, dropout=0.4):
        super().__init__()
        self.frame_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout * 0.6), nn.Linear(d_model, num_classes))
    def forward(self, x, labels=None):
        attn = F.softmax(self.frame_attn(x).squeeze(-1), dim=1)
        return self.net((x * attn.unsqueeze(-1)).sum(dim=1))


class SLTStage1(nn.Module):
    def __init__(self, num_classes, in_channels=16, d_model=384, nhead=8, num_transformer_layers=6, dropout=0.1, head_dropout=0.4, drop_path_rate=0.1):
        super().__init__()
        self.encoder = DSGCNEncoder(in_channels=in_channels, d_model=d_model, nhead=nhead, num_transformer_layers=num_transformer_layers, dropout=dropout, drop_path_rate=drop_path_rate)
        self.head = ClassifierHead(d_model=d_model, num_classes=num_classes, dropout=head_dropout)
    def forward(self, x, labels=None):
        return self.head(self.encoder(x))


def classify_isolated(features, s1_model, idx_to_gloss_s1, top_k=5):
    """Run Stage 1 isolated classifier on a single 32-frame clip."""
    x = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)
    s1_model.eval()
    with torch.no_grad():
        logits = s1_model(x)
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, top_k, dim=-1)
    results = []
    for prob, idx in zip(topk.values[0], topk.indices[0]):
        gloss = idx_to_gloss_s1.get(int(idx.item()), idx_to_gloss_s1.get(str(idx.item()), f"UNK_{idx.item()}"))
        results.append((gloss, prob.item()))
    return results


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_pipeline(video_path, s1_model, s1_idx_to_label, s2_model, s3_model, s3_tokenizer, idx_to_gloss):
    # Stage 0: Extract using extract.py's exact pipeline
    features = extract_from_video(video_path)

    n_clips = features.shape[0] // 32
    is_single = (n_clips <= 1)

    # Route: single clip → Stage 1, multi-clip → Stage 2 CTC
    if is_single and s1_model is not None:
        print(f"\n[STAGE 1] Isolated classification (1 clip)...")
        top5 = classify_isolated(features[:32], s1_model, s1_idx_to_label)
        print("   Top-5:")
        for gloss, prob in top5:
            print(f"     {gloss:20s} {prob:.4f}")
        glosses = [top5[0][0]]
        print(f"   Prediction: {glosses[0]} ({top5[0][1]:.4f})")
    else:
        print(f"\n[STAGE 2] CTC Recognition ({n_clips} clips)...")
        x = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)
        lens = torch.tensor([x.shape[1]], dtype=torch.long).to(DEVICE)

        s2_model.eval()
        with torch.no_grad():
            logits, out_lens = s2_model(x, lens)
            log_probs = torch.log_softmax(logits[0], dim=-1).cpu().numpy()
            n_tokens = out_lens[0].item()

        glosses = ctc_greedy_decode(log_probs[:n_tokens], idx_to_gloss)

    gloss_str = " ".join(glosses) if glosses else "(no signs detected)"
    print(f"   Glosses: {gloss_str}")

    # Stage 3: Translation
    print("\n[STAGE 3] Translation...")
    if not glosses:
        translation = "[No signs detected]"
    else:
        prompt = f"Translate this ASL gloss to natural conversational English: {' '.join(glosses)}"
        inputs = s3_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = s3_model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
        translation = s3_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"  Video   : {os.path.basename(video_path)}")
    print(f"  Glosses : {gloss_str}")
    print(f"  English : {translation}")
    print("=" * 60)
    return glosses, translation


def main():
    parser = argparse.ArgumentParser(description="SLT End-to-End Video Pipeline")
    parser.add_argument("videos", nargs="+", help="Path(s) to video file(s)")
    args = parser.parse_args()

    print("=" * 60)
    print("SLT END-TO-END PIPELINE")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load Stage 1 (isolated classifier)
    s1_model = None
    s1_idx_to_label = None
    print(f"\nLoading Stage 1 from {STAGE1_CKPT}...")
    if os.path.exists(STAGE1_CKPT):
        ckpt1 = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)
        s1_idx_to_label = ckpt1["idx_to_label"]
        s1_model = SLTStage1(
            num_classes=ckpt1["num_classes"],
            in_channels=ckpt1.get("in_channels", 16),
            d_model=ckpt1.get("d_model", 384),
            nhead=ckpt1.get("nhead", 8),
            num_transformer_layers=ckpt1.get("num_transformer_layers", 6),
        ).to(DEVICE)
        s1_model.load_state_dict(ckpt1["model_state_dict"], strict=False)
        s1_model.eval()
        print(f"   Loaded ({ckpt1['num_classes']} classes, d_model={ckpt1.get('d_model', 384)})")
    else:
        print(f"   Not found (Stage 1 disabled)")

    # Load Stage 2 (CTC)
    print(f"\nLoading Stage 2 from {STAGE2_CKPT}...")
    if not os.path.exists(STAGE2_CKPT):
        print(f"ERROR: Stage 2 checkpoint not found at {STAGE2_CKPT}")
        sys.exit(1)

    ckpt = torch.load(STAGE2_CKPT, map_location=DEVICE, weights_only=False)
    idx_to_gloss = ckpt["idx_to_gloss"]
    vocab_size = ckpt["vocab_size"]

    s2_model = SLTStage2CTC(vocab_size=vocab_size, d_model=384).to(DEVICE)
    s2_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    s2_model.eval()
    print(f"   Loaded ({vocab_size} classes, d_model=384)")

    # Load Stage 3 (T5)
    print(f"\nLoading Stage 3 from {STAGE3_DIR}...")
    if not os.path.exists(STAGE3_DIR):
        print(f"ERROR: Stage 3 model not found at {STAGE3_DIR}")
        sys.exit(1)

    s3_tokenizer = AutoTokenizer.from_pretrained(STAGE3_DIR)
    s3_model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3_DIR).to(DEVICE)
    s3_model.eval()
    print(f"   Loaded Flan-T5 ({sum(p.numel() for p in s3_model.parameters())/1e6:.1f}M params)")

    # Process each video
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"\nERROR: Video not found: {video_path}")
            continue
        try:
            run_pipeline(video_path, s1_model, s1_idx_to_label, s2_model, s3_model, s3_tokenizer, idx_to_gloss)
        except Exception as e:
            print(f"\nERROR processing {video_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
