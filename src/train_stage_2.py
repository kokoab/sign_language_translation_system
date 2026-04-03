"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT Stage 2 — Continuous Sign Language Recognition (CTC)        ║
║  Frozen DS-GCN Encoder (Stage 1) + BiLSTM + CTC Decoder          ║
║  Input : Variable-length sequences of 32-frame 47-point clips    ║
║  Output: Gloss sequences decoded via CTC                         ║
║  Target: Kaggle T4 (16GB VRAM)                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  v6.0: Updated to 47-node face-aware landmarks                   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
# Use GPU 0 by default; override with CUDA_VISIBLE_DEVICES env var before launching
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import json
import math
import logging
import shutil
from pathlib import Path
from collections import defaultdict
import random
import time as _time

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-S2")
torch.backends.cudnn.benchmark = True
# Enable TF32 Tensor Cores
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — DS-GCN ENCODER (Copied verbatim from Stage 1 v5.1)
# ══════════════════════════════════════════════════════════════════

_EDGES_SINGLE = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
_EDGES_HANDS = _EDGES_SINGLE + [(u+21, v+21) for u, v in _EDGES_SINGLE]

# Face nodes: 42=Nose, 43=Chin, 44=Forehead, 45=L_Ear, 46=R_Ear
NOSE_NODE = 42; CHIN_NODE = 43; FOREHEAD_NODE = 44; L_EAR_NODE = 45; R_EAR_NODE = 46
L_MOUTH_NODE = 47; R_MOUTH_NODE = 48; UPPER_LIP_NODE = 49; LOWER_LIP_NODE = 50
L_SHOULDER_NODE = 57; R_SHOULDER_NODE = 58; L_ELBOW_NODE = 59; R_ELBOW_NODE = 60
_EDGES_FACE = [
    (NOSE_NODE, CHIN_NODE), (NOSE_NODE, FOREHEAD_NODE),
    (NOSE_NODE, L_EAR_NODE), (NOSE_NODE, R_EAR_NODE),
    (CHIN_NODE, FOREHEAD_NODE),
]
_EDGES_HAND_FACE = [
    (0, NOSE_NODE), (0, CHIN_NODE), (0, FOREHEAD_NODE),
    (21, NOSE_NODE), (21, CHIN_NODE), (21, FOREHEAD_NODE),
    (4, NOSE_NODE), (4, FOREHEAD_NODE), (4, CHIN_NODE),
    (8, NOSE_NODE), (8, FOREHEAD_NODE), (8, CHIN_NODE),
    (25, NOSE_NODE), (25, FOREHEAD_NODE), (25, CHIN_NODE),
    (29, NOSE_NODE), (29, FOREHEAD_NODE), (29, CHIN_NODE),
]
_EDGES = _EDGES_HANDS + _EDGES_FACE + _EDGES_HAND_FACE
NUM_NODES = 61

def build_adjacency_matrices(num_nodes: int = NUM_NODES) -> torch.Tensor:
    def _norm(M):
        deg = M.sum(axis=1, keepdims=True).clip(min=1)
        return M / deg
    A_self = np.eye(num_nodes, dtype=np.float32)
    A_out  = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    A_in   = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in _EDGES:
        A_out[src, dst] = 1.0
        A_in[dst, src]  = 1.0
    return torch.from_numpy(np.stack([_norm(A_self), _norm(A_out), _norm(A_in)]))

class DSGCNBlock(nn.Module):
    def __init__(self, C_in, C_out, temporal_kernel=3, dropout=0.1, num_groups=8,
                 num_nodes=NUM_NODES, node_drop_rate=0.05):
        super().__init__()
        self.node_drop_rate = node_drop_rate
        self.dw_weights = nn.Parameter(torch.ones(3, C_in))
        nn.init.uniform_(self.dw_weights, 0.8, 1.2)
        self.pointwise = nn.Linear(3 * C_in, C_out, bias=False)
        self.temporal_conv = nn.Conv1d(C_out, C_out, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=C_out, bias=False)
        self.temporal_norm = nn.GroupNorm(num_groups, C_out)
        self.norm = nn.LayerNorm(C_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Linear(C_in, C_out, bias=False) if C_in != C_out else nn.Identity()
        # Phase 1A: Learnable adjacency residual
        self.adj_residual = nn.Parameter(torch.zeros(3, num_nodes, num_nodes))
        nn.init.normal_(self.adj_residual, std=0.01)

    def forward(self, x, A):
        B, T, N, C = x.shape
        residual = self.residual(x)
        # Phase 1B: Drop-Graph regularization
        if self.training and self.node_drop_rate > 0:
            node_mask = (torch.rand(B, 1, N, 1, device=x.device) > self.node_drop_rate).float()
            for gs, ge in [(0, 21), (21, 42), (42, 47)]:
                if node_mask[:, :, gs:ge, :].sum() < 1:
                    node_mask[:, :, gs, :] = 1.0
            x = x * node_mask
        # Phase 1A: Learnable adjacency residual (baked at eval too)
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

_THUMB_MCP = 2; _THUMB_IP = 3; _THUMB_TIP = 4; _INDEX_MCP = 5; _INDEX_PIP = 6; _INDEX_TIP = 8
_MIDDLE_MCP = 9; _MIDDLE_PIP = 10; _MIDDLE_TIP = 12; _RING_MCP = 13; _RING_PIP = 14; _RING_TIP = 16
_PINKY_MCP = 17; _PINKY_PIP = 18; _PINKY_TIP = 20
# 12 per hand x 2 = 24, + 10 hand-to-face features = 34 (base)
# + 30 joint angles + 6 palm orientation + 6 finger spread = 76 total
N_GEO_FEATURES = 114

# Phase 4A: Joint angle triplets (joint at middle vertex)
_JOINT_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),          # Thumb MCP, IP, tip
    (0,5,6),(5,6,7),(6,7,8),          # Index MCP, PIP, DIP
    (0,9,10),(9,10,11),(10,11,12),    # Middle
    (0,13,14),(13,14,15),(14,15,16),  # Ring
    (0,17,18),(17,18,19),(18,19,20),  # Pinky
]  # 15 triplets per hand -> 30 joint angle features

_SPREAD_MCPS = [5, 9, 13, 17]  # Index, Middle, Ring, Pinky MCP indices

# Phase 1C: Bone feature pairs (parent -> child)
_BONE_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

def compute_bone_features_np(x_np):
    """Numpy version for dataset __getitem__. x: [T, N, 10] -> [T, N, 16] where N=47 or 61.
    Must match train_stage_1.py's version exactly for consistent train/inference."""
    N = x_np.shape[-2]
    xyz = x_np[..., :3]
    bone = np.zeros_like(xyz)
    for p, c in _BONE_PAIRS:
        bone[..., c, :] = xyz[..., c, :] - xyz[..., p, :]
        bone[..., c+21, :] = xyz[..., c+21, :] - xyz[..., p+21, :]
    # Face bones: all face nodes relative to nose (node 42)
    face_end = min(N, 57) if N > 42 else min(N, 47)
    for fn in range(43, face_end):
        bone[..., fn, :] = xyz[..., fn, :] - xyz[..., 42, :]
    # Body bones: elbows relative to shoulders
    if N > 60:
        bone[..., 59, :] = xyz[..., 59, :] - xyz[..., 57, :]  # L elbow - L shoulder
        bone[..., 60, :] = xyz[..., 60, :] - xyz[..., 58, :]  # R elbow - R shoulder
    bone_motion = np.zeros_like(bone)
    T = xyz.shape[0]
    if T > 2:
        bone_motion[1:-1] = (bone[2:] - bone[:-2]) / 2.0
        bone_motion[0] = bone_motion[1]
        bone_motion[-1] = bone_motion[-2]
    return np.concatenate([x_np, bone, bone_motion], axis=-1).astype(np.float32)

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
    def __init__(self, in_channels=16, d_model=256, nhead=8, num_transformer_layers=4, dropout=0.1, drop_path_rate=0.1, causal=False):
        super().__init__()
        self.register_buffer('A', build_adjacency_matrices(NUM_NODES))
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Sequential(nn.Linear(in_channels, 96), nn.LayerNorm(96), nn.GELU())
        self.gcn1 = DSGCNBlock(96,  192, temporal_kernel=3, dropout=dropout)
        self.gcn2 = DSGCNBlock(192, 192, temporal_kernel=3, dropout=dropout)
        self.gcn3 = DSGCNBlock(192, d_model, temporal_kernel=5, dropout=dropout)
        self.node_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        dp_rates = [drop_path_rate * i / max(num_transformer_layers - 1, 1) for i in range(num_transformer_layers)]
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))

        self.transformer_norm = nn.LayerNorm(d_model)
        # Phase 2B: Causal masking option
        self.causal = causal
        if causal:
            self.register_buffer('causal_mask', torch.triu(torch.ones(32, 32), diagonal=1).bool())

    @staticmethod
    def _geo_dist(a, b): return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz, face_mask=None):
        d = self._geo_dist
        def get_hand_features(base):
            tips = [d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_TIP]), d(xyz[:,:,base+_INDEX_TIP], xyz[:,:,base+_MIDDLE_TIP]), d(xyz[:,:,base+_MIDDLE_TIP], xyz[:,:,base+_RING_TIP]), d(xyz[:,:,base+_RING_TIP], xyz[:,:,base+_PINKY_TIP]), d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_PINKY_TIP])]
            curls = [d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_TIP]) / (d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_IP]) + 1e-4), d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_TIP]) / (d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_PIP]) + 1e-4), d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_TIP]) / (d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_PIP]) + 1e-4), d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_TIP]) / (d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_PIP]) + 1e-4), d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_TIP]) / (d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_PIP]) + 1e-4)]
            cross_idx_mid = xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0]
            d_thumb_idxmcp = d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_MCP])
            return tips + curls + [cross_idx_mid, d_thumb_idxmcp]
        face_feats = [
            d(xyz[:,:,0], xyz[:,:,NOSE_NODE]), d(xyz[:,:,0], xyz[:,:,CHIN_NODE]), d(xyz[:,:,0], xyz[:,:,FOREHEAD_NODE]),
            d(xyz[:,:,21], xyz[:,:,NOSE_NODE]), d(xyz[:,:,21], xyz[:,:,CHIN_NODE]), d(xyz[:,:,21], xyz[:,:,FOREHEAD_NODE]),
            d(xyz[:,:,_INDEX_TIP], xyz[:,:,NOSE_NODE]), d(xyz[:,:,_INDEX_TIP], xyz[:,:,FOREHEAD_NODE]),
            d(xyz[:,:,21+_INDEX_TIP], xyz[:,:,NOSE_NODE]), d(xyz[:,:,21+_INDEX_TIP], xyz[:,:,FOREHEAD_NODE]),
        ]
        # Gate face geo features: zero them out when face is not detected
        if face_mask is not None:
            face_gate = face_mask[:, :, 0, 0]  # [B, T] — 1.0 if face detected
            face_feats = [f * face_gate for f in face_feats]

        # Phase 4A: Joint angles — 15 per hand x 2 = 30 features
        angle_feats = []
        for base in [0, 21]:
            for p, j, c in _JOINT_TRIPLETS:
                v1 = xyz[:, :, base+p, :] - xyz[:, :, base+j, :]
                v2 = xyz[:, :, base+c, :] - xyz[:, :, base+j, :]
                cos_a = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
                angle_feats.append(torch.acos(cos_a.clamp(-1 + 1e-6, 1 - 1e-6)))

        # Phase 4A: Palm orientation — unit normal to palm plane, 3 x 2 = 6 features
        palm_feats = []
        for base in [0, 21]:
            wrist = xyz[:, :, base, :]
            v1 = xyz[:, :, base + 5, :] - wrist   # wrist -> index MCP
            v2 = xyz[:, :, base + 17, :] - wrist  # wrist -> pinky MCP
            normal = torch.cross(v1, v2, dim=-1)
            normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
            for i in range(3):
                palm_feats.append(normal[..., i])

        # Phase 4A: Finger spread — angle between adjacent MCPs, 3 x 2 = 6 features
        spread_feats = []
        for base in [0, 21]:
            wrist = xyz[:, :, base, :]
            for i in range(len(_SPREAD_MCPS) - 1):
                v1 = xyz[:, :, base + _SPREAD_MCPS[i], :] - wrist
                v2 = xyz[:, :, base + _SPREAD_MCPS[i+1], :] - wrist
                cos_s = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
                spread_feats.append(torch.acos(cos_s.clamp(-1 + 1e-6, 1 - 1e-6)))

        # Wrist orientation: palm normal dot product with canonical directions
        orient_feats = []
        up = torch.tensor([0.0, 1.0, 0.0], device=xyz.device)
        forward = torch.tensor([0.0, 0.0, 1.0], device=xyz.device)
        for base in [0, 21]:
            wrist = xyz[:, :, base, :]
            v1 = xyz[:, :, base + 5, :] - wrist
            v2 = xyz[:, :, base + 17, :] - wrist
            normal = torch.cross(v1, v2, dim=-1)
            normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
            orient_feats.append((normal * up).sum(-1))
            orient_feats.append((normal * forward).sum(-1))

        # Wrist trajectory: velocity direction of each wrist
        traj_feats = []
        for base in [0, 21]:
            wrist_xyz = xyz[:, :, base, :]
            vel = torch.zeros_like(wrist_xyz)
            vel[:, 1:-1] = (wrist_xyz[:, 2:] - wrist_xyz[:, :-2]) / 2.0
            vel[:, 0] = vel[:, 1]; vel[:, -1] = vel[:, -2]
            speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            vel_dir = vel / speed
            for i in range(3):
                traj_feats.append(vel_dir[..., i])

        # Path curvature
        curve_feats = []
        for base in [0, 21]:
            wrist_xyz = xyz[:, :, base, :]
            vel = torch.zeros_like(wrist_xyz)
            vel[:, 1:-1] = (wrist_xyz[:, 2:] - wrist_xyz[:, :-2]) / 2.0
            vel[:, 0] = vel[:, 1]; vel[:, -1] = vel[:, -2]
            vel_norm = vel / (vel.norm(dim=-1, keepdim=True) + 1e-6)
            dot = (vel_norm[:, :-1] * vel_norm[:, 1:]).sum(-1)
            curvature = torch.acos(dot.clamp(-1 + 1e-6, 1 - 1e-6))
            curvature = F.pad(curvature, (0, 1), value=0.0)
            curve_feats.append(curvature)

        # Inter-hand features
        inter_feats = []
        l_wrist = xyz[:, :, 0, :]; r_wrist = xyz[:, :, 21, :]
        inter_feats.append(d(l_wrist, r_wrist))
        rel = r_wrist - l_wrist
        rel_norm = rel / (rel.norm(dim=-1, keepdim=True) + 1e-6)
        for i in range(2):
            inter_feats.append(rel_norm[..., i])

        # Wrist-to-face approach velocity
        approach_feats = []
        for base in [0, 21]:
            wrist_xyz = xyz[:, :, base, :]
            nose_xyz = xyz[:, :, NOSE_NODE, :]
            dist_to_face = (wrist_xyz - nose_xyz).norm(dim=-1)
            approach_vel = torch.zeros_like(dist_to_face)
            approach_vel[:, 1:-1] = (dist_to_face[:, 2:] - dist_to_face[:, :-2]) / 2.0
            approach_vel[:, 0] = approach_vel[:, 1]; approach_vel[:, -1] = approach_vel[:, -2]
            approach_feats.append(approach_vel)

        # Finger crossing
        cross_feats = []
        for base in [0, 21]:
            cross_feats.append(xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0])
            cross_feats.append(xyz[:,:,base+_MIDDLE_TIP,0] - xyz[:,:,base+_RING_TIP,0])
        # Hand symmetry
        sym_feats = []
        lv = torch.zeros_like(xyz[:,:,0,:]); lv[:,1:-1]=(xyz[:,2:,0,:]-xyz[:,:-2,0,:])/2.0; lv[:,0]=lv[:,1]; lv[:,-1]=lv[:,-2]
        rv = torch.zeros_like(xyz[:,:,21,:]); rv[:,1:-1]=(xyz[:,2:,21,:]-xyz[:,:-2,21,:])/2.0; rv[:,0]=rv[:,1]; rv[:,-1]=rv[:,-2]
        ld = lv/(lv.norm(dim=-1,keepdim=True)+1e-6); rd = rv/(rv.norm(dim=-1,keepdim=True)+1e-6)
        sym_feats.append((ld*rd).sum(-1))
        # Speed per hand
        speed_feats = []
        for base in [0, 21]:
            ww = xyz[:,:,base,:]; v = torch.zeros_like(ww); v[:,1:-1]=(ww[:,2:]-ww[:,:-2])/2.0; v[:,0]=v[:,1]; v[:,-1]=v[:,-2]
            speed_feats.append(v.norm(dim=-1))
        # Acceleration profile per hand
        accel_feats = []
        for base in [0, 21]:
            ww = xyz[:,:,base,:]; v = torch.zeros_like(ww); v[:,1:-1]=(ww[:,2:]-ww[:,:-2])/2.0; v[:,0]=v[:,1]; v[:,-1]=v[:,-2]
            a = torch.zeros_like(v); a[:,1:-1]=(v[:,2:]-v[:,:-2])/2.0; a[:,0]=a[:,1]; a[:,-1]=a[:,-2]
            accel_feats.append(a.norm(dim=-1))
        # Contact detection
        contact_feats = []
        hd = (xyz[:,:,0,:]-xyz[:,:,21,:]).norm(dim=-1)
        contact_feats.append(torch.sigmoid(5.0*(0.05-hd)))

        # Body-relative features (must match train_stage_1.py exactly)
        body_feats = []
        l_shoulder = xyz[:, :, L_SHOULDER_NODE, :]
        r_shoulder = xyz[:, :, R_SHOULDER_NODE, :]
        shoulder_mid = (l_shoulder + r_shoulder) / 2.0
        shoulder_width = d(l_shoulder, r_shoulder)
        for base in [0, 21]:
            wrist = xyz[:, :, base, :]
            hand_height = (wrist[:, :, 1] - shoulder_mid[:, :, 1]) / (shoulder_width + 1e-6)
            body_feats.append(hand_height)
            hand_lateral = (wrist[:, :, 0] - shoulder_mid[:, :, 0]) / (shoulder_width + 1e-6)
            body_feats.append(hand_lateral)
            if base == 0:
                body_feats.append(d(wrist, l_shoulder))
            else:
                body_feats.append(d(wrist, r_shoulder))
            if base == 0:
                body_feats.append(d(wrist, xyz[:, :, L_ELBOW_NODE, :]))
            else:
                body_feats.append(d(wrist, xyz[:, :, R_ELBOW_NODE, :]))
        body_feats.append(shoulder_width)
        mouth_mid = (xyz[:, :, UPPER_LIP_NODE, :] + xyz[:, :, LOWER_LIP_NODE, :]) / 2.0
        for base in [0, 21]:
            body_feats.append(d(xyz[:, :, base, :], mouth_mid))
        # 103 + 11 body = 114 total
        return torch.stack(get_hand_features(0) + get_hand_features(21) + face_feats + angle_feats + palm_feats + spread_feats + orient_feats + traj_feats + curve_feats + inter_feats + approach_feats + cross_feats + sym_feats + speed_feats + accel_feats + contact_feats + body_feats, dim=-1)

    def forward(self, x):
        xyz = x[:, :, :, :3]
        face_mask = x[:, :, 42:47, 9:10]  # [B, T, 5, 1]
        h = self.input_proj(self.input_norm(x))
        # Gate face node features by mask before GCN aggregation
        h[:, :, 42:47, :] = h[:, :, 42:47, :] * face_mask
        h = self.gcn3(self.gcn2(self.gcn1(h, self.A), self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)
        h = self.geo_proj(torch.cat([h, self.geo_norm(self._compute_geo_features(xyz, face_mask))], dim=-1)) + self.pos_enc
        # Phase 2B: Causal masking
        mask = self.causal_mask[:h.size(1), :h.size(1)] if self.causal else None
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            h = h + dp(layer(h, src_mask=mask) - h)
        return self.transformer_norm(h)

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — STAGE 2 CTC ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

class MultiScaleTCN(nn.Module):
    """Phase 2A: Multi-scale temporal convolutions replacing single AdaptiveAvgPool1d."""
    def __init__(self, d_model=256, out_tokens=4):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, k, padding=k//2, groups=d_model),
                nn.GroupNorm(8, d_model), nn.GELU()
            ) for k in [3, 5, 9]
        ])
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model), nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)

    def forward(self, x):  # x: [clips, d_model, 32]
        outs = [b(x) for b in self.branches]
        merged = torch.cat(outs, dim=1).permute(0, 2, 1)  # [clips, 32, 3*d]
        fused = self.fuse(merged).permute(0, 2, 1)         # [clips, d, 32]
        return self.pool(fused)                              # [clips, d, 4]


class SequenceTransformer(nn.Module):
    """Transformer encoder replacing BiLSTM for temporal sequence modeling.
    Pre-LayerNorm for stable training (Xiong et al. 2020)."""
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.3, max_len=512):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
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
    def __init__(self, vocab_size, stage1_ckpt=None, d_model=256, seq_layers=4, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Detect model type from checkpoint
        model_type = None
        if stage1_ckpt and Path(stage1_ckpt).exists():
            ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
            model_type = ckpt.get('model_type', None)

        if model_type and ('v14' in model_type or 'v15' in model_type):
            from model_v14 import DSGCNEncoderV14, compute_angle_features
            self.encoder = DSGCNEncoderV14(in_channels=16, d_model=d_model, num_tcn_blocks=4)
            self._compute_angle_features = compute_angle_features
            self._is_v14 = True
            log.info(f"Using v14/v15 encoder — DS-GCN-TCN angle-primary (d_model={d_model})")
        elif model_type and ('v12' in model_type or 'v13' in model_type):
            from model_v12 import DSGCNEncoderV12
            self.encoder = DSGCNEncoderV12(in_channels=16, d_model=d_model, num_tcn_blocks=4)
            log.info(f"Using v12 encoder — DS-GCN-TCN (d_model={d_model})")
        elif model_type and 'v11' in model_type:
            from model_v11 import DSGCNEncoderV11
            self.encoder = DSGCNEncoderV11(in_channels=16, d_model=d_model, num_transformer_layers=4)
            log.info(f"Using v11 encoder (d_model={d_model})")
        else:
            self.encoder = DSGCNEncoder(in_channels=16, d_model=d_model)

        if stage1_ckpt and Path(stage1_ckpt).exists():
            log.info(f"Loading pre-trained Stage 1 weights from {stage1_ckpt}")
            enc_state = {}
            for k, v in ckpt['model_state_dict'].items():
                k_clean = k.replace('_orig_mod.', '')
                if k_clean.startswith('encoder.'):
                    enc_state[k_clean.replace('encoder.', '')] = v
            self.encoder.load_state_dict(enc_state, strict=False)

        # Initially freeze encoder (will be unfrozen after epoch 30)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Multi-scale temporal convolutions
        self.temporal_pool = MultiScaleTCN(d_model, 4)

        # Transformer sequence encoder (replaces BiLSTM)
        self.seq_transformer = SequenceTransformer(
            d_model=d_model, nhead=8, num_layers=seq_layers, dropout=dropout)

        # CTC head
        self.classifier = nn.Linear(d_model, vocab_size)

        # InterCTC head at intermediate layer (layer 1 of seq_layers)
        self.inter_ctc_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, x_lens, return_inter=False):
        B = x.size(0)
        V, C = x.shape[2], x.shape[3]

        # Batched clip extraction: gather all 32-frame clips across all samples
        all_clips = []
        clip_counts = []
        for b in range(B):
            valid_x = x[b, :x_lens[b]]
            num_clips = valid_x.size(0) // 32
            remainder = valid_x.size(0) % 32
            if remainder > 0:
                pad_frames = torch.zeros(32 - remainder, V, C, device=valid_x.device)
                valid_x = torch.cat([valid_x, pad_frames], dim=0)
                num_clips += 1
            clips = valid_x.view(num_clips, 32, V, C)
            all_clips.append(clips)
            clip_counts.append(num_clips)

        # ONE batched encoder call instead of B separate calls
        all_clips_batch = torch.cat(all_clips, dim=0)  # [total_clips, 32, V, C]
        encoder_frozen = not any(p.requires_grad for p in self.encoder.parameters())

        # v14: compute angle features for the encoder
        enc_kwargs = {}
        if getattr(self, '_is_v14', False):
            angle_feats, angle_vel, _ = self._compute_angle_features(all_clips_batch)
            enc_kwargs = {'angle_features': angle_feats, 'angle_vel': angle_vel}

        if encoder_frozen:
            with torch.no_grad():
                enc_out = self.encoder(all_clips_batch, **enc_kwargs)
        else:
            enc_out = self.encoder(all_clips_batch, **enc_kwargs)

        # ONE batched temporal pool call
        enc_out = enc_out.permute(0, 2, 1)       # [total_clips, d_model, 32]
        pooled = self.temporal_pool(enc_out)       # [total_clips, d_model, 4]
        pooled = pooled.permute(0, 2, 1)           # [total_clips, 4, d_model]

        # Split back per sample
        out_seqs = []
        out_lens = []
        offset = 0
        for nc in clip_counts:
            seq_features = pooled[offset:offset + nc].reshape(nc * 4, -1)
            out_seqs.append(seq_features)
            out_lens.append(nc * 4)
            offset += nc

        padded_seqs = pad_sequence(out_seqs, batch_first=True)
        max_len = padded_seqs.size(1)

        # Create padding mask for Transformer
        padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= torch.tensor(out_lens, device=x.device).unsqueeze(1)

        # InterCTC: capture intermediate representation after layer 1
        inter_logits = None
        if return_inter and len(self.seq_transformer.layers) > 1:
            h = padded_seqs + self.seq_transformer.pos_enc[:, :max_len]
            h = self.seq_transformer.layers[0](h, src_key_padding_mask=padding_mask)
            inter_logits = self.inter_ctc_proj(self.seq_transformer.norm(h))
            # Continue through remaining layers
            for layer in self.seq_transformer.layers[1:]:
                h = layer(h, src_key_padding_mask=padding_mask)
            seq_out = self.seq_transformer.norm(h)
        else:
            seq_out = self.seq_transformer(padded_seqs, padding_mask=padding_mask)

        logits = self.classifier(seq_out)
        out_lens_t = torch.tensor(out_lens, dtype=torch.long, device=x.device)

        if return_inter and inter_logits is not None:
            return logits, out_lens_t, inter_logits
        return logits, out_lens_t

    def unfreeze_encoder(self, lr_scale=0.1):
        """Unfreeze encoder with scaled learning rate."""
        unfrozen = 0
        for param in self.encoder.parameters():
            param.requires_grad = True
            unfrozen += 1
        log.info(f"Unfroze {unfrozen} encoder parameters (use {lr_scale}x LR)")
        return unfrozen

# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — DATASET, UTILS, & AUGMENTATION
# ══════════════════════════════════════════════════════════════════

def temporal_resample(clip, target_frames=32):
    """Resample a clip to target_frames using vectorized linear interpolation."""
    T = clip.shape[0]
    if T == target_frames:
        return clip
    idx = np.linspace(0, T - 1, target_frames)
    lower = np.floor(idx).astype(int)
    upper = np.minimum(lower + 1, T - 1)
    alpha = (idx - lower).astype(np.float32)[:, None, None]
    return (clip[lower] * (1 - alpha) + clip[upper] * alpha).astype(np.float32)


def temporal_speed_warp_np(xyz, min_speed=0.75, max_speed=1.25):
    """Apply temporal speed warping using vectorized interpolation."""
    T = xyz.shape[0]
    speed = np.random.uniform(min_speed, max_speed)
    warp_amount = (speed - 1.0) * 0.5
    t_norm = np.linspace(0, 1, T)
    warped_t_norm = t_norm + warp_amount * t_norm * (1 - t_norm) * 4
    warped_t_norm = np.clip(warped_t_norm, 0, 1)
    warped_t_norm = np.sort(warped_t_norm)
    # Map back to frame indices
    warped_idx = warped_t_norm * (T - 1)
    # Vectorized linear interpolation (sample xyz at warped positions)
    lower = np.floor(warped_idx).astype(int)
    upper = np.minimum(lower + 1, T - 1)
    alpha = (warped_idx - lower).astype(np.float32)[:, None, None]
    return (xyz[lower] * (1 - alpha) + xyz[upper] * alpha).astype(np.float32)


def recompute_kinematics_np(xyz):
    """
    Recompute velocity and acceleration from XYZ (numpy version).
    """
    T = xyz.shape[0]
    vel = np.zeros_like(xyz)
    if T > 2:
        vel[1:-1] = (xyz[2:] - xyz[:-2]) / 2.0
        vel[0] = vel[1]
        vel[-1] = vel[-2]

    acc = np.zeros_like(xyz)
    if T > 2:
        acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
        acc[0] = acc[1]
        acc[-1] = acc[-2]

    return vel, acc


def apply_speed_warp_to_clip(clip, min_speed=0.75, max_speed=1.25):
    """Apply speed warp to a full clip [T, N, C]. Kinematics deferred to end of pipeline."""
    xyz = clip[:, :, :3]
    mask = clip[:, :, 9:10]
    warped_xyz = temporal_speed_warp_np(xyz, min_speed, max_speed)
    # Pack as XYZ + placeholder vel/acc + mask (kinematics recomputed at end)
    placeholder = np.zeros_like(warped_xyz)
    return np.concatenate([warped_xyz, placeholder, placeholder, mask], axis=-1).astype(np.float32)


def minimum_jerk_interpolation(start_xyz, end_xyz, n_frames):
    """Minimum-jerk trajectory (Flash & Hogan 1985): 5th-order polynomial.
    Produces bell-shaped velocity — accelerate, peak, decelerate.
    This is how humans actually move their hands between positions."""
    t = np.linspace(0, 1, n_frames)
    # 5th-order polynomial: 10t^3 - 15t^4 + 6t^5
    s = 10 * t**3 - 15 * t**4 + 6 * t**5
    alphas = s[:, None, None]  # [n_frames, 1, 1]
    return ((1 - alphas) * start_xyz + alphas * end_xyz).astype(np.float32)


def trim_holds(clip, trim_start=2, trim_end=2):
    """Remove static hold frames from clip edges (Hold-Movement-Hold model).
    Isolated signs have static holds at start/end that don't exist in continuous signing."""
    T = clip.shape[0]
    if T <= trim_start + trim_end + 8:
        return clip  # Too short to trim safely
    return clip[trim_start:T - trim_end]


def fast_smooth_sequence(xyz, window=5):
    """Vectorized moving-average smoothing to reduce concatenation artifacts.
    Replaces scipy filtfilt (100ms/sample) with numpy cumsum trick (~1ms/sample).
    Window=5 at 25fps = 0.2s smoothing. Gentler than 5Hz Butterworth — preserves
    fingerspelling frequencies (8-15Hz) while still smoothing stitch boundaries."""
    T, N, C = xyz.shape
    if T < window:
        return xyz.copy()
    pad_size = window // 2
    padded = np.pad(xyz, ((pad_size, pad_size), (0, 0), (0, 0)), mode='edge')
    # Prepend zero for correct cumsum-based moving average
    cs = np.cumsum(padded, axis=0, dtype=np.float64)
    cs = np.concatenate([np.zeros((1, N, C), dtype=np.float64), cs], axis=0)
    smoothed = (cs[window:] - cs[:-window]) / window
    return smoothed[:T].astype(np.float32)


def apply_temporal_drop(clip, drop_rate=0.15):
    """Randomly drop frames (1st place ICCV 2025 MSLR: -1.6% WER improvement)."""
    T = clip.shape[0]
    if T < 8:
        return clip
    n_keep = max(8, int(T * (1 - drop_rate)))
    keep_idx = sorted(random.sample(range(T), n_keep))
    return clip[keep_idx]


def apply_gaussian_jitter(xyz, std=0.01):
    """Add Gaussian noise to landmarks (1st place ICCV 2025 MSLR: simulates tracker noise)."""
    return xyz + np.random.randn(*xyz.shape).astype(np.float32) * std


class SyntheticCTCDataset(Dataset):
    """
    Enhanced synthetic CTC dataset with:
    - Transition frames between signs (CRITICAL for real-world performance)
    - Wider sequence length distribution (1-8 signs instead of 2-6)
    - Segment boundary jitter
    - Speed augmentation per clip
    """

    def __init__(self, data_path, manifest, gloss_to_idx, num_samples=5000,
                 min_len=1, max_len=8, transition_prob=1.0, jitter_frames=3,
                 speed_warp_prob=0.3, confused_glosses=None):
        self.data_path = Path(data_path)
        self.manifest = manifest
        self.gloss_to_idx = gloss_to_idx
        self.transition_prob = transition_prob
        self.jitter_frames = jitter_frames
        self.speed_warp_prob = speed_warp_prob

        self.gloss_files = defaultdict(list)
        for f, gloss in manifest.items():
            if gloss in gloss_to_idx:
                self.gloss_files[gloss].append(f)

        self.vocab_keys = list(self.gloss_files.keys())
        self.samples = []

        # Build sampling weights: upweight glosses that Stage 1 frequently confuses
        gloss_weights = {g: 1.0 for g in self.vocab_keys}
        if confused_glosses:
            for g in confused_glosses:
                if g in gloss_weights:
                    gloss_weights[g] = 3.0  # 3x more likely to appear in sequences
        weight_vals = [gloss_weights[g] for g in self.vocab_keys]
        total_w = sum(weight_vals)
        self._gloss_probs = [w / total_w for w in weight_vals]

        log.info(f"Generating {num_samples} synthetic continuous sequences (len {min_len}-{max_len})...")
        if confused_glosses:
            log.info(f"  Upweighting {len(confused_glosses)} confused glosses for harder sequences")

        # 10% single-sign sequences (important for edge cases)
        single_count = int(num_samples * 0.10)
        for _ in range(single_count):
            gloss = random.choices(self.vocab_keys, weights=self._gloss_probs, k=1)[0]
            self.samples.append(([random.choice(self.gloss_files[gloss])],
                                [gloss_to_idx[gloss]]))

        # 10% longer sequences (7-8 signs)
        long_count = int(num_samples * 0.10)
        for _ in range(long_count):
            seq_len = random.randint(7, max_len)
            seq_glosses = random.choices(self.vocab_keys, weights=self._gloss_probs, k=seq_len)
            seq_files = [random.choice(self.gloss_files[g]) for g in seq_glosses]
            self.samples.append((seq_files, [gloss_to_idx[g] for g in seq_glosses]))

        # 80% standard sequences (2-6 signs)
        remaining = num_samples - single_count - long_count
        for _ in range(remaining):
            seq_len = random.randint(max(2, min_len), min(6, max_len))
            seq_glosses = random.choices(self.vocab_keys, weights=self._gloss_probs, k=seq_len)
            seq_files = [random.choice(self.gloss_files[g]) for g in seq_glosses]
            self.samples.append((seq_files, [gloss_to_idx[g] for g in seq_glosses]))

        # Pre-load ALL unique .npy files into RAM (avoids disk I/O every epoch)
        unique_files = set()
        for files, _ in self.samples:
            unique_files.update(files)
        log.info(f"  Pre-loading {len(unique_files)} unique .npy files into RAM...")
        self._file_cache = {}
        for f in unique_files:
            try:
                arr = np.load(self.data_path / f).astype(np.float32)
                if arr.shape[0] == 32 and arr.shape[2] == 10 and arr.shape[1] in (47, 61):
                    self._file_cache[f] = arr
            except Exception:
                pass
        n_nodes = 61 if any(v.shape[1] == 61 for v in self._file_cache.values()) else 47
        log.info(f"  Cached {len(self._file_cache)} files ({n_nodes} nodes, ~{len(self._file_cache) * 32 * n_nodes * 10 * 4 / 1e9:.1f} GB RAM)")

    def _create_transition_frames(self, prev_clip, next_clip):
        """Create realistic transition frames using minimum-jerk trajectory.
        Duration varies with hand displacement (Fitts' Law). Uses 5th-order
        polynomial for bell-shaped velocity (Flash & Hogan 1985)."""
        end_xyz = prev_clip[-1, :, :3]    # [47, 3]
        start_xyz = next_clip[0, :, :3]   # [47, 3]

        # Variable duration: more frames for larger hand displacement (Fitts' Law)
        hand_dist = np.sqrt(((end_xyz[:42] - start_xyz[:42]) ** 2).sum(axis=-1)).mean()
        base_frames = max(4, min(12, int(hand_dist * 15)))  # Scale distance to frames
        trans_len = base_frames + random.randint(-1, 2)      # Small random variation
        trans_len = max(4, min(14, trans_len))

        # Minimum-jerk interpolation (biomechanically realistic)
        trans_xyz = minimum_jerk_interpolation(end_xyz, start_xyz, trans_len)

        # Compute kinematics from interpolated XYZ
        vel, acc = recompute_kinematics_np(trans_xyz)

        # Inherit mask from surrounding clips
        prev_mask = prev_clip[-1, :, 9:10]
        next_mask = next_clip[0, :, 9:10]
        trans_mask = np.maximum(prev_mask, next_mask)
        trans_mask = np.tile(trans_mask, (trans_len, 1, 1))

        transition = np.concatenate([trans_xyz, vel, acc, trans_mask], axis=-1)
        return transition.astype(np.float32)

    def _jitter_segment_boundary(self, clip):
        """
        Randomly shift where clips start/end to simulate imperfect segmentation.
        """
        if self.jitter_frames <= 0:
            return clip

        start_jitter = random.randint(-self.jitter_frames, self.jitter_frames)
        end_jitter = random.randint(-self.jitter_frames, self.jitter_frames)

        T = clip.shape[0]

        if start_jitter > 0:
            # Trim start
            clip = clip[start_jitter:]
        elif start_jitter < 0:
            # Pad start by repeating first frame
            pad = np.tile(clip[0:1], (-start_jitter, 1, 1))
            clip = np.concatenate([pad, clip], axis=0)

        if end_jitter > 0:
            # Pad end by repeating last frame
            pad = np.tile(clip[-1:], (end_jitter, 1, 1))
            clip = np.concatenate([clip, pad], axis=0)
        elif end_jitter < 0:
            # Trim end
            clip = clip[:end_jitter] if end_jitter < 0 else clip

        # Ensure we have some frames
        if clip.shape[0] < 8:
            return temporal_resample(clip, 32)

        # Resample back to 32 frames
        if clip.shape[0] != 32:
            clip = temporal_resample(clip, 32)

        return clip

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        files, target_glosses = self.samples[idx]
        arrays = []
        valid_targets = []
        num_signs = len(files)

        prev_clip = None
        for i, (f, tgt) in enumerate(zip(files, target_glosses)):
            arr = self._file_cache.get(f)
            if arr is None:
                continue
            arr = arr.copy()  # Don't mutate cached data

            # Quality filters
            xyz_q = arr[:, :42, :3]
            if np.abs(xyz_q).max() < 1e-6 or np.abs(xyz_q - xyz_q[0:1]).max() < 1e-4:
                continue
            if np.abs(xyz_q).max() > 10.0:
                continue

            # 1. Hold trimming: remove static frames from clip edges
            #    (isolated signs have holds that don't exist in continuous signing)
            arr = trim_holds(arr, trim_start=2, trim_end=2)

            # 2. Segment boundary jitter
            arr = self._jitter_segment_boundary(arr)

            # 3. Speed perturbation (0.8-1.2x per sign)
            if random.random() < self.speed_warp_prob:
                arr = apply_speed_warp_to_clip(arr, min_speed=0.8, max_speed=1.2)

            # 4. Prosodic lengthening: last sign in sequence gets 1.3x duration
            #    Kinematics deferred to end of pipeline
            if i == num_signs - 1 and num_signs > 1:
                arr = temporal_resample(arr, int(arr.shape[0] * 1.3))

            # 5. Minimum-jerk transitions between signs (always inject)
            if prev_clip is not None and random.random() < self.transition_prob:
                transition = self._create_transition_frames(prev_clip, arr)
                # Optionally insert a random hold/pause (10% chance, 2-5 frames)
                if random.random() < 0.10:
                    hold_len = random.randint(2, 5)
                    hold_frame = prev_clip[-1:]
                    hold = np.tile(hold_frame, (hold_len, 1, 1))
                    arrays.append(hold)
                arrays.append(transition)

            arrays.append(arr)
            valid_targets.append(tgt)
            prev_clip = arr

        # Fallback if all files are skipped
        if len(arrays) == 0:
            n_nodes = 61 if self._file_cache and next(iter(self._file_cache.values())).shape[1] == 61 else 47
            return np.zeros((32, n_nodes, 16), dtype=np.float32), []

        x = np.concatenate(arrays, axis=0)  # [T_total, 47, 10]

        # From here, work on XYZ + mask only. Kinematics computed once at the end.
        xyz = x[:, :, :3]
        mask = x[:, :, 9:10]

        # 6. Smooth stitched XYZ to reduce concatenation artifacts
        if xyz.shape[0] >= 5:
            xyz = fast_smooth_sequence(xyz, window=5)

        # 7. Temporal drop: randomly drop 15% of frames
        if random.random() < 0.5 and xyz.shape[0] >= 16:
            keep = sorted(random.sample(range(xyz.shape[0]), max(8, int(xyz.shape[0] * 0.85))))
            xyz = xyz[keep]
            mask = mask[keep]

        # 8. Gaussian jitter on XYZ landmarks (simulates tracker noise)
        if random.random() < 0.5:
            xyz = apply_gaussian_jitter(xyz, std=0.01)

        # 9. Single kinematics recomputation + bone features (warp XYZ first, then recompute)
        vel, acc = recompute_kinematics_np(xyz)
        x = np.concatenate([xyz, vel, acc, mask], axis=-1).astype(np.float32)
        x = compute_bone_features_np(x)
        return x, valid_targets

class PregenContinuousCTCDataset(Dataset):
    """Pre-generated continuous sequences extracted with extract_frames_continuous.
    Format matches inference exactly. Each .npy is [N*32, 61, 10].
    Manifest maps filename → gloss string like 'HELLO HOW YOU'."""

    def __init__(self, data_path, manifest, gloss_to_idx):
        self.data_path = Path(data_path)
        self.gloss_to_idx = gloss_to_idx
        self.samples = []

        for fname, gloss_str in manifest.items():
            glosses = gloss_str.split()
            target = [gloss_to_idx[g] for g in glosses if g in gloss_to_idx]
            if target and len(target) == len(glosses):
                self.samples.append((fname, target))

        log.info(f"PregenContinuousCTCDataset: {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, target_glosses = self.samples[idx]
        arr = np.load(self.data_path / fname).astype(np.float32)
        arr = compute_bone_features_np(arr)
        return arr, target_glosses


class RealPhraseCTCDataset(Dataset):
    """Dataset for real continuous signing phrase videos.
    Each .npy file is [N*32, 61, 10] — multiple 32-frame clips concatenated.
    Manifest maps filename → gloss string like 'GOOD MORNING'."""

    def __init__(self, data_path, manifest, gloss_to_idx, augment=True):
        self.data_path = Path(data_path)
        self.gloss_to_idx = gloss_to_idx
        self.augment = augment
        self.samples = []

        for fname, gloss_str in manifest.items():
            glosses = gloss_str.split()
            target = [gloss_to_idx[g] for g in glosses if g in gloss_to_idx]
            if target:
                self.samples.append((fname, target))

        log.info(f"RealPhraseCTCDataset: {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, target_glosses = self.samples[idx]
        arr = np.load(self.data_path / fname).astype(np.float32)  # [N*32, 61, 10]

        # Add bone features
        arr = compute_bone_features_np(arr)  # [N*32, 61, 16]

        return arr, target_glosses


def collate_ctc(batch):
    xs = [torch.from_numpy(b[0]) for b in batch if len(b[1]) > 0]
    ys = [torch.tensor(b[1], dtype=torch.long) for b in batch if len(b[1]) > 0]
    
    if len(xs) == 0: return None, None, None, None
    
    x_lens = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
    
    x_pad = pad_sequence(xs, batch_first=True) # [B, max_T, 47, 16]
    y_flat = torch.cat(ys) # CTC expects flattened targets
    
    return x_pad, y_flat, x_lens, y_lens

def _batch_rotation_matrices(batch_size, max_deg, device):
    angles = torch.randn(batch_size, 3, device=device) * max_deg
    rad = angles * (math.pi / 180.0)
    cx, sx = torch.cos(rad[:, 0]), torch.sin(rad[:, 0])
    cy, sy = torch.cos(rad[:, 1]), torch.sin(rad[:, 1])
    cz, sz = torch.cos(rad[:, 2]), torch.sin(rad[:, 2])
    zero, one = torch.zeros_like(cx), torch.ones_like(cx)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], dim=1).view(-1, 3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], dim=1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], dim=1).view(-1, 3, 3)
    return Rx @ Ry @ Rz

def online_augment(x, rotation_deg=10.0, scale_lo=0.85, scale_hi=1.15, noise_std=0.003):
    """
    Batch-level augmentation: rotation, scale, noise.
    Speed warping is handled per-clip in the dataset (SyntheticCTCDataset),
    so it is NOT duplicated here to avoid double-warping padded tensors.
    """
    B, T, N, C = x.shape
    device = x.device

    R = _batch_rotation_matrices(B, rotation_deg, device)

    spatial_features = x[..., :9]
    mask_features = x[..., 9:]
    xr = spatial_features.view(B, T, N, 3, 3)
    xr = torch.einsum('btngi,bij->btngj', xr, R)
    xr = xr.reshape(B, T, N, 9)

    x_rotated = torch.cat([xr, mask_features], dim=-1)
    # Scale and noise applied to spatial channels only, mask preserved
    scale = scale_lo + torch.rand(B, 1, 1, 1, device=device) * (scale_hi - scale_lo)
    spatial = x_rotated[..., :9] * scale + torch.randn(B, T, N, 9, device=device) * noise_std
    return torch.cat([spatial, x_rotated[..., 9:]], dim=-1)

def calculate_wer(reference, hypothesis):
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1), dtype=np.uint8)
    for i in range(len(reference) + 1): d[i][0] = i
    for j in range(len(hypothesis) + 1): d[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    return d[len(reference)][len(hypothesis)] / max(len(reference), 1)

def focal_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, gamma=2.0):
    """Focal CTC: downweight easy (blank-dominated) frames, focus on hard frames. (Feng 2019)"""
    ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True, reduction='none')
    per_sample_loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    # Normalize by target length for focal weighting
    norm_loss = per_sample_loss / target_lengths.float().clamp(min=1)
    pt = torch.exp(-norm_loss)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * per_sample_loss).mean()

def cr_ctc_loss(model, x_pad, x_lens, y_flat, y_lens, ctc_loss_fn, use_amp, augment_fn):
    """CR-CTC: Consistency Regularization on CTC. (Yao et al. ICLR 2025)
    Feed two differently-augmented views, enforce KL-divergence consistency."""
    # View 1: already augmented input
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits1, out_lens1 = model(x_pad, x_lens)
        log_probs1 = F.log_softmax(logits1, dim=-1)

    # View 2: re-augment the same input
    x_aug2 = augment_fn(x_pad)
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits2, _ = model(x_aug2, x_lens)
        log_probs2 = F.log_softmax(logits2, dim=-1)

    # CTC loss on view 1
    ctc = ctc_loss_fn(log_probs1.transpose(0, 1), y_flat, out_lens1, y_lens)

    # KL consistency between the two views (bidirectional)
    probs1 = log_probs1.detach().exp()
    kl_fwd = F.kl_div(log_probs2, probs1, reduction='batchmean', log_target=False)
    probs2 = log_probs2.detach().exp()
    kl_bwd = F.kl_div(log_probs1, probs2, reduction='batchmean', log_target=False)
    kl_loss = (kl_fwd + kl_bwd) / 2

    return ctc, kl_loss, log_probs1, out_lens1

def decode_ctc(log_probs, out_lens, blank=0):
    preds = log_probs.argmax(dim=-1).cpu().numpy()
    decoded_batch = []
    for b in range(preds.shape[0]):
        seq = preds[b, :out_lens[b]]
        decoded = []
        last_tok = blank
        for tok in seq:
            if tok != blank and tok != last_tok:
                decoded.append(tok)
            last_tok = tok
        decoded_batch.append(decoded)
    return decoded_batch

# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — TRAINING COMPONENTS (SCHEDULER, EMA, CKPT)
# ══════════════════════════════════════════════════════════════════

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.01, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.min_lr_ratio  = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs: scale = (e + 1) / self.warmup_epochs
        else:
            progress = ((e - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]

class ModelEMA:
    def __init__(self, model, decay=0.999):
        # Only track trainable parameters
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.decay = decay
        self.backup = {}

    def to(self, device):
        self.shadow = {n: p.to(device) for n, p in self.shadow.items()}
        return self

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])

def make_checkpoint(model, optimizer, scheduler, ema, epoch, val_wer, best_wer, trigger_times, gloss_to_idx, idx_to_gloss, vocab_size, d_model=384):
    unwrapped = model.module if hasattr(model, 'module') else model
    return {
        'model_state_dict':       unwrapped.state_dict(),
        'optimizer_state_dict':   optimizer.state_dict(),
        'scheduler_state_dict':   scheduler.state_dict(),
        'ema_shadow':             ema.shadow if ema else None,
        'epoch': epoch, 'best_wer': best_wer, 'trigger_times': trigger_times,
        'gloss_to_idx': gloss_to_idx, 'idx_to_gloss': idx_to_gloss, 'vocab_size': vocab_size,
        'd_model': d_model, 'val_wer': val_wer, 'stage': 2,
    }

# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def _auto_data_path():
    for p in ['/workspace/ASL_landmarks_rtmlib', '/workspace/ASL_landmarks_float16',
              '/kaggle/input/datasets/kokoab/batch-1/ASL_landmarks_float16',
              'ASL_landmarks_rtmlib', 'ASL_landmarks_float16']:
        if os.path.isdir(p): return p
    return 'ASL_landmarks_rtmlib'

def _auto_save_dir():
    for p in ['/workspace/output_stage2', '/kaggle/working']:
        if os.path.isdir(p): return p
    return './output_stage2'

def _auto_stage1_ckpt(save_dir):
    """Find Stage 1 checkpoint. Checks common locations (NOT save_dir, which is Stage 2's dir)."""
    for p in ['/workspace/output/best_model.pth',
              '/workspace/output_joint/best_model.pth',
              'models/output_rtmlib_joint/best_model.pth',
              'models/output_joint/best_model.pth',
              '/kaggle/input/datasets/kokoab/model-dataset/best_model.pth']:
        if Path(p).exists(): return str(p)
    return 'models/output_joint/best_model.pth'

def train_stage2(
    data_path = None,
    stage1_ckpt = None,
    save_dir = None,
    smoke_test = False,
    epochs = 60, batch_size = 32, lr = 5e-4, warmup_epochs = 5, patience = 35,
    unfreeze_epoch = 30, encoder_lr_scale = 0.1,
    use_focal_ctc = True, focal_gamma = 2.0,
    use_cr_ctc = True, cr_ctc_weight = 0.3,
    use_inter_ctc = True, inter_ctc_weight = 0.1,
    phrase_data = None,
    continuous_data = None,
):
    if data_path is None: data_path = _auto_data_path()
    if save_dir is None:  save_dir = _auto_save_dir()
    if stage1_ckpt is None: stage1_ckpt = _auto_stage1_ckpt(save_dir)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_CKPT, LAST_CKPT = save_dir / 'stage2_best_model.pth', save_dir / 'stage2_last_checkpoint.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    log.info(f"Device: {device} | Data: {data_path} | Stage1 ckpt: {stage1_ckpt} | Save: {save_dir}")

    if smoke_test:
        log.warning("🚬 SMOKE TEST MODE ACTIVATED! Running on subset for 3 epochs.")
        epochs, patience = 3, 3

    manifest_path = None
    for mp in [Path(data_path) / 'manifest.json', Path('manifest.json'), Path(data_path).parent / 'manifest.json']:
        if mp.exists():
            manifest_path = mp
            break
    if manifest_path is None:
        raise FileNotFoundError(f"CRITICAL: manifest.json not found in {data_path}, CWD, or parent. Upload it first!")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    unique_labels = sorted(list(set(manifest.values())))
    unique_labels = ['<BLANK>'] + unique_labels # CTC Blank token is Index 0
    gloss_to_idx = {gloss: i for i, gloss in enumerate(unique_labels)}
    idx_to_gloss = {i: gloss for gloss, i in gloss_to_idx.items()}
    vocab_size = len(gloss_to_idx)
    
    log.info(f"CTC Vocab Size: {vocab_size} (including BLANK at index 0)")

    # Load confused glosses from Stage 1 confusion matrix if available
    confused_glosses = None
    confusion_path = Path(data_path).parent / 'confused_glosses.json'
    if confusion_path.exists():
        with open(confusion_path, 'r') as f:
            confused_glosses = json.load(f)
        log.info(f"Loaded {len(confused_glosses)} confused glosses for weighted sampling")

    # Mix old synthetic (clean CTC alignment) + new continuous (inference format)
    old_synth = SyntheticCTCDataset(data_path, manifest, gloss_to_idx,
                                     num_samples=50 if smoke_test else 10000,
                                     confused_glosses=confused_glosses)

    if continuous_data and os.path.isdir(continuous_data):
        cont_manifest_path = Path(continuous_data) / 'manifest.json'
        with open(cont_manifest_path) as f:
            cont_manifest = json.load(f)
        cont_items = list(cont_manifest.items())
        random.shuffle(cont_items)
        n = len(cont_items)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        train_cont = dict(cont_items[:n_train])
        val_cont = dict(cont_items[n_train:n_train+n_val])
        test_cont = dict(cont_items[n_train+n_val:])
        new_cont = PregenContinuousCTCDataset(continuous_data, train_cont, gloss_to_idx)
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([old_synth, new_cont])
        log.info(f"Mixed training: {len(old_synth)} old synthetic + {len(new_cont)} continuous = {len(train_ds)}")
    else:
        train_ds = old_synth
        val_cont, test_cont = None, None

    # Mix in real phrase data if provided
    if phrase_data and os.path.isdir(phrase_data):
        phrase_manifest_path = Path(phrase_data) / 'manifest.json'
        if phrase_manifest_path.exists():
            with open(phrase_manifest_path) as f:
                phrase_manifest = json.load(f)
            # Split phrases: 70% train, 15% val, 15% test
            phrase_items = list(phrase_manifest.items())
            random.shuffle(phrase_items)
            n = len(phrase_items)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)
            train_phrases = dict(phrase_items[:n_train])
            val_phrases = dict(phrase_items[n_train:n_train+n_val])
            test_phrases = dict(phrase_items[n_train+n_val:])

            phrase_train_ds = RealPhraseCTCDataset(phrase_data, train_phrases, gloss_to_idx)
            phrase_val_ds = RealPhraseCTCDataset(phrase_data, val_phrases, gloss_to_idx, augment=False)
            phrase_test_ds = RealPhraseCTCDataset(phrase_data, test_phrases, gloss_to_idx, augment=False)

            from torch.utils.data import ConcatDataset
            train_ds = ConcatDataset([train_ds] + [phrase_train_ds] * 3)
            log.info(f"Mixed training: {len(train_ds)} total (continuous + 3x real phrases)")
            log.info(f"Real phrases: {len(train_phrases)} train, {len(val_phrases)} val, {len(test_phrases)} test")
        else:
            log.warning(f"No manifest.json in {phrase_data}")
            phrase_val_ds, phrase_test_ds = None, None
    else:
        phrase_val_ds, phrase_test_ds = None, None

    # Val/test: use continuous splits if available, else old synthetic
    if continuous_data and val_cont:
        val_ds_synth = PregenContinuousCTCDataset(continuous_data, val_cont, gloss_to_idx)
        test_ds_synth = PregenContinuousCTCDataset(continuous_data, test_cont, gloss_to_idx)
    else:
        val_ds_synth = SyntheticCTCDataset(data_path, manifest, gloss_to_idx, num_samples=20 if smoke_test else 2000)
        test_ds_synth = SyntheticCTCDataset(data_path, manifest, gloss_to_idx, num_samples=20 if smoke_test else 2000)

    # Combine synthetic + real for val/test
    if phrase_val_ds is not None:
        from torch.utils.data import ConcatDataset
        val_ds = ConcatDataset([val_ds_synth, phrase_val_ds])
        test_ds = ConcatDataset([test_ds_synth, phrase_test_ds])
    else:
        val_ds = val_ds_synth
        test_ds = test_ds_synth

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_ctc, num_workers=8, pin_memory=use_amp, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ctc, num_workers=4, pin_memory=use_amp, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_ctc, num_workers=4, pin_memory=use_amp, persistent_workers=True)

    # Auto-detect d_model from Stage 1 checkpoint
    s1_d_model = 384  # fallback default (matches trained Stage 1)
    if stage1_ckpt and Path(stage1_ckpt).exists():
        try:
            s1_ckpt = torch.load(stage1_ckpt, map_location='cpu', weights_only=False)
            s1_d_model = s1_ckpt.get('d_model', 384)
            log.info(f"Stage 1 checkpoint d_model={s1_d_model}")
        except Exception:
            pass

    model = SLTStage2CTC(vocab_size=vocab_size, stage1_ckpt=stage1_ckpt, d_model=s1_d_model)

    # Only pass trainable params to optimizer initially (encoder frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4,
                            fused=False)
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model)

    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    encoder_unfrozen = False

    start_epoch, best_wer, trigger_times = 1, float('inf'), 0
    
    # S2-9 Fix: Checkpoint Resume Logic
    if LAST_CKPT.exists():
        try:
            ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'ema_shadow' in ckpt and ckpt['ema_shadow'] is not None:
                ema.shadow = ckpt['ema_shadow']
            start_epoch, best_wer, trigger_times = ckpt['epoch'] + 1, ckpt.get('best_wer', float('inf')), ckpt.get('trigger_times', 0)
            log.info(f"Resumed from epoch {ckpt['epoch']} | Best WER so far: {best_wer:.2f}%")
        except Exception as e: 
            log.warning(f"Resume failed ({e}). Starting fresh.")

    model.to(device)
    ema.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor): state[k] = v.to(device)

    history = []
    converged_epoch = None
    log.info("Starting Stage 2 CTC Training...")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = _time.time()

        # Encoder unfreezing schedule (Section 2.6 of plan)
        if not encoder_unfrozen and epoch >= unfreeze_epoch:
            model.unfreeze_encoder(lr_scale=encoder_lr_scale)
            # Add encoder params to optimizer with scaled LR
            enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
            optimizer.add_param_group({'params': enc_params, 'lr': lr * encoder_lr_scale})
            # Rebuild scheduler to match new number of param groups
            scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=0, max_epochs=epochs - epoch)
            encoder_unfrozen = True
            ema = ModelEMA(model)  # Rebuild EMA with new trainable params
            ema.to(device)

        model.train()
        if not encoder_unfrozen:
            model.encoder.eval()  # Keep frozen encoder in eval mode
        epoch_loss = 0.0

        for i, (x_pad, y_flat, x_lens, y_lens) in enumerate(train_loader):
            if x_pad is None: continue

            x_pad = x_pad.to(device, non_blocking=True)
            y_flat = y_flat.to(device, non_blocking=True)
            x_lens = x_lens.to(device, non_blocking=True)
            y_lens = y_lens.to(device, non_blocking=True)

            x_pad = online_augment(x_pad)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Forward with InterCTC if enabled
                if use_inter_ctc:
                    result = model(x_pad, x_lens, return_inter=True)
                    if len(result) == 3:
                        logits, out_lens, inter_logits = result
                    else:
                        logits, out_lens = result
                        inter_logits = None
                else:
                    logits, out_lens = model(x_pad, x_lens)
                    inter_logits = None

                assert (out_lens >= y_lens).all(), \
                    f"CTC violation: out_lens={out_lens.tolist()} vs y_lens={y_lens.tolist()}"

                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

                # Primary CTC loss (Focal or standard)
                if use_focal_ctc:
                    loss = focal_ctc_loss(log_probs, y_flat, out_lens, y_lens,
                                          blank=0, gamma=focal_gamma)
                else:
                    loss = ctc_loss_fn(log_probs, y_flat, out_lens, y_lens)

                # InterCTC auxiliary loss (Section 2.3)
                if use_inter_ctc and inter_logits is not None:
                    inter_log_probs = F.log_softmax(inter_logits, dim=-1).transpose(0, 1)
                    inter_loss = ctc_loss_fn(inter_log_probs, y_flat, out_lens, y_lens)
                    loss = loss + inter_ctc_weight * inter_loss

            scaler.scale(loss).backward()

            # CR-CTC: consistency regularization (Section 2.2)
            if use_cr_ctc and epoch > warmup_epochs:
                x_aug2 = online_augment(x_pad)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits2, _ = model(x_aug2, x_lens)
                    log_probs2 = F.log_softmax(logits2, dim=-1)
                    log_probs1_detached = F.log_softmax(logits.detach(), dim=-1)
                    kl_loss = F.kl_div(log_probs2, log_probs1_detached.exp(),
                                       reduction='batchmean', log_target=False)
                scaler.scale(kl_loss * cr_ctc_weight).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            epoch_loss += loss.item()

            if epoch == 1 and i == 0 and use_amp:
                log.info(f"GPU Mem Usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        epoch_time = _time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)
        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s"
            
        # Validation Loop
        ema.apply(model)
        model.eval()
        total_wer = 0
        total_samples = 0
        seq_correct = 0
        val_loss_total = 0
        val_loss_count = 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for x_pad, y_flat, x_lens, y_lens in val_loader:
                if x_pad is None: continue
                x_pad = x_pad.to(device, non_blocking=True)
                x_lens = x_lens.to(device, non_blocking=True)
                y_lens = y_lens.to(device, non_blocking=True)
                y_flat = y_flat.to(device, non_blocking=True)
                logits, out_lens = model(x_pad, x_lens)
                log_probs = F.log_softmax(logits, dim=-1)

                # CTC loss for val
                ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)(
                    log_probs.transpose(0, 1), y_flat.cpu(), out_lens.cpu(), y_lens.cpu())
                val_loss_total += ctc_loss.item()
                val_loss_count += 1

                decoded_preds = decode_ctc(log_probs, out_lens, blank=0)

                targets = []
                idx_t = 0
                for length in y_lens:
                    targets.append(y_flat[idx_t:idx_t+length].cpu().tolist())
                    idx_t += length

                for ref, hyp in zip(targets, decoded_preds):
                    wer = calculate_wer(ref, hyp)
                    total_wer += wer
                    total_samples += 1
                    if ref == hyp:
                        seq_correct += 1

        ema.restore(model)

        train_loss = epoch_loss / len(train_loader)
        val_wer = (total_wer / max(total_samples, 1)) * 100
        val_loss = val_loss_total / max(val_loss_count, 1)
        seq_acc = (seq_correct / max(total_samples, 1)) * 100

        frozen_str = "frozen" if not encoder_unfrozen else "unfrozen"
        log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | LR: {cur_lr:.2e} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | WER: {val_wer:.2f}% | SeqAcc: {seq_acc:.1f}% | Enc: {frozen_str}")
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_wer": round(val_wer, 2),
            "seq_accuracy": round(seq_acc, 2),
            "lr": round(cur_lr, 8),
        })

        # Checkpoint (WER: lower is better!)
        should_save_last = (epoch % 5 == 0 or epoch == epochs)
        if val_wer < best_wer or should_save_last:
            ckpt = make_checkpoint(model, optimizer, scheduler, ema, epoch, val_wer, min(val_wer, best_wer), trigger_times, gloss_to_idx, idx_to_gloss, vocab_size, d_model=s1_d_model)
            if should_save_last:
                torch.save(ckpt, LAST_CKPT)
            if val_wer < best_wer:
                best_wer, trigger_times = val_wer, 0
                converged_epoch = epoch
                torch.save(ckpt, BEST_CKPT)
                log.info(f"  ✨ Best Checkpoint Saved! (WER: {best_wer:.2f}%)")
            else: trigger_times += 1
        else: trigger_times += 1

        if trigger_times >= patience:
            log.info(f"🛑 CONVERGENCE REACHED: Early stopping triggered at Epoch {epoch}.")
            break

    log.info(f"🏆 Best model peaked at Epoch {converged_epoch} with Validation WER: {best_wer:.2f}%")

    # Final test evaluation using best checkpoint
    if BEST_CKPT.exists():
        best_ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        if 'ema_shadow' in best_ckpt and best_ckpt['ema_shadow'] is not None:
            for n, p in model.named_parameters():
                if n in best_ckpt['ema_shadow']:
                    p.data.copy_(best_ckpt['ema_shadow'][n])
        model.eval()
        test_wer_total, test_samples = 0, 0
        example_decodes = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for x_pad, y_flat, x_lens, y_lens in test_loader:
                if x_pad is None: continue
                x_pad = x_pad.to(device, non_blocking=True)
                x_lens = x_lens.to(device, non_blocking=True)
                y_lens = y_lens.to(device, non_blocking=True)
                logits, out_lens = model(x_pad, x_lens)
                log_probs = F.log_softmax(logits, dim=-1)
                decoded_preds = decode_ctc(log_probs, out_lens, blank=0)
                targets, idx = [], 0
                for length in y_lens:
                    targets.append(y_flat[idx:idx+length].cpu().tolist())
                    idx += length
                for ref, hyp in zip(targets, decoded_preds):
                    wer = calculate_wer(ref, hyp)
                    test_wer_total += wer
                    test_samples += 1
                    if len(example_decodes) < 20:
                        ref_glosses = [idx_to_gloss.get(t, f"?{t}") for t in ref]
                        hyp_glosses = [idx_to_gloss.get(t, f"?{t}") for t in hyp]
                        example_decodes.append({
                            "ref": " ".join(ref_glosses),
                            "hyp": " ".join(hyp_glosses),
                            "wer": round(wer, 3),
                        })
        test_wer = (test_wer_total / max(test_samples, 1)) * 100
        log.info(f"🧪 Final Test Set WER: {test_wer:.2f}%")

        # Log example decoded sequences
        log.info("Example decoded sequences (ref -> hyp):")
        for ex in example_decodes[:10]:
            match = "✓" if ex["wer"] == 0 else "✗"
            log.info(f"  {match} [{ex['ref']}] -> [{ex['hyp']}] (WER={ex['wer']})")

        history.append({"test_wer": round(test_wer, 2), "example_decodes": example_decodes})

    with open(save_dir / 'stage2_history.json', 'w') as f: json.dump(history, f, indent=2)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="SLT Stage 2 — Continuous Sign Recognition (CTC)")
    p.add_argument('--stage1_ckpt', default=None, help='Path to Stage 1 best_model.pth')
    p.add_argument('--data_path', default=None, help='Path to .npy landmark directory (overrides auto-detect)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--patience', type=int, default=35)
    p.add_argument('--unfreeze_epoch', type=int, default=30)
    p.add_argument('--save_dir', default=None)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--no_cr_ctc', action='store_true', help='Disable CR-CTC (saves ~50%% GPU memory)')
    p.add_argument('--phrase_data', default=None, help='Path to real phrase data (ASL_phrases_extracted)')
    p.add_argument('--continuous_data', default=None, help='Path to pre-generated continuous data (ASL_continuous_synthetic)')
    args = p.parse_args()
    train_stage2(
        stage1_ckpt=args.stage1_ckpt,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        unfreeze_epoch=args.unfreeze_epoch,
        save_dir=args.save_dir,
        smoke_test=args.smoke,
        use_cr_ctc=not args.no_cr_ctc,
        phrase_data=args.phrase_data,
        continuous_data=args.continuous_data,
    )