"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT Stage 1 — Isolated Sign Classification (Face-Aware Edition) ║
║  DS-GCN Spatial Encoder + Transformer Encoder + Classifier Head  ║
║  Input : [B, 32, 47, 10] (Hands + Face: xyz + vel + acc + mask)  ║
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
# Enable TF32 Tensor Cores for FP32 matmuls (3-4x faster, negligible precision loss)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import numpy as np
import json
import math
import random
import logging
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split 

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("SLT-S1")

torch.backends.cudnn.benchmark = True

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — DUAL-HAND GRAPH & DS-GCN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

# Base edges for a single hand
_EDGES_SINGLE = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]
# Replicate for right hand (offset +21)
_EDGES_HANDS = _EDGES_SINGLE + [(u+21, v+21) for u, v in _EDGES_SINGLE]

# Face node indices: 42=Nose, 43=Chin, 44=Forehead, 45=L_Ear, 46=R_Ear
NOSE_NODE = 42; CHIN_NODE = 43; FOREHEAD_NODE = 44; L_EAR_NODE = 45; R_EAR_NODE = 46

# Face internal edges
_EDGES_FACE = [
    (NOSE_NODE, CHIN_NODE), (NOSE_NODE, FOREHEAD_NODE),
    (NOSE_NODE, L_EAR_NODE), (NOSE_NODE, R_EAR_NODE),
    (CHIN_NODE, FOREHEAD_NODE),
]

# Wrist-to-face edges (spatial context for hand-face interaction)
_EDGES_HAND_FACE = [
    (0, NOSE_NODE), (0, CHIN_NODE), (0, FOREHEAD_NODE),   # L_WRIST -> face
    (21, NOSE_NODE), (21, CHIN_NODE), (21, FOREHEAD_NODE), # R_WRIST -> face
    # Fingertips to key face points (for signs touching face)
    (4, NOSE_NODE), (4, FOREHEAD_NODE), (4, CHIN_NODE),    # L_THUMB_TIP
    (8, NOSE_NODE), (8, FOREHEAD_NODE), (8, CHIN_NODE),    # L_INDEX_TIP
    (25, NOSE_NODE), (25, FOREHEAD_NODE), (25, CHIN_NODE),  # R_THUMB_TIP
    (29, NOSE_NODE), (29, FOREHEAD_NODE), (29, CHIN_NODE),  # R_INDEX_TIP
]

_EDGES = _EDGES_HANDS + _EDGES_FACE + _EDGES_HAND_FACE

# 47 Nodes total (21 Left + 21 Right + 5 Face)
NUM_NODES = 47

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
        K = 3
        self.node_drop_rate = node_drop_rate
        self.dw_weights    = nn.Parameter(torch.ones(K, C_in))
        nn.init.uniform_(self.dw_weights, 0.8, 1.2)
        self.pointwise     = nn.Linear(K * C_in, C_out, bias=False)
        self.temporal_conv = nn.Conv1d(C_out, C_out, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=C_out, bias=False)
        self.temporal_norm = nn.GroupNorm(num_groups, C_out)
        self.norm          = nn.LayerNorm(C_out)
        self.act           = nn.GELU()
        self.drop          = nn.Dropout(dropout)
        self.residual      = nn.Linear(C_in, C_out, bias=False) if C_in != C_out else nn.Identity()
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
        h   = agg.permute(1, 2, 3, 0, 4).reshape(B, T, N, 3 * C)
        h     = self.drop(self.pointwise(h))
        C_out = h.shape[-1]
        h_t   = h.permute(0, 2, 3, 1).reshape(B * N, C_out, T)
        h_t   = self.temporal_norm(self.temporal_conv(h_t))
        h     = h_t.reshape(B, N, C_out, T).permute(0, 3, 1, 2)
        return self.act(self.norm(h + residual))

_THUMB_MCP  = 2;  _THUMB_IP   = 3;  _THUMB_TIP  = 4
_INDEX_MCP  = 5;  _INDEX_PIP  = 6;  _INDEX_TIP  = 8
_MIDDLE_MCP = 9;  _MIDDLE_PIP = 10; _MIDDLE_TIP = 12
_RING_MCP   = 13; _RING_PIP   = 14; _RING_TIP   = 16
_PINKY_MCP  = 17; _PINKY_PIP  = 18; _PINKY_TIP  = 20

# 12 per hand x 2 = 24, + 10 hand-to-face features = 34 (base)
# + 30 joint angles + 6 palm orientation + 6 finger spread = 76 total
N_GEO_FEATURES = 76

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

def compute_bone_features(x):
    """Append bone direction (3ch) + bone-motion (3ch) to input tensor.
    x: [B, T, 47, 10] or [T, 47, 10] -> [..., 16]
    Bone vectors are translation-invariant hand structure.
    Bone-motion vectors capture how the hand structure changes over time.
    """
    squeeze = False
    if x.ndim == 3:
        x = x.unsqueeze(0)
        squeeze = True
    xyz = x[..., :3]
    bone = torch.zeros_like(xyz)
    for p, c in _BONE_PAIRS:
        bone[..., c, :] = xyz[..., c, :] - xyz[..., p, :]
        bone[..., c+21, :] = xyz[..., c+21, :] - xyz[..., p+21, :]
    # Face bones: relative to nose (node 42)
    for fn in [43, 44, 45, 46]:
        bone[..., fn, :] = xyz[..., fn, :] - xyz[..., 42, :]
    # Bone-motion: central difference of bone vectors across time
    bone_motion = torch.zeros_like(bone)
    bone_motion[:, 1:-1] = (bone[:, 2:] - bone[:, :-2]) / 2.0
    bone_motion[:, 0] = bone_motion[:, 1]
    bone_motion[:, -1] = bone_motion[:, -2]
    result = torch.cat([x, bone, bone_motion], dim=-1)
    if squeeze:
        result = result.squeeze(0)
    return result

def compute_bone_features_np(x_np):
    """Numpy version for dataset __getitem__. x: [T, 47, 10] -> [T, 47, 16]"""
    xyz = x_np[..., :3]
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
        self.transformer_layers, self.drop_paths = nn.ModuleList(), nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))
        self.transformer_norm = nn.LayerNorm(d_model)
        # Phase 2B: Causal masking option
        self.causal = causal
        if causal:
            self.register_buffer('causal_mask', torch.triu(torch.ones(32, 32), diagonal=1).bool())
        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'transformer' in name: continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    @staticmethod
    def _geo_dist(a, b): return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz, face_mask=None):
        d = self._geo_dist

        def get_hand_features(base):
            tips = [d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_TIP]), d(xyz[:,:,base+_INDEX_TIP], xyz[:,:,base+_MIDDLE_TIP]), d(xyz[:,:,base+_MIDDLE_TIP], xyz[:,:,base+_RING_TIP]), d(xyz[:,:,base+_RING_TIP], xyz[:,:,base+_PINKY_TIP]), d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_PINKY_TIP])]
            curls = [d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_TIP]) / (d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_IP]) + 1e-4), d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_TIP]) / (d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_PIP]) + 1e-4), d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_TIP]) / (d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_PIP]) + 1e-4), d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_TIP]) / (d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_PIP]) + 1e-4), d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_TIP]) / (d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_PIP]) + 1e-4)]
            cross_idx_mid  = xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0]
            d_thumb_idxmcp = d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_MCP])
            return tips + curls + [cross_idx_mid, d_thumb_idxmcp]

        feats_hand1 = get_hand_features(0)   # Left Hand: 12 features
        feats_hand2 = get_hand_features(21)  # Right Hand: 12 features

        # Hand-to-face distance features (for signs interacting with face)
        face_feats = [
            d(xyz[:,:,0], xyz[:,:,NOSE_NODE]),       # L_wrist to nose
            d(xyz[:,:,0], xyz[:,:,CHIN_NODE]),        # L_wrist to chin
            d(xyz[:,:,0], xyz[:,:,FOREHEAD_NODE]),    # L_wrist to forehead
            d(xyz[:,:,21], xyz[:,:,NOSE_NODE]),       # R_wrist to nose
            d(xyz[:,:,21], xyz[:,:,CHIN_NODE]),       # R_wrist to chin
            d(xyz[:,:,21], xyz[:,:,FOREHEAD_NODE]),   # R_wrist to forehead
            d(xyz[:,:,_INDEX_TIP], xyz[:,:,NOSE_NODE]),       # L_index to nose
            d(xyz[:,:,_INDEX_TIP], xyz[:,:,FOREHEAD_NODE]),   # L_index to forehead
            d(xyz[:,:,21+_INDEX_TIP], xyz[:,:,NOSE_NODE]),    # R_index to nose
            d(xyz[:,:,21+_INDEX_TIP], xyz[:,:,FOREHEAD_NODE]),# R_index to forehead
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

        # 34 base + 30 angles + 6 palm + 6 spread = 76 total
        return torch.stack(feats_hand1 + feats_hand2 + face_feats + angle_feats + palm_feats + spread_feats, dim=-1)

    def forward(self, x, full_x=None):
        C = x.shape[-1]
        # For full 16-ch joint stream: extract XYZ and face mask directly
        # For auxiliary streams (4-ch): mask is last channel, no XYZ for geo features
        if C >= 10:
            xyz = x[:, :, :, :3]
            face_mask = x[:, :, 42:47, 9:10]
            has_geo = True
        else:
            face_mask = x[:, :, 42:47, -1:]  # mask is always last channel in subset
            has_geo = False
            # If full_x provided (for geo features), use it
            if full_x is not None:
                xyz = full_x[:, :, :, :3]
                has_geo = True

        h = self.input_proj(self.input_norm(x))
        # Gate face node features by mask before GCN aggregation
        h[:, :, 42:47, :] = h[:, :, 42:47, :] * face_mask
        h = self.gcn3(self.gcn2(self.gcn1(h, self.A), self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)
        if has_geo:
            h = self.geo_proj(torch.cat([h, self.geo_norm(self._compute_geo_features(xyz, face_mask))], dim=-1)) + self.pos_enc
        else:
            # No geo features: project with zeros as placeholder
            geo_zeros = torch.zeros(h.shape[0], h.shape[1], N_GEO_FEATURES, device=h.device)
            h = self.geo_proj(torch.cat([h, geo_zeros], dim=-1)) + self.pos_enc
        # Phase 2B: Causal masking
        mask = self.causal_mask[:h.size(1), :h.size(1)] if self.causal else None
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            h = h + dp(layer(h, src_mask=mask) - h)
        return self.transformer_norm(h)

class ClassifierHead(nn.Module):
    def __init__(self, d_model=256, num_classes=29, dropout=0.4):
        super().__init__()
        self.frame_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        self.net = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout * 0.6), nn.Linear(d_model, num_classes))
    def forward(self, x):
        attn = F.softmax(self.frame_attn(x).squeeze(-1), dim=1)
        return self.net((x * attn.unsqueeze(-1)).sum(dim=1))

class ArcFaceHead(nn.Module):
    """ArcFace angular margin classifier with gradual warmup (Deng et al. CVPR 2019).
    Warmup: Phase 1 (ep 1-10): pure CE (m=0, s=1). Phase 2 (ep 11-30): linear ramp to
    m=0.5, s=30. Phase 3 (ep 31+): full ArcFace. Prevents collapse on small datasets."""
    def __init__(self, d_model=256, num_classes=29, s=30.0, m=0.5, dropout=0.4,
                 ce_warmup=10, margin_ramp=20):
        super().__init__()
        self.frame_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, d_model))
        nn.init.xavier_uniform_(self.weight)
        self.s_target, self.m_target = s, m
        self.num_classes = num_classes
        self.ce_warmup, self.margin_ramp = ce_warmup, margin_ramp
        self._cur_s, self._cur_m = 1.0, 0.0
    def set_epoch(self, epoch):
        if epoch <= self.ce_warmup:
            self._cur_m, self._cur_s = 0.0, 1.0
        elif epoch <= self.ce_warmup + self.margin_ramp:
            p = (epoch - self.ce_warmup) / self.margin_ramp
            self._cur_m = self.m_target * p
            self._cur_s = 10.0 + (self.s_target - 10.0) * p
        else:
            self._cur_m, self._cur_s = self.m_target, self.s_target
    def forward(self, x, labels=None):
        attn = F.softmax(self.frame_attn(x).squeeze(-1), dim=1)
        pooled = self.drop(self.norm((x * attn.unsqueeze(-1)).sum(dim=1)))
        cosine = F.linear(F.normalize(pooled, dim=1), F.normalize(self.weight, dim=1))
        if labels is not None and self.training and self._cur_m > 0:
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            one_hot = F.one_hot(labels, self.num_classes).float()
            cosine = torch.cos(theta + one_hot * self._cur_m)
        return cosine * self._cur_s

class AngleStreamModel(nn.Module):
    """Transformer-only model for frame-level angle features (no GCN).
    Input: [B, T, in_features] where in_features=43 (42 angle/palm/spread + mask)."""
    def __init__(self, num_classes, in_features=43, d_model=192, nhead=8,
                 num_transformer_layers=4, dropout=0.1, head_dropout=0.3, drop_path_rate=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_features), nn.Linear(in_features, d_model), nn.GELU())
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        dp_rates = [drop_path_rate * i / max(num_transformer_layers - 1, 1) for i in range(num_transformer_layers)]
        self.layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))
        self.norm = nn.LayerNorm(d_model)
        self.head = ClassifierHead(d_model=d_model, num_classes=num_classes, dropout=head_dropout)
    def forward(self, x, labels=None):
        h = self.proj(x) + self.pos_enc[:, :x.size(1)]
        for layer, dp in zip(self.layers, self.drop_paths):
            h = h + dp(layer(h) - h)
        return self.head(self.norm(h))

class SLTStage1(nn.Module):
    def __init__(self, num_classes, in_channels=16, d_model=256, nhead=8, num_transformer_layers=4, dropout=0.1, head_dropout=0.4, drop_path_rate=0.1, use_arcface=False):
        super().__init__()
        self.encoder = DSGCNEncoder(in_channels=in_channels, d_model=d_model, nhead=nhead, num_transformer_layers=num_transformer_layers, dropout=dropout, drop_path_rate=drop_path_rate)
        if use_arcface:
            self.head = ArcFaceHead(d_model=d_model, num_classes=num_classes, dropout=head_dropout)
        else:
            self.head = ClassifierHead(d_model=d_model, num_classes=num_classes, dropout=head_dropout)
        self.use_arcface = use_arcface
    def forward(self, x, labels=None):
        enc = self.encoder(x)
        if self.use_arcface:
            return self.head(enc, labels=labels)
        return self.head(enc)

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — DATASET, LOADER, & EMA
# ══════════════════════════════════════════════════════════════════

def is_single_hand(tensor: torch.Tensor) -> bool:
    """Check if only one hand is active (for curriculum learning).

    Args:
        tensor: Shape [32, 47, 10] - 32 frames, 47 joints, 10 features
                Feature index 9 is the hand mask (1.0 = active, 0.0 = inactive)
    """
    l_active = tensor[:, :21, 9].max() > 0.5
    r_active = tensor[:, 21:42, 9].max() > 0.5
    return bool(l_active) != bool(r_active)


class SignDataset(Dataset):
    def __init__(self, data_path: str, label_to_idx: dict, manifest: dict, cache_path: str = None):
        self.label_to_idx = label_to_idx
        self.num_classes  = len(label_to_idx)
        data_path = Path(data_path)

        if cache_path and Path(cache_path).exists():
            log.info(f"Loading cache: {cache_path}")
            cache = torch.load(cache_path, weights_only=True)
            if cache.get('label_to_idx') == label_to_idx and cache.get('num_files') == len(manifest):
                self.data, self.targets = cache['data'], cache['targets']
                self.filenames = cache.get('filenames', []) 
                log.info(f"  {len(self.targets)} samples | RAM: ~{self.data.nbytes / 1e6:.0f} MB")
                return
            log.warning("Dataset/Manifest changed — rebuilding cache from .npy files.")

        log.info("Building dataset from .npy files using strict manifest...")
        data_list, target_list, filename_list = [], [], []
        skipped_manifest = 0
        skipped_quality = 0
        
        for fname in sorted(f for f in os.listdir(data_path) if f.endswith('.npy')):
            if fname not in manifest:
                skipped_manifest += 1
                continue
                
            label = manifest[fname]
            if label not in label_to_idx:
                skipped_manifest += 1
                continue
                
            arr = np.load(data_path / fname).astype(np.float32)

            # Accept ONLY (32, 47, 10)
            if arr.shape != (32, 47, 10):
                continue

            # Quality filters — skip obviously broken files
            xyz = arr[:, :42, :3]
            if np.abs(xyz).max() < 1e-6:       # all-zeros
                skipped_quality += 1
                continue
            if np.abs(xyz - xyz[0:1]).max() < 1e-4:  # no motion (static)
                skipped_quality += 1
                continue
            if np.abs(xyz).max() > 10.0:       # spatial outlier (landmarks way outside normal range)
                skipped_quality += 1
                continue

            data_list.append(arr)
            target_list.append(label_to_idx[label])
            filename_list.append(fname)

        if len(data_list) == 0:
            raise ValueError("CRITICAL: Every file was skipped! Ensure your .npy files are (32, 47, 10).")

        raw_data = torch.from_numpy(np.stack(data_list))
        self.targets   = torch.tensor(target_list, dtype=torch.long)
        self.filenames = filename_list
        log.info(f"  {len(self.targets)} samples loaded | {skipped_manifest} skipped (manifest) | {skipped_quality} skipped (quality)")

        # Precompute bone features once (avoids recomputing every epoch)
        log.info("  Precomputing bone features (one-time cost)...")
        bone_list = []
        for i in range(0, len(raw_data), 512):
            batch = raw_data[i:i+512]
            bone_list.append(compute_bone_features(batch))
        self.data = torch.cat(bone_list, dim=0)
        log.info(f"  Done. Shape: {list(self.data.shape)} | RAM: ~{self.data.nbytes / 1e9:.1f} GB")

        if cache_path:
            torch.save({'data': self.data, 'targets': self.targets,
                        'label_to_idx': label_to_idx, 'filenames': self.filenames,
                        'num_files': len(manifest)}, cache_path)

    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def class_weights(self, temperature: float = 0.5) -> torch.Tensor:
        counts = Counter(self.targets.tolist())
        raw = torch.tensor([len(self.targets) / counts[t.item()] for t in self.targets], dtype=torch.float32)
        return raw ** temperature

class ModelEMA:
    def __init__(self, model, decay=0.999):
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

# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — TRAINING, EVAL, SCHEDULER, AUGMENTATION
# ══════════════════════════════════════════════════════════════════

def temporal_speed_warp(xyz_tensor, min_speed=0.75, max_speed=1.25):
    """
    Warp temporal axis to simulate fast/slow signing. Fully vectorized on GPU.
    MUST be applied to XYZ BEFORE computing velocity/acceleration.

    Args:
        xyz_tensor: [B, T, N, 3] raw XYZ positions (torch tensor), N=47
        min_speed: Minimum speed multiplier (0.75 = 25% slower)
        max_speed: Maximum speed multiplier (1.25 = 25% faster)

    Returns:
        Warped tensor with same shape, same device
    """
    squeeze = False
    if xyz_tensor.ndim == 3:
        xyz_tensor = xyz_tensor.unsqueeze(0)
        squeeze = True

    B, T, N, C = xyz_tensor.shape
    device = xyz_tensor.device

    # Per-sample random speed factors [B]
    speeds = torch.empty(B, device=device).uniform_(min_speed, max_speed)
    warp_amounts = (speeds - 1.0) * 0.5  # [B]

    # Original time grid [T]
    orig_t = torch.linspace(0, 1, T, device=device)

    # Warped time grid per sample: [B, T]
    warped_t = orig_t.unsqueeze(0) + warp_amounts.unsqueeze(1) * orig_t.unsqueeze(0) * (1 - orig_t.unsqueeze(0)) * 4
    warped_t = warped_t.clamp(0, 1)

    # Convert warped_t to grid_sample coordinates: map [0,1] -> [-1,1]
    grid_1d = warped_t * 2 - 1  # [B, T]

    # Reshape for grid_sample: input [B, C, T_in], grid [B, T_out, 1]
    # Flatten N*C into channel dim
    flat = xyz_tensor.permute(0, 2, 3, 1).reshape(B, N * C, T)  # [B, N*C, T]
    flat = flat.unsqueeze(-1)  # [B, N*C, T, 1] — treat as 2D with H=T, W=1

    grid = torch.stack([
        torch.zeros_like(grid_1d),  # W dimension (unused, set to 0)
        grid_1d,                     # H dimension (time)
    ], dim=-1).unsqueeze(2)  # [B, T, 1, 2]

    warped = F.grid_sample(flat, grid, mode='bilinear', padding_mode='border', align_corners=True)
    # [B, N*C, T, 1] -> [B, T, N, C]
    warped = warped.squeeze(-1).reshape(B, N, C, T).permute(0, 3, 1, 2)

    if squeeze:
        warped = warped.squeeze(0)
    return warped


def recompute_kinematics(xyz):
    """
    Recompute velocity and acceleration from XYZ positions.

    Args:
        xyz: [B, T, N, 3] or [T, N, 3] XYZ tensor

    Returns:
        Tuple of (velocity, acceleration) with same shape as input
    """
    squeeze = False
    if xyz.ndim == 3:
        xyz = xyz.unsqueeze(0)
        squeeze = True

    B, T, N, _ = xyz.shape
    device = xyz.device if hasattr(xyz, 'device') else None

    # Central difference for velocity
    vel = torch.zeros_like(xyz) if device else np.zeros_like(xyz)
    if T > 2:
        if device:
            vel[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) / 2.0
        else:
            vel[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) / 2.0
        vel[:, 0] = vel[:, 1]
        vel[:, -1] = vel[:, -2]

    # Central difference for acceleration
    acc = torch.zeros_like(xyz) if device else np.zeros_like(xyz)
    if T > 2:
        if device:
            acc[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / 2.0
        else:
            acc[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / 2.0
        acc[:, 0] = acc[:, 1]
        acc[:, -1] = acc[:, -2]

    if squeeze:
        vel = vel[0]
        acc = acc[0]

    return vel, acc


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

def _rebuild_derived_channels(xyz, mask):
    """Rebuild all derived channels from XYZ + mask -> 16-channel tensor.
    xyz: [B, T, N, 3], mask: [B, T, N, 1] -> [B, T, N, 16]
    """
    vel, acc = recompute_kinematics(xyz)
    base = torch.cat([xyz, vel, acc, mask], dim=-1)  # [B, T, N, 10]
    return compute_bone_features(base)  # [B, T, N, 16]


# Finger chains for signer normalization: 5 chains per hand (thumb through pinky)
_FINGER_CHAINS = [
    (0, 1, 2, 3, 4),    # thumb
    (0, 5, 6, 7, 8),    # index
    (0, 9, 10, 11, 12),  # middle
    (0, 13, 14, 15, 16), # ring
    (0, 17, 18, 19, 20), # pinky
]


def online_augment(x, rotation_deg=8.0, scale_lo=0.88, scale_hi=1.12, noise_std=0.002,
                   speed_warp_prob=0.3, min_speed=0.8, max_speed=1.2,
                   signer_norm_prob=0.2, temporal_mask_prob=0.15, temporal_mask_frames=3):
    """
    Enhanced augmentation with temporal speed warping, signer normalization,
    rotation, scale, and noise. Operates on 16-channel bone-augmented tensors.

    All XYZ-modifying steps happen first, then derived channels (vel, acc,
    bone, bone-motion) are rebuilt once at the end.

    Args:
        x: [B, T, N, C] tensor with C=16 (xyz, vel, acc, mask, bone, bone_motion)
        rotation_deg: Max rotation in degrees
        scale_lo/hi: Scale range
        noise_std: Gaussian noise std
        speed_warp_prob: Probability of applying speed warp
        min_speed/max_speed: Speed warp range
        signer_norm_prob: Probability of applying finger chain normalization
    """
    B, T, N, C = x.shape
    device = x.device
    xyz = x[..., :3].clone()  # [B, T, N, 3]
    mask = x[..., 9:10]       # [B, T, N, 1]

    # Step 1: Speed warp (modifies temporal structure of XYZ)
    if random.random() < speed_warp_prob:
        xyz = temporal_speed_warp(xyz, min_speed, max_speed)

    # Step 2: Signer normalization — normalize each finger chain to unit length
    # per-sample (not batch-averaged) to learn hand-size invariance
    if random.random() < signer_norm_prob:
        for base in [0, 21]:  # both hands
            for chain in _FINGER_CHAINS:
                nodes = [base + c for c in chain]
                # Chain length per sample: [B, T]
                chain_len = sum(
                    torch.norm(xyz[..., nodes[i+1], :] - xyz[..., nodes[i], :], dim=-1)
                    for i in range(len(chain) - 1)
                )  # [B, T]
                # Scale relative to chain root, per-sample per-frame
                safe_len = chain_len.clamp(min=1e-4).unsqueeze(-1)  # [B, T, 1]
                root = xyz[..., nodes[0], :]  # [B, T, 3]
                for ni in range(1, len(chain)):
                    disp = xyz[..., nodes[ni], :] - root
                    xyz[..., nodes[ni], :] = root + disp / safe_len

    # Step 3: Rotation
    R = _batch_rotation_matrices(B, rotation_deg, device)
    xyz = torch.einsum('btni,bij->btnj', xyz, R)

    # Step 4: Scale and noise (applied to XYZ only)
    scale = scale_lo + torch.rand(B, 1, 1, 1, device=device) * (scale_hi - scale_lo)
    xyz = xyz * scale + torch.randn_like(xyz) * noise_std

    # Step 5: Temporal masking — zero out random contiguous frames
    if random.random() < temporal_mask_prob:
        n_mask = random.randint(2, temporal_mask_frames)
        start = random.randint(0, T - n_mask)
        xyz[:, start:start+n_mask] = 0.0

    # Step 6: Rebuild all derived channels from modified XYZ
    return _rebuild_derived_channels(xyz, mask)

def focal_cross_entropy(logits, targets, gamma=2.0, label_smoothing=0.0):
    """Focal loss: downweights easy examples, focuses on hard ones."""
    ce = F.cross_entropy(logits, targets, label_smoothing=label_smoothing, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()

def balanced_softmax_loss(logits, targets, log_prior, label_smoothing=0.10):
    """Balanced Softmax: adjusts logits by log class prior before CE. (Ren et al. 2020)"""
    adjusted = logits + log_prior.unsqueeze(0)
    return F.cross_entropy(adjusted, targets, label_smoothing=label_smoothing)

def apply_mixup(x, y, alpha=0.1, cutmix_prob=0.5):
    if alpha <= 0: return x, y, y, 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    if cutmix_prob > 0 and random.random() < cutmix_prob:
        # Temporal CutMix: swap a contiguous frame window
        T = x.size(1)
        lam = torch.distributions.Beta(1.0, 1.0).sample().item()
        cut_len = max(1, int(T * (1 - lam)))
        start = random.randint(0, T - cut_len)
        x_mix = x.clone()
        x_mix[:, start:start+cut_len] = x[idx, start:start+cut_len]
        lam = 1.0 - cut_len / T  # actual lambda based on swapped frames
    else:
        # Standard Mixup
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, y, y[idx], lam

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup. NO warm restarts — smooth single decay."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr_ratio=0.01, last_epoch=-1, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.max_epochs    = max_epochs
        self.min_lr_ratio  = min_lr_ratio
        # Accept but ignore T_0, T_mult for backward compatibility
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            scale = (e + 1) / self.warmup_epochs
        else:
            progress = (e - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            scale = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]

@torch.no_grad()
def evaluate(model, loader, device, use_amp, stream_name='joint', stream_channels=None, geo_encoder=None):
    model.eval()
    total_loss, correct_1, correct_5, sample_total = 0.0, 0, 0, 0
    is_angle = stream_name in ('angle', 'angle_motion')

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Stream-specific input transform (same as training but no augmentation)
        if is_angle and geo_encoder is not None:
            xyz = x[..., :3]
            face_mask = x[:, :, 42:47, 9:10]
            geo = geo_encoder._compute_geo_features(xyz, face_mask)
            angles = geo[..., 34:]
            mask_scalar = x[:, :, :42, 9].max(dim=2).values.unsqueeze(-1)
            x = torch.cat([angles, mask_scalar], dim=-1)
            if stream_name == 'angle_motion':
                motion = torch.zeros_like(x)
                motion[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
                motion[:, 0] = motion[:, 1]; motion[:, -1] = motion[:, -2]
                x = motion
        elif stream_channels is not None:
            x = x[..., stream_channels]
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss   = F.cross_entropy(logits, y)
            total_loss += loss.item()
            
        preds = logits.argmax(dim=1)
        correct_1 += (preds == y).sum().item()
        
        _, top5_preds = logits.topk(5, dim=1)
        correct_5 += (top5_preds == y.unsqueeze(1)).any(dim=1).sum().item()
        
        sample_total += y.size(0)
        
    return {
        "val_loss": total_loss / len(loader), 
        "acc": (correct_1 / max(sample_total, 1)) * 100,
        "top5_acc": (correct_5 / max(sample_total, 1)) * 100
    }

def _strip_compiled_prefix(state_dict):
    """Strip _orig_mod. prefix added by torch.compile for clean checkpoint keys."""
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

def make_checkpoint(model, optimizer, scheduler, ema, epoch, val_acc, best_acc, trigger_times, label_to_idx, idx_to_label, num_classes, in_channels, d_model, nhead, num_transformer_layers):
    unwrapped = model.module if hasattr(model, 'module') else model
    clean_sd = _strip_compiled_prefix(unwrapped.state_dict())
    clean_ema = _strip_compiled_prefix(ema.shadow) if ema and ema.shadow else None
    return {
        'encoder_state_dict':     {k: v for k, v in clean_sd.items() if k.startswith('encoder.')},
        'head_state_dict':        {k: v for k, v in clean_sd.items() if k.startswith('head.')},
        'model_state_dict':       clean_sd,
        'optimizer_state_dict':   optimizer.state_dict(),
        'scheduler_state_dict':   scheduler.state_dict(),
        'ema_shadow':             clean_ema,
        'epoch': epoch, 'best_acc': best_acc, 'trigger_times': trigger_times,
        'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label, 'num_classes': num_classes,
        'in_channels': in_channels, 'd_model': d_model, 'nhead': nhead, 'num_transformer_layers': num_transformer_layers,
        'val_acc': val_acc, 'stage': 1,
    }

def _auto_data_path():
    for p in ['/workspace/ASL_landmarks_rtmlib', '/workspace/ASL_landmarks_float16',
              '/kaggle/input/datasets/kokoab/batch-1/ASL_landmarks_float16',
              'ASL_landmarks_rtmlib', 'ASL_landmarks_float16']:
        if os.path.isdir(p): return p
    return 'ASL_landmarks_rtmlib'

def _auto_save_dir():
    for p in ['/workspace/output', '/kaggle/working']:
        if os.path.isdir(p): return p
    return './output'

STREAM_CHANNELS = {
    'joint': None,                    # all 16 channels
    'bone': [10, 11, 12, 9],          # bone_dir(3) + mask(1)
    'velocity': [3, 4, 5, 9],         # vel(3) + mask(1)
    'bone_motion': [13, 14, 15, 9],   # bone_motion(3) + mask(1)
    # angle and angle_motion use frame-level features, not channel subsets
}

def train(
    data_path = None,
    save_dir  = None,

    smoke_test = False,

    # Stream config
    stream_name = 'joint',
    use_balanced_softmax = False,
    use_arcface = False,

    epochs = 150, batch_size = 256, accum_steps = 4, lr = 3e-4, weight_decay = 0.01,
    warmup_epochs = 10, label_smoothing = 0.05, grad_clip = 5.0, patience = 25,
    val_every = 1,
    focal_gamma = 0.0,
    sampler_temperature = 0.5, in_channels = 16, d_model = 384, nhead = 8,
    num_transformer_layers = 6, dropout = 0.10, head_dropout = 0.45,
    mixup_alpha = 0.1,
    cutmix_prob = 0.15,
    curriculum_learning = False,
    curriculum_phase1_epochs = 50,
    curriculum_phase2_epochs = 50,
):
    if data_path is None: data_path = _auto_data_path()
    if save_dir is None:  save_dir = _auto_save_dir()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    BEST_CKPT, LAST_CKPT = save_dir / 'best_model.pth', save_dir / 'last_checkpoint.pth'

    # Stream configuration
    is_angle_stream = stream_name in ('angle', 'angle_motion')
    stream_channels = STREAM_CHANNELS.get(stream_name)  # None for joint (all 16) or angle streams
    if stream_channels is not None:
        in_channels = len(stream_channels)
    log.info(f"Device: {device} | AMP: {use_amp} | Stream: {stream_name} | in_ch: {in_channels} | Data: {data_path} | Save: {save_dir}")

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
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {str(v): k for k, v in label_to_idx.items()}
    
    log.info(f"Loaded strict manifest mapping {len(manifest)} files to {len(unique_labels)} classes.")
    
    cache_path = str(Path(data_path) / 'ds_cache.pt')  # Shared cache across all streams
    full_ds    = SignDataset(data_path, label_to_idx, manifest=manifest, cache_path=cache_path)
    
    # Filter out classes with fewer than 2 samples (can't stratify with singletons)
    all_targets = full_ds.targets.cpu().numpy()
    class_counts = Counter(all_targets.tolist())
    singleton_classes = {c for c, n in class_counts.items() if n < 2}
    if singleton_classes:
        valid_mask = np.array([t not in singleton_classes for t in all_targets])
        indices = [i for i in range(len(full_ds)) if valid_mask[i]]
        dropped_labels = [idx_to_label.get(str(c), f"?{c}") for c in singleton_classes]
        log.warning(f"Dropped {len(singleton_classes)} singleton classes ({len(all_targets) - len(indices)} samples): {dropped_labels}")
    else:
        indices = list(range(len(full_ds)))

    strat_targets = all_targets[indices]
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=strat_targets
    )
    temp_targets = all_targets[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=temp_targets
    )

    if smoke_test:
        train_idx = train_idx[:400]
        val_idx = val_idx[:100]
        test_idx = test_idx[:100]

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    test_ds  = Subset(full_ds, test_idx)
    log.info(f"Dataset Stratified Split: 70% Train ({len(train_ds)}) | 15% Val ({len(val_ds)}) | 15% Test ({len(test_ds)})")

    sample_weights = full_ds.class_weights(temperature=sampler_temperature)[train_ds.indices]

    # Curriculum learning: identify single-hand vs two-hand signs
    single_hand_mask = None
    if curriculum_learning:
        log.info("Analyzing dataset for curriculum learning (single-hand vs two-hand)...")
        single_hand_mask = torch.zeros(len(train_idx), dtype=torch.bool)
        for i, idx in enumerate(train_idx):
            single_hand_mask[i] = is_single_hand(full_ds.data[idx])
        n_single = single_hand_mask.sum().item()
        n_two = len(train_idx) - n_single
        log.info(f"  Single-hand signs: {n_single} | Two-hand signs: {n_two}")
        if n_single == 0:
            log.info("  No single-hand signs found (RTMW always detects both hands). Disabling curriculum.")
            curriculum_learning = False
            single_hand_mask = None
        else:
            log.info(f"  Phase 1 (epochs 1-{curriculum_phase1_epochs}): Single-hand only")
            log.info(f"  Phase 2 (epochs {curriculum_phase1_epochs+1}-{curriculum_phase1_epochs+curriculum_phase2_epochs}): Gradual mix")
            log.info(f"  Phase 3 (epochs {curriculum_phase1_epochs+curriculum_phase2_epochs+1}+): Full dataset")

    # GPU preload: only if single-stream mode AND enough VRAM. Otherwise pinned CPU.
    on_gpu = False
    multi_stream = os.environ.get('SLT_MULTI_STREAM', '1') == '0'  # Default: multi-stream (CPU)
    if multi_stream and device.type == 'cuda':
        ds_bytes = full_ds.data.element_size() * full_ds.data.nelement()
        gpu_total = torch.cuda.get_device_properties(device).total_mem
        if ds_bytes < gpu_total * 0.4:
            full_ds.data = full_ds.data.to(device)
            full_ds.targets = full_ds.targets.to(device)
            on_gpu = True
            log.info(f"Dataset preloaded to GPU ({ds_bytes / 1e9:.1f} GB)")
    if not on_gpu:
        full_ds.data = full_ds.data.pin_memory()
        full_ds.targets = full_ds.targets.pin_memory()
        log.info("Dataset on pinned CPU memory (multi-stream safe)")

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    nw = 0 if on_gpu else min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=nw, pin_memory=(not on_gpu), drop_last=True, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=(not on_gpu), persistent_workers=(nw > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=(not on_gpu), persistent_workers=(nw > 0))

    # Compute balanced softmax log-prior if needed
    log_prior = None
    if use_balanced_softmax:
        all_targets_list = full_ds.targets.cpu().numpy().tolist()
        cc = Counter(all_targets_list)
        counts_tensor = torch.zeros(full_ds.num_classes)
        for cls_idx, cnt in cc.items():
            counts_tensor[cls_idx] = cnt
        log_prior = torch.log(counts_tensor / counts_tensor.sum() + 1e-9).to(device)
        log.info(f"Balanced Softmax enabled (log-prior computed for {full_ds.num_classes} classes)")

    # Create model based on stream type
    if is_angle_stream:
        angle_in_features = 43  # 42 angle/palm/spread geo features + 1 mask
        model = AngleStreamModel(num_classes=full_ds.num_classes, in_features=angle_in_features, d_model=d_model, nhead=nhead, num_transformer_layers=num_transformer_layers, dropout=dropout, head_dropout=head_dropout)
        _geo_encoder = DSGCNEncoder(in_channels=16, d_model=64, nhead=8, num_transformer_layers=1)
        _geo_encoder.to(device).eval()
        log.info(f"AngleStreamModel created (in_features={angle_in_features}, d_model={d_model})")
    else:
        model = SLTStage1(num_classes=full_ds.num_classes, in_channels=in_channels, d_model=d_model, nhead=nhead, num_transformer_layers=num_transformer_layers, dropout=dropout, head_dropout=head_dropout)
        _geo_encoder = None

    # === CRITICAL ORDER: model.to(device) FIRST, then optimizer, then compile, then EMA ===

    # 1. Move model to GPU
    model.to(device)

    # 2. Create optimizer (fused=True needs CUDA tensors — model is on GPU now)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98),
                            fused=(device.type == 'cuda'))

    # 3. Scheduler
    use_onecycle = False
    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    if stream_name != 'joint':
        # Aux streams use cosine warmup (OneCycleLR was too aggressive)
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
        log.info(f"Using CosineWarmupScheduler (aux stream, {epochs} epochs)")
    else:
        scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs)
        log.info(f"Using CosineWarmupScheduler with smooth cosine decay ({epochs} epochs, warmup={warmup_epochs})")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 4. Resume from checkpoint (before compile — clean keys)
    start_epoch, best_acc, trigger_times = 1, 0.0, 0
    resumed_ema_shadow = None
    if LAST_CKPT.exists():
        try:
            ckpt = torch.load(LAST_CKPT, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if 'ema_shadow' in ckpt and ckpt['ema_shadow'] is not None:
                resumed_ema_shadow = ckpt['ema_shadow']
            start_epoch, best_acc, trigger_times = ckpt['epoch'] + 1, ckpt.get('best_acc', 0.0), ckpt.get('trigger_times', 0)
            log.info(f"Resumed from epoch {ckpt['epoch']} | Best so far: {best_acc:.2f}%")
        except Exception as e:
            log.warning(f"Resume failed ({e}). Starting fresh.")

    # 5. torch.compile AFTER model.to(device) and checkpoint loading
    #    Creates _orig_mod. prefix on parameters — EMA must be initialized AFTER this
    use_compile = hasattr(torch, 'compile') and not os.environ.get('TORCH_COMPILE_DISABLE')
    if use_compile:
        try:
            log.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode="default")
        except Exception as e:
            log.warning(f"torch.compile failed ({e}). Running without compilation.")

    # 6. EMA initialized AFTER compile so shadow keys match _orig_mod. prefixed params
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw_model = raw_model.module if hasattr(raw_model, 'module') else raw_model
    ema = ModelEMA(raw_model)
    if resumed_ema_shadow is not None:
        ema.shadow = resumed_ema_shadow
    ema.to(device)

    history = []
    converged_epoch = None

    def get_curriculum_weights(epoch, base_weights, single_hand_mask):
        """Get adjusted sample weights based on curriculum phase."""
        if single_hand_mask is None:
            return base_weights

        weights = base_weights.clone()

        if epoch <= curriculum_phase1_epochs:
            # Phase 1: Only single-hand signs
            weights[~single_hand_mask] = 0.0
        elif epoch <= curriculum_phase1_epochs + curriculum_phase2_epochs:
            # Phase 2: Gradually mix in two-hand signs
            progress = (epoch - curriculum_phase1_epochs) / curriculum_phase2_epochs
            weights[~single_hand_mask] *= progress
        # Phase 3: Full dataset (no modification needed)

        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum() * len(weights)

        return weights

    import time as _time
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = _time.time()
        # Update sampler with curriculum-adjusted weights
        if curriculum_learning and single_hand_mask is not None:
            curr_weights = get_curriculum_weights(epoch, sample_weights, single_hand_mask)
            sampler = WeightedRandomSampler(curr_weights, len(curr_weights), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                       num_workers=nw, pin_memory=(not on_gpu and use_amp), drop_last=True)

        model.train()
        # ArcFace warmup schedule: update margin/scale per epoch
        if use_arcface and hasattr(raw_model, 'head') and hasattr(raw_model.head, 'set_epoch'):
            raw_model.head.set_epoch(epoch)
            if epoch <= 5 or epoch == 11 or epoch == 31:
                log.info(f"  ArcFace: m={raw_model.head._cur_m:.3f}, s={raw_model.head._cur_s:.1f}")
        epoch_loss_gpu = torch.tensor(0.0, device=device)
        raw_loss_gpu = torch.tensor(0.0, device=device)
        raw_loss_count = 0

        optimizer.zero_grad(set_to_none=True)

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x = online_augment(x)  # Must happen on full 16-ch tensor

            # Stream-specific input preparation
            if is_angle_stream:
                with torch.no_grad():
                    xyz = x[..., :3]
                    face_mask = x[:, :, 42:47, 9:10]
                    geo = _geo_encoder._compute_geo_features(xyz, face_mask)  # [B, T, 76]
                    angles = geo[..., 34:]  # [B, T, 42] — angles + palm + spread
                    mask_scalar = x[:, :, :42, 9].max(dim=2).values.unsqueeze(-1)  # [B, T, 1]
                    x = torch.cat([angles, mask_scalar], dim=-1)  # [B, T, 43]
                    if stream_name == 'angle_motion':
                        motion = torch.zeros_like(x)
                        motion[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
                        motion[:, 0] = motion[:, 1]
                        motion[:, -1] = motion[:, -2]
                        x = motion
            elif stream_channels is not None:
                x = x[..., stream_channels]  # Channel subset for bone/velocity/bone_motion

            x, y_a, y_b, lam = apply_mixup(x, y, alpha=mixup_alpha, cutmix_prob=cutmix_prob)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Forward pass — ArcFace needs labels
                if use_arcface:
                    logits = model(x, labels=y_a)
                else:
                    logits = model(x)

                # Loss computation
                if use_balanced_softmax and log_prior is not None:
                    loss_a = balanced_softmax_loss(logits, y_a, log_prior, label_smoothing=label_smoothing)
                    loss_b = balanced_softmax_loss(logits, y_b, log_prior, label_smoothing=label_smoothing)
                elif focal_gamma > 0:
                    loss_a = focal_cross_entropy(logits, y_a, gamma=focal_gamma, label_smoothing=label_smoothing)
                    loss_b = focal_cross_entropy(logits, y_b, gamma=focal_gamma, label_smoothing=label_smoothing)
                else:
                    loss_a = F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                    loss_b = F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                loss = (lam * loss_a + (1 - lam) * loss_b) / accum_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(raw_model)

            epoch_loss_gpu += loss.detach() * accum_steps

            # Raw train loss (no augmentation penalty) for apples-to-apples comparison with val_loss
            if (i + 1) % 10 == 0:
                with torch.no_grad():
                    raw_ce = F.cross_entropy(logits.detach(), y.detach() if lam == 1.0 else y_a.detach())
                    raw_loss_gpu += raw_ce.detach()
                    raw_loss_count += 1

            if epoch == 1 and i == 0 and use_amp:
                log.info(f"GPU 0 Mem Usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        train_loss = epoch_loss_gpu.item() / len(train_loader)  # single GPU sync per epoch

        # Skip validation on non-val epochs for speed
        run_val = (epoch % val_every == 0) or (epoch <= 5) or (epoch == epochs)
        if run_val:
            ema.apply(raw_model)
            val_metrics = evaluate(model, val_loader, device, use_amp, stream_name=stream_name, stream_channels=stream_channels, geo_encoder=_geo_encoder)
            ema.restore(raw_model)
            val_loss, val_acc, val_top5 = val_metrics["val_loss"], val_metrics["acc"], val_metrics["top5_acc"]
        else:
            val_loss, val_acc, val_top5 = -1, -1, -1

        epoch_time = _time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)
        eta_str = f"{int(eta//3600)}h{int((eta%3600)//60):02d}m" if eta >= 3600 else f"{int(eta//60)}m{int(eta%60):02d}s"
        if run_val:
            log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | LR: {cur_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_acc:.2f}% | Val Top-5: {val_top5:.2f}%")
        else:
            log.info(f"Ep {epoch:03d} | {epoch_time:.0f}s | ETA {eta_str} | LR: {cur_lr:.2e} | Train Loss: {train_loss:.4f}")

        raw_train_loss = raw_loss_gpu.item() / max(raw_loss_count, 1)
        history.append({"epoch": epoch, "val_acc": round(val_acc, 3), "val_top5": round(val_top5, 3), "train_loss": round(train_loss, 5), "raw_train_loss": round(raw_train_loss, 5), "val_loss": round(val_loss, 5), "lr": round(cur_lr, 8)})

        if not run_val:
            continue  # skip checkpoint/early-stop logic on non-val epochs

        should_save_last = (epoch % 5 == 0 or epoch == epochs)
        if val_acc > best_acc or should_save_last:
            ckpt = make_checkpoint(model, optimizer, scheduler, ema, epoch, val_acc, max(val_acc, best_acc), trigger_times, label_to_idx, idx_to_label, full_ds.num_classes, in_channels, d_model, nhead, num_transformer_layers)

            if should_save_last:
                torch.save(ckpt, LAST_CKPT)

            if val_acc > best_acc:
                best_acc, trigger_times = val_acc, 0
                converged_epoch = epoch
                torch.save(ckpt, BEST_CKPT)
                log.info(f"  ✨ Best Checkpoint Saved! (Top-1 Acc: {best_acc:.2f}%)")
            else:
                trigger_times += 1
        else:
            trigger_times += 1

        if trigger_times >= patience:
            log.info(f"🛑 CONVERGENCE REACHED: Early stopping triggered at Epoch {epoch}.")
            break

    log.info(f"🏆 Best model peaked at Epoch {converged_epoch} with Validation Accuracy: {best_acc:.2f}%")

    # Final test evaluation using best checkpoint
    if BEST_CKPT.exists():
        best_ckpt = torch.load(BEST_CKPT, map_location=device, weights_only=False)
        raw_model.load_state_dict(best_ckpt['model_state_dict'])
        if 'ema_shadow' in best_ckpt and best_ckpt['ema_shadow'] is not None:
            for n, p in raw_model.named_parameters():
                if n in best_ckpt['ema_shadow']:
                    p.data.copy_(best_ckpt['ema_shadow'][n])
        model.eval()
        test_metrics = evaluate(model, test_loader, device, use_amp, stream_name=stream_name, stream_channels=stream_channels, geo_encoder=_geo_encoder)
        test_acc, test_top5 = test_metrics["acc"], test_metrics["top5_acc"]
        log.info(f"🧪 Final Test Set: Top-1 Acc: {test_acc:.2f}% | Top-5 Acc: {test_top5:.2f}%")
        history.append({"test_acc": round(test_acc, 3), "test_top5": round(test_top5, 3)})

        # Per-class accuracy + confusion matrix on test set
        per_class_correct = Counter()
        per_class_total = Counter()
        confusion = {}  # {true_label: {pred_label: count}}
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                # Apply same stream transform as evaluate()
                if is_angle_stream and _geo_encoder is not None:
                    xyz = x[..., :3]; fm = x[:, :, 42:47, 9:10]
                    geo = _geo_encoder._compute_geo_features(xyz, fm)
                    ang = geo[..., 34:]; ms = x[:, :, :42, 9].max(dim=2).values.unsqueeze(-1)
                    x = torch.cat([ang, ms], dim=-1)
                    if stream_name == 'angle_motion':
                        mot = torch.zeros_like(x); mot[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
                        mot[:, 0] = mot[:, 1]; mot[:, -1] = mot[:, -2]; x = mot
                elif stream_channels is not None:
                    x = x[..., stream_channels]
                preds = model(x).argmax(dim=1)
                for pred, target in zip(preds.cpu().tolist(), y.cpu().tolist()):
                    true_lbl = idx_to_label.get(str(target), f"UNK_{target}")
                    pred_lbl = idx_to_label.get(str(pred), f"UNK_{pred}")
                    per_class_total[true_lbl] += 1
                    if pred == target:
                        per_class_correct[true_lbl] += 1
                    # Confusion matrix
                    if true_lbl not in confusion:
                        confusion[true_lbl] = {}
                    confusion[true_lbl][pred_lbl] = confusion[true_lbl].get(pred_lbl, 0) + 1

        per_class_acc = {}
        for lbl in sorted(per_class_total.keys()):
            acc = per_class_correct[lbl] / max(per_class_total[lbl], 1) * 100
            per_class_acc[lbl] = {"correct": per_class_correct[lbl], "total": per_class_total[lbl], "acc": round(acc, 1)}

        # Log weakest classes with their top confusions
        sorted_by_acc = sorted(per_class_acc.items(), key=lambda x: x[1]["acc"])
        log.info("Bottom 15 classes by test accuracy:")
        for lbl, stats in sorted_by_acc[:15]:
            conf = confusion.get(lbl, {})
            # Top 3 misclassifications (excluding correct)
            misses = sorted(((p, c) for p, c in conf.items() if p != lbl), key=lambda x: -x[1])[:3]
            miss_str = ", ".join(f"{p}({c})" for p, c in misses)
            log.info(f"  {lbl}: {stats['acc']}% ({stats['correct']}/{stats['total']}) -> confused with: {miss_str}")

        history.append({"per_class_acc": per_class_acc})

        # Save confusion matrix
        with open(save_dir / 'confusion_matrix.json', 'w') as f:
            json.dump(confusion, f, indent=2)
        log.info(f"Confusion matrix saved to {save_dir / 'confusion_matrix.json'}")

    with open(save_dir / 'history.json', 'w') as f: json.dump(history, f, indent=2)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SLT Stage 1 — Multi-Stream Training")
    p.add_argument('--stream', default='joint', choices=['joint', 'bone', 'velocity', 'bone_motion', 'angle', 'angle_motion'])
    p.add_argument('--data_path', default=None, help='Path to .npy landmark directory (overrides auto-detect)')
    p.add_argument('--d_model', type=int, default=384)
    p.add_argument('--num_layers', type=int, default=6)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--save_dir', default=None)
    p.add_argument('--balanced_softmax', action='store_true', help='Use Balanced Softmax loss')
    p.add_argument('--arcface', action='store_true', help='Use ArcFace head with gradual warmup')
    p.add_argument('--smoke', action='store_true', help='Smoke test (3 epochs, small data)')
    args = p.parse_args()
    train(
        stream_name=args.stream,
        data_path=args.data_path,
        d_model=args.d_model,
        num_transformer_layers=args.num_layers,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        accum_steps=args.accum_steps,
        save_dir=args.save_dir,
        use_balanced_softmax=args.balanced_softmax,
        use_arcface=args.arcface,
        smoke_test=args.smoke,
    )