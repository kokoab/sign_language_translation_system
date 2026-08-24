"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v14 — DS-GCN-TCN with Angle-Primary Features               ║
║  Key change from v12/v13:                                        ║
║    - Primary input: angle-based features (signer-invariant)      ║
║    - XYZ used only for geo features, not as GCN input            ║
║    - Same GCN blocks, TCN, ArcFace                               ║
║    - Same .npy format [32, 61, 10] — angles computed at runtime  ║
║                                                                  ║
║  Evidence: cosine similarity test showed                         ║
║    XYZ:    0.63 (webcam vs training) — signer-dependent          ║
║    Angles: 0.90 — signer-invariant                               ║
║    Curls:  0.93 — nearly identical                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from train_stage_1 import (
    NUM_NODES, N_GEO_FEATURES,
    FACE_START, FACE_END, BODY_START, BODY_END,
    build_adjacency_matrices, DropPath,
    DSGCNBlock,
)

# Joint angle triplets (same as in _compute_geo_features)
_JOINT_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),          # Thumb
    (0,5,6),(5,6,7),(6,7,8),          # Index
    (0,9,10),(9,10,11),(10,11,12),    # Middle
    (0,13,14),(13,14,15),(14,15,16),  # Ring
    (0,17,18),(17,18,19),(18,19,20),  # Pinky
]  # 15 triplets per hand = 30 total

# Fingertip-MCP pairs for curl ratios
_CURL_PAIRS = [
    (2, 4, 2, 3),    # Thumb: MCP->TIP / MCP->IP
    (5, 8, 5, 6),    # Index: MCP->TIP / MCP->PIP
    (9, 12, 9, 10),  # Middle
    (13, 16, 13, 14), # Ring
    (17, 20, 17, 18), # Pinky
]  # 5 per hand = 10 total

# MCP indices for finger spread
_SPREAD_MCPS = [5, 9, 13, 17]  # 3 adjacent pairs per hand = 6 total

# Number of angle-based features per frame per node
# We compute features globally (not per-node), then broadcast or use as frame-level
N_ANGLE_FEATURES = 30 + 10 + 6 + 6 + 4 + 2 + 1  # = 59
# 30 joint angles, 10 curls, 6 spreads, 6 palm normals, 4 wrist orientation, 2 inter-hand, 1 hand symmetry


def compute_angle_features(x):
    """Compute signer-invariant angle features from raw input tensor.

    Args:
        x: [B, T, N, C] where C >= 10 (xyz + vel + acc + mask)

    Returns:
        angle_feats: [B, T, N_ANGLE_FEATURES] — frame-level angle features
        vel_feats: [B, T, N_ANGLE_FEATURES] — temporal derivatives of angle features
        mask: [B, T, N, 1] — hand/face/body masks
    """
    B, T, N, C = x.shape
    xyz = x[:, :, :, :3]  # [B, T, N, 3]
    mask = x[:, :, :, 9:10]  # [B, T, N, 1]

    def _dist(a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _angle(v1, v2):
        cos_a = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
        return torch.acos(cos_a.clamp(-1 + 1e-6, 1 - 1e-6))

    all_feats = []

    # 1. Joint angles: 15 per hand × 2 = 30
    for base in [0, 21]:
        for p, j, c in _JOINT_TRIPLETS:
            v1 = xyz[:, :, base+p, :] - xyz[:, :, base+j, :]
            v2 = xyz[:, :, base+c, :] - xyz[:, :, base+j, :]
            angle = _angle(v1, v2)  # [B, T]
            all_feats.append(angle)

    # 2. Curl ratios: 5 per hand × 2 = 10
    for base in [0, 21]:
        for mcp, tip, mcp2, pip in _CURL_PAIRS:
            d_tip = _dist(xyz[:, :, base+mcp, :], xyz[:, :, base+tip, :])
            d_pip = _dist(xyz[:, :, base+mcp2, :], xyz[:, :, base+pip, :])
            curl = d_tip / (d_pip + 1e-4)  # [B, T]
            all_feats.append(curl)

    # 3. Finger spreads: 3 per hand × 2 = 6
    for base in [0, 21]:
        wrist = xyz[:, :, base, :]
        for i in range(len(_SPREAD_MCPS) - 1):
            v1 = xyz[:, :, base + _SPREAD_MCPS[i], :] - wrist
            v2 = xyz[:, :, base + _SPREAD_MCPS[i+1], :] - wrist
            spread = _angle(v1, v2)  # [B, T]
            all_feats.append(spread)

    # 4. Palm normals (direction, scale-invariant): 3 per hand × 2 = 6
    for base in [0, 21]:
        wrist = xyz[:, :, base, :]
        v1 = xyz[:, :, base + 5, :] - wrist   # wrist -> index MCP
        v2 = xyz[:, :, base + 17, :] - wrist  # wrist -> pinky MCP
        normal = torch.cross(v1, v2, dim=-1)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
        for i in range(3):
            all_feats.append(normal[..., i])  # [B, T]

    # 5. Wrist orientation (palm facing direction): 2 per hand × 2 = 4
    up = torch.tensor([0.0, 1.0, 0.0], device=xyz.device)
    forward = torch.tensor([0.0, 0.0, 1.0], device=xyz.device)
    for base in [0, 21]:
        wrist = xyz[:, :, base, :]
        v1 = xyz[:, :, base + 5, :] - wrist
        v2 = xyz[:, :, base + 17, :] - wrist
        normal = torch.cross(v1, v2, dim=-1)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
        all_feats.append((normal * up).sum(-1))       # palm-up/down
        all_feats.append((normal * forward).sum(-1))   # palm-forward/back

    # 6. Inter-hand features: 2
    inter_dist = _dist(xyz[:, :, 0, :], xyz[:, :, 21, :])  # wrist-to-wrist
    # Normalize by hand size to make scale-invariant
    l_hand_size = _dist(xyz[:, :, 0, :], xyz[:, :, 9, :])  # L wrist to L middle MCP
    r_hand_size = _dist(xyz[:, :, 21, :], xyz[:, :, 30, :])
    avg_hand_size = (l_hand_size + r_hand_size) / 2 + 1e-4
    all_feats.append(inter_dist / avg_hand_size)  # normalized inter-hand distance

    # Inter-hand relative direction (unit vector)
    inter_dir = xyz[:, :, 21, :] - xyz[:, :, 0, :]
    inter_dir = inter_dir / (inter_dir.norm(dim=-1, keepdim=True) + 1e-6)
    all_feats.append(inter_dir[..., 1])  # vertical component (above/below)

    # 7. Hand symmetry: 1
    # Cosine similarity between left and right hand angle vectors
    # (high for symmetric signs like BOOK, low for asymmetric like HELLO)
    l_angles = torch.stack(all_feats[:15], dim=-1)  # first 15 = left hand angles
    r_angles = torch.stack(all_feats[15:30], dim=-1)  # next 15 = right hand angles
    sym = F.cosine_similarity(l_angles, r_angles, dim=-1)  # [B, T]
    all_feats.append(sym)

    # Stack all features: [B, T, N_ANGLE_FEATURES]
    angle_feats = torch.stack(all_feats, dim=-1)  # [B, T, 59]

    # Compute temporal derivatives (velocity of angle features)
    vel = torch.zeros_like(angle_feats)
    vel[:, 1:-1] = (angle_feats[:, 2:] - angle_feats[:, :-2]) / 2.0
    vel[:, 0] = vel[:, 1]
    vel[:, -1] = vel[:, -2]

    return angle_feats, vel, mask


# ── Reuse GCN components from v12 ──────────────────────

class ChannelSEAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        se = x.mean(dim=2).mean(dim=1)
        scale = torch.sigmoid(self.fc2(F.gelu(self.fc1(se))))
        return x * scale.unsqueeze(1).unsqueeze(2) + x


class DSGCNBlockWithSE(nn.Module):
    def __init__(self, C_in, C_out, temporal_kernel=3, dropout=0.1, num_nodes=NUM_NODES):
        super().__init__()
        self.gcn = DSGCNBlock(C_in, C_out, temporal_kernel=temporal_kernel,
                              dropout=dropout, num_nodes=num_nodes)
        self.se = ChannelSEAttention(C_out)

    def forward(self, x, A):
        x = self.gcn(x, A)
        x = self.se(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1, num_groups=8):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding,
                               dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding,
                               dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        residual = x
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return h + residual


class TemporalTCN(nn.Module):
    def __init__(self, d_model, num_blocks=4, kernel_size=3, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        dilations = [2 ** i for i in range(num_blocks)]
        dp_rates = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]
        self.blocks = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for d, dp in zip(dilations, dp_rates):
            self.blocks.append(TCNBlock(d_model, kernel_size=kernel_size, dilation=d, dropout=dropout))
            self.drop_paths.append(DropPath(dp))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = x.permute(0, 2, 1)
        for block, dp in zip(self.blocks, self.drop_paths):
            h = h + dp(block(h) - h)
        h = h.permute(0, 2, 1)
        return self.norm(h)


# ── v14 Encoder ─────────────────────────────────────────

class DSGCNEncoderV14(nn.Module):
    """DS-GCN + TCN with angle-primary features.

    Input: [B, T, N, 16] (same .npy format as v12/v13)
    Internally computes angle features [B, T, 59*2] and uses as primary.
    GCN still operates on per-node features, but augmented with angle context.
    """
    def __init__(self, in_channels=16, d_model=384, num_tcn_blocks=4,
                 dropout=0.1, drop_path_rate=0.1,
                 nhead=8, num_transformer_layers=4):
        super().__init__()
        self.register_buffer('A', build_adjacency_matrices(NUM_NODES))

        # GCN still processes per-node features (XYZ + bone + mask = 16ch)
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Sequential(nn.Linear(in_channels, 96), nn.LayerNorm(96), nn.GELU())

        self.gcn1 = DSGCNBlock(96, 192, temporal_kernel=3, dropout=dropout, num_nodes=NUM_NODES)
        self.gcn2 = DSGCNBlock(192, 384, temporal_kernel=5, dropout=dropout, num_nodes=NUM_NODES)
        self.gcn3 = DSGCNBlockWithSE(384, d_model, temporal_kernel=5, dropout=dropout)
        self.gcn4 = DSGCNBlockWithSE(d_model, d_model, temporal_kernel=7, dropout=dropout)

        self.node_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))

        # Angle features projection (primary signer-invariant features)
        # 59 angle features + 59 angle velocities = 118
        n_angle_total = N_ANGLE_FEATURES * 2  # features + velocities
        self.angle_norm = nn.LayerNorm(n_angle_total)
        self.angle_proj = nn.Linear(d_model + n_angle_total, d_model)

        # TCN temporal encoding
        self.tcn = TemporalTCN(d_model, num_blocks=num_tcn_blocks, kernel_size=3,
                               dropout=dropout, drop_path_rate=drop_path_rate)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'tcn' in name or 'se' in name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, angle_features=None, angle_vel=None):
        """x: [B, T, N, 16] -> [B, T, d_model]"""
        # Face and body masking (non-inplace)
        face_mask = x[:, :, FACE_START:FACE_END, 9:10]
        body_mask = x[:, :, BODY_START:BODY_END, 9:10]

        B = x.size(0)
        mask = torch.ones(B, 1, x.size(2), 1, device=x.device, dtype=x.dtype)
        mask = mask.clone()
        mask[:, :, FACE_START:FACE_END, :] = face_mask[:, :1, :, :]
        mask[:, :, BODY_START:BODY_END, :] = body_mask[:, :1, :, :]

        h = self.input_proj(self.input_norm(x))
        h = h * mask

        # GCN blocks (still on per-node features)
        h = self.gcn1(h, self.A)
        h = self.gcn2(h, self.A)
        h = self.gcn3(h, self.A)
        h = self.gcn4(h, self.A)

        # Node attention pooling
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)  # [B, T, d_model]

        # Fuse with angle features (PRIMARY signer-invariant features)
        if angle_features is not None and angle_vel is not None:
            angle_combined = torch.cat([angle_features, angle_vel], dim=-1)  # [B, T, 118]
            h = self.angle_proj(torch.cat([h, self.angle_norm(angle_combined)], dim=-1))

        # TCN temporal encoding
        h = self.tcn(h)
        return h


class SLTStage1V14(nn.Module):
    """DS-GCN-TCN with angle-primary features and ArcFace."""
    def __init__(self, num_classes, d_model=384, num_tcn_blocks=4,
                 dropout=0.1, head_dropout=0.30, drop_path_rate=0.1,
                 in_channels=16, use_arcface=True,
                 nhead=8, num_transformer_layers=4):
        super().__init__()
        self.encoder = DSGCNEncoderV14(
            in_channels=in_channels, d_model=d_model, num_tcn_blocks=num_tcn_blocks,
            dropout=dropout, drop_path_rate=drop_path_rate)

        # Frame attention pooling
        self.frame_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))

        # Classifier
        self.num_classes = num_classes
        self.use_arcface = use_arcface
        if use_arcface:
            self.head_norm = nn.LayerNorm(d_model)
            self.head_drop = nn.Dropout(head_dropout)
            self.head_weight = nn.Parameter(torch.empty(num_classes, d_model))
            nn.init.xavier_uniform_(self.head_weight)
            self.arcface_s = 30.0
            self.arcface_m = 0.5
            self._cur_s = 1.0
            self._cur_m = 0.0
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(head_dropout),
                nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(head_dropout * 0.6),
                nn.Linear(d_model, num_classes))

    def set_epoch(self, epoch, ce_warmup=10, margin_ramp=20):
        if not self.use_arcface: return
        if epoch <= ce_warmup:
            self._cur_m, self._cur_s = 0.0, 1.0
        elif epoch <= ce_warmup + margin_ramp:
            p = (epoch - ce_warmup) / margin_ramp
            self._cur_m = self.arcface_m * p
            self._cur_s = 10.0 + (self.arcface_s - 10.0) * p
        else:
            self._cur_m, self._cur_s = self.arcface_m, self.arcface_s

    def forward(self, x, labels=None, return_embeddings=False):
        """x: [B, T, N, 16]"""
        # Compute angle features (signer-invariant)
        angle_feats, angle_vel, _ = compute_angle_features(x)

        # Encode with both node features and angle features
        h = self.encoder(x, angle_features=angle_feats, angle_vel=angle_vel)

        # Pool
        attn = F.softmax(self.frame_attn(h).squeeze(-1), dim=1)
        pooled = (h * attn.unsqueeze(-1)).sum(dim=1)

        # Classify
        if self.use_arcface:
            pooled_norm = self.head_drop(self.head_norm(pooled))
            cosine = F.linear(F.normalize(pooled_norm, dim=1),
                            F.normalize(self.head_weight, dim=1))
            logits_plain = cosine * self._cur_s  # plain cosine similarity (no margin)
            if labels is not None and self.training and self._cur_m > 0:
                theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
                one_hot = F.one_hot(labels, self.num_classes).float()
                cosine = torch.cos(theta + one_hot * self._cur_m)
            logits = cosine * self._cur_s
        else:
            logits = self.classifier(pooled)
            logits_plain = logits

        if return_embeddings:
            return logits, pooled
        if self.training and labels is not None:
            return logits, logits_plain
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SLTStage1V14(num_classes=310, d_model=384)
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 32, 61, 16)
    labels = torch.randint(0, 310, (2,))

    model.train()
    model.set_epoch(35)
    logits = model(x, labels=labels)
    print(f"Train: logits={logits.shape}")

    model.eval()
    model.set_epoch(200)
    with torch.no_grad():
        logits, emb = model(x, return_embeddings=True)
    print(f"Eval: logits={logits.shape}, embeddings={emb.shape}")

    # Verify encoder output
    angle_feats, angle_vel, _ = compute_angle_features(x)
    enc_out = model.encoder(x, angle_features=angle_feats, angle_vel=angle_vel)
    print(f"Encoder output: {enc_out.shape}")
    assert enc_out.shape == (2, 32, 384)

    # Compare with v12
    from model_v12 import SLTStage1V12
    v12 = SLTStage1V12(num_classes=310, d_model=384)
    print(f"\nv12 params: {count_parameters(v12):,}")
    print(f"v14 params: {count_parameters(model):,}")
    print("v14 forward pass OK")
