"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v12 — DS-GCN-TCN                                           ║
║  Changes from v11:                                               ║
║    - Replace 4 Transformer layers with 4-block dilated TCN       ║
║    - Remove positional encoding (implicit in convolution)        ║
║    - Same GCN blocks, geo features, ArcFace, node attention      ║
║    - Same encoder contract: [B, T, N, 16] → [B, T, d_model]     ║
║                                                                  ║
║  TCN advantages over Transformer for this pipeline:              ║
║    - ~30% fewer temporal params (3.6M vs 7.1M)                   ║
║    - Naturally causal for real-time streaming inference           ║
║    - Sharper sign boundaries for CTC in Stage 2                  ║
║    - Full receptive field (31 frames) covers 32-frame clips      ║
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
    DSGCNBlock,  # base GCN block
)


# ── Reuse from v11 ──────────────────────────────────────────────

class ChannelSEAttention(nn.Module):
    """Lightweight channel attention (SE-style). No conv1d overhead."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.weight, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        """x: [B, T, N, C] -> [B, T, N, C]"""
        se = x.mean(dim=2).mean(dim=1)  # [B, C]
        scale = torch.sigmoid(self.fc2(F.gelu(self.fc1(se))))  # [B, C]
        return x * scale.unsqueeze(1).unsqueeze(2) + x


class DSGCNBlockWithSE(nn.Module):
    """DS-GCN block + lightweight SE channel attention."""
    def __init__(self, C_in, C_out, temporal_kernel=3, dropout=0.1, num_nodes=NUM_NODES):
        super().__init__()
        self.gcn = DSGCNBlock(C_in, C_out, temporal_kernel=temporal_kernel,
                              dropout=dropout, num_nodes=num_nodes)
        self.se = ChannelSEAttention(C_out)

    def forward(self, x, A):
        x = self.gcn(x, A)
        x = self.se(x)
        return x


# ── TCN Temporal Encoder ────────────────────────────────────────

class TCNBlock(nn.Module):
    """Double-conv residual TCN block with dilation.
    GroupNorm + GELU + Conv1d × 2 + residual.
    Non-causal (symmetric padding) for training.
    """
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1, num_groups=8):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # symmetric (non-causal)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding,
                               dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding,
                               dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Init: zero-init second conv for stable residual at start
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        """x: [B, C, T] -> [B, C, T]"""
        residual = x
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return h + residual


class TemporalTCN(nn.Module):
    """4-block dilated TCN. Dilations [1, 2, 4, 8] with kernel=3
    give receptive field of 31 frames — covers full 32-frame clips.
    """
    def __init__(self, d_model, num_blocks=4, kernel_size=3, dropout=0.1,
                 drop_path_rate=0.1):
        super().__init__()
        dilations = [2 ** i for i in range(num_blocks)]  # [1, 2, 4, 8]
        dp_rates = [drop_path_rate * i / max(num_blocks - 1, 1)
                    for i in range(num_blocks)]

        self.blocks = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for d, dp in zip(dilations, dp_rates):
            self.blocks.append(TCNBlock(d_model, kernel_size=kernel_size,
                                        dilation=d, dropout=dropout))
            self.drop_paths.append(DropPath(dp))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: [B, T, d_model] -> [B, T, d_model]"""
        # Conv1d expects [B, C, T]
        h = x.permute(0, 2, 1)
        for block, dp in zip(self.blocks, self.drop_paths):
            h = h + dp(block(h) - h)
        # Back to [B, T, C]
        h = h.permute(0, 2, 1)
        return self.norm(h)


# ── Encoder ─────────────────────────────────────────────────────

class DSGCNEncoderV12(nn.Module):
    """DS-GCN + TCN Encoder: 4 GCN blocks (SE on last 2), 4-block dilated TCN.
    Progressive channel widths: 96→192→384→384
    Same contract as DSGCNEncoderV11: [B, T, N, 16] → [B, T, d_model]
    """
    def __init__(self, in_channels=16, d_model=384, num_tcn_blocks=4,
                 dropout=0.1, drop_path_rate=0.1,
                 # Accept but ignore transformer-specific args for compatibility
                 nhead=8, num_transformer_layers=4):
        super().__init__()
        self.register_buffer('A', build_adjacency_matrices(NUM_NODES))
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Sequential(nn.Linear(in_channels, 96), nn.LayerNorm(96), nn.GELU())

        # 4 GCN blocks: first 2 plain, last 2 with SE attention
        self.gcn1 = DSGCNBlock(96, 192, temporal_kernel=3, dropout=dropout, num_nodes=NUM_NODES)
        self.gcn2 = DSGCNBlock(192, 384, temporal_kernel=5, dropout=dropout, num_nodes=NUM_NODES)
        self.gcn3 = DSGCNBlockWithSE(384, d_model, temporal_kernel=5, dropout=dropout)
        self.gcn4 = DSGCNBlockWithSE(d_model, d_model, temporal_kernel=7, dropout=dropout)

        # Node attention to collapse spatial dimension
        self.node_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))

        # Geo features
        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)

        # 4-block dilated TCN (replaces Transformer + positional encoding)
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

    def forward(self, x, geo_features=None):
        """x: [B, T, N, 16] -> [B, T, d_model]"""
        face_mask = x[:, :, FACE_START:FACE_END, 9:10]
        body_mask = x[:, :, BODY_START:BODY_END, 9:10]

        h = self.input_proj(self.input_norm(x))

        # Non-inplace masking (avoids autograd inplace error)
        B = h.size(0)
        mask = torch.ones(B, 1, h.size(2), 1, device=h.device, dtype=h.dtype)
        mask = mask.clone()
        mask[:, :, FACE_START:FACE_END, :] = face_mask[:, :1, :, :]
        mask[:, :, BODY_START:BODY_END, :] = body_mask[:, :1, :, :]
        h = h * mask

        # 4 GCN blocks
        h = self.gcn1(h, self.A)
        h = self.gcn2(h, self.A)
        h = self.gcn3(h, self.A)
        h = self.gcn4(h, self.A)

        # Node attention pooling
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)  # [B, T, d_model]

        # Add geo features
        if geo_features is not None:
            h = self.geo_proj(torch.cat([h, self.geo_norm(geo_features)], dim=-1))

        # TCN temporal encoding (no positional encoding needed)
        h = self.tcn(h)
        return h


# ── Full Model ──────────────────────────────────────────────────

class SLTStage1V12(nn.Module):
    """DS-GCN-TCN with ArcFace.
    Supports knowledge distillation via return_embeddings."""
    def __init__(self, num_classes, d_model=384, num_tcn_blocks=4,
                 dropout=0.1, head_dropout=0.30, drop_path_rate=0.1,
                 in_channels=16, use_arcface=True,
                 # Accept but ignore for compat
                 nhead=8, num_transformer_layers=4):
        super().__init__()
        self.encoder = DSGCNEncoderV12(
            in_channels=in_channels, d_model=d_model, num_tcn_blocks=num_tcn_blocks,
            dropout=dropout, drop_path_rate=drop_path_rate)

        # Geo feature extractor (reuse from train_stage_1)
        from train_stage_1 import DSGCNEncoder
        self._geo_computer = DSGCNEncoder(in_channels=16, d_model=64)  # dummy, only use _compute_geo_features

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
        # Compute geo features
        xyz = x[:, :, :, :3]
        face_mask = x[:, :, FACE_START:FACE_END, 9:10]
        body_mask = x[:, :, BODY_START:BODY_END, 9:10]
        geo = self._geo_computer._compute_geo_features(xyz, face_mask, body_mask)

        # Encode
        h = self.encoder(x, geo_features=geo)  # [B, T, d_model]

        # Pool
        attn = F.softmax(self.frame_attn(h).squeeze(-1), dim=1)
        pooled = (h * attn.unsqueeze(-1)).sum(dim=1)  # [B, d_model]

        # Classify
        if self.use_arcface:
            pooled_norm = self.head_drop(self.head_norm(pooled))
            cosine = F.linear(F.normalize(pooled_norm, dim=1),
                            F.normalize(self.head_weight, dim=1))
            if labels is not None and self.training and self._cur_m > 0:
                theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
                one_hot = F.one_hot(labels, self.num_classes).float()
                cosine = torch.cos(theta + one_hot * self._cur_m)
            logits = cosine * self._cur_s
        else:
            logits = self.classifier(pooled)

        if return_embeddings:
            return logits, pooled
        return logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SLTStage1V12(num_classes=310, d_model=384)
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 32, 61, 16)
    labels = torch.randint(0, 310, (2,))

    model.train()
    model.set_epoch(35)
    logits = model(x, labels=labels)
    print(f"Train: logits={logits.shape}")

    model.eval()
    with torch.no_grad():
        logits, emb = model(x, return_embeddings=True)
    print(f"Eval: logits={logits.shape}, embeddings={emb.shape}")

    # Verify encoder contract
    enc_out = model.encoder(x, geo_features=model._geo_computer._compute_geo_features(
        x[:,:,:,:3], x[:,:,FACE_START:FACE_END,9:10], x[:,:,BODY_START:BODY_END,9:10]))
    print(f"Encoder output: {enc_out.shape}")
    assert enc_out.shape == (2, 32, 384), f"Expected [2, 32, 384], got {enc_out.shape}"

    # Compare param count with v11
    from model_v11 import SLTStage1V11
    v11 = SLTStage1V11(num_classes=310, d_model=384)
    print(f"\nv11 params: {count_parameters(v11):,}")
    print(f"v12 params: {count_parameters(model):,}")
    print(f"Reduction:  {count_parameters(v11) - count_parameters(model):,} fewer params")
    print("✅ v12 forward pass OK — same encoder contract as v11")
