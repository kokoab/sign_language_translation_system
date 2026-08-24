"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v11 — Improved DS-GCN-Transformer                          ║
║  Changes from v9:                                                ║
║    - 4 GCN blocks (was 3) with progressive widths                ║
║    - Channel attention (SE) on last 2 blocks                     ║
║    - 4 transformer layers                                        ║
║    - Everything else same (114 geo, ArcFace, 61 nodes, 16ch)    ║
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


class DSGCNEncoderV11(nn.Module):
    """Improved DS-GCN Encoder: 4 blocks, SE on last 2, 4 transformer layers.
    Progressive channel widths: 96→192→384→384"""
    def __init__(self, in_channels=16, d_model=384, nhead=8, num_transformer_layers=4,
                 dropout=0.1, drop_path_rate=0.1):
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

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # 4 Transformer layers
        dp_rates = [drop_path_rate * i / max(num_transformer_layers - 1, 1)
                    for i in range(num_transformer_layers)]
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))
        self.transformer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'transformer' in name or 'se' in name:
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

        h[:, :, FACE_START:FACE_END, :] = h[:, :, FACE_START:FACE_END, :] * face_mask
        h[:, :, BODY_START:BODY_END, :] = h[:, :, BODY_START:BODY_END, :] * body_mask

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

        # Positional encoding + transformer
        h = h + self.pos_enc[:, :h.size(1)]
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            h = h + dp(layer(h) - h)
        return self.transformer_norm(h)


class SLTStage1V11(nn.Module):
    """Improved DS-GCN-Transformer with ArcFace.
    Supports knowledge distillation via return_embeddings."""
    def __init__(self, num_classes, d_model=384, nhead=8, num_transformer_layers=4,
                 dropout=0.1, head_dropout=0.30, drop_path_rate=0.1,
                 in_channels=16, use_arcface=True):
        super().__init__()
        self.encoder = DSGCNEncoderV11(
            in_channels=in_channels, d_model=d_model, nhead=nhead,
            num_transformer_layers=num_transformer_layers,
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
    model = SLTStage1V11(num_classes=310, d_model=384)
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
    print("✅ v11 forward pass OK")
