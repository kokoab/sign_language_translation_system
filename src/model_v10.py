"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v10 — Multi-Stream Fusion Model (Single Model)             ║
║  3 GCN branches (joint, bone, velocity) fused with attention     ║
║  + ArcFace + Center Loss + 114 geo features                     ║
║  Input : [B, 32, 61, 16]  Output: [B, num_classes]              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import everything we need from train_stage_1
from train_stage_1 import (
    NUM_NODES, N_GEO_FEATURES,
    FACE_START, FACE_END, BODY_START, BODY_END,
    NOSE_NODE, CHIN_NODE, FOREHEAD_NODE,
    L_SHOULDER_NODE, R_SHOULDER_NODE, L_ELBOW_NODE, R_ELBOW_NODE,
    UPPER_LIP_NODE, LOWER_LIP_NODE,
    build_adjacency_matrices, DSGCNBlock, DropPath,
    _BONE_PAIRS, _JOINT_TRIPLETS, _SPREAD_MCPS,
    _THUMB_MCP, _THUMB_IP, _THUMB_TIP,
    _INDEX_MCP, _INDEX_PIP, _INDEX_TIP,
    _MIDDLE_MCP, _MIDDLE_PIP, _MIDDLE_TIP,
    _RING_MCP, _RING_PIP, _RING_TIP,
    _PINKY_MCP, _PINKY_PIP, _PINKY_TIP,
)


class BranchGCN(nn.Module):
    """Lightweight GCN branch for a specific feature stream (joint/bone/velocity)."""
    def __init__(self, in_channels, d_branch=192, dropout=0.1):
        super().__init__()
        self.register_buffer('A', build_adjacency_matrices(NUM_NODES))
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 96),
            nn.LayerNorm(96),
            nn.GELU()
        )
        self.gcn1 = DSGCNBlock(96, 128, temporal_kernel=3, dropout=dropout)
        self.gcn2 = DSGCNBlock(128, d_branch, temporal_kernel=5, dropout=dropout)
        # Node attention to collapse spatial dimension
        self.node_attn = nn.Sequential(
            nn.Linear(d_branch, d_branch // 4),
            nn.GELU(),
            nn.Linear(d_branch // 4, 1)
        )

    def forward(self, x, mask=None):
        """x: [B, T, N, C_in] -> [B, T, d_branch]"""
        h = self.input_proj(x)
        if mask is not None:
            h = h * mask
        h = self.gcn2(self.gcn1(h, self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)  # [B, T, N]
        return (h * attn.unsqueeze(-1)).sum(dim=2)  # [B, T, d_branch]


class AttentionFusion(nn.Module):
    """Learned attention fusion of multiple branch outputs."""
    def __init__(self, n_branches, d_branch, d_out):
        super().__init__()
        self.n_branches = n_branches
        # Per-branch projection to common space
        self.projs = nn.ModuleList([
            nn.Linear(d_branch, d_out) for _ in range(n_branches)
        ])
        # Attention weights: learn which branch matters per frame
        self.gate = nn.Sequential(
            nn.Linear(d_out * n_branches, d_out),
            nn.GELU(),
            nn.Linear(d_out, n_branches),
        )
        self.out_proj = nn.Linear(d_out, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, branch_outputs):
        """branch_outputs: list of [B, T, d_branch] tensors"""
        # Project each branch to common space
        projected = [proj(bo) for proj, bo in zip(self.projs, branch_outputs)]  # list of [B, T, d_out]

        # Compute attention weights
        concat = torch.cat(projected, dim=-1)  # [B, T, d_out * n_branches]
        gate_logits = self.gate(concat)  # [B, T, n_branches]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, T, n_branches]

        # Weighted sum
        stacked = torch.stack(projected, dim=-1)  # [B, T, d_out, n_branches]
        fused = (stacked * gate_weights.unsqueeze(2)).sum(dim=-1)  # [B, T, d_out]

        return self.norm(self.out_proj(fused))


class CenterLoss(nn.Module):
    """Center Loss: pulls embeddings toward learnable class centers.
    Complements ArcFace by structuring the embedding space."""
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        """features: [B, D], labels: [B]"""
        batch_centers = self.centers[labels]  # [B, D]
        return ((features - batch_centers) ** 2).sum(dim=1).mean()


class GeoFeatureExtractor(nn.Module):
    """Computes 114 geometric features from XYZ coordinates.
    Extracted as a module so it can be shared and the code is cleaner."""
    def __init__(self):
        super().__init__()
        # No learnable parameters — pure geometric computation

    @staticmethod
    def _dist(a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def forward(self, xyz, face_mask=None, body_mask=None):
        """xyz: [B, T, N, 3] -> [B, T, 114]"""
        d = self._dist

        def get_hand_features(base):
            tips = [d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_TIP]),
                    d(xyz[:,:,base+_INDEX_TIP], xyz[:,:,base+_MIDDLE_TIP]),
                    d(xyz[:,:,base+_MIDDLE_TIP], xyz[:,:,base+_RING_TIP]),
                    d(xyz[:,:,base+_RING_TIP], xyz[:,:,base+_PINKY_TIP]),
                    d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_PINKY_TIP])]
            curls = [d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_TIP]) / (d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_IP]) + 1e-4),
                     d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_TIP]) / (d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_PIP]) + 1e-4),
                     d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_TIP]) / (d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_PIP]) + 1e-4),
                     d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_TIP]) / (d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_PIP]) + 1e-4),
                     d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_TIP]) / (d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_PIP]) + 1e-4)]
            cross = xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0]
            d_ti = d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_MCP])
            return tips + curls + [cross, d_ti]

        f1 = get_hand_features(0)
        f2 = get_hand_features(21)

        # Hand-to-face (10)
        face_feats = [
            d(xyz[:,:,0], xyz[:,:,NOSE_NODE]), d(xyz[:,:,0], xyz[:,:,CHIN_NODE]),
            d(xyz[:,:,0], xyz[:,:,FOREHEAD_NODE]),
            d(xyz[:,:,21], xyz[:,:,NOSE_NODE]), d(xyz[:,:,21], xyz[:,:,CHIN_NODE]),
            d(xyz[:,:,21], xyz[:,:,FOREHEAD_NODE]),
            d(xyz[:,:,_INDEX_TIP], xyz[:,:,NOSE_NODE]),
            d(xyz[:,:,_INDEX_TIP], xyz[:,:,FOREHEAD_NODE]),
            d(xyz[:,:,21+_INDEX_TIP], xyz[:,:,NOSE_NODE]),
            d(xyz[:,:,21+_INDEX_TIP], xyz[:,:,FOREHEAD_NODE]),
        ]
        if face_mask is not None:
            fg = face_mask[:,:,0,0]
            face_feats = [f * fg for f in face_feats]

        # Joint angles (30)
        angle_feats = []
        for base in [0, 21]:
            for p, j, c in _JOINT_TRIPLETS:
                v1 = xyz[:,:,base+p,:] - xyz[:,:,base+j,:]
                v2 = xyz[:,:,base+c,:] - xyz[:,:,base+j,:]
                cos_a = (v1*v2).sum(-1) / (v1.norm(dim=-1)*v2.norm(dim=-1) + 1e-6)
                angle_feats.append(torch.acos(cos_a.clamp(-1+1e-6, 1-1e-6)))

        # Palm orientation (6)
        palm_feats = []
        for base in [0, 21]:
            w = xyz[:,:,base,:]
            v1 = xyz[:,:,base+5,:] - w; v2 = xyz[:,:,base+17,:] - w
            n = torch.cross(v1, v2, dim=-1)
            n = n / (n.norm(dim=-1, keepdim=True) + 1e-6)
            for i in range(3): palm_feats.append(n[...,i])

        # Finger spread (6)
        spread_feats = []
        for base in [0, 21]:
            w = xyz[:,:,base,:]
            for i in range(len(_SPREAD_MCPS)-1):
                v1 = xyz[:,:,base+_SPREAD_MCPS[i],:] - w
                v2 = xyz[:,:,base+_SPREAD_MCPS[i+1],:] - w
                cos_s = (v1*v2).sum(-1) / (v1.norm(dim=-1)*v2.norm(dim=-1) + 1e-6)
                spread_feats.append(torch.acos(cos_s.clamp(-1+1e-6, 1-1e-6)))

        # Wrist orientation (4)
        orient_feats = []
        up = torch.tensor([0.,1.,0.], device=xyz.device)
        fwd = torch.tensor([0.,0.,1.], device=xyz.device)
        for base in [0, 21]:
            w = xyz[:,:,base,:]; v1 = xyz[:,:,base+5,:]-w; v2 = xyz[:,:,base+17,:]-w
            n = torch.cross(v1, v2, dim=-1); n = n/(n.norm(dim=-1,keepdim=True)+1e-6)
            orient_feats.append((n*up).sum(-1)); orient_feats.append((n*fwd).sum(-1))

        # Wrist trajectory (6)
        traj_feats = []
        for base in [0, 21]:
            ww = xyz[:,:,base,:]; vel = torch.zeros_like(ww)
            vel[:,1:-1] = (ww[:,2:]-ww[:,:-2])/2.0; vel[:,0]=vel[:,1]; vel[:,-1]=vel[:,-2]
            sp = vel.norm(dim=-1,keepdim=True).clamp(min=1e-6); vd = vel/sp
            for i in range(3): traj_feats.append(vd[...,i])

        # Path curvature (2)
        curve_feats = []
        for base in [0, 21]:
            ww = xyz[:,:,base,:]; vel = torch.zeros_like(ww)
            vel[:,1:-1] = (ww[:,2:]-ww[:,:-2])/2.0; vel[:,0]=vel[:,1]; vel[:,-1]=vel[:,-2]
            vn = vel/(vel.norm(dim=-1,keepdim=True)+1e-6)
            dot = (vn[:,:-1]*vn[:,1:]).sum(-1)
            curve_feats.append(F.pad(torch.acos(dot.clamp(-1+1e-6,1-1e-6)), (0,1), value=0.0))

        # Inter-hand (3)
        inter_feats = [d(xyz[:,:,0,:], xyz[:,:,21,:])]
        rel = xyz[:,:,21,:]-xyz[:,:,0,:]; rn = rel/(rel.norm(dim=-1,keepdim=True)+1e-6)
        inter_feats.append(rn[...,0]); inter_feats.append(rn[...,1])

        # Face approach velocity (2)
        approach_feats = []
        for base in [0, 21]:
            ww = xyz[:,:,base,:]; dist = (ww-xyz[:,:,NOSE_NODE,:]).norm(dim=-1)
            av = torch.zeros_like(dist); av[:,1:-1]=(dist[:,2:]-dist[:,:-2])/2.0; av[:,0]=av[:,1]; av[:,-1]=av[:,-2]
            approach_feats.append(av)

        # Finger crossing (4)
        cross_feats = []
        for base in [0, 21]:
            cross_feats.append(xyz[:,:,base+_INDEX_TIP,0]-xyz[:,:,base+_MIDDLE_TIP,0])
            cross_feats.append(xyz[:,:,base+_MIDDLE_TIP,0]-xyz[:,:,base+_RING_TIP,0])

        # Hand symmetry (1)
        lv = torch.zeros_like(xyz[:,:,0,:]); lv[:,1:-1]=(xyz[:,2:,0,:]-xyz[:,:-2,0,:])/2.0; lv[:,0]=lv[:,1]; lv[:,-1]=lv[:,-2]
        rv = torch.zeros_like(xyz[:,:,21,:]); rv[:,1:-1]=(xyz[:,2:,21,:]-xyz[:,:-2,21,:])/2.0; rv[:,0]=rv[:,1]; rv[:,-1]=rv[:,-2]
        ld = lv/(lv.norm(dim=-1,keepdim=True)+1e-6); rd = rv/(rv.norm(dim=-1,keepdim=True)+1e-6)
        sym_feats = [(ld*rd).sum(-1)]

        # Speed (2)
        speed_feats = []
        for base in [0, 21]:
            ww = xyz[:,:,base,:]; v = torch.zeros_like(ww); v[:,1:-1]=(ww[:,2:]-ww[:,:-2])/2.0; v[:,0]=v[:,1]; v[:,-1]=v[:,-2]
            speed_feats.append(v.norm(dim=-1))

        # Acceleration (2)
        accel_feats = []
        for base in [0, 21]:
            ww = xyz[:,:,base,:]; v = torch.zeros_like(ww); v[:,1:-1]=(ww[:,2:]-ww[:,:-2])/2.0; v[:,0]=v[:,1]; v[:,-1]=v[:,-2]
            a = torch.zeros_like(v); a[:,1:-1]=(v[:,2:]-v[:,:-2])/2.0; a[:,0]=a[:,1]; a[:,-1]=a[:,-2]
            accel_feats.append(a.norm(dim=-1))

        # Contact (1)
        hd = (xyz[:,:,0,:]-xyz[:,:,21,:]).norm(dim=-1)
        contact_feats = [torch.sigmoid(5.0*(0.05-hd))]

        # Body-relative (11)
        body_feats = []
        l_sh = xyz[:,:,L_SHOULDER_NODE,:]; r_sh = xyz[:,:,R_SHOULDER_NODE,:]
        sh_mid = (l_sh+r_sh)/2.0; sh_w = d(l_sh, r_sh)
        for base in [0, 21]:
            w = xyz[:,:,base,:]
            body_feats.append((w[:,:,1]-sh_mid[:,:,1])/(sh_w+1e-6))  # height
            body_feats.append((w[:,:,0]-sh_mid[:,:,0])/(sh_w+1e-6))  # lateral
            body_feats.append(d(w, l_sh if base==0 else r_sh))       # shoulder dist
            body_feats.append(d(w, xyz[:,:,L_ELBOW_NODE if base==0 else R_ELBOW_NODE,:]))  # elbow dist
        body_feats.append(sh_w)
        mouth_mid = (xyz[:,:,UPPER_LIP_NODE,:]+xyz[:,:,LOWER_LIP_NODE,:])/2.0
        for base in [0, 21]:
            body_feats.append(d(xyz[:,:,base,:], mouth_mid))

        if body_mask is not None:
            bg = body_mask[:,:,0,0]
            body_feats = [f*bg for f in body_feats]

        # Total: 24 + 10 + 30 + 6 + 6 + 4 + 6 + 2 + 3 + 2 + 4 + 1 + 2 + 2 + 1 + 11 = 114
        return torch.stack(f1 + f2 + face_feats + angle_feats + palm_feats + spread_feats +
                          orient_feats + traj_feats + curve_feats + inter_feats + approach_feats +
                          cross_feats + sym_feats + speed_feats + accel_feats + contact_feats +
                          body_feats, dim=-1)


class SLTStage1MultiStream(nn.Module):
    """Multi-stream fusion model — 3 GCN branches fused with attention.
    Effectively an ensemble inside a single model."""

    def __init__(self, num_classes, d_model=512, nhead=8, num_transformer_layers=6,
                 dropout=0.1, head_dropout=0.30, drop_path_rate=0.1,
                 d_branch=192, use_arcface=True):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # === 3 GCN Branches ===
        # Branch 1: Joint stream (XYZ + mask = 4 effective channels from 16)
        self.branch_joint = BranchGCN(in_channels=16, d_branch=d_branch, dropout=dropout)
        # Branch 2: Bone stream (bone_dir + mask = channels 10,11,12,9)
        self.branch_bone = BranchGCN(in_channels=4, d_branch=d_branch, dropout=dropout)
        # Branch 3: Velocity stream (vel + mask = channels 3,4,5,9)
        self.branch_vel = BranchGCN(in_channels=4, d_branch=d_branch, dropout=dropout)

        # === Attention Fusion ===
        self.fusion = AttentionFusion(n_branches=3, d_branch=d_branch, d_out=d_model)

        # === Geo Features ===
        self.geo_extractor = GeoFeatureExtractor()
        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)

        # === Transformer ===
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        dp_rates = [drop_path_rate * i / max(num_transformer_layers-1, 1) for i in range(num_transformer_layers)]
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True))
            self.drop_paths.append(DropPath(dp))
        self.transformer_norm = nn.LayerNorm(d_model)

        # === Classifier Head (ArcFace) ===
        self.frame_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))

        if use_arcface:
            self.head_norm = nn.LayerNorm(d_model)
            self.head_drop = nn.Dropout(head_dropout)
            self.head_weight = nn.Parameter(torch.empty(num_classes, d_model))
            nn.init.xavier_uniform_(self.head_weight)
            self.arcface_s = 30.0
            self.arcface_m = 0.5
            self._cur_s = 1.0
            self._cur_m = 0.0
            self.use_arcface = True
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(head_dropout),
                nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(head_dropout * 0.6),
                nn.Linear(d_model, num_classes)
            )
            self.use_arcface = False

        # === Center Loss ===
        self.center_loss = CenterLoss(num_classes, d_model)

        # === Branch channel indices ===
        self.bone_channels = [10, 11, 12, 9]    # bone_dir_xyz + mask
        self.vel_channels = [3, 4, 5, 9]         # vel_xyz + mask

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'transformer' in name: continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def set_epoch(self, epoch, ce_warmup=10, margin_ramp=20):
        """ArcFace warmup schedule."""
        if not self.use_arcface: return
        if epoch <= ce_warmup:
            self._cur_m, self._cur_s = 0.0, 1.0
        elif epoch <= ce_warmup + margin_ramp:
            p = (epoch - ce_warmup) / margin_ramp
            self._cur_m = self.arcface_m * p
            self._cur_s = 10.0 + (self.arcface_s - 10.0) * p
        else:
            self._cur_m, self._cur_s = self.arcface_m, self.arcface_s

    def _pool_and_classify(self, h, labels=None):
        """Frame attention pooling + classification."""
        attn = F.softmax(self.frame_attn(h).squeeze(-1), dim=1)
        pooled = (h * attn.unsqueeze(-1)).sum(dim=1)  # [B, d_model]

        if self.use_arcface:
            pooled = self.head_drop(self.head_norm(pooled))
            cosine = F.linear(F.normalize(pooled, dim=1), F.normalize(self.head_weight, dim=1))
            if labels is not None and self.training and self._cur_m > 0:
                theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
                one_hot = F.one_hot(labels, self.num_classes).float()
                cosine = torch.cos(theta + one_hot * self._cur_m)
            logits = cosine * self._cur_s
        else:
            logits = self.classifier(pooled)

        return logits, pooled

    def forward(self, x, labels=None):
        """
        x: [B, T, N, 16] (full 16-channel input with bone features)
        labels: [B] class labels (needed for ArcFace during training)
        Returns: logits [B, num_classes], center_loss scalar
        """
        B, T, N, C = x.shape

        # Extract feature subsets for each branch
        x_joint = x  # all 16 channels
        x_bone = x[..., self.bone_channels]   # [B, T, N, 4]
        x_vel = x[..., self.vel_channels]      # [B, T, N, 4]

        # Masks for face and body gating
        face_mask = x[:, :, FACE_START:FACE_END, 9:10]
        body_mask = x[:, :, BODY_START:BODY_END, 9:10]

        # Run 3 GCN branches in parallel
        feat_joint = self.branch_joint(x_joint, mask=None)     # [B, T, d_branch]
        feat_bone = self.branch_bone(x_bone, mask=None)        # [B, T, d_branch]
        feat_vel = self.branch_vel(x_vel, mask=None)           # [B, T, d_branch]

        # Attention fusion
        fused = self.fusion([feat_joint, feat_bone, feat_vel])  # [B, T, d_model]

        # Add geo features
        xyz = x[:, :, :, :3]
        geo = self.geo_extractor(xyz, face_mask, body_mask)     # [B, T, 114]
        fused = self.geo_proj(torch.cat([fused, self.geo_norm(geo)], dim=-1))

        # Add positional encoding
        fused = fused + self.pos_enc[:, :T]

        # Transformer
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            fused = fused + dp(layer(fused) - fused)
        h = self.transformer_norm(fused)

        # Classify
        logits, pooled = self._pool_and_classify(h, labels)

        # Center loss (only during training)
        if self.training and labels is not None:
            c_loss = self.center_loss(pooled.detach(), labels)
            return logits, c_loss

        return logits


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = SLTStage1MultiStream(num_classes=310, d_model=512)
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 32, 61, 16)
    labels = torch.randint(0, 310, (2,))

    model.train()
    model.set_epoch(35)
    logits, c_loss = model(x, labels=labels)
    print(f"Train: logits={logits.shape}, center_loss={c_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(x)
    print(f"Eval: logits={logits.shape}")
    print("✅ Forward pass OK")
