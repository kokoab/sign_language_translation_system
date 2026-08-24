"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT v16 — Squeezeformer Encoder                                ║
║  Replaces DSGCN-V14 based on exhaustive benchmarking:            ║
║    - 10x faster CPU inference (12ms vs 117ms)                    ║
║    - 2x faster GPU training                                      ║
║    - Better convergence (33.8% vs 12.4% in 1000-step test)      ║
║    - Simpler architecture (no graph ops, no angle features)      ║
║                                                                  ║
║  Architecture origin:                                            ║
║    - Squeezeformer: Kim et al., NeurIPS 2022                    ║
║    - Adapted for skeleton SLR by Christof Henkel                ║
║      (Kaggle ASL Fingerspelling 2023, 1st place)                ║
║                                                                  ║
║  Input: [B, T, 61, C] where C = 5 channels                     ║
║    ch0: X, ch1: Y, ch2: Z (3D body / perspective hands)        ║
║    ch3: mask, ch4: palm_scale                                    ║
║                                                                  ║
║  Tested configurations and results (1000 steps, 100 classes):   ║
║    Squeezeformer d=256: 33.8% val (BEST)                        ║
║    DecoupledGCN:        19.1% val                                ║
║    DSGCN-V14 (old):     12.4% val                                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ── Constants ──
NUM_NODES = 61          # 21 LH + 21 RH + 15 face + 4 body
DEFAULT_IN_CHANNELS = 9 # X, Y, Z, mask, palm_scale, vel_x, vel_y, acc_x, acc_y
DEFAULT_DIM = 256       # d_model (tested: d=256 > d=192 > d=128 > d=384)
DEFAULT_DEPTH = 4       # number of Squeezeformer blocks
DEFAULT_HEADS = 8       # attention heads (256 / 8 = 32 per head)
DEFAULT_CONV_KERNEL = 15  # depthwise conv kernel for temporal patterns
DEFAULT_DROPOUT = 0.1


class DropPath(nn.Module):
    """Stochastic Depth (drop entire residual block during training)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)) / keep
        return x * mask


class SqueezeformerBlock(nn.Module):
    """
    Squeezeformer block: FF → Attention → Conv → FF → LayerNorm

    From Kim et al. NeurIPS 2022 "Squeezeformer: An Efficient Transformer
    for Automatic Speech Recognition".

    Key design:
    - Half-residual on FF layers (×0.5) prevents feature explosion
    - Conv uses depthwise separable (groups=dim) for efficiency
    - GLU gating on conv branch
    - LayerNorm at end of block (post-norm)
    - DropPath (stochastic depth) for regularization
    """
    def __init__(self, dim, num_heads=DEFAULT_HEADS, conv_kernel=DEFAULT_CONV_KERNEL,
                 dropout=DEFAULT_DROPOUT, drop_path=0.0):
        super().__init__()

        # Feed-Forward 1 (with half-residual)
        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Multi-Head Self-Attention
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        # Convolutional module (depthwise separable + GLU)
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, 1),     # pointwise expansion with GLU
            nn.GLU(dim=1),                   # gates: dim*2 → dim
            nn.Conv1d(dim, dim, conv_kernel, # depthwise temporal conv
                      padding=conv_kernel // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),          # pointwise projection
            nn.Dropout(dropout),
        )

        # Feed-Forward 2 (with half-residual)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        # Post-norm
        self.norm = nn.LayerNorm(dim)

        # DropPath (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, T, dim]
            mask: optional attention mask [T, T] or [B*H, T, T]
        Returns:
            [B, T, dim]
        """
        # FF1 with half-residual
        x = x + self.drop_path(0.5 * self.ff1(x))

        # Self-Attention
        a = self.attn_norm(x)
        a, _ = self.attn(a, a, a, attn_mask=mask, need_weights=False)
        x = x + self.drop_path(self.attn_drop(a))

        # Conv module
        c = self.conv_norm(x).transpose(1, 2)  # [B, dim, T]
        c = self.conv(c).transpose(1, 2)        # [B, T, dim]
        x = x + self.drop_path(c)

        # FF2 with half-residual
        x = x + self.drop_path(0.5 * self.ff2(x))

        return self.norm(x)


def compute_angle_features(x):
    """Compute signer-invariant angle features from skeleton input.

    These features have 0.90 cosine similarity across different signers
    (vs 0.63 for raw XYZ), making them critical for signer-independent
    generalization.

    Args:
        x: [B, T, V, C] where V >= 42 (need both hands) and C >= 2 (need XY)
    Returns:
        [B, T, 59] angle features
        [B, T, 59] angle velocity (temporal derivative)
    """
    B, T, V, C = x.shape
    xyz = x[:, :, :, :min(C, 3)]  # use XY or XYZ
    if C < 3:
        # Pad to 3D with zeros for Z
        xyz = F.pad(xyz, (0, 3 - C))

    def _dist(a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _angle(v1, v2):
        cos_a = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
        return torch.acos(cos_a.clamp(-1 + 1e-6, 1 - 1e-6))

    # Joint angle triplets (15 per hand)
    _TRIPLETS = [
        (0,1,2),(1,2,3),(2,3,4),          # Thumb
        (0,5,6),(5,6,7),(6,7,8),          # Index
        (0,9,10),(9,10,11),(10,11,12),    # Middle
        (0,13,14),(13,14,15),(14,15,16),  # Ring
        (0,17,18),(17,18,19),(18,19,20),  # Pinky
    ]
    _CURL_PAIRS = [(2,4,2,3),(5,8,5,6),(9,12,9,10),(13,16,13,14),(17,20,17,18)]
    _SPREAD_MCPS = [5, 9, 13, 17]

    all_feats = []

    # 1. Joint angles: 15 per hand × 2 = 30
    for base in [0, 21]:
        for p, j, c in _TRIPLETS:
            v1 = xyz[:,:,base+p,:] - xyz[:,:,base+j,:]
            v2 = xyz[:,:,base+c,:] - xyz[:,:,base+j,:]
            all_feats.append(_angle(v1, v2))

    # 2. Curl ratios: 5 per hand × 2 = 10
    for base in [0, 21]:
        for mcp, tip, mcp2, pip in _CURL_PAIRS:
            d_tip = _dist(xyz[:,:,base+mcp,:], xyz[:,:,base+tip,:])
            d_pip = _dist(xyz[:,:,base+mcp2,:], xyz[:,:,base+pip,:])
            all_feats.append(d_tip / (d_pip + 1e-4))

    # 3. Finger spreads: 3 per hand × 2 = 6
    for base in [0, 21]:
        wrist = xyz[:,:,base,:]
        for i in range(len(_SPREAD_MCPS) - 1):
            v1 = xyz[:,:,base+_SPREAD_MCPS[i],:] - wrist
            v2 = xyz[:,:,base+_SPREAD_MCPS[i+1],:] - wrist
            all_feats.append(_angle(v1, v2))

    # 4. Palm normals: 3 per hand × 2 = 6
    for base in [0, 21]:
        wrist = xyz[:,:,base,:]
        v1 = xyz[:,:,base+5,:] - wrist
        v2 = xyz[:,:,base+17,:] - wrist
        normal = torch.cross(v1, v2, dim=-1)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
        for i in range(3):
            all_feats.append(normal[..., i])

    # 5. Inter-hand: 2
    inter_dist = _dist(xyz[:,:,0,:], xyz[:,:,21,:])
    hand_size = (_dist(xyz[:,:,0,:], xyz[:,:,9,:]) + _dist(xyz[:,:,21,:], xyz[:,:,30,:])) / 2 + 1e-4
    all_feats.append(inter_dist / hand_size)
    inter_dir = xyz[:,:,21,:] - xyz[:,:,0,:]
    inter_dir = inter_dir / (inter_dir.norm(dim=-1, keepdim=True) + 1e-6)
    all_feats.append(inter_dir[..., 1])

    # 6. Hand symmetry: 1
    l_angles = torch.stack(all_feats[:15], dim=-1)
    r_angles = torch.stack(all_feats[15:30], dim=-1)
    all_feats.append(F.cosine_similarity(l_angles, r_angles, dim=-1))

    # Stack: [B, T, 59]
    angle_feats = torch.stack(all_feats, dim=-1)

    # Temporal derivative
    vel = torch.zeros_like(angle_feats)
    vel[:, 1:-1] = (angle_feats[:, 2:] - angle_feats[:, :-2]) / 2.0
    vel[:, 0] = vel[:, 1]
    vel[:, -1] = vel[:, -2]

    return angle_feats, vel  # each [B, T, 59]


# Number of angle features: 30 angles + 10 curls + 6 spreads + 6 normals + 2 inter + 1 symmetry = 55
# With velocity: 55 × 2 = 110
N_ANGLE_FEATURES = 55


class SqueezeformerEncoder(nn.Module):
    """
    Squeezeformer encoder for skeleton sequences.

    Takes [B, T, V, C] skeleton input, computes pairwise hand distances
    and signer-invariant angle features, flattens to per-frame feature
    vector, projects to d_model, and processes through Squeezeformer blocks.

    Output: [B, T, dim] per-frame embeddings.
    """
    def __init__(self, in_channels=DEFAULT_IN_CHANNELS, num_nodes=NUM_NODES,
                 dim=DEFAULT_DIM, depth=DEFAULT_DEPTH,
                 num_heads=DEFAULT_HEADS, conv_kernel=DEFAULT_CONV_KERNEL,
                 dropout=DEFAULT_DROPOUT, drop_path=0.0,
                 use_pairwise=False, use_angles=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.dim = dim
        self.use_pairwise = use_pairwise
        self.use_angles = use_angles

        # 34 curated pairwise distances per hand × 2 hands = 68
        pairwise_dim = len(self.CURATED_PAIRS) * 2 if use_pairwise else 0
        # 55 angle features + 55 angle velocities = 110
        angle_dim = N_ANGLE_FEATURES * 2 if use_angles else 0
        in_dim = num_nodes * in_channels + pairwise_dim + angle_dim

        # Input projection: flatten V*C + pairwise → dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

        # Positional encoding (learnable)
        self.pos_enc = nn.Parameter(torch.zeros(1, 512, dim))  # max 512 frames
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Squeezeformer blocks with linearly increasing drop_path
        dp_rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList([
            SqueezeformerBlock(dim, num_heads, conv_kernel, dropout, drop_path=dp_rates[i])
            for i in range(depth)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # Curated pairwise indices: anatomically meaningful distances only
    # Instead of all 210 per hand, use ~40 per hand (80 total)
    # These capture: fingertip-to-fingertip, fingertip-to-thumb, MCP spread, palm shape
    CURATED_PAIRS = [
        # Fingertip distances (10 per hand — most discriminative)
        (4, 8), (4, 12), (4, 16), (4, 20),   # thumb tip to each fingertip
        (8, 12), (8, 16), (8, 20),            # index tip to others
        (12, 16), (12, 20),                    # middle tip to others
        (16, 20),                              # ring tip to pinky tip
        # Finger curl: tip to MCP (5 per hand)
        (4, 2), (8, 5), (12, 9), (16, 13), (20, 17),
        # MCP spread (4 per hand)
        (5, 9), (9, 13), (13, 17), (5, 17),
        # Wrist to fingertips (5 per hand)
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
        # Thumb opposition (3 per hand)
        (4, 5), (4, 9), (4, 13),
        # Palm diagonal (3 per hand)
        (0, 12), (5, 20), (0, 9),
        # Inter-finger DIP distances (4 per hand)
        (7, 11), (11, 15), (15, 19), (3, 7),
        # Total: 39 per hand × 2 = 78
    ]

    def _compute_pairwise(self, x):
        """Compute curated pairwise distances for both hands.
        Args: x [B, T, V, C] where C >= 2 (XY minimum)
        Returns: [B, T, 68] (34 pairs × 2 hands)
        """
        B, T = x.shape[0], x.shape[1]
        dists = []
        for hs in [0, 21]:  # left hand, right hand
            hand = x[:, :, hs:hs+21, :min(x.shape[-1], 3)]  # [B, T, 21, 2or3]
            if hand.shape[-1] < 3:
                hand = F.pad(hand, (0, 3 - hand.shape[-1]))
            hand_dists = []
            for i, j in self.CURATED_PAIRS:
                d = torch.sqrt(((hand[:, :, i] - hand[:, :, j]) ** 2).sum(-1) + 1e-8)
                hand_dists.append(d)
            dists.append(torch.stack(hand_dists, dim=-1))  # [B, T, 39]
        return torch.cat(dists, dim=-1)  # [B, T, 78]

    def forward(self, x):
        """
        Args:
            x: [B, T, V, C] skeleton input
        Returns:
            [B, T, dim] per-frame embeddings
        """
        B, T, V, C = x.shape

        # Flatten nodes × channels
        h = x.reshape(B, T, V * C)

        # Add pairwise hand distances
        if self.use_pairwise:
            pw = self._compute_pairwise(x)
            h = torch.cat([h, pw], dim=-1)

        # Add signer-invariant angle features
        if self.use_angles:
            angle_feats, angle_vel = compute_angle_features(x)
            h = torch.cat([h, angle_feats, angle_vel], dim=-1)

        # Project to d_model
        h = self.input_proj(h)

        # Add positional encoding
        h = h + self.pos_enc[:, :T, :]

        # Squeezeformer blocks
        for block in self.blocks:
            h = block(h)

        return h


class SLTStage1V16(nn.Module):
    """
    Stage 1: Isolated sign classification.

    Input: [B, 32, 61, C] → SqueezeformerEncoder → frame attention pool → classifier

    Uses learned frame attention (same as v14) to weight important frames
    before classification.
    """
    def __init__(self, num_classes, in_channels=DEFAULT_IN_CHANNELS,
                 num_nodes=NUM_NODES, dim=DEFAULT_DIM, depth=DEFAULT_DEPTH,
                 dropout=DEFAULT_DROPOUT, head_dropout=0.3, drop_path=0.0,
                 use_pairwise=False, use_angles=False):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = SqueezeformerEncoder(
            in_channels=in_channels, num_nodes=num_nodes,
            dim=dim, depth=depth, dropout=dropout, drop_path=drop_path,
            use_pairwise=use_pairwise, use_angles=use_angles)

        # Frame attention pooling (learned, same pattern as v14 ClassifierHead)
        self.frame_attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(head_dropout * 0.6),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x, return_embeddings=False):
        """
        Args:
            x: [B, T, V, C]
            return_embeddings: if True, also return [B, dim] embeddings
        Returns:
            logits [B, num_classes]
            (optional) embeddings [B, dim]
        """
        enc = self.encoder(x)  # [B, T, dim]

        # Frame attention pooling
        attn = F.softmax(self.frame_attn(enc).squeeze(-1), dim=1)  # [B, T]
        pooled = (enc * attn.unsqueeze(-1)).sum(dim=1)  # [B, dim]

        logits = self.classifier(pooled)

        if return_embeddings:
            return logits, pooled
        return logits


class MultiScaleTCNV16(nn.Module):
    """
    Multi-scale temporal convolutional network for Stage 2.
    Compresses per-clip encoder output from T tokens to out_tokens.

    Uses 3 parallel depthwise conv branches (kernel 3, 5, 9) to capture
    patterns at different temporal scales, fuses, then pools.
    """
    def __init__(self, dim, out_tokens=4, num_groups=8):
        super().__init__()
        self.out_tokens = out_tokens

        # 3 parallel branches with different kernel sizes
        self.branch3 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.GroupNorm(num_groups, dim), nn.GELU())
        self.branch5 = nn.Sequential(
            nn.Conv1d(dim, dim, 5, padding=2, groups=dim),
            nn.GroupNorm(num_groups, dim), nn.GELU())
        self.branch9 = nn.Sequential(
            nn.Conv1d(dim, dim, 9, padding=4, groups=dim),
            nn.GroupNorm(num_groups, dim), nn.GELU())

        # Fusion: 3*dim → dim
        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

        # Pool to fixed number of tokens
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)

    def forward(self, x):
        """
        Args:
            x: [B, T, dim] per-frame encoder output for one clip
        Returns:
            [B, out_tokens, dim]
        """
        xt = x.transpose(1, 2)  # [B, dim, T]
        b3 = self.branch3(xt).transpose(1, 2)  # [B, T, dim]
        b5 = self.branch5(xt).transpose(1, 2)
        b9 = self.branch9(xt).transpose(1, 2)

        fused = self.fuse(torch.cat([b3, b5, b9], dim=-1))  # [B, T, dim]

        # Pool: [B, T, dim] → [B, out_tokens, dim]
        pooled = self.pool(fused.transpose(1, 2)).transpose(1, 2)

        return pooled


class SLTStage2V16CTC(nn.Module):
    """
    Stage 2: Continuous sign recognition with CTC.

    Input: [B, T, 61, C] where T is variable (multiple of 32 frames)

    Pipeline:
    1. Split into 32-frame clips
    2. Encode each clip with SqueezeformerEncoder (shared with Stage 1)
    3. MultiScaleTCN compresses each clip to 4 tokens
    4. Squeezeformer sequence model across clips
    5. CTC head

    The encoder is frozen for the first 30 epochs, then unfrozen with 0.1x LR.
    """
    def __init__(self, vocab_size, stage1_ckpt=None, in_channels=DEFAULT_IN_CHANNELS,
                 num_nodes=NUM_NODES, dim=DEFAULT_DIM, encoder_depth=DEFAULT_DEPTH,
                 seq_layers=4, out_tokens=4, dropout=0.3,
                 use_pairwise=False, use_angles=False):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.out_tokens = out_tokens
        self.clip_len = 32

        # Encoder (shared with Stage 1, can be loaded from checkpoint)
        self.encoder = SqueezeformerEncoder(
            in_channels=in_channels, num_nodes=num_nodes,
            dim=dim, depth=encoder_depth, dropout=0.1,
            use_pairwise=use_pairwise, use_angles=use_angles)

        # Load Stage 1 encoder weights if provided
        if stage1_ckpt is not None:
            self._load_stage1_encoder(stage1_ckpt)

        # Initially freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # MultiScale TCN: compress 32-frame clips to 4 tokens each
        self.tcn = MultiScaleTCNV16(dim, out_tokens=out_tokens)

        # Sequence Squeezeformer: model cross-clip temporal relationships
        self.seq_pos_enc = nn.Parameter(torch.zeros(1, 512, dim))
        nn.init.trunc_normal_(self.seq_pos_enc, std=0.02)

        self.seq_blocks = nn.ModuleList([
            SqueezeformerBlock(dim, dropout=dropout)
            for _ in range(seq_layers)
        ])
        self.seq_norm = nn.LayerNorm(dim)

        # CTC classifier
        self.ctc_head = nn.Linear(dim, vocab_size)  # vocab includes blank token

        # InterCTC (intermediate CTC for deeper gradient signal)
        self.inter_ctc_proj = nn.Linear(dim, vocab_size)

    def _load_stage1_encoder(self, ckpt_path):
        """Load encoder weights from Stage 1 checkpoint."""
        import logging
        log = logging.getLogger(__name__)

        from pathlib import Path
        if not Path(ckpt_path).exists():
            log.warning(f"Stage 1 checkpoint not found: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Extract encoder state dict
        enc_state = {}
        sd = ckpt.get('ema_shadow') or ckpt.get('model_state_dict', {})
        for k, v in sd.items():
            k_clean = k.replace('_orig_mod.', '')
            if k_clean.startswith('encoder.'):
                enc_state[k_clean.replace('encoder.', '')] = v

        if enc_state:
            missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
            loaded = len(enc_state) - len(unexpected)
            log.info(f"Loaded {loaded}/{len(enc_state)} encoder weights from Stage 1")
            if missing:
                log.warning(f"Missing encoder keys: {len(missing)}")
            if unexpected:
                log.warning(f"Unexpected encoder keys: {len(unexpected)}")

    def unfreeze_encoder(self, lr_scale=0.1):
        """Unfreeze encoder for fine-tuning with reduced LR."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        return lr_scale

    def forward(self, x, input_lengths=None):
        """
        Args:
            x: [B, T, V, C] where T is variable (multiple of clip_len)
            input_lengths: [B] actual lengths before padding (optional)
        Returns:
            logits: [B, S, vocab_size] where S = num_clips * out_tokens
            inter_logits: [B, S, vocab_size] intermediate CTC logits
        """
        B, T, V, C = x.shape

        # Split into 32-frame clips
        num_clips = T // self.clip_len
        if T % self.clip_len != 0:
            # Pad to multiple of clip_len
            pad_len = self.clip_len - (T % self.clip_len)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            num_clips = x.shape[1] // self.clip_len

        # Reshape to process all clips at once
        clips = x.reshape(B * num_clips, self.clip_len, V, C)

        # Encode all clips
        clip_enc = self.encoder(clips)  # [B*nc, clip_len, dim]

        # Compress each clip to out_tokens
        clip_tokens = self.tcn(clip_enc)  # [B*nc, out_tokens, dim]

        # Reshape back to per-sample sequences
        seq = clip_tokens.reshape(B, num_clips * self.out_tokens, self.dim)

        # Add positional encoding
        S = seq.shape[1]
        seq = seq + self.seq_pos_enc[:, :S, :]

        # Sequence Squeezeformer blocks
        inter_logits = None
        for i, block in enumerate(self.seq_blocks):
            seq = block(seq)
            # InterCTC after first block
            if i == 0:
                inter_logits = self.inter_ctc_proj(seq)

        seq = self.seq_norm(seq)

        # CTC head
        logits = self.ctc_head(seq)  # [B, S, vocab_size]

        return logits, inter_logits


def make_checkpoint_v16(model, optimizer, scheduler, ema, epoch, metric_value,
                        best_metric, trigger_times, label_maps, vocab_size,
                        dim=DEFAULT_DIM, in_channels=DEFAULT_IN_CHANNELS, stage=1,
                        depth=DEFAULT_DEPTH, use_angles=None, use_pairwise=None):
    """Create a checkpoint dict for saving.

    depth / use_angles / use_pairwise: architecture flags. If use_angles /
    use_pairwise are None, we read them from the model's encoder attribute so
    the caller doesn't have to pass them explicitly.
    """
    unwrapped = model.module if hasattr(model, 'module') else model
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in unwrapped.state_dict().items()}
    clean_ema = None
    if ema and hasattr(ema, 'shadow') and ema.shadow:
        clean_ema = {k.replace('_orig_mod.', ''): v for k, v in ema.shadow.items()}

    enc = getattr(unwrapped, 'encoder', unwrapped)
    if use_angles is None:
        use_angles = bool(getattr(enc, 'use_angles', False))
    if use_pairwise is None:
        use_pairwise = bool(getattr(enc, 'use_pairwise', False))

    ckpt = {
        'model_state_dict': clean_sd,
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'ema_shadow': clean_ema,
        'epoch': epoch,
        'best_metric': best_metric,
        'trigger_times': trigger_times,
        'd_model': dim,
        'depth': depth,
        'in_channels': in_channels,
        'num_nodes': NUM_NODES,
        'use_angles': use_angles,
        'use_pairwise': use_pairwise,
        'model_type': 'Squeezeformer-v16',
        'stage': stage,
        'vocab_size': vocab_size if stage == 2 else None,
    }

    # Stage-specific fields
    if stage == 1:
        ckpt['val_acc'] = metric_value
        ckpt['best_acc'] = best_metric
        ckpt['num_classes'] = vocab_size  # reuse vocab_size for num_classes
        if 'label_to_idx' in label_maps:
            ckpt['label_to_idx'] = label_maps['label_to_idx']
        if 'idx_to_label' in label_maps:
            ckpt['idx_to_label'] = label_maps['idx_to_label']
    else:
        ckpt['val_wer'] = metric_value
        ckpt['best_wer'] = best_metric
        if 'gloss_to_idx' in label_maps:
            ckpt['gloss_to_idx'] = label_maps['gloss_to_idx']
        if 'idx_to_gloss' in label_maps:
            ckpt['idx_to_gloss'] = label_maps['idx_to_gloss']

    return ckpt
