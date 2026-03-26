"""
SLT Inference — .npy landmarks → Stage 1 (isolated) or Stage 2 (CTC) → Stage 3 (T5) → English
Takes pre-extracted .npy files (from slt-extract container) and runs inference.

Single 32-frame clips use Stage 1 (isolated sign classifier).
Multi-clip sequences use Stage 2 (CTC continuous recognition).

Usage:
    python run_inference.py /app/input/file.npy
    python run_inference.py /app/input/           # all .npy files
"""
import sys, os, glob, json, argparse, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =====================================================================
# PATHS
# =====================================================================
# Resolve paths relative to project root (works both in Docker and natively)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
STAGE1_CKPT = os.environ.get("STAGE1_CKPT", os.path.join(_PROJECT_ROOT, "models", "output_joint", "best_model.pth"))
STAGE2_CKPT = os.environ.get("STAGE2_CKPT", os.path.join(_PROJECT_ROOT, "models", "output", "stage2_best_model.pth"))
STAGE3_DIR = os.environ.get("STAGE3_DIR", os.path.join(_PROJECT_ROOT, "weights", "slt_final_t5_model"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(_PROJECT_ROOT, "output"))
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =====================================================================
# BONE FEATURES (computed at load time, matching training dataloader)
# =====================================================================
_BONE_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]

def compute_bone_features(x_np):
    """Append bone direction (3ch) + bone-motion (3ch) to [T, 47, 10] → [T, 47, 16]."""
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

# =====================================================================
# MODEL ARCHITECTURE (must match training exactly)
# =====================================================================
# Geometric feature indices
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
        enc = self.encoder(x)
        return self.head(enc)


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

        all_clips_batch = torch.cat(all_clips, dim=0)
        with torch.no_grad():
            enc_out = self.encoder(all_clips_batch)       # [total_clips, 32, d_model]

        enc_out = enc_out.permute(0, 2, 1)                # [total_clips, d_model, 32]
        pooled = self.temporal_pool(enc_out)                # [total_clips, d_model, 4]
        pooled = pooled.permute(0, 2, 1)                   # [total_clips, 4, d_model]

        out_seqs = []
        out_lens = []
        offset = 0
        for nc in clip_counts:
            seq_features = pooled[offset:offset + nc].reshape(nc * 4, -1)  # [nc*4, d_model]
            out_seqs.append(seq_features)
            out_lens.append(nc * 4)
            offset += nc

        padded_seqs = pad_sequence(out_seqs, batch_first=True)
        max_len = padded_seqs.size(1)
        padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= \
            torch.tensor(out_lens, device=x.device).unsqueeze(1)

        transformer_out = self.seq_transformer(padded_seqs, padding_mask=padding_mask)
        logits = self.classifier(transformer_out)
        return logits, torch.tensor(out_lens, device=x.device)


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
# MAIN
# =====================================================================
def load_models():
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    # Stage 1 (isolated sign classifier)
    s1_model = None
    s1_idx_to_label = None
    print(f"\nStage 1: {STAGE1_CKPT}")
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
        print(f"  Loaded ({ckpt1['num_classes']} classes, d_model={ckpt1.get('d_model', 384)})")
    else:
        print(f"  Not found (Stage 1 disabled)")

    # Stage 2 (CTC continuous recognition)
    print(f"\nStage 2: {STAGE2_CKPT}")
    if not os.path.exists(STAGE2_CKPT):
        print(f"ERROR: Not found: {STAGE2_CKPT}")
        sys.exit(1)

    ckpt2 = torch.load(STAGE2_CKPT, map_location=DEVICE, weights_only=False)
    idx_to_gloss = ckpt2["idx_to_gloss"]
    vocab_size = ckpt2["vocab_size"]
    s2_d_model = ckpt2.get("d_model", 384)
    s2_model = SLTStage2CTC(vocab_size=vocab_size, d_model=s2_d_model).to(DEVICE)
    s2_model.load_state_dict(ckpt2["model_state_dict"], strict=False)
    s2_model.eval()
    print(f"  Loaded ({vocab_size} classes)")

    # Stage 3 (T5 translation)
    print(f"\nStage 3: {STAGE3_DIR}")
    if not os.path.exists(STAGE3_DIR):
        print(f"ERROR: Not found: {STAGE3_DIR}")
        sys.exit(1)
    s3_tokenizer = AutoTokenizer.from_pretrained(STAGE3_DIR)
    s3_model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3_DIR).to(DEVICE)
    s3_model.eval()
    param_count = sum(p.numel() for p in s3_model.parameters()) / 1e6
    print(f"  Loaded Flan-T5 ({param_count:.1f}M params)")

    return s1_model, s1_idx_to_label, s2_model, s3_model, s3_tokenizer, idx_to_gloss


def _run_stage2_ctc(data, s2_model, idx_to_gloss):
    """Run Stage 2 CTC on [N*32, 47, 16] data with TTA. Returns (glosses, confidence)."""
    x = torch.from_numpy(data).unsqueeze(0).float().to(DEVICE)
    lens = torch.tensor([x.shape[1]], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits, out_lens = s2_model(x, lens)
        log_probs_orig = torch.log_softmax(logits[0], dim=-1)
        # Mirror TTA
        x_mirror = _mirror_tta(x)
        logits_m, _ = s2_model(x_mirror, lens)
        log_probs_mirror = torch.log_softmax(logits_m[0], dim=-1)
        # Average in probability space
        avg_probs = (log_probs_orig.exp() + log_probs_mirror.exp()) / 2.0
        log_probs = avg_probs.log().cpu().numpy()
        n_tokens = out_lens[0].item()
    glosses = ctc_greedy_decode(log_probs[:n_tokens], idx_to_gloss)
    if glosses:
        max_probs = log_probs[:n_tokens].max(axis=-1)
        conf = float(np.exp(np.mean(max_probs)))
    else:
        conf = 0.0
    return glosses, conf


def _mirror_tta(x):
    """Create mirrored version: swap left/right hands (0-20 <-> 21-41), flip X.
    x: [B, T, 47, C] tensor. Returns mirrored copy."""
    m = x.clone()
    # Swap left and right hand nodes
    m[:, :, 0:21] = x[:, :, 21:42]
    m[:, :, 21:42] = x[:, :, 0:21]
    # Flip X coordinate (channel 0) for hands
    m[:, :, :42, 0] *= -1
    # Flip X velocity (channel 3) and X acceleration (channel 6) if present
    if m.shape[-1] > 3:
        m[:, :, :42, 3] *= -1  # vel_x
    if m.shape[-1] > 6:
        m[:, :, :42, 6] *= -1  # acc_x
    # Flip bone direction X (channel 10) and bone motion X (channel 13) if present
    if m.shape[-1] > 10:
        m[:, :, :42, 10] *= -1  # bone_dir_x
    if m.shape[-1] > 13:
        m[:, :, :42, 13] *= -1  # bone_motion_x
    return m


def _run_stage1(data, s1_model, s1_idx_to_label):
    """Run Stage 1 on [32, 47, 16] data with TTA (mirror averaging).
    Returns (gloss, confidence, top5)."""
    x = torch.from_numpy(data[:32]).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        # Original
        probs_orig = torch.softmax(s1_model(x), dim=-1)
        # Mirror TTA
        x_mirror = _mirror_tta(x)
        probs_mirror = torch.softmax(s1_model(x_mirror), dim=-1)
        # Average
        probs = (probs_orig + probs_mirror) / 2.0
        topk = torch.topk(probs, 5, dim=-1)
    top5 = []
    for prob, idx in zip(topk.values[0], topk.indices[0]):
        label = s1_idx_to_label.get(str(idx.item()), s1_idx_to_label.get(idx.item(), f"UNK_{idx.item()}"))
        top5.append((label, prob.item()))
    return top5[0][0], top5[0][1], top5


def _process_raw_window(raw_xyz, l_ever, r_ever, face_ever):
    """Process a raw [32, 47, 3] XYZ window into [32, 47, 16] features.
    Each window processed independently — matches how training data was created."""
    from scipy.signal import savgol_filter
    seq = raw_xyz.copy().astype(np.float32)

    # Normalize (same as extract.py normalize_sequence)
    L_WRIST, R_WRIST = 0, 21
    L_MIDDLE_MCP, R_MIDDLE_MCP = 9, 30
    valid_wrists = []
    if l_ever: valid_wrists.append(seq[:, L_WRIST, :])
    if r_ever: valid_wrists.append(seq[:, R_WRIST, :])
    if valid_wrists:
        all_w = np.concatenate(valid_wrists, axis=0)
        nonzero = all_w[np.linalg.norm(all_w, axis=-1) > 1e-6]
        center = np.median(nonzero, axis=0) if len(nonzero) > 0 else np.zeros(3)
    else:
        center = np.zeros(3)
    if l_ever: seq[:, 0:21] -= center
    if r_ever: seq[:, 21:42] -= center
    seq[:, 42:47] -= center
    bone_lengths = []
    if l_ever: bone_lengths.extend(np.linalg.norm(seq[:, L_MIDDLE_MCP] - seq[:, L_WRIST], axis=-1))
    if r_ever: bone_lengths.extend(np.linalg.norm(seq[:, R_MIDDLE_MCP] - seq[:, R_WRIST], axis=-1))
    if bone_lengths:
        filtered = [b for b in bone_lengths if b > 1e-6]
        if filtered:
            seq /= (np.median(filtered) + 1e-8)

    # Kinematics (Savitzky-Golay)
    F = seq.shape[0]
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    wl = min(7, F if F % 2 == 1 else F - 1)
    if wl >= 3:
        for p in range(seq.shape[1]):
            for c in range(3):
                vel[:, p, c] = savgol_filter(seq[:, p, c], window_length=wl, polyorder=2, deriv=1)
                acc[:, p, c] = savgol_filter(seq[:, p, c], window_length=wl, polyorder=2, deriv=2)

    # Mask
    mask = np.zeros((F, 47, 1), dtype=np.float32)
    if l_ever: mask[:, 0:21, 0] = 1.0
    if r_ever: mask[:, 21:42, 0] = 1.0
    if face_ever: mask[:, 42:47, 0] = 1.0

    features_10ch = np.concatenate([seq, vel, acc, mask], axis=-1).astype(np.float32)
    return compute_bone_features(features_10ch)


def _sliding_window_stage1(raw_xyz, l_ever, r_ever, face_ever,
                            s1_model, s1_idx_to_label,
                            window=32, stride=8, conf_threshold=0.15):
    """Slide Stage 1 across raw XYZ, processing each window independently.
    Based on Zuo et al. 2024 'Towards Online CSLR'.
    Returns deduplicated gloss sequence."""
    T = raw_xyz.shape[0]
    if T < window:
        return [], 0.0

    # Step 1: Resample raw XYZ to a reasonable frame rate for sliding
    # Use stride=8 for 75% overlap
    predictions = []
    for start in range(0, T - window + 1, stride):
        clip_xyz = raw_xyz[start:start + window]  # [32, 47, 3]
        clip_16ch = _process_raw_window(clip_xyz, l_ever, r_ever, face_ever)

        x = torch.from_numpy(clip_16ch).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            probs_orig = torch.softmax(s1_model(x), dim=-1)
            probs_mirror = torch.softmax(s1_model(_mirror_tta(x)), dim=-1)
            probs = (probs_orig + probs_mirror) / 2.0
            conf, idx = probs.max(dim=-1)
            conf = conf.item()
            label = s1_idx_to_label.get(str(idx.item()), s1_idx_to_label.get(idx.item(), "UNK"))
        predictions.append((start, label, conf))

    if not predictions:
        return [], 0.0

    # Step 2: Background elimination
    labels = [(l if c >= conf_threshold else '_BLANK_') for _, l, c in predictions]
    confs = [c if c >= conf_threshold else 0.0 for _, _, c in predictions]

    # Step 3: Majority voting deduplication
    glosses = []
    gloss_confs = []
    i = 0
    while i < len(labels):
        if labels[i] == '_BLANK_':
            i += 1
            continue
        current = labels[i]
        group_confs = [confs[i]]
        j = i + 1
        # Allow blanks within a sign (brief low-conf dips)
        blank_run = 0
        while j < len(labels):
            if labels[j] == current:
                group_confs.append(confs[j])
                blank_run = 0
            elif labels[j] == '_BLANK_':
                blank_run += 1
                if blank_run > 2:  # more than 2 consecutive blanks = sign boundary
                    break
            else:
                break  # different sign
            j += 1

        if len(group_confs) >= 2:
            glosses.append(current)
            gloss_confs.append(float(np.mean(group_confs)))

        i = j

    avg_conf = float(np.mean(gloss_confs)) if gloss_confs else 0.0
    return glosses, avg_conf


def run_inference(npy_path, s1_model, s1_idx_to_label, s2_model, s3_model, s3_tokenizer, idx_to_gloss):
    basename = os.path.splitext(os.path.basename(npy_path))[0]

    # Skip _continuous files — they're loaded automatically when the base file is processed
    if basename.endswith('_continuous'):
        return None

    print(f"\n[INFERENCE] {basename}")

    # Load isolated .npy [32, 47, 10]
    data = np.load(npy_path).astype(np.float32)
    print(f"  Loaded: {data.shape}")

    if data.shape[-1] == 10:
        data = compute_bone_features(data)

    # --- Hypothesis 1: Stage 1 isolated (whole video = 1 sign) ---
    s1_gloss, s1_conf, s1_top5 = None, 0.0, []
    if s1_model is not None:
        s1_gloss, s1_conf, s1_top5 = _run_stage1(data, s1_model, s1_idx_to_label)
        print(f"\n  [Stage 1] {s1_gloss} ({s1_conf:.4f})")
        for g, p in s1_top5:
            print(f"    {g:20s} {p:.4f}")

    # --- Hypothesis 2: Stage 2 CTC on 1 clip ---
    s2_iso_glosses, s2_iso_conf = _run_stage2_ctc(data, s2_model, idx_to_gloss)
    if s2_iso_glosses:
        print(f"\n  [Stage 2, 1-clip] {' '.join(s2_iso_glosses)} (conf={s2_iso_conf:.4f})")

    # --- Hypothesis 3: Sliding window Stage 1 on full-length sequence ---
    candidates = []
    if s1_gloss:
        candidates.append(([s1_gloss], s1_conf, 'stage1'))
    if s2_iso_glosses:
        candidates.append((s2_iso_glosses, s2_iso_conf, 'stage2-1clip'))

    raw_path = npy_path.replace('.npy', '') + '_raw.npy'
    raw_meta_path = raw_path.replace('.npy', '_meta.json')
    if os.path.exists(raw_path) and os.path.exists(raw_meta_path) and s1_model is not None:
        import json
        raw_xyz = np.load(raw_path).astype(np.float32)
        with open(raw_meta_path) as f:
            meta = json.load(f)
        sw_glosses, sw_conf = _sliding_window_stage1(
            raw_xyz, meta['l_ever'], meta['r_ever'], meta['face_ever'],
            s1_model, s1_idx_to_label
        )
        if sw_glosses:
            print(f"  [Sliding Window S1] {' '.join(sw_glosses)} (conf={sw_conf:.4f})")
            candidates.append((sw_glosses, sw_conf, 'sliding-window'))

    base_path = npy_path.replace('.npy', '')
    full_path = f"{base_path}_full.npy"
    if os.path.exists(full_path):
        full_data = np.load(full_path).astype(np.float32)
        if full_data.shape[-1] == 10:
            full_data = compute_bone_features(full_data)
        n_clips = full_data.shape[0] // 32
        s2_full_glosses, s2_full_conf = _run_stage2_ctc(full_data, s2_model, idx_to_gloss)
        if s2_full_glosses:
            print(f"  [Stage 2, full ({n_clips} auto-clips)] {' '.join(s2_full_glosses)} (conf={s2_full_conf:.4f})")
            candidates.append((s2_full_glosses, s2_full_conf, f'stage2-full-{n_clips}clip'))

    # --- Hypothesis 4+: Stage 2 CTC on segmented hypotheses (n=2,3,4) ---
    for n in range(2, 5):
        hyp_path = f"{base_path}_n{n}.npy"
        if not os.path.exists(hyp_path):
            continue
        cont_data = np.load(hyp_path).astype(np.float32)
        if cont_data.shape[-1] == 10:
            cont_data = compute_bone_features(cont_data)
        n_clips = cont_data.shape[0] // 32
        s2_glosses, s2_conf = _run_stage2_ctc(cont_data, s2_model, idx_to_gloss)
        tag = f'stage2-n{n}'
        if s2_glosses:
            print(f"  [Stage 2, n={n}] {' '.join(s2_glosses)} (conf={s2_conf:.4f})")
            candidates.append((s2_glosses, s2_conf, tag))

    # --- Pick best hypothesis ---
    best_glosses, best_conf, best_source = [], 0.0, 'none'
    for glosses, conf, source in candidates:
        if conf > best_conf:
            best_glosses, best_conf, best_source = glosses, conf, source

    if not best_glosses:
        best_glosses = [s1_gloss] if s1_gloss else []

    gloss_str = " ".join(best_glosses) if best_glosses else "(no signs detected)"
    print(f"\n  Winner: [{best_source}] {gloss_str} ({best_conf:.4f})")

    # Stage 3: Translation
    print("\n[STAGE 3] Translation...")
    if not best_glosses:
        translation = "[No signs detected]"
    else:
        prompt = f"Translate this ASL gloss to natural conversational English: {' '.join(best_glosses)}"
        inputs = s3_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = s3_model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
        translation = s3_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print(f"  File    : {basename}")
    print(f"  Glosses : {gloss_str}")
    print(f"  English : {translation}")
    print("=" * 60)

    return {"file": basename, "glosses": best_glosses, "source": best_source, "english": translation}


def main():
    parser = argparse.ArgumentParser(description="SLT Inference (.npy → English)")
    parser.add_argument("input", help=".npy file or folder of .npy files")
    parser.add_argument("--output", default="/app/output", help="Output directory")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find .npy files
    if os.path.isdir(args.input):
        npys = sorted(glob.glob(os.path.join(args.input, "*.npy")))
        print(f"Found {len(npys)} .npy files in {args.input}")
    else:
        npys = [args.input]

    if not npys:
        print("No .npy files found.")
        sys.exit(1)

    # Load models once
    s1_model, s1_idx_to_label, s2_model, s3_model, s3_tokenizer, idx_to_gloss = load_models()

    # Process each file
    results = []
    for npy_path in npys:
        if not os.path.exists(npy_path):
            print(f"\nERROR: Not found: {npy_path}")
            continue
        try:
            t0 = time.time()
            result = run_inference(npy_path, s1_model, s1_idx_to_label, s2_model, s3_model, s3_tokenizer, idx_to_gloss)
            result["time_sec"] = round(time.time() - t0, 1)
            results.append(result)
        except Exception as e:
            print(f"\nERROR processing {npy_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"file": os.path.basename(npy_path), "error": str(e)})

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  FAIL  {r['file']}: {r['error']}")
        else:
            print(f"  OK    {r['file']}: {r['english']} ({r['time_sec']}s)")


if __name__ == "__main__":
    main()
