"""
SLT Test — Batch WER Evaluation
Architecture matches train_stage_1.py / train_stage_2.py exactly (47-node, 76 geo, face-aware, 16-ch bone).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import time

# ======================================================================
#  SECTION 1 -- FULL ARCHITECTURE (mirrors train_stage_1.py / train_stage_2.py)
# ======================================================================

# Base edges for a single hand
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
NUM_NODES = 47

_THUMB_MCP = 2; _THUMB_IP = 3; _THUMB_TIP = 4
_INDEX_MCP = 5; _INDEX_PIP = 6; _INDEX_TIP = 8
_MIDDLE_MCP = 9; _MIDDLE_PIP = 10; _MIDDLE_TIP = 12
_RING_MCP = 13; _RING_PIP = 14; _RING_TIP = 16
_PINKY_MCP = 17; _PINKY_PIP = 18; _PINKY_TIP = 20

# 12 per hand x 2 = 24, + 10 hand-to-face = 34 (base)
# + 30 joint angles + 6 palm orientation + 6 finger spread = 76 total
N_GEO_FEATURES = 76

# Phase 4A: Joint angle triplets (joint at middle vertex)
_JOINT_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),
    (0,5,6),(5,6,7),(6,7,8),
    (0,9,10),(9,10,11),(10,11,12),
    (0,13,14),(13,14,15),(14,15,16),
    (0,17,18),(17,18,19),(18,19,20),
]

_SPREAD_MCPS = [5, 9, 13, 17]

# Phase 1C: Bone feature pairs (parent -> child)
_BONE_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]


def compute_bone_features_np(x_np):
    """Numpy version: x: [T, 47, 10] -> [T, 47, 16]"""
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
        # Phase 1A: Learnable adjacency residual
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
            self.transformer_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            ))
            self.drop_paths.append(DropPath(dp))
        self.transformer_norm = nn.LayerNorm(d_model)
        # Phase 2B: Causal masking
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
        if face_mask is not None:
            face_gate = face_mask[:, :, 0, 0]
            face_feats = [f * face_gate for f in face_feats]

        # Phase 4A: Joint angles — 15 per hand x 2 = 30 features
        angle_feats = []
        for base in [0, 21]:
            for p, j, c in _JOINT_TRIPLETS:
                v1 = xyz[:, :, base+p, :] - xyz[:, :, base+j, :]
                v2 = xyz[:, :, base+c, :] - xyz[:, :, base+j, :]
                cos_a = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
                angle_feats.append(torch.acos(cos_a.clamp(-1 + 1e-6, 1 - 1e-6)))

        # Phase 4A: Palm orientation — 3 x 2 = 6 features
        palm_feats = []
        for base in [0, 21]:
            wrist = xyz[:, :, base, :]
            v1 = xyz[:, :, base + 5, :] - wrist
            v2 = xyz[:, :, base + 17, :] - wrist
            normal = torch.cross(v1, v2, dim=-1)
            normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
            for i in range(3):
                palm_feats.append(normal[..., i])

        # Phase 4A: Finger spread — 3 x 2 = 6 features
        spread_feats = []
        for base in [0, 21]:
            wrist = xyz[:, :, base, :]
            for i in range(len(_SPREAD_MCPS) - 1):
                v1 = xyz[:, :, base + _SPREAD_MCPS[i], :] - wrist
                v2 = xyz[:, :, base + _SPREAD_MCPS[i+1], :] - wrist
                cos_s = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-6)
                spread_feats.append(torch.acos(cos_s.clamp(-1 + 1e-6, 1 - 1e-6)))

        return torch.stack(get_hand_features(0) + get_hand_features(21) + face_feats + angle_feats + palm_feats + spread_feats, dim=-1)

    def forward(self, x):
        xyz = x[:, :, :, :3]
        face_mask = x[:, :, 42:47, 9:10]
        h = self.input_proj(self.input_norm(x))
        h[:, :, 42:47, :] = h[:, :, 42:47, :] * face_mask
        h = self.gcn3(self.gcn2(self.gcn1(h, self.A), self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)
        h = self.geo_proj(torch.cat([h, self.geo_norm(self._compute_geo_features(xyz, face_mask))], dim=-1)) + self.pos_enc
        mask = self.causal_mask[:h.size(1), :h.size(1)] if self.causal else None
        for layer, dp in zip(self.transformer_layers, self.drop_paths):
            h = h + dp(layer(h, src_mask=mask) - h)
        return self.transformer_norm(h)


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
        self.fuse = nn.Sequential(nn.Linear(d_model * 3, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(out_tokens)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        min_t = min(o.shape[2] for o in outs)
        cat = torch.cat([o[:, :, :min_t] for o in outs], dim=1)
        cat = cat.permute(0, 2, 1)
        fused = self.fuse(cat).permute(0, 2, 1)
        return self.pool(fused)


class SLTStage2CTC(nn.Module):
    def __init__(self, vocab_size, d_model=256, lstm_hidden=512, lstm_layers=2):
        super().__init__()
        self.encoder = DSGCNEncoder(in_channels=16, d_model=d_model)
        self.temporal_pool = MultiScaleTCN(d_model, 4)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x):
        B = x.shape[0]
        V, C = x.shape[2], x.shape[3]
        num_clips = x.shape[1] // 32
        if num_clips == 0: return None
        clips = x[:, :num_clips*32].view(B * num_clips, 32, V, C)
        enc_out = self.encoder(clips).permute(0, 2, 1)
        pooled = self.temporal_pool(enc_out).permute(0, 2, 1)
        seq_features = pooled.reshape(B, num_clips * 4, -1)
        lstm_out, _ = self.lstm(seq_features)
        return self.classifier(lstm_out)


# ======================================================================
#  SECTION 2 -- BATCH WER EVALUATION
# ======================================================================

def compute_edit_distance(ref_words, hyp_words):
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1): d[i, 0] = i
    for j in range(len(hyp_words) + 1): d[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)
    return d[len(ref_words), len(hyp_words)]


def run_batch_evaluation():
    S2_PATH = "weights/stage2_best_model.pth"
    DATA_DIR = "ASL_landmarks_float16/"
    NUM_EVAL_SENTENCES = 200

    checkpoint = torch.load(S2_PATH, map_location='cpu', weights_only=False)
    idx_to_gloss = checkpoint['idx_to_gloss']

    model = SLTStage2CTC(vocab_size=len(idx_to_gloss))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    ema_shadow = checkpoint.get('ema_shadow')
    if ema_shadow:
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name])

    model.eval()
    print("Model loaded! Starting Batch Evaluation...\n" + "="*50)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    if len(files) < 10:
        print("Not enough .npy files.")
        return

    total_edit_distance = 0
    total_reference_words = 0
    perfect_sentences = 0
    start_time = time.time()

    for test_num in range(1, NUM_EVAL_SENTENCES + 1):
        seq_len = random.randint(2, 5)
        test_files = random.sample(files, seq_len)
        target_words = [f.split('_')[0] for f in test_files]

        clips = []
        for filename in test_files:
            clip = np.load(os.path.join(DATA_DIR, filename)).astype(np.float32)
            if clip.shape[0] < 32:
                clip = np.pad(clip, ((0, 32 - clip.shape[0]), (0,0), (0,0)))
            elif clip.shape[0] > 32:
                clip = clip[:32]
            # Apply bone features to get 16-channel input
            clip = compute_bone_features_np(clip)
            clips.append(clip)

        sentence_landmarks = np.concatenate(clips, axis=0)
        input_tensor = torch.from_numpy(sentence_landmarks).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            if logits is None: continue
            pred_ids = torch.argmax(logits, dim=-1).squeeze(0).numpy()
            decoded = []
            last = -1
            for idx in pred_ids:
                if idx != last and idx != 0:
                    word = idx_to_gloss.get(int(idx), idx_to_gloss.get(str(idx), f"UNKNOWN_{idx}"))
                    decoded.append(word)
                last = idx

        edits = compute_edit_distance(target_words, decoded)
        total_edit_distance += edits
        total_reference_words += len(target_words)
        if edits == 0:
            perfect_sentences += 1
        if test_num % 20 == 0:
            current_wer = (total_edit_distance / total_reference_words) * 100
            print(f"Processed {test_num}/{NUM_EVAL_SENTENCES} sentences | Current WER: {current_wer:.2f}%")

    final_wer = (total_edit_distance / total_reference_words) * 100
    perfect_ratio = (perfect_sentences / NUM_EVAL_SENTENCES) * 100
    elapsed = time.time() - start_time

    print("\n" + "="*50)
    print("BATCH EVALUATION RESULTS")
    print("="*50)
    print(f"Total Sentences Tested : {NUM_EVAL_SENTENCES}")
    print(f"Total Words Tested     : {total_reference_words}")
    print(f"Perfectly Translated   : {perfect_sentences} ({perfect_ratio:.1f}%)")
    print(f"Time Taken             : {elapsed:.1f} seconds")
    print(f"FINAL WER              : {final_wer:.2f}%")
    print("="*50)


if __name__ == "__main__":
    run_batch_evaluation()
