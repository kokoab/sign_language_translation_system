import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import time

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — FULL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════

_THUMB_MCP = 2; _THUMB_IP = 3; _THUMB_TIP = 4; _INDEX_MCP = 5; _INDEX_PIP = 6; _INDEX_TIP = 8
_MIDDLE_MCP = 9; _MIDDLE_PIP = 10; _MIDDLE_TIP = 12; _RING_MCP = 13; _RING_PIP = 14; _RING_TIP = 16
_PINKY_MCP = 17; _PINKY_PIP = 18; _PINKY_TIP = 20
N_GEO_FEATURES = 24 

def build_adjacency_matrices(num_nodes: int = 42) -> torch.Tensor:
    A_self = np.eye(num_nodes, dtype=np.float32)
    return torch.from_numpy(np.stack([A_self, A_self, A_self])) 

class DSGCNBlock(nn.Module):
    def __init__(self, C_in, C_out, temporal_kernel=3, dropout=0.1, num_groups=8):
        super().__init__()
        self.dw_weights = nn.Parameter(torch.ones(3, C_in))
        self.pointwise = nn.Linear(3 * C_in, C_out, bias=False)
        self.temporal_conv = nn.Conv1d(C_out, C_out, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=C_out, bias=False)
        self.temporal_norm = nn.GroupNorm(num_groups, C_out)
        self.norm = nn.LayerNorm(C_out)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Linear(C_in, C_out, bias=False) if C_in != C_out else nn.Identity()

    def forward(self, x, A):
        B, T, N, C = x.shape
        residual = self.residual(x)
        agg = torch.einsum('knm,btnc->kbtnc', A, x)                
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
    def forward(self, x):
        return x 

class DSGCNEncoder(nn.Module):
    def __init__(self, in_channels=10, d_model=256, nhead=8, num_transformer_layers=4, dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.register_buffer('A', build_adjacency_matrices(42))
        self.input_norm = nn.LayerNorm(in_channels)
        self.input_proj = nn.Sequential(nn.Linear(in_channels, 64), nn.LayerNorm(64), nn.GELU())
        self.gcn1 = DSGCNBlock(64,  128, temporal_kernel=3, dropout=dropout)
        self.gcn2 = DSGCNBlock(128, 128, temporal_kernel=3, dropout=dropout)
        self.gcn3 = DSGCNBlock(128, d_model, temporal_kernel=5, dropout=dropout)
        self.node_attn = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1))
        
        self.geo_norm = nn.LayerNorm(N_GEO_FEATURES)
        self.geo_proj = nn.Linear(d_model + N_GEO_FEATURES, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, 32, d_model))
        
        dp_rates = [drop_path_rate * i / max(num_transformer_layers - 1, 1) for i in range(num_transformer_layers)]
        self.transformer_layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for dp in dp_rates:
            self.transformer_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True
            ))
            self.drop_paths.append(DropPath(dp))
            
        self.transformer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _geo_dist(a, b): return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-6)

    def _compute_geo_features(self, xyz):
        d = self._geo_dist
        def get_hand_features(base):
            tips = [d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_TIP]), d(xyz[:,:,base+_INDEX_TIP], xyz[:,:,base+_MIDDLE_TIP]), d(xyz[:,:,base+_MIDDLE_TIP], xyz[:,:,base+_RING_TIP]), d(xyz[:,:,base+_RING_TIP], xyz[:,:,base+_PINKY_TIP]), d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_PINKY_TIP])]
            curls = [d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_TIP]) / (d(xyz[:,:,base+_THUMB_MCP], xyz[:,:,base+_THUMB_IP]) + 1e-4), d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_TIP]) / (d(xyz[:,:,base+_INDEX_MCP], xyz[:,:,base+_INDEX_PIP]) + 1e-4), d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_TIP]) / (d(xyz[:,:,base+_MIDDLE_MCP], xyz[:,:,base+_MIDDLE_PIP]) + 1e-4), d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_TIP]) / (d(xyz[:,:,base+_RING_MCP], xyz[:,:,base+_RING_PIP]) + 1e-4), d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_TIP]) / (d(xyz[:,:,base+_PINKY_MCP], xyz[:,:,base+_PINKY_PIP]) + 1e-4)]
            cross_idx_mid = xyz[:,:,base+_INDEX_TIP,0] - xyz[:,:,base+_MIDDLE_TIP,0]
            d_thumb_idxmcp = d(xyz[:,:,base+_THUMB_TIP], xyz[:,:,base+_INDEX_MCP])
            return tips + curls + [cross_idx_mid, d_thumb_idxmcp]
        return torch.stack(get_hand_features(0) + get_hand_features(21), dim=-1)

    def forward(self, x):
        B, T, N, C = x.shape
        xyz = x[:, :, :, :3] 
        h = self.input_proj(self.input_norm(x))
        h = self.gcn3(self.gcn2(self.gcn1(h, self.A), self.A), self.A)
        attn = F.softmax(self.node_attn(h).squeeze(-1), dim=2)
        h = (h * attn.unsqueeze(-1)).sum(dim=2)
        real_geo = self._compute_geo_features(xyz)
        h = self.geo_proj(torch.cat([h, self.geo_norm(real_geo)], dim=-1)) + self.pos_enc
        for layer, dp in zip(self.transformer_layers, self.drop_paths): 
            h = h + dp(layer(h) - h)
        return self.transformer_norm(h)

class SLTStage2CTC(nn.Module):
    def __init__(self, vocab_size, d_model=256, lstm_hidden=512, lstm_layers=2):
        super().__init__()
        self.encoder = DSGCNEncoder(in_channels=10, d_model=d_model)
        self.temporal_pool = nn.AdaptiveAvgPool1d(4)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, 
                            num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x):
        B, T, V, C = x.shape
        num_clips = T // 32
        if num_clips == 0: return None
        clips = x[:, :num_clips*32].view(B * num_clips, 32, V, C)
        enc_out = self.encoder(clips).permute(0, 2, 1) 
        pooled = self.temporal_pool(enc_out).permute(0, 2, 1) 
        seq_features = pooled.reshape(B, num_clips * 4, -1)
        lstm_out, _ = self.lstm(seq_features)
        return self.classifier(lstm_out)

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — BATCH WER EVALUATION
# ══════════════════════════════════════════════════════════════════

def compute_edit_distance(ref_words, hyp_words):
    """Calculates Levenshtein distance between two lists of words."""
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1): d[i, 0] = i
    for j in range(len(hyp_words) + 1): d[0, j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost = 0
            else:
                cost = 1
            d[i, j] = min(d[i-1, j] + 1,      # Deletion
                          d[i, j-1] + 1,      # Insertion
                          d[i-1, j-1] + cost) # Substitution
    return d[len(ref_words), len(hyp_words)]

def run_batch_evaluation():
    S2_PATH = "weights/stage2_best_model.pth"
    DATA_DIR = "ASL_landmarks_float16/"
    NUM_EVAL_SENTENCES = 200  # Number of random sentences to generate
    
    checkpoint = torch.load(S2_PATH, map_location='cpu', weights_only=False)
    idx_to_gloss = checkpoint['idx_to_gloss']
    
    model = SLTStage2CTC(vocab_size=len(idx_to_gloss))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    ema_shadow = checkpoint.get('ema_shadow')
    if ema_shadow:
        for name, param in model.named_parameters():
            if name in ema_shadow:
                param.data.copy_(ema_shadow[name])

    model.eval()
    print("✅ Model loaded! Starting Batch Evaluation...\n" + "="*50)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    if len(files) < 10: return print("❌ Not enough .npy files.")
    
    total_edit_distance = 0
    total_reference_words = 0
    perfect_sentences = 0
    
    start_time = time.time()
    
    for test_num in range(1, NUM_EVAL_SENTENCES + 1):
        # Generate a random sentence length between 2 and 5 words
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
                
        # Calculate Errors
        edits = compute_edit_distance(target_words, decoded)
        total_edit_distance += edits
        total_reference_words += len(target_words)
        
        if edits == 0:
            perfect_sentences += 1
            
        # Print progress every 20 sentences
        if test_num % 20 == 0:
            current_wer = (total_edit_distance / total_reference_words) * 100
            print(f"🔄 Processed {test_num}/{NUM_EVAL_SENTENCES} sentences | Current WER: {current_wer:.2f}%")

    # Final Stats
    final_wer = (total_edit_distance / total_reference_words) * 100
    perfect_ratio = (perfect_sentences / NUM_EVAL_SENTENCES) * 100
    elapsed = time.time() - start_time
    
    print("\n" + "═"*50)
    print(" 📊 BATCH EVALUATION RESULTS")
    print("═"*50)
    print(f"Total Sentences Tested : {NUM_EVAL_SENTENCES}")
    print(f"Total Words Tested     : {total_reference_words}")
    print(f"Perfectly Translated   : {perfect_sentences} ({perfect_ratio:.1f}%)")
    print(f"Time Taken             : {elapsed:.1f} seconds")
    print(f"🎯 FINAL WER           : {final_wer:.2f}%")
    print("═"*50)

if __name__ == "__main__":
    run_batch_evaluation()