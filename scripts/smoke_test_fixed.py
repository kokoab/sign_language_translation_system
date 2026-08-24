"""Smoke test: run one training epoch of train_stage_2_v16_fixed.py locally to catch errors."""
import sys, os, json
sys.path.insert(0, 'src_v16')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random

from model_v16 import SLTStage2V16CTC
from train_stage_2_v16 import (
    SyntheticCTCDataset, RealPhraseCTCDataset, collate_ctc,
    compute_wer, ctc_greedy_decode,
)
from train_stage_2_v16_fixed import PartialPhraseDataset, IsolatedSignDataset

print('[1] Loading datasets...')
with open('models/manifest_v16.json') as f: g2i = json.load(f)

# Small synthetic set
synth = SyntheticCTCDataset('src_v16/ASL_landmarks_v16', split='train',
                             manifest_path='models/manifest_v16.json', num_sequences=100)
print(f'  synth: {len(synth)}')

# Phrase split
base = RealPhraseCTCDataset('src_v16/ASL_phrases_v16', g2i)
idx = list(range(len(base.files)))
random.Random(42).shuffle(idx)
n_val = max(1, int(len(idx) * 0.15))
train_idx = idx[n_val:n_val+30]  # take just 30 for smoke test
val_idx = idx[:5]

pds = PartialPhraseDataset('src_v16/ASL_phrases_v16', g2i, indices=train_idx)
pds_val = PartialPhraseDataset('src_v16/ASL_phrases_v16', g2i, indices=val_idx)
print(f'  partial phrase: {len(pds)} train + {len(pds_val)} val')

iso = IsolatedSignDataset('src_v16/ASL_landmarks_v16', 'models/manifest_v16.json',
                           g2i, max_samples=50, split='train')
iso_val = IsolatedSignDataset('src_v16/ASL_landmarks_v16', 'models/manifest_v16.json',
                               g2i, max_samples=20, split='val')
print(f'  isolated: {len(iso)} train + {len(iso_val)} val')

train_ds = ConcatDataset([synth, pds, iso])
# Val set: just synth + a bit of phrase + iso
synth_val = SyntheticCTCDataset('src_v16/ASL_landmarks_v16', split='val',
                                 manifest_path='models/manifest_v16.json', num_sequences=30)
val_ds = ConcatDataset([synth_val, pds_val, iso_val])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                           collate_fn=collate_ctc, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                         collate_fn=collate_ctc, num_workers=0)

print(f'\n[2] Building model...')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = SLTStage2V16CTC(vocab_size=synth.vocab_size,
                         stage1_ckpt='src_v16/output_v16_d384/best_model.pth',
                         in_channels=5, dim=384).to(device)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

print(f'\n[3] One training step...')
model.train()
for x, targets, in_lens, tgt_lens in train_loader:
    x, targets = x.to(device), targets.to(device)
    in_lens, tgt_lens = in_lens.to(device), tgt_lens.to(device)
    logits, inter_logits = model(x)
    log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
    loss = ctc_loss_fn(log_probs, targets, in_lens, tgt_lens)
    if inter_logits is not None:
        inter_lp = F.log_softmax(inter_logits, dim=-1).permute(1, 0, 2)
        loss = loss + 0.1 * ctc_loss_fn(inter_lp, targets, in_lens, tgt_lens)
    loss.backward()
    print(f'  train loss: {loss.item():.4f}  x.shape={tuple(x.shape)}  targets={tgt_lens.tolist()}')
    optimizer.step()
    optimizer.zero_grad()
    break

print(f'\n[4] Validation loop (the part that was failing)...')
model.eval()
wer_total = 0.0
wer_count = 0
val_loss = 0.0
with torch.no_grad():
    for x, targets, in_lens, tgt_lens in val_loader:
        x, targets = x.to(device), targets.to(device)
        in_lens, tgt_lens = in_lens.to(device), tgt_lens.to(device)
        logits, _ = model(x)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        loss = ctc_loss_fn(log_probs, targets, in_lens, tgt_lens)
        val_loss += loss.item()

        # WER (batched)
        probs = F.softmax(logits, dim=-1)
        decoded_all = ctc_greedy_decode(probs, blank=0)
        refs = []
        ti = 0
        for b in range(x.shape[0]):
            L = int(tgt_lens[b])
            refs.append(targets[ti:ti+L].cpu().tolist())
            ti += L
        batch_wer = compute_wer(decoded_all, refs)
        total_ref_words = sum(len(r) for r in refs)
        wer_total += batch_wer * total_ref_words / 100.0
        wer_count += total_ref_words
        print(f'  batch WER: {batch_wer:.2f}%  (refs: {[len(r) for r in refs]}, preds: {[len(d) for d in decoded_all]})')

val_wer = 100.0 * wer_total / max(1, wer_count)
print(f'\n[5] Final: val_loss={val_loss/len(val_loader):.4f}  val_wer={val_wer:.2f}%')
print('\nALL CHECKS PASS — training script should run without errors')
