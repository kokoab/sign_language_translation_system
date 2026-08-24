"""Test a single video with v15 pipeline (DSGCN-V14 Stage 1 + Stage 2 CTC).
Uses Apple Vision extraction like v16, but converts to 16ch features.
"""
import os, sys, cv2, types
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

# Fake mediapipe to prevent import errors
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

import numpy as np
import torch
import torch.nn.functional as F
from train_stage_1 import compute_bone_features_np
from model_v14 import SLTStage1V14
from train_stage_2 import SLTStage2CTC
from extract_apple_vision import extract_frames_continuous, extract_frames_isolated


def ctc_decode(logits, blank=0):
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if logits.ndim == 3:
        logits = logits[0]
    preds = logits.argmax(axis=-1)
    out, prev = [], -1
    for p in preds:
        if p != prev and p != blank:
            out.append(int(p))
        prev = p
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'sample_videos/HELLO_HOW_YOU_training.mp4'
    print(f'Testing with v15: {path}')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load Stage 1 v15
    s1_ckpt = torch.load('models/output_v15_clean/best_model.pth', map_location=device, weights_only=False)
    s1 = SLTStage1V14(
        num_classes=s1_ckpt['num_classes'],
        d_model=s1_ckpt['d_model'],
        use_arcface=True,
    ).to(device)
    sd = s1_ckpt.get('ema_shadow', s1_ckpt['model_state_dict'])
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s1.load_state_dict(sd, strict=False); s1.eval()
    l2i = s1_ckpt['label_to_idx']
    s1_i2l = {int(v): k for k, v in l2i.items()}
    print(f'Stage 1 v15 loaded: {len(s1_i2l)} classes, d_model={s1_ckpt["d_model"]}')

    # Load Stage 2 v15
    s2_path = 'models/output_stage2_v15_reextracted/stage2_best_model.pth'
    if not os.path.exists(s2_path):
        s2_path = 'models/output/stage2_best_model.pth'
    s2_ckpt = torch.load(s2_path, map_location=device, weights_only=False)
    s2 = SLTStage2CTC(
        vocab_size=s2_ckpt.get('vocab_size', s2_ckpt.get('num_classes', s1_ckpt['num_classes']) + 1),
        d_model=s2_ckpt.get('d_model', 384),
        encoder_type=s2_ckpt.get('encoder_type', 'DS-GCN-TCN-v15'),
    ).to(device)
    sd = s2_ckpt.get('ema_shadow') or s2_ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s2.load_state_dict(sd, strict=False); s2.eval()
    print(f'Stage 2 v15 loaded: {s2_path}')

    # Load video
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    print(f'Source: {len(frames)} frames')

    # Extract with AV (v15 format: 10ch)
    arr_10ch = extract_frames_continuous(frames)
    if arr_10ch is None:
        print('Extraction failed'); return
    arr_10ch = arr_10ch.astype(np.float32)
    T = arr_10ch.shape[0]
    nc = T // 32
    print(f'Extracted: {arr_10ch.shape} ({nc} clips)')

    # Compute bone features → 16ch
    arr_16ch = compute_bone_features_np(arr_10ch)
    print(f'With bone features: {arr_16ch.shape}')

    # Stage 1 per-clip
    print(f'\nStage 1 v15 (per-clip top-3):')
    for i in range(nc):
        clip = arr_16ch[i*32:(i+1)*32]
        x = torch.from_numpy(clip)[None, ...].to(device)
        with torch.no_grad():
            logits = s1(x)
        probs = torch.softmax(logits, dim=-1)[0]
        top3 = probs.topk(3)
        preds = ', '.join(f'{s1_i2l[idx.item()]}({conf.item()*100:.0f}%)'
                          for conf, idx in zip(top3.values, top3.indices))
        print(f'  clip {i+1} frames {i*32}-{(i+1)*32}: {preds}')

    # Stage 2 v15 on full sequence
    x = torch.from_numpy(arr_16ch)[None, ...].to(device)
    x_lens = torch.tensor([T], dtype=torch.long, device=device)
    with torch.no_grad():
        s2_logits, s2_lens = s2(x, x_lens)
    log_probs = torch.log_softmax(s2_logits[0], dim=-1).cpu().numpy()
    preds = log_probs[:s2_lens[0].item()].argmax(axis=-1)
    decoded = []
    prev = 0
    for p in preds:
        if p != 0 and p != prev:
            decoded.append(s1_i2l.get(int(p), f'<{p}>'))
        prev = p
    print(f'\nStage 2 v15 ({nc} clips): {" ".join(decoded) if decoded else "(empty)"}')


if __name__ == '__main__':
    main()
