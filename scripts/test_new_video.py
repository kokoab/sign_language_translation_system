"""Test a single video with v16 pipeline (Stage 1 + Stage 2 phrase-trained)."""
import os, sys, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))

import numpy as np
import torch
from model_v16 import SLTStage1V16, SLTStage2V16CTC
from extract_v16 import extract_continuous_v16


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
    print(f'Testing: {path}')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load Stage 1
    s1_ckpt = torch.load('src_v16/output_v16_d384/best_model.pth', map_location=device, weights_only=False)
    s1 = SLTStage1V16(
        num_classes=s1_ckpt['num_classes'],
        in_channels=s1_ckpt['in_channels'],
        dim=s1_ckpt['d_model'],
    ).to(device)
    sd = s1_ckpt.get('ema_shadow', s1_ckpt['model_state_dict'])
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s1.load_state_dict(sd, strict=False); s1.eval()
    s1_i2l = {v: k for k, v in s1_ckpt['label_to_idx'].items()}

    # Load Stage 2 (fixed — anti-hallucination)
    s2_ckpt = torch.load('/Users/frnzlo/Downloads/results (2)/models/output_stage2_v16_fixed/stage2_best_model.pth',
                         map_location=device, weights_only=False)
    s2 = SLTStage2V16CTC(
        vocab_size=s2_ckpt.get('vocab_size', 311), stage1_ckpt=None,
        in_channels=s2_ckpt.get('in_channels', 5), dim=s2_ckpt.get('d_model', 384),
    ).to(device)
    sd = s2_ckpt.get('ema_shadow') or s2_ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s2.load_state_dict(sd, strict=False); s2.eval()
    g2i = s2_ckpt.get('label_to_idx') or s2_ckpt.get('gloss_to_idx')
    s2_i2g = {int(v): k for k, v in g2i.items()}

    # Load video
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    print(f'Source: {len(frames)} frames')

    # Extract
    arr = extract_continuous_v16(frames, segment_len=28)
    if arr is None:
        print('Extraction failed'); return
    arr = arr.astype(np.float32)
    T = arr.shape[0]
    nc = T // 32
    print(f'Extracted: {arr.shape} ({nc} clips)')

    # Pad if needed
    if T % 32 != 0:
        pad = np.zeros((((T+31)//32)*32 - T, 61, 5), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
        T = arr.shape[0]; nc = T // 32

    # Stage 1 per-clip predictions
    print(f'\nStage 1 (per-clip top-3):')
    print(f'{"Clip":>5} {"Frames":>10}  Top-3 predictions')
    for i in range(nc):
        clip = arr[i*32:(i+1)*32]
        x = torch.from_numpy(clip)[None, ...].to(device)
        with torch.no_grad():
            logits = s1(x)
        probs = torch.softmax(logits, dim=-1)[0]
        top3 = probs.topk(3)
        preds = ', '.join(f'{s1_i2l[idx.item()]}({conf.item()*100:.0f}%)'
                          for conf, idx in zip(top3.values, top3.indices))
        print(f'{i+1:>5} {i*32:>4}-{(i+1)*32:>4}  {preds}')

    # Stage 2 full
    x = torch.from_numpy(arr)[None, ...].to(device)
    with torch.no_grad():
        s2_logits, _ = s2(x)
    decoded = [s2_i2g.get(i, f'<{i}>') for i in ctc_decode(s2_logits, blank=0)]
    print(f'\nStage 2 ({nc} clips): {" ".join(decoded) if decoded else "(empty)"}')


if __name__ == '__main__':
    main()
