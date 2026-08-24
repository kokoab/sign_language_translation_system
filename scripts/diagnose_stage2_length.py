"""Test how Stage 2 behavior changes with input length.
Takes one working video (HELLO_HOW_YOU_training.mp4 that gave correct HELLO HOW YOU)
and feeds it at various trim points to see when decoding breaks.
"""
import os, sys, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))

import numpy as np
import torch
from model_v16 import SLTStage2V16CTC
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
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt_path = '/Users/frnzlo/Downloads/results (1)/models/output_stage2_v16_phrases/stage2_best_model.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SLTStage2V16CTC(
        vocab_size=ckpt.get('vocab_size', 311), stage1_ckpt=None,
        in_channels=ckpt.get('in_channels', 5), dim=ckpt.get('d_model', 384),
    ).to(device)
    sd = ckpt.get('ema_shadow') or ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    g2i = ckpt.get('label_to_idx') or ckpt.get('gloss_to_idx')
    i2g = {int(v): k for k, v in g2i.items()}

    # Extract the full working video once
    path = 'sample_videos/HELLO_HOW_YOU_training.mp4'
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    print(f'Loaded {len(frames)} source frames from {path}')

    arr = extract_continuous_v16(frames, segment_len=28).astype(np.float32)
    T = arr.shape[0]
    print(f'Extracted {T} frames ({T//32} clips)\n')

    # Test at various clip counts (truncating)
    print(f'{"Clips":>6} {"T":>4}  Output')
    print('-' * 70)
    for num_clips in [1, 2, 3, 4, 5, 6, 7]:
        if num_clips * 32 > T:
            break
        x_trim = arr[:num_clips * 32]
        x = torch.from_numpy(x_trim)[None, ...].to(device)
        with torch.no_grad():
            logits, _ = model(x)
        decoded = [i2g.get(i, f'<{i}>') for i in ctc_decode(logits, blank=0)]
        print(f'{num_clips:>6} {num_clips*32:>4}  {" ".join(decoded) if decoded else "(empty)"}')

    # Also test taking different START positions (2-clip sliding window)
    print(f'\n{"="*70}')
    print('Sliding 2-clip window across the video:')
    print('-' * 70)
    for start in range(0, T - 64 + 1, 32):
        x_win = arr[start:start + 64]
        x = torch.from_numpy(x_win)[None, ...].to(device)
        with torch.no_grad():
            logits, _ = model(x)
        decoded = [i2g.get(i, f'<{i}>') for i in ctc_decode(logits, blank=0)]
        print(f'frames [{start:3d}:{start+64:3d}]  {" ".join(decoded) if decoded else "(empty)"}')


if __name__ == '__main__':
    main()
