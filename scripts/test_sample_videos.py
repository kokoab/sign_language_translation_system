"""Test sample videos end-to-end: extract with Apple Vision, run Stage 1 + Stage 2.
Uses the new phrase-trained Stage 2 checkpoint.
"""
import os, sys, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))

import numpy as np
import torch
from model_v16 import SLTStage1V16, SLTStage2V16CTC
from extract_v16 import extract_continuous_v16


def ctc_greedy_decode(logits, blank=0):
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
    print(f'Device: {device}')

    # Load Stage 1
    s1_ckpt_path = '/Users/frnzlo/Documents/machine_learning/SLT/src_v16/output_v16_d384/best_model.pth'
    print(f'Loading Stage 1: {s1_ckpt_path}')
    s1_ckpt = torch.load(s1_ckpt_path, map_location=device, weights_only=False)
    s1_model = SLTStage1V16(
        num_classes=s1_ckpt['num_classes'],
        in_channels=s1_ckpt['in_channels'],
        dim=s1_ckpt['d_model'],
    ).to(device)
    sd = s1_ckpt.get('ema_shadow', s1_ckpt['model_state_dict'])
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s1_model.load_state_dict(sd, strict=False)
    s1_model.eval()
    s1_idx_to_label = {v: k for k, v in s1_ckpt['label_to_idx'].items()}

    # Load Stage 2 (new phrase-trained checkpoint)
    s2_ckpt_path = '/Users/frnzlo/Downloads/results (1)/models/output_stage2_v16_phrases/stage2_best_model.pth'
    print(f'Loading Stage 2: {s2_ckpt_path}')
    s2_ckpt = torch.load(s2_ckpt_path, map_location=device, weights_only=False)
    s2_model = SLTStage2V16CTC(
        vocab_size=s2_ckpt.get('vocab_size', 311),
        stage1_ckpt=None,
        in_channels=s2_ckpt.get('in_channels', 5),
        dim=s2_ckpt.get('d_model', 384),
    ).to(device)
    sd = s2_ckpt.get('ema_shadow') or s2_ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s2_model.load_state_dict(sd, strict=False)
    s2_model.eval()
    g2i = s2_ckpt.get('label_to_idx') or s2_ckpt.get('gloss_to_idx')
    s2_idx_to_gloss = {int(v): k for k, v in g2i.items()}

    print()

    # Process each video
    video_dir = 'sample_videos'
    videos = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.mov'))])

    for vid in videos:
        path = os.path.join(video_dir, vid)
        print(f'━━━ {vid} ━━━')

        # Load frames
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
        print(f'  Loaded {len(frames)} frames')

        # Extract
        arr = extract_continuous_v16(frames, segment_len=28)
        if arr is None:
            print(f'  Extraction failed (no hands detected)')
            continue
        print(f'  Extracted: {arr.shape}')

        arr_f32 = arr.astype(np.float32)
        T = arr_f32.shape[0]

        # Stage 1 — take first 32 frames as isolated prediction
        x_iso = torch.from_numpy(arr_f32[:32])[None, ...].to(device)
        with torch.no_grad():
            s1_logits = s1_model(x_iso)
        probs = torch.softmax(s1_logits, dim=-1)[0]
        top5 = probs.topk(5)
        print(f'  Stage 1 (first 32 frames):')
        for conf, idx in zip(top5.values, top5.indices):
            print(f'    {s1_idx_to_label[idx.item()]:<25}  {conf.item()*100:.1f}%')

        # Stage 2 — full continuous sequence
        if T % 32 != 0:
            pad = np.zeros((((T + 31) // 32) * 32 - T, 61, 5), dtype=np.float32)
            arr_f32 = np.concatenate([arr_f32, pad], axis=0)

        x = torch.from_numpy(arr_f32)[None, ...].to(device)
        with torch.no_grad():
            s2_logits, _ = s2_model(x)
        decoded_idx = ctc_greedy_decode(s2_logits, blank=0)
        decoded = [s2_idx_to_gloss.get(i, f'<{i}>') for i in decoded_idx]

        num_clips = x.shape[1] // 32
        print(f'  Stage 2 ({num_clips} clips): {" ".join(decoded) if decoded else "(empty)"}')
        print()


if __name__ == '__main__':
    main()
