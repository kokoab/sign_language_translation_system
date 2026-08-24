"""Test a video through v15 AND v16 pipelines with full demo disambiguation rules.
Headless — no GUI, just prints results.
"""
import os, sys, cv2, types, warnings
warnings.filterwarnings('ignore')

# Fake mediapipe
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

sys.path.insert(0, 'src')
sys.path.insert(0, 'src_v16')
sys.path.insert(0, 'scripts')

import numpy as np
import torch

# ─── Disambiguation rules (from demo_classify.py) ─────────
def disambiguate(ctc_words, data_f32, ch_mask=9, ch_xyz=slice(0,3)):
    """Apply demo_classify.py's motion-based disambiguation rules.
    Works with both 10ch (v15, mask=ch9) and 5ch (v16, mask=ch3)."""
    n_clips = data_f32.shape[0] // 32
    fixed = []
    for wi, word in enumerate(ctc_words):
        seg_idx = min(int(wi / max(len(ctc_words), 1) * n_clips), n_clips - 1)
        seg = data_f32[seg_idx*32:(seg_idx+1)*32]
        lh_m = seg[:, 0, ch_mask].mean()
        rh_m = seg[:, 21, ch_mask].mean()
        wr = 0 if lh_m > rh_m else 21
        ns = 42
        end_f = min(seg.shape[0]-2, 28)
        yt = seg[end_f, wr, 1] - seg[4, wr, 1]
        fd_m = np.linalg.norm(seg[seg.shape[0]//2, wr, ch_xyz] - seg[seg.shape[0]//2, ns, ch_xyz])
        fd_e = np.linalg.norm(seg[end_f, wr, ch_xyz] - seg[end_f, ns, ch_xyz])
        hh = seg[:, wr, 1].mean() - seg[:, ns, 1].mean()
        motion = np.abs(np.diff(seg[:, wr, ch_xyz], axis=0)).sum()
        x_osc = np.abs(np.diff(seg[:, wr, 0])).sum()
        y_osc = np.abs(np.diff(seg[:, wr, 1])).sum()

        if word in ('SIX', 'W'):
            word = 'SIX' if motion > 0.4 and hh > 1.4 else 'W' if motion < 0.3 else word
        elif word in ('O', 'ZERO'):
            word = 'ZERO' if hh > 1.0 else 'O'
        elif word in ('SCHOOL', 'COOK'):
            word = 'COOK' if yt > 0.1 else 'SCHOOL'
        elif word in ('YES', 'S'):
            word = 'YES' if y_osc > 0.5 and motion > 0.5 else 'S' if motion < 0.4 else word
        elif word in ('NO', 'Z'):
            word = 'NO' if yt < -0.2 and x_osc < 0.5 else 'Z' if x_osc > 0.7 and yt > 0.2 else word
        elif word in ('PLEASE', 'SORRY'):
            tips = [wr+4, wr+8, wr+12, wr+16, wr+20]
            openness = np.mean([np.linalg.norm(seg[:, t, ch_xyz] - seg[:, wr, ch_xyz], axis=-1).mean() for t in tips])
            word = 'PLEASE' if openness > 1.5 else 'SORRY'
        elif word in ('FATHER', 'MOTHER'):
            word = 'FATHER' if hh < 1.0 else 'MOTHER'
        elif word in ('GOOD', 'THANKYOU'):
            # palm scale check (v16 ch4) or motion check
            if data_f32.shape[-1] >= 5:
                ps = seg[:, wr:wr+21, 4].mean()
                word = 'THANKYOU' if ps > 1.1 else 'GOOD'
            else:
                word = 'THANKYOU' if yt < -0.15 else 'GOOD'
        elif word in ('HELLO', 'HIS_HER'):
            # HELLO = wave near forehead, HIS_HER = flat hand push
            word = 'HELLO' if motion > 0.5 and hh > 0.8 else 'HIS_HER' if motion < 0.3 else word
        fixed.append(word)

    # Dedup consecutive
    deduped = []
    for w in fixed:
        if not deduped or deduped[-1] != w:
            deduped.append(w)
    return deduped


def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames


def test_v15(frames, device):
    """V15 pipeline: AV extraction (10ch) → bone features (16ch) → DSGCN-V14 → Stage 2 CTC"""
    print('\n' + '=' * 70)
    print('V15 PIPELINE (DSGCN-V14, 16ch, Apple Vision)')
    print('=' * 70)

    from extract_apple_vision import extract_frames_continuous
    from train_stage_1 import compute_bone_features_np
    from model_v14 import SLTStage1V14
    from train_stage_2 import SLTStage2CTC

    # Extract
    result = extract_frames_continuous(frames)
    if result is None:
        print('  Extraction failed'); return
    result = result.astype(np.float32)
    T = result.shape[0]
    nc = T // 32
    print(f'  Extracted: {result.shape} ({nc} clips)')

    # Load Stage 1
    s1_ckpt = torch.load('models/output_v15_clean/best_model.pth', map_location=device, weights_only=False)
    s1 = SLTStage1V14(num_classes=s1_ckpt['num_classes'], d_model=s1_ckpt['d_model'], use_arcface=True).to(device)
    sd = s1_ckpt.get('ema_shadow', s1_ckpt['model_state_dict'])
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s1.load_state_dict(sd, strict=False); s1.eval()
    l2i = s1_ckpt['label_to_idx']
    i2l = {int(v): k for k, v in l2i.items()}

    # Bone features
    result_16ch = compute_bone_features_np(result)
    x = torch.from_numpy(result_16ch).unsqueeze(0).to(device)

    # Per-clip Stage 1
    print(f'\n  Stage 1 v15 per-clip:')
    for ci in range(min(nc, 10)):
        clip = x[:, ci*32:(ci+1)*32]
        with torch.no_grad():
            logits = s1(clip)
        probs = torch.softmax(logits, dim=-1)[0]
        top3 = probs.topk(3)
        preds = ', '.join(f'{i2l[idx.item()]}({conf.item()*100:.0f}%)' for conf, idx in zip(top3.values, top3.indices))
        print(f'    clip {ci+1}: {preds}')

    # Stage 2
    s2_path = 'models/output_stage2_v15_reextracted/stage2_best_model.pth'
    if os.path.exists(s2_path):
        s2_ckpt = torch.load(s2_path, map_location=device, weights_only=False)
        s2 = SLTStage2CTC(
            vocab_size=s2_ckpt.get('vocab_size', s1_ckpt['num_classes'] + 1),
            d_model=s2_ckpt.get('d_model', 384),
            encoder_type=s2_ckpt.get('encoder_type', 'DS-GCN-TCN-v15'),
        ).to(device)
        sd = s2_ckpt.get('ema_shadow') or s2_ckpt['model_state_dict']
        sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
        s2.load_state_dict(sd, strict=False); s2.eval()

        with torch.no_grad():
            x_lens = torch.tensor([T], dtype=torch.long, device=device)
            s2_logits, s2_lens = s2(x, x_lens)
            log_probs = torch.log_softmax(s2_logits[0], dim=-1).cpu().numpy()
            preds = log_probs[:s2_lens[0].item()].argmax(axis=-1)
            decoded = []
            prev = 0
            for p in preds:
                if p != 0 and p != prev:
                    decoded.append(i2l.get(int(p), f'<{p}>'))
                prev = p

        print(f'\n  Stage 2 v15 raw: {" ".join(decoded) if decoded else "(empty)"}')
        if decoded:
            fixed = disambiguate(decoded, result, ch_mask=9, ch_xyz=slice(0, 3))
            print(f'  Stage 2 v15 + disambiguation: {" ".join(fixed)}')
    else:
        print(f'  Stage 2 v15 checkpoint not found at {s2_path}')


def test_v16(frames, device):
    """V16 pipeline: AV extraction (5ch) → Squeezeformer → Stage 2 CTC (fixed)"""
    print('\n' + '=' * 70)
    print('V16 PIPELINE (Squeezeformer, 5ch, Apple Vision)')
    print('=' * 70)

    from extract_v16 import extract_continuous_v16
    from model_v16 import SLTStage1V16, SLTStage2V16CTC

    # Extract
    result = extract_continuous_v16(frames, segment_len=28)
    if result is None:
        print('  Extraction failed'); return
    result = result.astype(np.float32)
    T = result.shape[0]
    nc = T // 32
    print(f'  Extracted: {result.shape} ({nc} clips)')

    # Load Stage 1
    s1_ckpt = torch.load('src_v16/output_v16_d384/best_model.pth', map_location=device, weights_only=False)
    s1 = SLTStage1V16(
        num_classes=s1_ckpt['num_classes'], in_channels=s1_ckpt['in_channels'], dim=s1_ckpt['d_model'],
    ).to(device)
    sd = s1_ckpt.get('ema_shadow', s1_ckpt['model_state_dict'])
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s1.load_state_dict(sd, strict=False); s1.eval()
    i2l = {int(v): k for k, v in s1_ckpt['label_to_idx'].items()}

    # Per-clip Stage 1
    print(f'\n  Stage 1 v16 per-clip:')
    for ci in range(min(nc, 10)):
        clip = torch.from_numpy(result[ci*32:(ci+1)*32])[None, ...].to(device)
        with torch.no_grad():
            logits = s1(clip)
        probs = torch.softmax(logits, dim=-1)[0]
        top3 = probs.topk(3)
        preds = ', '.join(f'{i2l[idx.item()]}({conf.item()*100:.0f}%)' for conf, idx in zip(top3.values, top3.indices))
        print(f'    clip {ci+1}: {preds}')

    # Load Stage 2 (fixed anti-hallucination)
    s2_path = '/Users/frnzlo/Downloads/results (2)/models/output_stage2_v16_fixed/stage2_best_model.pth'
    s2_ckpt = torch.load(s2_path, map_location=device, weights_only=False)
    s2 = SLTStage2V16CTC(
        vocab_size=s2_ckpt.get('vocab_size', 311), stage1_ckpt=None,
        in_channels=s2_ckpt.get('in_channels', 5), dim=s2_ckpt.get('d_model', 384),
    ).to(device)
    sd = s2_ckpt.get('ema_shadow') or s2_ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    s2.load_state_dict(sd, strict=False); s2.eval()
    g2i = s2_ckpt.get('label_to_idx') or s2_ckpt.get('gloss_to_idx')
    s2_i2g = {int(v): k for k, v in g2i.items()}

    # Pad to multiple of 32
    arr = result.copy()
    if T % 32 != 0:
        pad = np.zeros((((T + 31) // 32) * 32 - T, 61, 5), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)

    x = torch.from_numpy(arr)[None, ...].to(device)
    with torch.no_grad():
        logits, _ = s2(x)
    probs = logits.detach().cpu().numpy()[0]
    preds = probs.argmax(axis=-1)
    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != 0:
            decoded.append(s2_i2g.get(int(p), f'<{p}>'))
        prev = p

    print(f'\n  Stage 2 v16 raw: {" ".join(decoded) if decoded else "(empty)"}')
    if decoded:
        fixed = disambiguate(decoded, result, ch_mask=3, ch_xyz=slice(0, 3))
        print(f'  Stage 2 v16 + disambiguation: {" ".join(fixed)}')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'sample_videos/HELLO_HOW_YOU_training.mp4'
    print(f'Video: {path}')
    frames = load_video(path)
    print(f'Frames: {len(frames)}')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    test_v15(frames, device)
    test_v16(frames, device)


if __name__ == '__main__':
    main()
