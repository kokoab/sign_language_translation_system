"""
Test aspect ratio fix for extraction normalization.

Problem: Apple Vision returns [0,1] normalized coords relative to frame dimensions.
Portrait 540x960 → X compressed, Y stretched vs landscape 640x480.
After palm-length normalization, portrait coords are 2-3x larger than landscape.
Model trained on landscape fails on portrait input.

Fix: Before normalization, scale coordinates to square aspect ratio space.

Test plan:
1. Extract the problem video WITH and WITHOUT the fix
2. Compare feature ranges against training data
3. Run Stage 1 on both, compare predictions
4. Run on 100 random training videos to verify no regression
"""
import os, sys, cv2, time, gc
sys.path.insert(0, 'src_v16')

import numpy as np
import torch
from model_v16 import SLTStage1V16

# We'll monkey-patch normalize_sequence to test the fix without modifying extract_v16.py

import extract_v16

# Save original
_orig_normalize = extract_v16.normalize_sequence

def normalize_sequence_fixed(xy_seq, mask_seq, frame_w=None, frame_h=None):
    """Aspect-ratio corrected normalization.
    Scales coordinates so X and Y represent equal physical distances
    before centering and palm-length normalization.
    """
    T, N, _ = xy_seq.shape

    # Step 0: Correct for aspect ratio if dimensions provided
    if frame_w is not None and frame_h is not None:
        max_dim = max(frame_w, frame_h)
        xy_seq = xy_seq.copy()
        xy_seq[:, :, 0] *= frame_w / max_dim  # X scaled down in portrait
        xy_seq[:, :, 1] *= frame_h / max_dim  # Y scaled down in landscape

    # Rest is identical to original
    lh_active = mask_seq[:, extract_v16.LHAND_START].mean() > 0.3
    rh_active = mask_seq[:, extract_v16.RHAND_START].mean() > 0.3

    centers = []
    if lh_active:
        centers.append(xy_seq[:, extract_v16.LHAND_START, :])
    if rh_active:
        centers.append(xy_seq[:, extract_v16.RHAND_START, :])

    if centers:
        center = np.median(np.concatenate(centers, axis=0), axis=0)
    else:
        center = np.array([0.5, 0.5])

    xy_seq = xy_seq - center[None, None, :]

    palm_lengths = []
    for hand_start in [extract_v16.LHAND_START, extract_v16.RHAND_START]:
        wrist = xy_seq[:, hand_start, :]
        mid_mcp = xy_seq[:, hand_start + 9, :]
        pl = np.sqrt(((mid_mcp - wrist) ** 2).sum(axis=-1))
        valid = pl > 0.01
        if valid.sum() > 3:
            palm_lengths.append(np.median(pl[valid]))

    if palm_lengths:
        scale = np.mean(palm_lengths)
    else:
        scale = 1.0

    if scale > 0.01:
        xy_seq = xy_seq / scale

    return xy_seq


def extract_with_fix(frames, use_fix=True):
    """Extract with or without the aspect ratio fix."""
    if use_fix:
        h, w = frames[0].shape[:2]
        # Monkey-patch: inject frame dimensions into normalize_sequence
        def patched_normalize(xy_seq, mask_seq):
            return normalize_sequence_fixed(xy_seq, mask_seq, frame_w=w, frame_h=h)
        extract_v16.normalize_sequence = patched_normalize
    else:
        extract_v16.normalize_sequence = _orig_normalize

    result = extract_v16.extract_frames_v16(frames)

    # Restore original
    extract_v16.normalize_sequence = _orig_normalize
    return result


def extract_continuous_with_fix(frames, use_fix=True, segment_len=28):
    """Extract continuous with or without fix."""
    if use_fix:
        h, w = frames[0].shape[:2]
        def patched_normalize(xy_seq, mask_seq):
            return normalize_sequence_fixed(xy_seq, mask_seq, frame_w=w, frame_h=h)
        extract_v16.normalize_sequence = patched_normalize
    else:
        extract_v16.normalize_sequence = _orig_normalize

    result = extract_v16.extract_continuous_v16(frames, segment_len=segment_len)

    extract_v16.normalize_sequence = _orig_normalize
    return result


def load_stage1():
    device = torch.device('cpu')  # Use CPU for consistency
    ckpt = torch.load('src_v16/output_v16_d384/best_model.pth', map_location=device, weights_only=False)
    model = SLTStage1V16(
        num_classes=ckpt['num_classes'], in_channels=ckpt['in_channels'], dim=ckpt['d_model'],
    ).to(device)
    sd = ckpt.get('ema_shadow', ckpt['model_state_dict'])
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False); model.eval()
    i2l = {int(v): k for k, v in ckpt['label_to_idx'].items()}
    return model, i2l


def main():
    model, i2l = load_stage1()

    # ══════════════════════════════════════════════════════════
    # TEST 1: Problem video (portrait 540x960)
    # ══════════════════════════════════════════════════════════
    print('=' * 72)
    print('TEST 1: Problem video (portrait 540x960)')
    print('=' * 72)

    path = '/Users/frnzlo/Downloads/AQN1WD8awPGFbyTSpbhF4cX2Y86PkNtIYdq9tUPyAE8hcQ_FVCRYRrBp2Qt6qjuud2-rL0tzxaVfUO-7TntMNSIa2cHI4QDYAAsaOxJu0g.mp4'
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    print(f'  Video: {w}x{h}, {len(frames)} frames')

    # Extract first 60 frames (HELLO region) with and without fix
    for label, use_fix in [('WITHOUT fix (original)', False), ('WITH fix', True)]:
        arr = extract_with_fix(frames[:60], use_fix=use_fix)
        if arr is None:
            print(f'  {label}: extraction failed'); continue
        arr = arr.astype(np.float32)
        x = torch.from_numpy(arr)[None, ...]
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        top5 = probs.topk(5)
        print(f'\n  {label}:')
        print(f'    X range: [{arr[:,:,0].min():.3f}, {arr[:,:,0].max():.3f}]')
        print(f'    Y range: [{arr[:,:,1].min():.3f}, {arr[:,:,1].max():.3f}]')
        for c, i in zip(top5.values, top5.indices):
            print(f'      {i2l[i.item()]:<20} {c.item()*100:.1f}%')
        gc.collect()

    # ══════════════════════════════════════════════════════════
    # TEST 2: Training data (landscape 640x480) — verify no regression
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('TEST 2: Training data landscape videos — regression check')
    print('=' * 72)

    # Get 100 training .npy files and their true labels
    import json
    with open('src_v16/manifest_v16_files_deep_cleaned.json') as f:
        manifest = json.load(f)

    import random
    rng = random.Random(42)
    items = list(manifest.items())
    rng.shuffle(items)
    test_items = items[:100]

    iso_dir = 'src_v16/ASL_landmarks_v16'

    # Test original .npy files (no re-extraction, just model accuracy)
    correct_orig = 0
    for fname, cls in test_items:
        fpath = os.path.join(iso_dir, fname)
        if not os.path.exists(fpath): continue
        arr = np.load(fpath).astype(np.float32)
        if arr.shape != (32, 61, 5): continue
        x = torch.from_numpy(arr)[None, ...]
        with torch.no_grad():
            pred = model(x).argmax(dim=-1).item()
        if i2l[pred] == cls:
            correct_orig += 1

    print(f'  Original .npy accuracy: {correct_orig}/100 = {correct_orig}%')
    print(f'  (These are already extracted, fix doesnt apply to stored .npy files)')

    # ══════════════════════════════════════════════════════════
    # TEST 3: Re-extract training videos with fix, check accuracy
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('TEST 3: Re-extract landscape training videos WITH fix')
    print('=' * 72)

    video_dir = 'data/raw_videos/ASL VIDEOS'
    if not os.path.isdir(video_dir):
        print('  Training video directory not found, skipping')
        return

    # Pick 100 random signs to re-extract
    classes = sorted([d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))])
    rng = random.Random(42)
    test_signs = rng.sample(classes, min(50, len(classes)))

    correct_no_fix = 0
    correct_with_fix = 0
    total = 0
    range_diffs = []

    for cls in test_signs:
        cls_dir = os.path.join(video_dir, cls)
        vids = [f for f in os.listdir(cls_dir) if f.endswith('.mp4')]
        if not vids: continue
        vid = vids[0]  # take first video per class
        vpath = os.path.join(cls_dir, vid)

        cap = cv2.VideoCapture(vpath)
        vframes = []
        while True:
            ret, f = cap.read()
            if not ret: break
            vframes.append(f)
        cap.release()
        if len(vframes) < 8: continue

        vh, vw = vframes[0].shape[:2]

        # Extract WITHOUT fix
        arr_orig = extract_with_fix(vframes, use_fix=False)
        if arr_orig is None: continue
        arr_orig = arr_orig.astype(np.float32)

        # Extract WITH fix
        arr_fixed = extract_with_fix(vframes, use_fix=True)
        if arr_fixed is None: continue
        arr_fixed = arr_fixed.astype(np.float32)

        # Track range differences
        x_range_orig = arr_orig[:,:,0].max() - arr_orig[:,:,0].min()
        x_range_fixed = arr_fixed[:,:,0].max() - arr_fixed[:,:,0].min()
        range_diffs.append((cls, vw, vh, x_range_orig, x_range_fixed))

        # Predict both
        x_o = torch.from_numpy(arr_orig)[None, ...]
        x_f = torch.from_numpy(arr_fixed)[None, ...]
        with torch.no_grad():
            pred_o = i2l[model(x_o).argmax(dim=-1).item()]
            pred_f = i2l[model(x_f).argmax(dim=-1).item()]

        if pred_o == cls: correct_no_fix += 1
        if pred_f == cls: correct_with_fix += 1
        total += 1

        if total % 10 == 0:
            print(f'  [{total}/50] no_fix={correct_no_fix} with_fix={correct_with_fix}')
        gc.collect()

    print(f'\n  Results on {total} freshly extracted training videos:')
    print(f'    WITHOUT fix: {correct_no_fix}/{total} = {100*correct_no_fix/max(total,1):.1f}%')
    print(f'    WITH fix:    {correct_with_fix}/{total} = {100*correct_with_fix/max(total,1):.1f}%')

    # Show range differences by aspect ratio
    landscape = [(c, w, h, ro, rf) for c, w, h, ro, rf in range_diffs if w >= h]
    portrait = [(c, w, h, ro, rf) for c, w, h, ro, rf in range_diffs if w < h]
    print(f'\n  Landscape videos ({len(landscape)}):')
    if landscape:
        avg_ro = np.mean([ro for _, _, _, ro, _ in landscape])
        avg_rf = np.mean([rf for _, _, _, _, rf in landscape])
        print(f'    Avg X range: original={avg_ro:.3f}, fixed={avg_rf:.3f}, diff={abs(avg_rf-avg_ro)/avg_ro*100:.1f}%')
    print(f'  Portrait videos ({len(portrait)}):')
    if portrait:
        avg_ro = np.mean([ro for _, _, _, ro, _ in portrait])
        avg_rf = np.mean([rf for _, _, _, _, rf in portrait])
        print(f'    Avg X range: original={avg_ro:.3f}, fixed={avg_rf:.3f}, diff={abs(avg_rf-avg_ro)/avg_ro*100:.1f}%')

    # ══════════════════════════════════════════════════════════
    # TEST 4: Problem video full continuous with fix
    # ══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('TEST 4: Problem video full continuous extraction with fix')
    print('=' * 72)

    path = '/Users/frnzlo/Downloads/AQN1WD8awPGFbyTSpbhF4cX2Y86PkNtIYdq9tUPyAE8hcQ_FVCRYRrBp2Qt6qjuud2-rL0tzxaVfUO-7TntMNSIa2cHI4QDYAAsaOxJu0g.mp4'
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()

    arr = extract_continuous_with_fix(frames, use_fix=True)
    if arr is None:
        print('  Extraction failed'); return
    arr = arr.astype(np.float32)
    T = arr.shape[0]
    nc = T // 32
    print(f'  Extracted: {arr.shape} ({nc} clips)')
    print(f'  X range: [{arr[:,:,0].min():.3f}, {arr[:,:,0].max():.3f}]')
    print(f'  Y range: [{arr[:,:,1].min():.3f}, {arr[:,:,1].max():.3f}]')

    # Per-clip Stage 1
    print(f'\n  Stage 1 per-clip (WITH fix):')
    for ci in range(min(nc, 10)):
        clip = torch.from_numpy(arr[ci*32:(ci+1)*32])[None, ...]
        with torch.no_grad():
            logits = model(clip)
        probs = torch.softmax(logits, dim=-1)[0]
        top3 = probs.topk(3)
        preds = ', '.join(f'{i2l[idx.item()]}({conf.item()*100:.0f}%)' for conf, idx in zip(top3.values, top3.indices))
        print(f'    clip {ci+1}: {preds}')

    print('\n  Ground truth: HELLO GOOD MORNING HOW YOU')


if __name__ == '__main__':
    main()
