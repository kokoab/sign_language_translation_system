"""
ATLAS v16 — Unified Inference Pipeline
Extracts landmarks from video, runs Stage 1 + Stage 2, outputs glosses.

Fixes over ad-hoc scripts:
  1. Smart segmentation: sliding window (stride=14) instead of fixed 28-frame chunks
  2. Dead frame trimming: detects pre-signing setup and trims
  3. Mirror TTA: averages original + hand-swapped predictions
  4. Beam search CTC decoding (width=25) instead of greedy

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python active/v16/inference_v16.py video.mp4
    KMP_DUPLICATE_LIB_OK=TRUE python active/v16/inference_v16.py video.mp4 --stage1_only
"""
import os, sys, argparse, time, gc
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v16 import SLTStage1V16, SLTStage2V16CTC
from extract_v16 import extract_frames_v16


# ═══════════════════════════════════════════════════════════
# Dead Frame Detection
# ═══════════════════════════════════════════════════════════

def detect_signing_range(frames, sample_interval=5):
    """Detect where signing starts and ends by measuring hand presence.
    Samples every Nth frame for speed. Returns (start, end) frame indices.
    """
    n = len(frames)
    if n < 16:
        return 0, n

    # Quick scan: extract every sample_interval-th frame, check hand mask
    mask_scores = np.zeros(n, dtype=np.float32)
    for i in range(0, n, sample_interval):
        chunk_end = min(i + sample_interval, n)
        if chunk_end - i < 4:
            continue
        arr = extract_frames_v16(frames[i:chunk_end])
        if arr is not None:
            # Hand mask average (nodes 0-41 = both hands, channel 3 = mask)
            mask_scores[i:chunk_end] = arr[:min(chunk_end - i, 32), :42, 3].mean()
        gc.collect()

    # Smooth
    k = max(3, n // 20)
    smoothed = np.convolve(mask_scores, np.ones(k) / k, mode='same')

    # Find first and last frame above threshold
    threshold = max(smoothed.max() * 0.3, 0.15)
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return 0, n

    start = max(0, above[0] - sample_interval)
    end = min(n, above[-1] + sample_interval)
    return start, end


# ═══════════════════════════════════════════════════════════
# Smart Continuous Extraction
# ═══════════════════════════════════════════════════════════

def extract_continuous_smart(frames, segment_len=28, stride=14, trim_dead=True):
    """Extract continuous video with sliding window and dead frame trimming.

    Improvements over extract_continuous_v16:
      1. Sliding window (stride=14) instead of non-overlapping segments
      2. Dead frame trimming at start/end
      3. No <= 40 frame shortcut

    Args:
        frames: list of BGR frames
        segment_len: window size in frames (default 28)
        stride: step size between windows (default 14 = 50% overlap)
        trim_dead: whether to detect and trim setup/teardown frames

    Returns:
        np.array [N*32, 61, 5] float16, or None
    """
    if len(frames) < 8:
        return None

    # Trim dead frames
    if trim_dead and len(frames) > 40:
        start, end = detect_signing_range(frames)
        frames = frames[start:end]
        if len(frames) < 8:
            return None

    n_frames = len(frames)

    # Single clip if very short
    if n_frames <= segment_len:
        return extract_frames_v16(frames)

    # Sliding window extraction
    clips = []
    for start in range(0, max(1, n_frames - segment_len + 1), stride):
        end = min(start + segment_len, n_frames)
        if end - start < 8:
            break
        clip = extract_frames_v16(frames[start:end])
        if clip is not None:
            clips.append(clip)
        # Memory management
        if len(clips) % 5 == 0:
            gc.collect()

    if not clips:
        return None

    return np.concatenate(clips, axis=0).astype(np.float16)


# ═══════════════════════════════════════════════════════════
# Motion-Based Sign Boundary Detection
# ═══════════════════════════════════════════════════════════

def find_best_split_points(features, n_splits):
    """Find n_splits lowest-energy frames to split the sequence.
    Ported from src/camera_inference.py:274-299, adapted for v16 format.

    Args:
        features: [T, 61, C] extracted features
        n_splits: number of split points to find
    Returns:
        list of frame indices
    """
    T = features.shape[0]
    min_seg = max(8, T // (n_splits + 2))

    # Compute motion energy from XY of hand nodes only (0-41)
    xy = features[:, :42, :2]  # [T, 42, 2]
    energy = np.zeros(T)
    diff = xy[1:] - xy[:-1]
    energy[1:] = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=1)

    # Smooth
    k = min(7, max(3, T // 6))
    smoothed = np.convolve(energy, np.ones(k) / k, mode='same')

    # Search for minima in the middle portion
    margin = max(min_seg, int(T * 0.15))
    search_start = margin
    search_end = T - margin

    if search_start >= search_end:
        return [int(T * i / (n_splits + 1)) for i in range(1, n_splits + 1)]

    frame_energies = [(i, smoothed[i]) for i in range(search_start, search_end)]
    frame_energies.sort(key=lambda x: x[1])

    selected = []
    for pos, _ in frame_energies:
        if len(selected) >= n_splits:
            break
        if all(abs(pos - s) >= min_seg for s in selected):
            selected.append(pos)

    if len(selected) < n_splits:
        selected = [int(T * i / (n_splits + 1)) for i in range(1, n_splits + 1)]

    selected.sort()
    return selected


# ═══════════════════════════════════════════════════════════
# Mirror TTA (Test-Time Augmentation)
# ═══════════════════════════════════════════════════════════

def mirror_tta_v16(x):
    """Create mirrored version: swap left/right hands, flip X.
    Adapted from src/camera_inference.py:429-445 for v16 channel layout.

    v16 channels: X(0), Y(1), Z(2), mask(3), palm_scale(4),
                  vel_x(5), vel_y(6), acc_x(7), acc_y(8)

    Args:
        x: [B, T, 61, C] tensor (C=5 or C=9)
    Returns:
        mirrored copy
    """
    m = x.clone()
    # Swap left hand (0-20) with right hand (21-41)
    m[:, :, 0:21] = x[:, :, 21:42]
    m[:, :, 21:42] = x[:, :, 0:21]
    # Flip X coordinate
    m[:, :, :42, 0] *= -1
    # Flip velocity_x if present
    if m.shape[-1] > 5:
        m[:, :, :42, 5] *= -1
    # Flip acceleration_x if present
    if m.shape[-1] > 7:
        m[:, :, :42, 7] *= -1
    # Do NOT flip: Y(1), Z(2), mask(3), palm_scale(4), vel_y(6), acc_y(8)
    return m


# ═══════════════════════════════════════════════════════════
# CTC Beam Search
# ═══════════════════════════════════════════════════════════

def ctc_beam_search(log_probs, beam_width=25, blank=0):
    """CTC prefix beam search.
    Ported from src/camera_inference.py:385-427.

    Args:
        log_probs: [T, V] log probabilities (numpy)
        beam_width: number of hypotheses to keep
        blank: blank token index
    Returns:
        list of (log_prob, token_tuple) sorted by score
    """
    T, V = log_probs.shape
    beams = {(): (0.0, float('-inf'))}

    for t in range(T):
        new_beams = {}
        for prefix, (pb, pnb) in beams.items():
            p = np.logaddexp(pb, pnb)
            for c in range(V):
                lp = log_probs[t, c]
                if c == blank:
                    key = prefix
                    new_pb = p + lp
                    if key in new_beams:
                        new_beams[key] = (np.logaddexp(new_beams[key][0], new_pb), new_beams[key][1])
                    else:
                        new_beams[key] = (new_pb, float('-inf'))
                else:
                    if len(prefix) > 0 and prefix[-1] == c:
                        new_pnb = pb + lp
                        key = prefix
                    else:
                        new_pnb = p + lp
                        key = prefix + (c,)
                    if key in new_beams:
                        new_beams[key] = (new_beams[key][0], np.logaddexp(new_beams[key][1], new_pnb))
                    else:
                        new_beams[key] = (float('-inf'), new_pnb)
                    if len(prefix) > 0 and prefix[-1] == c:
                        key2 = prefix + (c,)
                        new_pnb2 = p + lp
                        if key2 in new_beams:
                            new_beams[key2] = (new_beams[key2][0], np.logaddexp(new_beams[key2][1], new_pnb2))
                        else:
                            new_beams[key2] = (float('-inf'), new_pnb2)

        scored = [(np.logaddexp(pb, pnb), pf) for pf, (pb, pnb) in new_beams.items()]
        scored.sort(key=lambda x: -x[0])
        beams = {pf: new_beams[pf] for _, pf in scored[:beam_width]}

    results = []
    for prefix, (pb, pnb) in beams.items():
        results.append((np.logaddexp(pb, pnb), prefix))
    results.sort(key=lambda x: -x[0])
    return results


def ctc_greedy_decode(logits_np, blank=0):
    """Simple greedy CTC decode. Fallback when beam search is too slow."""
    preds = logits_np.argmax(axis=-1)
    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            decoded.append(int(p))
        prev = p
    return decoded


# ═══════════════════════════════════════════════════════════
# Post-Processing: Dedup + Disambiguation
# ═══════════════════════════════════════════════════════════

def dedup_consecutive(glosses):
    """Remove consecutive duplicate glosses from overlapping windows.
    e.g., ['HELLO', 'HOW', 'HOW', 'YOU', 'YOU', 'YOU'] → ['HELLO', 'HOW', 'YOU']
    """
    if not glosses:
        return glosses
    deduped = [glosses[0]]
    for g in glosses[1:]:
        if g != deduped[-1]:
            deduped.append(g)
    return deduped


def disambiguate_glosses(glosses, features, ch_mask=3, ch_xyz=slice(0, 3)):
    """Apply motion-based disambiguation rules.
    Ported from src/demo_classify.py:1180-1248, adapted for v16 5ch format.

    Args:
        glosses: list of gloss strings
        features: [T, 61, C] extracted features (before vel/acc)
        ch_mask: channel index for detection mask (v16=3, v15=9)
        ch_xyz: slice for XYZ channels (v16=slice(0,3))
    """
    if not glosses:
        return glosses

    n_clips = features.shape[0] // 32
    if n_clips == 0:
        return glosses

    fixed = []
    for wi, word in enumerate(glosses):
        seg_idx = min(int(wi / max(len(glosses), 1) * n_clips), n_clips - 1)
        seg = features[seg_idx * 32:(seg_idx + 1) * 32]

        if seg.shape[0] < 8:
            fixed.append(word)
            continue

        # Determine active hand
        lh_m = seg[:, 0, ch_mask].mean()
        rh_m = seg[:, 21, ch_mask].mean()
        wr = 0 if lh_m > rh_m else 21
        ns = 42  # nose
        end_f = min(seg.shape[0] - 2, 28)

        # Motion features
        yt = seg[end_f, wr, 1] - seg[4, wr, 1]
        hh = seg[:, wr, 1].mean() - seg[:, ns, 1].mean()
        motion = np.abs(np.diff(seg[:, wr, ch_xyz], axis=0)).sum()
        x_osc = np.abs(np.diff(seg[:, wr, 0])).sum()
        y_osc = np.abs(np.diff(seg[:, wr, 1])).sum()

        # Disambiguation rules
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
            tips = [wr + 4, wr + 8, wr + 12, wr + 16, wr + 20]
            openness = np.mean([np.linalg.norm(
                seg[:, t, ch_xyz] - seg[:, wr, ch_xyz], axis=-1).mean() for t in tips])
            word = 'PLEASE' if openness > 1.5 else 'SORRY'
        elif word in ('FATHER', 'MOTHER'):
            word = 'FATHER' if hh < 1.0 else 'MOTHER'
        elif word in ('HELLO', 'HIS_HER'):
            word = 'HELLO' if motion > 0.5 and hh > 0.8 else 'HIS_HER' if motion < 0.3 else word

        fixed.append(word)

    return dedup_consecutive(fixed)


# ═══════════════════════════════════════════════════════════
# Compute velocity/acceleration channels
# ═══════════════════════════════════════════════════════════

def add_vel_acc(arr):
    """Add velocity and acceleration channels to [T, 61, 5] → [T, 61, 9].
    Same computation as SignDatasetV16.__getitem__ in train_stage_1_v16.py:205-216.
    """
    xy = arr[:, :, :2]  # [T, N, 2]
    vel = np.zeros_like(xy)
    vel[1:-1] = (xy[2:] - xy[:-2]) / 2.0
    vel[0] = vel[1] if len(vel) > 1 else vel[0]
    vel[-1] = vel[-2] if len(vel) > 1 else vel[-1]
    acc = np.zeros_like(xy)
    acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
    acc[0] = acc[1] if len(acc) > 1 else acc[0]
    acc[-1] = acc[-2] if len(acc) > 1 else acc[-1]
    return np.concatenate([arr, vel, acc], axis=-1)


# ═══════════════════════════════════════════════════════════
# Main Inference
# ═══════════════════════════════════════════════════════════

def _strict_load_check(name, ckpt, missing, unexpected):
    """Fail loud on architecture mismatch. load_state_dict(strict=False) silently
    drops mismatched keys, which is how a use_angles/depth mismatch becomes a
    silent accuracy regression instead of a crash. Whitelist only EMA/optimizer
    leftovers, treat anything else as a real architecture mismatch.
    """
    real_missing = [k for k in missing if not k.endswith('.num_batches_tracked')]
    real_unexpected = [k for k in unexpected if not k.endswith('.num_batches_tracked')]
    if real_missing or real_unexpected:
        flags = (f"use_angles={ckpt.get('use_angles', '?')}, "
                 f"use_pairwise={ckpt.get('use_pairwise', '?')}, "
                 f"depth={ckpt.get('depth', '?')}, "
                 f"d_model={ckpt.get('d_model', '?')}, "
                 f"in_channels={ckpt.get('in_channels', '?')}")
        raise RuntimeError(
            f"{name} state_dict mismatch ({flags}):\n"
            f"  missing ({len(real_missing)}): {real_missing[:5]}...\n"
            f"  unexpected ({len(real_unexpected)}): {real_unexpected[:5]}...\n"
            f"This usually means the checkpoint's architecture flags don't match "
            f"the model you constructed."
        )


def load_models(stage1_ckpt, stage2_ckpt, device):
    """Load Stage 1 and Stage 2 models."""
    # Stage 1
    s1_ckpt = torch.load(stage1_ckpt, map_location=device, weights_only=False)
    s1_use_angles   = s1_ckpt.get('use_angles',   False)
    s1_use_pairwise = s1_ckpt.get('use_pairwise', False)
    s1_depth        = s1_ckpt.get('depth', 4)
    s1 = SLTStage1V16(
        num_classes=s1_ckpt['num_classes'],
        in_channels=s1_ckpt['in_channels'],
        dim=s1_ckpt['d_model'],
        depth=s1_depth,
        use_angles=s1_use_angles,
        use_pairwise=s1_use_pairwise,
    ).to(device)
    sd = s1_ckpt.get('ema_shadow') or s1_ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    missing, unexpected = s1.load_state_dict(sd, strict=False)
    _strict_load_check('Stage 1', s1_ckpt, missing, unexpected)
    s1.eval()
    s1_i2l = {int(v): k for k, v in s1_ckpt['label_to_idx'].items()}
    in_channels = s1_ckpt['in_channels']

    # Stage 2
    s2 = None
    s2_i2g = None
    if stage2_ckpt and os.path.exists(stage2_ckpt):
        s2_ckpt = torch.load(stage2_ckpt, map_location=device, weights_only=False)
        s2 = SLTStage2V16CTC(
            vocab_size=s2_ckpt.get('vocab_size', 311),
            stage1_ckpt=None,
            in_channels=s2_ckpt.get('in_channels', in_channels),
            dim=s2_ckpt.get('d_model', 384),
            encoder_depth=s2_ckpt.get('depth', 4),
            use_angles=s2_ckpt.get('use_angles', False),
            use_pairwise=s2_ckpt.get('use_pairwise', False),
        ).to(device)
        sd2 = s2_ckpt.get('ema_shadow') or s2_ckpt['model_state_dict']
        sd2 = {k.replace('_orig_mod.', ''): v for k, v in sd2.items()}
        missing, unexpected = s2.load_state_dict(sd2, strict=False)
        _strict_load_check('Stage 2', s2_ckpt, missing, unexpected)
        s2.eval()
        g2i = s2_ckpt.get('label_to_idx') or s2_ckpt.get('gloss_to_idx')
        s2_i2g = {int(v): k for k, v in g2i.items()}

    return s1, s1_i2l, in_channels, s2, s2_i2g


def run_inference(video_path, s1, s1_i2l, in_channels, s2=None, s2_i2g=None,
                  device='cpu', use_tta=True, beam_width=25, verbose=True):
    """Full inference pipeline on a video file.

    Returns:
        dict with keys: stage1_per_clip, stage2_glosses, stage2_beam_results
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if len(frames) < 8:
        return {'error': f'Too few frames ({len(frames)})'}

    w, h = frames[0].shape[1], frames[0].shape[0]
    if verbose:
        print(f'Video: {w}x{h}, {len(frames)} frames, {len(frames)/30:.1f}s')

    t0 = time.time()

    # Smart extraction with dead frame trimming + sliding window
    arr = extract_continuous_smart(frames, segment_len=28, stride=14, trim_dead=True)
    if arr is None:
        return {'error': 'Extraction failed — no hands detected'}
    arr = arr.astype(np.float32)

    T = arr.shape[0]
    nc = T // 32
    extract_time = time.time() - t0

    if verbose:
        print(f'Extracted: {arr.shape} ({nc} clips) in {extract_time:.1f}s')

    # Add velocity/acceleration if model expects more than 5 channels
    if in_channels > 5:
        # Process per 32-frame clip to get proper vel/acc
        clips_9ch = []
        for ci in range(nc):
            clip_5ch = arr[ci * 32:(ci + 1) * 32]
            clips_9ch.append(add_vel_acc(clip_5ch))
        arr_input = np.concatenate(clips_9ch, axis=0)
    else:
        arr_input = arr

    # Truncate to in_channels if needed
    if arr_input.shape[-1] > in_channels:
        arr_input = arr_input[..., :in_channels]

    results = {
        'video': video_path,
        'frames': len(frames),
        'clips': nc,
        'extract_time': extract_time,
    }

    # ── Stage 1: per-clip classification ──
    t1 = time.time()
    stage1_preds = []
    for ci in range(nc):
        clip = torch.from_numpy(arr_input[ci * 32:(ci + 1) * 32])[None, ...].to(device)
        with torch.no_grad():
            logits = s1(clip)

            if use_tta:
                clip_m = mirror_tta_v16(clip)
                logits_m = s1(clip_m)
                probs = (F.softmax(logits, dim=-1) + F.softmax(logits_m, dim=-1)) / 2
            else:
                probs = F.softmax(logits, dim=-1)

        probs = probs[0]
        top3 = probs.topk(3)
        preds = [(s1_i2l[idx.item()], conf.item() * 100)
                 for conf, idx in zip(top3.values, top3.indices)]
        stage1_preds.append(preds)

        if verbose:
            pred_str = ', '.join(f'{n}({c:.0f}%)' for n, c in preds)
            print(f'  clip {ci + 1:2d}: {pred_str}')

    results['stage1_per_clip'] = stage1_preds
    results['stage1_time'] = time.time() - t1

    # ── Stage 2: continuous CTC ──
    if s2 is not None:
        t2 = time.time()

        # Pad to multiple of 32
        if T % 32 != 0:
            pad = np.zeros((((T + 31) // 32) * 32 - T,) + arr_input.shape[1:], dtype=np.float32)
            arr_padded = np.concatenate([arr_input, pad], axis=0)
        else:
            arr_padded = arr_input

        x = torch.from_numpy(arr_padded)[None, ...].to(device)

        with torch.no_grad():
            logits, _ = s2(x)

            if use_tta:
                x_m = mirror_tta_v16(x)
                logits_m, _ = s2(x_m)
                avg_probs = (F.softmax(logits, dim=-1) + F.softmax(logits_m, dim=-1)) / 2
                log_probs = avg_probs.log()[0].cpu().numpy()
            else:
                log_probs = F.log_softmax(logits, dim=-1)[0].cpu().numpy()

        # Beam search CTC decode
        beam_results = ctc_beam_search(log_probs, beam_width=beam_width, blank=0)
        if beam_results:
            best_score, best_tokens = beam_results[0]
            glosses = [s2_i2g.get(int(t), f'<{t}>') for t in best_tokens]
        else:
            glosses = []

        # Also greedy for comparison
        greedy = ctc_greedy_decode(log_probs, blank=0)
        greedy_glosses = [s2_i2g.get(int(t), f'<{t}>') for t in greedy]

        # Post-processing: dedup + disambiguation
        glosses = dedup_consecutive(glosses)
        glosses = disambiguate_glosses(glosses, arr, ch_mask=3, ch_xyz=slice(0, 3))
        greedy_glosses = dedup_consecutive(greedy_glosses)
        greedy_glosses = disambiguate_glosses(greedy_glosses, arr, ch_mask=3, ch_xyz=slice(0, 3))

        results['stage2_beam'] = ' '.join(glosses) if glosses else '(empty)'
        results['stage2_greedy'] = ' '.join(greedy_glosses) if greedy_glosses else '(empty)'
        results['stage2_beam_results'] = [(score, [s2_i2g.get(int(t), f'<{t}>') for t in toks])
                                           for score, toks in beam_results[:5]]
        results['stage2_time'] = time.time() - t2

        if verbose:
            print(f'\n  Stage 2 beam:   {results["stage2_beam"]}')
            print(f'  Stage 2 greedy: {results["stage2_greedy"]}')
            if len(beam_results) > 1:
                print(f'  Top 3 hypotheses:')
                for score, toks in beam_results[:3]:
                    gs = [s2_i2g.get(int(t), f'<{t}>') for t in toks]
                    print(f'    {score:.2f}: {" ".join(gs)}')

    total_time = time.time() - t0
    results['total_time'] = total_time
    if verbose:
        print(f'\n  Total: {total_time:.1f}s (extract={extract_time:.1f}s)')

    return results


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='ATLAS v16 Inference')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--stage1_ckpt', default='src_v16/output_v16_d384/best_model.pth')
    parser.add_argument('--stage2_ckpt', default=None,
                        help='Stage 2 checkpoint (default: auto-detect)')
    parser.add_argument('--stage1_only', action='store_true', help='Skip Stage 2')
    parser.add_argument('--no_tta', action='store_true', help='Disable mirror TTA')
    parser.add_argument('--beam_width', type=int, default=25)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--no_trim', action='store_true', help='Disable dead frame trimming')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    print(f'Device: {device}')

    # Auto-detect Stage 2 checkpoint
    if args.stage2_ckpt is None and not args.stage1_only:
        for candidate in [
            '/Users/frnzlo/Downloads/results (2)/models/output_stage2_v16_fixed/stage2_best_model.pth',
            'models/output_stage2_v16_fixed/stage2_best_model.pth',
            'models/output_stage2_v16_phrases/stage2_best_model.pth',
            'models/output_stage2_v16/stage2_best_model.pth',
        ]:
            if os.path.exists(candidate):
                args.stage2_ckpt = candidate
                break

    s1, s1_i2l, in_channels, s2, s2_i2g = load_models(
        args.stage1_ckpt,
        args.stage2_ckpt if not args.stage1_only else None,
        device,
    )

    results = run_inference(
        args.video, s1, s1_i2l, in_channels, s2, s2_i2g,
        device=device,
        use_tta=not args.no_tta,
        beam_width=args.beam_width,
    )

    if 'error' in results:
        print(f'ERROR: {results["error"]}')
        sys.exit(1)


if __name__ == '__main__':
    main()
