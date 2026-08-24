"""Test Stage 2 on phrases, bucketed by input frame length.
Short videos (32 frames) have 2-3 signs compressed, which is an unfair test.
Longer videos (128+) have proper segmentation and should work well.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))

import numpy as np
import torch
from collections import defaultdict

from model_v16 import SLTStage2V16CTC

PHRASE_GLOSSES = {
    "GOOD_MORNING":           ["GOOD", "MORNING"],
    "HELLO_HOW_YOU":          ["HELLO", "HOW", "YOU"],
    "I_WANT_FOOD":            ["I", "WANT", "EAT_FOOD"],
    "MY_NAME":                ["MY", "NAME"],
    "PLEASE_HELP_ME":         ["PLEASE", "HELP", "I"],
    "SORRY_I_LATE":           ["SORRY", "I", "LATE"],
    "THANKYOU_FRIEND":        ["THANKYOU", "FRIEND"],
    "TOMORROW_SCHOOL_GO":     ["TOMORROW", "SCHOOL", "GO"],
    "YESTERDAY_TEACHER_MEET": ["YESTERDAY", "TEACHER", "MEET"],
}


def ctc_greedy_decode(logits, blank=0):
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    if logits.ndim == 3:
        logits = logits[0]
    preds = logits.argmax(axis=-1)
    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            decoded.append(int(p))
        prev = p
    return decoded


def wer(ref, hyp):
    if len(ref) == 0:
        return 100.0 if len(hyp) > 0 else 0.0
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return 100.0 * dp[r][h] / r


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt_path = '/Users/frnzlo/Downloads/results (1)/models/output_stage2_v16_phrases/stage2_best_model.pth'
    print(f'Device: {device}')
    print(f'Loading: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab_size = ckpt.get('vocab_size', 311)
    dim = ckpt.get('d_model', 384)
    in_channels = ckpt.get('in_channels', 5)
    gloss_to_idx = ckpt.get('label_to_idx') or ckpt.get('gloss_to_idx')
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}

    model = SLTStage2V16CTC(
        vocab_size=vocab_size, stage1_ckpt=None,
        in_channels=in_channels, dim=dim,
    ).to(device)
    sd = ckpt.get('ema_shadow') or ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    for candidate in ['ASL_phrases_v16', 'src_v16/ASL_phrases_v16']:
        if os.path.isdir(candidate):
            phrase_dir = candidate
            break
    else:
        print("ERROR: ASL_phrases_v16 not found")
        sys.exit(1)
    print(f'Phrase dir: {phrase_dir}')
    files = sorted([f for f in os.listdir(phrase_dir) if f.endswith('.npy')])
    print(f'Testing {len(files)} files\n')

    # Group results by (phrase, length bucket)
    bucket_stats = defaultdict(lambda: {'total': 0, 'exact': 0, 'wer_sum': 0.0, 'examples': []})

    with torch.no_grad():
        for fname in files:
            stem = fname.replace('.npy', '')
            phrase_name = None
            for p in PHRASE_GLOSSES:
                if stem.startswith(p + '_'):
                    phrase_name = p
                    break
            if phrase_name is None:
                continue
            expected = PHRASE_GLOSSES[phrase_name]

            arr = np.load(os.path.join(phrase_dir, fname)).astype(np.float32)
            if arr.ndim != 3 or arr.shape[1] != 61:
                continue
            if arr.shape[-1] == 10:
                continue  # skip old format
            T = arr.shape[0]

            # Pad to multiple of 32 if needed
            if T % 32 != 0:
                pad_to = ((T + 31) // 32) * 32
                pad = np.zeros((pad_to - T, 61, 5), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)

            x = torch.from_numpy(arr).unsqueeze(0).to(device)
            logits, _ = model(x)
            decoded_idx = ctc_greedy_decode(logits, blank=0)
            decoded = [idx_to_gloss.get(i, f'<{i}>') for i in decoded_idx]

            sample_wer = wer(expected, decoded)
            exact = decoded == expected

            key = (phrase_name, T)
            bucket_stats[key]['total'] += 1
            bucket_stats[key]['wer_sum'] += sample_wer
            if exact:
                bucket_stats[key]['exact'] += 1
            if len(bucket_stats[key]['examples']) < 2:
                bucket_stats[key]['examples'].append({
                    'expected': expected,
                    'decoded': decoded,
                    'wer': sample_wer,
                })

    # Report bucketed
    print(f'{"Phrase":<25} {"T":>4} {"N":>4} {"Exact":>8} {"WER":>8}  Sample')
    print('-' * 110)

    by_length = defaultdict(lambda: {'total': 0, 'exact': 0, 'wer_sum': 0.0})

    for (phrase, T), s in sorted(bucket_stats.items()):
        pct = 100 * s['exact'] / s['total']
        aw = s['wer_sum'] / s['total']
        ex = s['examples'][0] if s['examples'] else {'decoded': []}
        dec = ' '.join(ex.get('decoded', []))
        print(f'{phrase:<25} {T:>4} {s["total"]:>4} {pct:>7.1f}% {aw:>7.1f}%  {dec[:40]}')

        by_length[T]['total'] += s['total']
        by_length[T]['exact'] += s['exact']
        by_length[T]['wer_sum'] += s['wer_sum']

    print(f'\n{"="*70}\nSUMMARY BY VIDEO LENGTH\n{"="*70}')
    print(f'{"Frames":>8} {"Clips":>6} {"N":>5} {"Exact Match":>13} {"Avg WER":>10}')
    print('-' * 50)
    total_all = exact_all = wer_all = 0
    for T in sorted(by_length):
        s = by_length[T]
        pct = 100 * s['exact'] / s['total']
        aw = s['wer_sum'] / s['total']
        print(f'{T:>8} {T//32:>6} {s["total"]:>5} {pct:>12.1f}% {aw:>9.1f}%')
        total_all += s['total']
        exact_all += s['exact']
        wer_all += s['wer_sum']
    print('-' * 50)
    print(f'{"TOTAL":>8} {"":>6} {total_all:>5} {100*exact_all/total_all:>12.1f}% {wer_all/total_all:>9.1f}%')

    # Compute just on videos with ≥2 clips (where Stage 2 can actually segment)
    multi_clip = {T: v for T, v in by_length.items() if T >= 64}
    if multi_clip:
        mt = sum(v['total'] for v in multi_clip.values())
        me = sum(v['exact'] for v in multi_clip.values())
        mw = sum(v['wer_sum'] for v in multi_clip.values())
        print(f'\n{"="*70}')
        print(f'FAIR TEST: Multi-clip videos only (≥64 frames = ≥2 clips)')
        print(f'{"="*70}')
        print(f'N = {mt}, Exact = {100*me/mt:.1f}%, Avg WER = {mw/mt:.1f}%')


if __name__ == '__main__':
    main()
