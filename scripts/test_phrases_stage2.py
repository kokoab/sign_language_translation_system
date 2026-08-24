"""Test Stage 2 Squeezeformer-CTC on all 781 phrase videos.
These phrases were NOT used in Stage 2 training (which used synthetic sequences only).
This tests whether the model generalizes to real continuous signing.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter

from model_v16 import SLTStage2V16CTC

# Phrase → expected gloss sequence (from extract_phrases_v16.py)
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


def convert_10ch_to_5ch(arr):
    """Convert old v15 format [T, 61, 10] to v16 [T, 61, 5].
    Same logic as Stage 2 dataset _load_clip method.
    """
    T, N, C = arr.shape
    assert C == 10, f"Expected 10ch, got {C}"
    new_arr = np.zeros((T, N, 5), dtype=np.float32)
    new_arr[:, :, 0] = arr[:, :, 0]  # X
    new_arr[:, :, 1] = arr[:, :, 1]  # Y
    # ch2: Z — compute perspective Z from palm scale
    for hs in [0, 21]:
        wrist = arr[:, hs, :2].astype(np.float32)
        mcp = arr[:, hs + 9, :2].astype(np.float32)
        palm_len = np.sqrt(((mcp - wrist) ** 2).sum(axis=-1))
        valid = palm_len > 0.01
        if valid.sum() >= 3:
            ref = np.median(palm_len[valid])
            if ref > 0.01:
                for t in range(T):
                    if palm_len[t] > 0.01:
                        new_arr[t, hs:hs+21, 2] = np.log(ref / palm_len[t])
    new_arr[:, :, 3] = arr[:, :, 9]  # mask
    # ch4: palm scale
    for hs in [0, 21]:
        wrist = arr[:, hs, :2].astype(np.float32)
        mcp = arr[:, hs + 9, :2].astype(np.float32)
        palm_len = np.sqrt(((mcp - wrist) ** 2).sum(axis=-1))
        valid = palm_len > 0.01
        if valid.sum() >= 3:
            ref = np.median(palm_len[valid])
            if ref > 0.01:
                for t in range(T):
                    ps = palm_len[t] / ref if palm_len[t] > 0.01 else 1.0
                    new_arr[t, hs:hs+21, 4] = ps
            else:
                new_arr[:, hs:hs+21, 4] = 1.0
        else:
            new_arr[:, hs:hs+21, 4] = 1.0
    new_arr[:, 42:, 4] = 1.0
    return new_arr


def ctc_greedy_decode(logits, blank=0):
    """Greedy CTC decode: argmax, then merge duplicates and remove blanks."""
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
    """Word Error Rate via Levenshtein distance."""
    if len(ref) == 0:
        return 100.0 if len(hyp) > 0 else 0.0
    r, h = len(ref), len(hyp)
    dp = [[0] * (h + 1) for _ in range(r + 1)]
    for i in range(r + 1): dp[i][0] = i
    for j in range(h + 1): dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return 100.0 * dp[r][h] / r


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load Stage 2 checkpoint
    for candidate in [
        'models/output_stage2_v16/stage2_best_model.pth',
        'output_stage2_v16/stage2_best_model.pth',
        'src_v16/output_stage2_v16/stage2_best_model.pth',
        '/Users/frnzlo/Downloads/models 2/output_stage2_v16/stage2_best_model.pth',
    ]:
        if os.path.exists(candidate):
            ckpt_path = candidate
            break
    else:
        print("ERROR: Cannot find Stage 2 checkpoint.")
        sys.exit(1)

    print(f"Loading Stage 2: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vocab_size = ckpt.get('vocab_size', 311)
    dim = ckpt.get('d_model', 384)
    in_channels = ckpt.get('in_channels', 5)
    gloss_to_idx = ckpt.get('label_to_idx') or ckpt.get('gloss_to_idx')
    if gloss_to_idx is None:
        print("ERROR: No gloss_to_idx in checkpoint")
        sys.exit(1)
    idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}

    print(f"  vocab_size={vocab_size}, dim={dim}, in_channels={in_channels}")

    model = SLTStage2V16CTC(
        vocab_size=vocab_size, stage1_ckpt=None,
        in_channels=in_channels, dim=dim,
    ).to(device)

    sd = ckpt.get('ema_shadow') or ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARNING: missing {len(missing)} keys")
    model.eval()

    # Verify gloss indices
    for phrase, glosses in PHRASE_GLOSSES.items():
        for g in glosses:
            if g not in gloss_to_idx:
                print(f"  WARNING: gloss '{g}' not in vocab (from {phrase})")

    # Load all phrase files — prefer v16 extracted, fall back to v15
    for candidate in ['ASL_phrases_v16', 'ASL_phrases_reextracted']:
        if os.path.isdir(candidate):
            phrase_dir = candidate
            break
    else:
        print(f"ERROR: No phrase directory found (tried ASL_phrases_v16, ASL_phrases_reextracted)")
        sys.exit(1)
    print(f"Using phrase directory: {phrase_dir}")

    files = sorted([f for f in os.listdir(phrase_dir) if f.endswith('.npy')])
    print(f"\nFound {len(files)} phrase files in {phrase_dir}")

    # Run inference
    results = []
    per_phrase_stats = {p: {'total': 0, 'correct': 0, 'wer_sum': 0.0, 'decoded': []} for p in PHRASE_GLOSSES}

    with torch.no_grad():
        for i, fname in enumerate(files):
            # Parse phrase name from filename. Try both formats:
            #   v16: PHRASE_NAME_hash.npy
            #   v15: PHRASE_NAME_NNNN_hash.npy
            stem = fname.replace('.npy', '')
            phrase_name = None
            for p in PHRASE_GLOSSES:
                if stem.startswith(p + '_'):
                    phrase_name = p
                    break
            if phrase_name is None:
                continue

            expected_glosses = PHRASE_GLOSSES[phrase_name]
            expected_indices = [gloss_to_idx[g] for g in expected_glosses if g in gloss_to_idx]

            try:
                arr = np.load(os.path.join(phrase_dir, fname)).astype(np.float32)
                if arr.ndim != 3 or arr.shape[1] != 61:
                    continue

                # Convert 10ch → 5ch if needed
                if arr.shape[-1] == 10:
                    arr = convert_10ch_to_5ch(arr)
                elif arr.shape[-1] != 5:
                    continue

                # Pad to multiple of 32 if needed
                T = arr.shape[0]
                if T % 32 != 0:
                    pad_to = ((T + 31) // 32) * 32
                    pad = np.zeros((pad_to - T, 61, 5), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)

                x = torch.from_numpy(arr).unsqueeze(0).to(device)
                logits, _ = model(x)
                decoded_indices = ctc_greedy_decode(logits, blank=0)
                decoded_glosses = [idx_to_gloss.get(i, f'<{i}>') for i in decoded_indices]

                sample_wer = wer(expected_glosses, decoded_glosses)
                exact_match = decoded_glosses == expected_glosses

                per_phrase_stats[phrase_name]['total'] += 1
                per_phrase_stats[phrase_name]['wer_sum'] += sample_wer
                if exact_match:
                    per_phrase_stats[phrase_name]['correct'] += 1
                if len(per_phrase_stats[phrase_name]['decoded']) < 5:
                    per_phrase_stats[phrase_name]['decoded'].append({
                        'expected': expected_glosses,
                        'decoded': decoded_glosses,
                        'wer': sample_wer,
                    })

                results.append({
                    'file': fname,
                    'phrase': phrase_name,
                    'expected': expected_glosses,
                    'decoded': decoded_glosses,
                    'wer': sample_wer,
                    'exact_match': exact_match,
                })
            except Exception as e:
                print(f"  [{i}/{len(files)}] FAIL {fname}: {e}")
                continue

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(files)}]")

    # Compute overall stats
    total = len(results)
    exact_matches = sum(1 for r in results if r['exact_match'])
    avg_wer = sum(r['wer'] for r in results) / max(total, 1)

    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS ({total} phrases)")
    print(f"{'='*70}")
    print(f"Exact match rate: {exact_matches}/{total} = {100*exact_matches/max(total,1):.2f}%")
    print(f"Average WER:      {avg_wer:.2f}%")
    print(f"\nPer-phrase breakdown:")
    print(f"{'Phrase':<25} {'N':>5} {'Exact':>8} {'WER':>8}  Sample decodes")
    print('-' * 90)
    for phrase, stats in sorted(per_phrase_stats.items()):
        if stats['total'] == 0:
            continue
        pct_exact = 100 * stats['correct'] / stats['total']
        avg_w = stats['wer_sum'] / stats['total']
        print(f"{phrase:<25} {stats['total']:>5} {pct_exact:>7.1f}% {avg_w:>7.2f}%")
        for d in stats['decoded'][:2]:
            exp = ' '.join(d['expected'])
            dec = ' '.join(d['decoded'])
            print(f"    exp: {exp}")
            print(f"    got: {dec}  (WER {d['wer']:.1f}%)")

    # Save detailed report
    out = {
        'total_phrases': total,
        'exact_match_rate': 100 * exact_matches / max(total, 1),
        'average_wer': avg_wer,
        'per_phrase': {p: {
            'total': s['total'],
            'exact_matches': s['correct'],
            'exact_match_pct': 100 * s['correct'] / max(s['total'], 1),
            'avg_wer': s['wer_sum'] / max(s['total'], 1),
            'samples': s['decoded'][:5],
        } for p, s in per_phrase_stats.items() if s['total'] > 0},
    }
    out_path = 'mobile_export/reports/phrase_generalization_test.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved report: {out_path}")


if __name__ == '__main__':
    main()
