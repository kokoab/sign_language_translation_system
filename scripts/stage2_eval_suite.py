"""Comprehensive Stage 2 evaluation suite.
Tests the current model on several scenarios to diagnose the template memorization problem.
Run this BEFORE and AFTER each fix to compare.

Scenarios:
  1. Held-out phrases (117 files) — basic phrase WER
  2. Single-sign inputs (isolated clips) — tests if model outputs just 1 sign or hallucinates phrase
  3. Sample videos (5 recordings) — real-world test
  4. Cross-phrase swap — give model HOW signs and see if it adds HELLO/YOU
"""
import os, sys, random, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))

import numpy as np
import torch
from collections import defaultdict, Counter

from model_v16 import SLTStage2V16CTC
from train_stage_2_v16 import RealPhraseCTCDataset


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


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = SLTStage2V16CTC(
        vocab_size=ckpt.get('vocab_size', 311), stage1_ckpt=None,
        in_channels=ckpt.get('in_channels', 5), dim=ckpt.get('d_model', 384),
    ).to(device)
    sd = ckpt.get('ema_shadow') or ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False); model.eval()
    g2i = ckpt.get('label_to_idx') or ckpt.get('gloss_to_idx')
    i2g = {int(v): k for k, v in g2i.items()}
    return model, i2g, g2i


def pad_and_infer(model, arr, device):
    T = arr.shape[0]
    if T % 32 != 0:
        pad = np.zeros((((T + 31) // 32) * 32 - T, 61, 5), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    x = torch.from_numpy(arr.astype(np.float32))[None, ...].to(device)
    with torch.no_grad():
        logits, _ = model(x)
    return ctc_decode(logits, blank=0)


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


def eval_heldout_phrases(model, i2g, g2i, device):
    """Scenario 1: Held-out phrase videos."""
    phrase_dir = None
    for c in ['ASL_phrases_v16', 'src_v16/ASL_phrases_v16']:
        if os.path.isdir(c): phrase_dir = c; break
    if not phrase_dir: return None

    phrase_ds = RealPhraseCTCDataset(phrase_dir, g2i)
    n = len(phrase_ds.files)
    idx = list(range(n))
    random.Random(42).shuffle(idx)
    val_idx = sorted(idx[:max(1, int(n * 0.15))])

    results = defaultdict(lambda: {'total': 0, 'exact': 0, 'wer_sum': 0.0})
    all_decoded = []

    for i in val_idx:
        fpath = phrase_ds.files[i]
        fname = os.path.basename(fpath)
        phrase_name = next((p for p in PHRASE_GLOSSES if fname.startswith(p + '_')), None)
        if phrase_name is None: continue
        expected = PHRASE_GLOSSES[phrase_name]
        arr = np.load(fpath).astype(np.float32)
        if arr.shape[-1] != 5: continue
        T = arr.shape[0]
        decoded_idx = pad_and_infer(model, arr, device)
        decoded = [i2g.get(j, '?') for j in decoded_idx]
        w = wer(expected, decoded)
        exact = decoded == expected
        key = (phrase_name, 'multi-clip' if T >= 64 else 'single-clip')
        results[key]['total'] += 1
        if exact: results[key]['exact'] += 1
        results[key]['wer_sum'] += w
        all_decoded.append((fname, expected, decoded, w))
    return results, all_decoded


def eval_isolated_signs(model, i2g, g2i, device, n_samples=50):
    """Scenario 2: Feed isolated sign clips. Does Stage 2 output just 1 gloss or hallucinate a phrase?"""
    iso_dir = 'src_v16/ASL_landmarks_v16'
    if not os.path.isdir(iso_dir):
        return None

    # Pick samples from phrase vocabulary signs only
    phrase_signs = set()
    for glosses in PHRASE_GLOSSES.values():
        phrase_signs.update(glosses)

    results = defaultdict(lambda: {'total': 0, 'exact_1tok': 0, 'hallucinated': 0, 'decoded_counts': Counter()})
    rng = random.Random(42)

    for sign in sorted(phrase_signs):
        files = [f for f in os.listdir(iso_dir) if f.startswith(sign + '_')]
        rng.shuffle(files)
        for fname in files[:n_samples]:
            try:
                arr = np.load(os.path.join(iso_dir, fname)).astype(np.float32)
                if arr.shape != (32, 61, 5): continue
                decoded_idx = pad_and_infer(model, arr, device)
                decoded = tuple(i2g.get(j, '?') for j in decoded_idx)
                results[sign]['total'] += 1
                if decoded == (sign,):
                    results[sign]['exact_1tok'] += 1
                if len(decoded) > 1:
                    results[sign]['hallucinated'] += 1
                results[sign]['decoded_counts'][decoded] += 1
            except Exception:
                continue
    return results


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/Users/frnzlo/Downloads/results (1)/models/output_stage2_v16_phrases/stage2_best_model.pth'
    print(f'Evaluating: {ckpt_path}')
    print(f'Device: {device}\n')

    model, i2g, g2i = load_model(ckpt_path, device)

    # Scenario 1
    print('=' * 70)
    print('SCENARIO 1: Held-out phrase videos (117 files, same signer as train)')
    print('=' * 70)
    r1, decoded = eval_heldout_phrases(model, i2g, g2i, device)
    if r1:
        overall_t = sum(v['total'] for v in r1.values())
        overall_e = sum(v['exact'] for v in r1.values())
        overall_w = sum(v['wer_sum'] for v in r1.values())
        print(f'Overall: N={overall_t}, exact={100*overall_e/overall_t:.1f}%, WER={overall_w/overall_t:.1f}%')
        # Break down by multi-clip vs single-clip
        mc = {k: v for k, v in r1.items() if k[1] == 'multi-clip'}
        sc = {k: v for k, v in r1.items() if k[1] == 'single-clip'}
        if mc:
            t = sum(v['total'] for v in mc.values())
            e = sum(v['exact'] for v in mc.values())
            w = sum(v['wer_sum'] for v in mc.values())
            print(f'  Multi-clip:  N={t}, exact={100*e/t:.1f}%, WER={w/t:.1f}%')
        if sc:
            t = sum(v['total'] for v in sc.values())
            e = sum(v['exact'] for v in sc.values())
            w = sum(v['wer_sum'] for v in sc.values())
            print(f'  Single-clip: N={t}, exact={100*e/t:.1f}%, WER={w/t:.1f}%')

    # Scenario 2
    print('\n' + '=' * 70)
    print('SCENARIO 2: Isolated sign clips (does Stage 2 hallucinate phrase templates?)')
    print('=' * 70)
    print('If training is healthy: isolated HELLO → decode as [HELLO]')
    print('If template-memorized: isolated HELLO → decode as [HELLO, HOW, YOU]\n')
    r2 = eval_isolated_signs(model, i2g, g2i, device, n_samples=30)
    if r2:
        print(f'{"Sign":<15} {"N":>4} {"Single-tok":>11} {"Hallucin.":>10}  Most common outputs')
        print('-' * 120)
        total_t = 0; total_e = 0; total_h = 0
        for sign, s in sorted(r2.items()):
            top = s['decoded_counts'].most_common(2)
            top_str = '; '.join(f'{list(d)} × {c}' for d, c in top)
            print(f'{sign:<15} {s["total"]:>4} {100*s["exact_1tok"]/max(s["total"],1):>10.1f}% '
                  f'{100*s["hallucinated"]/max(s["total"],1):>9.1f}%  {top_str[:70]}')
            total_t += s['total']; total_e += s['exact_1tok']; total_h += s['hallucinated']
        print('-' * 120)
        print(f'{"TOTAL":<15} {total_t:>4} {100*total_e/max(total_t,1):>10.1f}% {100*total_h/max(total_t,1):>9.1f}%')
        print(f'\nHallucination rate (isolated sign → multi-gloss output): {100*total_h/max(total_t,1):.1f}%')
        print(f'Clean single-token rate (isolated sign → just that sign): {100*total_e/max(total_t,1):.1f}%')


if __name__ == '__main__':
    main()
