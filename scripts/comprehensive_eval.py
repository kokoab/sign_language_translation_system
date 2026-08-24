"""
Comprehensive end-to-end evaluation of the v16 pipeline after all inference fixes.

Tests:
  A. Real-world sample videos (6 videos with known ground truth)
  B. Stage 1 regression on 200 training videos (fresh extraction)
  C. Stage 2 held-out phrase WER (117 files)
  D. Hallucination rate on isolated signs (phrase vocabulary)
  E. Full new-pipeline test on problem video

Compares BEFORE (old extract_continuous_v16 + no TTA + greedy CTC)
        vs
        AFTER  (sliding window + TTA + beam search + dedup + disambiguation)
"""
import os, sys, json, random, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from model_v16 import SLTStage1V16, SLTStage2V16CTC
from extract_v16 import extract_frames_v16, extract_continuous_v16
from inference_v16 import (
    run_inference, load_models, mirror_tta_v16, ctc_beam_search,
    ctc_greedy_decode, dedup_consecutive, disambiguate_glosses,
    extract_continuous_smart, add_vel_acc,
)

# ── Ground truth for sample videos ──
SAMPLE_GT = {
    'HELLO_HOW_YOU_training.mp4': ['HELLO', 'HOW', 'YOU'],
    'HELLO_training.mp4': ['HELLO'],
    'HOW_YOU_training.mp4': ['HOW', 'YOU'],
    'how you.mp4': ['HOW', 'YOU'],
    'thank you.mp4': ['THANKYOU'],
}

PROBLEM_VIDEO = '/Users/frnzlo/Downloads/AQN1WD8awPGFbyTSpbhF4cX2Y86PkNtIYdq9tUPyAE8hcQ_FVCRYRrBp2Qt6qjuud2-rL0tzxaVfUO-7TntMNSIa2cHI4QDYAAsaOxJu0g.mp4'
PROBLEM_GT = ['HELLO', 'GOOD', 'MORNING', 'HOW', 'YOU']

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
            dp[i][j] = dp[i-1][j-1] if ref[i-1] == hyp[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return 100.0 * dp[r][h] / r


def load_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    return frames


def compute_recall(pred, ref):
    """What fraction of reference signs appear in prediction (order-independent)."""
    if not ref: return 100.0
    matches = sum(1 for r in ref if r in pred)
    return 100.0 * matches / len(ref)


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}\n')

    s1_ckpt = 'src_v16/output_v16_d384/best_model.pth'
    s2_ckpt = '/Users/frnzlo/Downloads/results (2)/models/output_stage2_v16_fixed/stage2_best_model.pth'

    print('Loading models...')
    s1, s1_i2l, in_channels, s2, s2_i2g = load_models(s1_ckpt, s2_ckpt, device)
    print(f'  Stage 1: in_channels={in_channels}')
    print(f'  Stage 2: vocab={len(s2_i2g)}')

    results = {}

    # ═══════════════════════════════════════════════════════════
    # TEST A: Sample videos
    # ═══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('TEST A: Sample videos (6 videos with known ground truth)')
    print('=' * 72)

    test_a = []
    for fname, gt in SAMPLE_GT.items():
        path = os.path.join('sample_videos', fname)
        if not os.path.exists(path):
            continue
        res = run_inference(path, s1, s1_i2l, in_channels, s2, s2_i2g,
                             device=device, use_tta=True, beam_width=25, verbose=False)
        if 'error' in res:
            print(f'  {fname}: ERROR {res["error"]}')
            continue
        pred_greedy = res['stage2_greedy'].split()
        pred_beam = res['stage2_beam'].split()
        r = {
            'video': fname,
            'gt': gt,
            'greedy': pred_greedy,
            'beam': pred_beam,
            'wer_greedy': wer(gt, pred_greedy),
            'wer_beam': wer(gt, pred_beam),
            'recall_greedy': compute_recall(pred_greedy, gt),
            'recall_beam': compute_recall(pred_beam, gt),
            'exact_greedy': pred_greedy == gt,
            'exact_beam': pred_beam == gt,
        }
        test_a.append(r)
        print(f'  {fname:<35} GT={" ".join(gt)}')
        print(f'    greedy: {" ".join(pred_greedy) or "(empty)"}  WER={r["wer_greedy"]:.0f}%  recall={r["recall_greedy"]:.0f}%  {"✓" if r["exact_greedy"] else ""}')
        gc.collect()

    # Problem video
    print(f'\n  Problem video (portrait 540x960):')
    print(f'    GT: {" ".join(PROBLEM_GT)}')
    res = run_inference(PROBLEM_VIDEO, s1, s1_i2l, in_channels, s2, s2_i2g,
                        device=device, use_tta=True, beam_width=25, verbose=False)
    if 'error' not in res:
        pred = res['stage2_greedy'].split()
        pred_b = res['stage2_beam'].split()
        problem_result = {
            'gt': PROBLEM_GT,
            'greedy': pred,
            'beam': pred_b,
            'recall_greedy': compute_recall(pred, PROBLEM_GT),
            'recall_beam': compute_recall(pred_b, PROBLEM_GT),
            'wer_greedy': wer(PROBLEM_GT, pred),
            'wer_beam': wer(PROBLEM_GT, pred_b),
        }
        print(f'    greedy: {" ".join(pred)}')
        print(f'    recall: {problem_result["recall_greedy"]:.0f}%  WER: {problem_result["wer_greedy"]:.0f}%')
    else:
        problem_result = {'error': res['error']}
        print(f'    ERROR: {res["error"]}')

    results['test_a_samples'] = test_a
    results['test_a_problem'] = problem_result

    # Summary
    exact_count_greedy = sum(1 for r in test_a if r['exact_greedy'])
    avg_wer_greedy = np.mean([r['wer_greedy'] for r in test_a])
    avg_recall_greedy = np.mean([r['recall_greedy'] for r in test_a])
    print(f'\n  Summary on {len(test_a)} sample videos:')
    print(f'    Exact match:  {exact_count_greedy}/{len(test_a)}')
    print(f'    Avg WER:      {avg_wer_greedy:.1f}%')
    print(f'    Avg recall:   {avg_recall_greedy:.1f}%')

    # ═══════════════════════════════════════════════════════════
    # TEST B: Stage 1 regression on 200 training videos
    # ═══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('TEST B: Stage 1 regression (200 training videos, fresh extraction + TTA)')
    print('=' * 72)

    video_dir = 'data/raw_videos/ASL VIDEOS'
    classes = sorted([d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))])
    rng = random.Random(123)
    test_classes = rng.sample(classes, min(200, len(classes)))

    correct_fresh = 0
    correct_tta = 0
    correct_npy = 0
    total = 0

    iso_dir = 'src_v16/ASL_landmarks_v16'
    t_start = time.time()
    for ci, cls in enumerate(test_classes):
        cls_dir = os.path.join(video_dir, cls)
        vids = [f for f in os.listdir(cls_dir) if f.endswith('.mp4')]
        if not vids: continue
        vid = vids[ci % len(vids)]  # rotate through videos

        frames = load_frames(os.path.join(cls_dir, vid))
        if len(frames) < 4: continue
        arr = extract_frames_v16(frames)
        if arr is None: continue
        arr = arr.astype(np.float32)

        # Match input_channels (may need vel/acc)
        if in_channels > 5:
            arr_input = add_vel_acc(arr)[..., :in_channels]
        else:
            arr_input = arr[..., :in_channels]

        x = torch.from_numpy(arr_input)[None, ...].to(device)
        with torch.no_grad():
            logits = s1(x)
            pred_fresh = s1_i2l[logits.argmax(dim=-1).item()]
            # With TTA
            x_m = mirror_tta_v16(x)
            logits_m = s1(x_m)
            avg = (F.softmax(logits, dim=-1) + F.softmax(logits_m, dim=-1)) / 2
            pred_tta = s1_i2l[avg.argmax(dim=-1).item()]

        # Stored .npy reference
        npy_files = [f for f in os.listdir(iso_dir) if f.startswith(cls + '_')]
        if npy_files:
            npy = np.load(os.path.join(iso_dir, npy_files[0])).astype(np.float32)
            if in_channels > 5:
                npy = add_vel_acc(npy)[..., :in_channels]
            else:
                npy = npy[..., :in_channels]
            x_npy = torch.from_numpy(npy)[None, ...].to(device)
            with torch.no_grad():
                pred_npy = s1_i2l[s1(x_npy).argmax(dim=-1).item()]
            if pred_npy == cls: correct_npy += 1

        if pred_fresh == cls: correct_fresh += 1
        if pred_tta == cls: correct_tta += 1
        total += 1

        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f'  [{ci+1}/{len(test_classes)}] fresh={correct_fresh}/{total} tta={correct_tta}/{total} npy={correct_npy}/{total}  ({elapsed:.0f}s)')
        gc.collect()

    print(f'\n  Final ({total} videos):')
    print(f'    Stored .npy:     {correct_npy}/{total} = {100*correct_npy/total:.1f}%')
    print(f'    Fresh extracted: {correct_fresh}/{total} = {100*correct_fresh/total:.1f}%')
    print(f'    Fresh + TTA:     {correct_tta}/{total} = {100*correct_tta/total:.1f}%')
    results['test_b_regression'] = {
        'total': total,
        'npy_accuracy': 100 * correct_npy / total,
        'fresh_accuracy': 100 * correct_fresh / total,
        'tta_accuracy': 100 * correct_tta / total,
    }

    # ═══════════════════════════════════════════════════════════
    # TEST C: Held-out phrase WER (uses pre-extracted npy)
    # ═══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('TEST C: Held-out phrase WER (117 pre-extracted files)')
    print('=' * 72)

    phrase_dir = 'src_v16/ASL_phrases_v16'
    from train_stage_2_v16 import RealPhraseCTCDataset
    g2i_s2 = {v: k for k, v in s2_i2g.items()}
    base = RealPhraseCTCDataset(phrase_dir, g2i_s2)
    idx = list(range(len(base.files)))
    random.Random(42).shuffle(idx)
    val_idx = sorted(idx[:max(1, int(len(idx) * 0.15))])

    wer_sum = 0.0
    exact = 0
    total_c = 0
    multi_clip_wer = 0.0
    multi_clip_exact = 0
    multi_clip_total = 0

    for i in val_idx:
        fpath = base.files[i]
        fname = os.path.basename(fpath)
        phrase_name = next((p for p in PHRASE_GLOSSES if fname.startswith(p + '_')), None)
        if phrase_name is None: continue
        gt = PHRASE_GLOSSES[phrase_name]
        arr = np.load(fpath).astype(np.float32)
        if arr.shape[-1] != 5 or arr.shape[1] != 61: continue
        T = arr.shape[0]

        if in_channels > 5:
            arr_in = add_vel_acc(arr)[..., :in_channels]
        else:
            arr_in = arr[..., :in_channels]

        if T % 32 != 0:
            pad = np.zeros((((T + 31) // 32) * 32 - T,) + arr_in.shape[1:], dtype=np.float32)
            arr_in = np.concatenate([arr_in, pad], axis=0)

        x = torch.from_numpy(arr_in)[None, ...].to(device)
        with torch.no_grad():
            logits, _ = s2(x)
            # Mirror TTA
            x_m = mirror_tta_v16(x)
            logits_m, _ = s2(x_m)
            avg_probs = (F.softmax(logits, dim=-1) + F.softmax(logits_m, dim=-1)) / 2
            log_probs = avg_probs.log()[0].cpu().numpy()

        # Beam search + post-processing
        beam_results = ctc_beam_search(log_probs, beam_width=25, blank=0)
        pred = [s2_i2g.get(int(t), f'<{t}>') for t in beam_results[0][1]] if beam_results else []
        pred = dedup_consecutive(pred)
        pred = disambiguate_glosses(pred, arr, ch_mask=3, ch_xyz=slice(0, 3))

        w = wer(gt, pred)
        is_exact = pred == gt
        wer_sum += w
        if is_exact: exact += 1
        total_c += 1

        if T >= 64:
            multi_clip_wer += w
            if is_exact: multi_clip_exact += 1
            multi_clip_total += 1

    print(f'  Overall: {total_c} files, exact={exact} ({100*exact/total_c:.1f}%), WER={wer_sum/total_c:.1f}%')
    print(f'  Multi-clip (T>=64): {multi_clip_total} files, exact={multi_clip_exact} ({100*multi_clip_exact/max(multi_clip_total,1):.1f}%), WER={multi_clip_wer/max(multi_clip_total,1):.1f}%')
    results['test_c_heldout'] = {
        'total': total_c,
        'exact_pct': 100 * exact / max(total_c, 1),
        'wer': wer_sum / max(total_c, 1),
        'multi_clip_total': multi_clip_total,
        'multi_clip_exact_pct': 100 * multi_clip_exact / max(multi_clip_total, 1),
        'multi_clip_wer': multi_clip_wer / max(multi_clip_total, 1),
    }

    # ═══════════════════════════════════════════════════════════
    # TEST D: Hallucination check on isolated signs
    # ═══════════════════════════════════════════════════════════
    print('\n' + '=' * 72)
    print('TEST D: Hallucination check (isolated signs should decode to 1 gloss)')
    print('=' * 72)

    phrase_signs = set()
    for glosses in PHRASE_GLOSSES.values():
        phrase_signs.update(glosses)

    hall_total = 0
    hall_count = 0
    single_correct = 0
    per_sign = {}
    rng = random.Random(42)

    for sign in sorted(phrase_signs):
        files = [f for f in os.listdir(iso_dir) if f.startswith(sign + '_')]
        rng.shuffle(files)
        for fname in files[:20]:  # 20 samples per sign
            try:
                arr = np.load(os.path.join(iso_dir, fname)).astype(np.float32)
                if arr.shape != (32, 61, 5): continue

                if in_channels > 5:
                    arr_in = add_vel_acc(arr)[..., :in_channels]
                else:
                    arr_in = arr[..., :in_channels]

                x = torch.from_numpy(arr_in)[None, ...].to(device)
                with torch.no_grad():
                    logits, _ = s2(x)
                    x_m = mirror_tta_v16(x)
                    logits_m, _ = s2(x_m)
                    avg_probs = (F.softmax(logits, dim=-1) + F.softmax(logits_m, dim=-1)) / 2
                    log_probs = avg_probs.log()[0].cpu().numpy()

                beam_results = ctc_beam_search(log_probs, beam_width=25, blank=0)
                pred = [s2_i2g.get(int(t), f'<{t}>') for t in beam_results[0][1]] if beam_results else []
                pred = dedup_consecutive(pred)
                pred = disambiguate_glosses(pred, arr, ch_mask=3, ch_xyz=slice(0, 3))

                hall_total += 1
                if len(pred) > 1:
                    hall_count += 1
                if pred == [sign]:
                    single_correct += 1
                per_sign.setdefault(sign, {'total': 0, 'hall': 0, 'correct': 0})
                per_sign[sign]['total'] += 1
                if len(pred) > 1: per_sign[sign]['hall'] += 1
                if pred == [sign]: per_sign[sign]['correct'] += 1
            except Exception:
                continue

    print(f'  Total isolated samples: {hall_total}')
    print(f'  Hallucination rate: {100*hall_count/max(hall_total,1):.1f}% ({hall_count}/{hall_total})')
    print(f'  Clean single-token correct: {100*single_correct/max(hall_total,1):.1f}% ({single_correct}/{hall_total})')
    print(f'  Worst signs (hallucination):')
    worst = sorted(per_sign.items(), key=lambda x: -x[1]['hall']/max(x[1]['total'],1))[:5]
    for sign, s in worst:
        print(f'    {sign:<15} hall={s["hall"]}/{s["total"]} correct={s["correct"]}/{s["total"]}')
    results['test_d_hallucination'] = {
        'total': hall_total,
        'hallucination_pct': 100 * hall_count / max(hall_total, 1),
        'single_correct_pct': 100 * single_correct / max(hall_total, 1),
    }

    # Save report
    out_path = 'mobile_export/reports/comprehensive_eval.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n\nFull report saved to: {out_path}')


if __name__ == '__main__':
    main()
