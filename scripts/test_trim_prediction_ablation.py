"""
Prediction-level ablation: does dead-frame trimming help or hurt?

For each video, run full Stage 1 + Stage 2 inference TWICE:
  A) trim_dead=True   (current default)
  B) trim_dead=False  (use all frames)

Compare:
  - Stage 1 top-1 per clip (did trimming change predictions?)
  - Stage 2 beam output (gloss sequences)
  - Ground truth match when derivable from filename/parent dir

Ground truth sources:
  - sample_videos/HELLO_training.mp4 → "HELLO" (uppercase tokens before _training)
  - data/raw_videos/ASL VIDEOS/HELLO/*.mp4 → "HELLO" (parent dir)
  - data/raw_videos/PHRASES/GOOD_MORNING/*.mp4 → "GOOD MORNING" (parent dir, _ → space)

Usage:
  KMP_DUPLICATE_LIB_OK=TRUE conda run -n sign_ai python scripts/test_trim_prediction_ablation.py
"""
import os, sys, random, json, cv2, gc, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src_v16"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn.functional as F

from src_v16.inference_v16 import (
    extract_continuous_smart, mirror_tta_v16, ctc_beam_search,
    ctc_greedy_decode, dedup_consecutive, disambiguate_glosses,
    add_vel_acc, load_models,
)

random.seed(42)

SAMPLE_DIR   = ROOT / "sample_videos"
ISOLATED_DIR = ROOT / "data" / "raw_videos" / "ASL VIDEOS"
PHRASE_DIR   = ROOT / "data" / "raw_videos" / "PHRASES"

S1_CKPT = str(ROOT / "src_v16" / "output_v16_d384" / "best_model.pth")
S2_CKPT = "/Users/frnzlo/Downloads/results (2)/models/output_stage2_v16_fixed/stage2_best_model.pth"

N_ISOLATED = 20
N_PHRASE   = 20
DEVICE     = "cpu"


def load_frames(path, max_frames=600):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        frames.append(f)
    cap.release()
    return frames


def ground_truth(path, category):
    if category == "sample":
        stem = path.stem
        # HELLO_training, HELLO_HOW_YOU_training, "how you", "thank you"
        if stem.endswith("_training"):
            stem = stem[:-len("_training")]
            return stem.replace("_", " ").upper()
        return stem.replace("_", " ").upper()
    elif category == "isolated":
        return path.parent.name.upper()
    elif category == "phrase":
        return path.parent.name.replace("_", " ").upper()
    return ""


def wer(ref, hyp):
    """Word error rate. ref and hyp are token lists."""
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / m


def run_pipeline(frames, s1, s1_i2l, in_channels, s2, s2_i2g, trim_dead):
    """Run Stage 1 + Stage 2 with or without dead-frame trim. Return predictions."""
    arr = extract_continuous_smart(frames, segment_len=28, stride=14, trim_dead=trim_dead)
    if arr is None:
        return None
    arr = arr.astype(np.float32)
    T = arr.shape[0]
    nc = T // 32
    if nc < 1:
        return None

    # Channel expansion if Stage 1 expects 9ch
    if in_channels > 5:
        clips_9ch = [add_vel_acc(arr[ci * 32:(ci + 1) * 32]) for ci in range(nc)]
        arr_input = np.concatenate(clips_9ch, axis=0)
    else:
        arr_input = arr
    if arr_input.shape[-1] > in_channels:
        arr_input = arr_input[..., :in_channels]

    # Stage 1
    s1_preds = []
    for ci in range(nc):
        clip = torch.from_numpy(arr_input[ci * 32:(ci + 1) * 32])[None, ...].to(DEVICE)
        with torch.no_grad():
            logits = s1(clip)
            clip_m = mirror_tta_v16(clip)
            logits_m = s1(clip_m)
            probs = (F.softmax(logits, dim=-1) + F.softmax(logits_m, dim=-1)) / 2
        top1_idx = probs[0].argmax().item()
        top1_conf = probs[0, top1_idx].item()
        s1_preds.append((s1_i2l[top1_idx], top1_conf))

    # Stage 2
    s2_glosses = []
    if s2 is not None:
        if T % 32 != 0:
            pad = np.zeros((((T + 31) // 32) * 32 - T,) + arr_input.shape[1:], dtype=np.float32)
            arr_padded = np.concatenate([arr_input, pad], axis=0)
        else:
            arr_padded = arr_input
        x = torch.from_numpy(arr_padded)[None, ...].to(DEVICE)
        with torch.no_grad():
            logits, _ = s2(x)
            x_m = mirror_tta_v16(x)
            logits_m, _ = s2(x_m)
            avg_probs = (F.softmax(logits, dim=-1) + F.softmax(logits_m, dim=-1)) / 2
            log_probs = avg_probs.log()[0].cpu().numpy()
        beam = ctc_beam_search(log_probs, beam_width=25, blank=0)
        if beam:
            _, toks = beam[0]
            gs = [s2_i2g.get(int(t), f"<{t}>") for t in toks]
            gs = dedup_consecutive(gs)
            gs = disambiguate_glosses(gs, arr, ch_mask=3, ch_xyz=slice(0, 3))
            s2_glosses = gs

    return {
        "n_frames": T,
        "n_clips": nc,
        "stage1": [p[0] for p in s1_preds],
        "stage1_conf": [p[1] for p in s1_preds],
        "stage2": s2_glosses,
    }


def sample_videos():
    samples = []
    for f in sorted(SAMPLE_DIR.glob("*.mp4")):
        samples.append((f, "sample"))
    iso_classes = [d for d in ISOLATED_DIR.iterdir() if d.is_dir()]
    random.shuffle(iso_classes)
    picked = 0
    for cls in iso_classes:
        vids = list(cls.glob("*.mp4")) + list(cls.glob("*.mov"))
        if vids:
            samples.append((random.choice(vids), "isolated"))
            picked += 1
            if picked >= N_ISOLATED:
                break
    phrase_classes = [d for d in PHRASE_DIR.iterdir() if d.is_dir()]
    random.shuffle(phrase_classes)
    picked = 0
    for cls in phrase_classes:
        vids = list(cls.glob("*.mp4")) + list(cls.glob("*.mov"))
        if vids:
            samples.append((random.choice(vids), "phrase"))
            picked += 1
            if picked >= N_PHRASE:
                break
    return samples


def main():
    print("Loading models ...")
    s1, s1_i2l, in_channels, s2, s2_i2g = load_models(S1_CKPT, S2_CKPT, DEVICE)
    print(f"Stage 1 in_channels={in_channels}  classes={len(s1_i2l)}")
    print(f"Stage 2 vocab={len(s2_i2g) if s2_i2g else 'N/A'}\n")

    vids = sample_videos()
    print(f"Testing {len(vids)} videos\n")

    rows = []
    for i, (path, cat) in enumerate(vids):
        gt = ground_truth(path, cat)
        print(f"[{i+1}/{len(vids)}] {cat:8s} {path.name:50s} GT={gt}")
        try:
            frames = load_frames(path)
            if len(frames) < 16:
                print("  SKIP (too short)")
                continue
            t0 = time.time()
            on  = run_pipeline(frames, s1, s1_i2l, in_channels, s2, s2_i2g, trim_dead=True)
            off = run_pipeline(frames, s1, s1_i2l, in_channels, s2, s2_i2g, trim_dead=False)
            if on is None or off is None:
                print("  SKIP (extraction failed)")
                continue

            gt_tokens = gt.split()
            # Stage 2 WER + exact match
            wer_on  = wer(gt_tokens, on["stage2"])
            wer_off = wer(gt_tokens, off["stage2"])
            exact_on  = on["stage2"] == gt_tokens
            exact_off = off["stage2"] == gt_tokens

            rows.append({
                "path": path.name, "category": cat, "gt": gt,
                "frames": len(frames),
                "on_frames": on["n_frames"], "off_frames": off["n_frames"],
                "on_clips": on["n_clips"], "off_clips": off["n_clips"],
                "on_s1": on["stage1"], "off_s1": off["stage1"],
                "on_s2": on["stage2"], "off_s2": off["stage2"],
                "wer_on": wer_on, "wer_off": wer_off,
                "exact_on": exact_on, "exact_off": exact_off,
                "time": round(time.time() - t0, 1),
            })
            mark = "  " if on["stage2"] == off["stage2"] else "* "
            print(f"  {mark}trim ON : clips={on['n_clips']}  S2={' '.join(on['stage2']) or '(empty)'}  WER={wer_on:.2f}")
            print(f"  {mark}trim OFF: clips={off['n_clips']}  S2={' '.join(off['stage2']) or '(empty)'}  WER={wer_off:.2f}")
        except Exception as e:
            print(f"  ERR {e}")
        gc.collect()

    # Aggregate
    print(f"\n{'='*80}\nSummary ({len(rows)} videos)\n{'='*80}\n")

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    for cat, items in by_cat.items():
        n = len(items)
        diff = sum(1 for r in items if r["on_s2"] != r["off_s2"])
        exact_on  = sum(1 for r in items if r["exact_on"])
        exact_off = sum(1 for r in items if r["exact_off"])
        wer_on_mean  = np.mean([r["wer_on"]  for r in items])
        wer_off_mean = np.mean([r["wer_off"] for r in items])
        wins_on  = sum(1 for r in items if r["wer_on"]  < r["wer_off"])
        wins_off = sum(1 for r in items if r["wer_off"] < r["wer_on"])
        ties     = sum(1 for r in items if r["wer_on"] == r["wer_off"])

        print(f"{cat.upper()} ({n} videos)")
        print(f"  Predictions differ:   {diff}/{n}")
        print(f"  Exact match  ON/OFF:  {exact_on}/{n}  vs  {exact_off}/{n}")
        print(f"  Mean WER     ON/OFF:  {wer_on_mean:.3f}  vs  {wer_off_mean:.3f}")
        print(f"  Head-to-head: ON wins {wins_on}, OFF wins {wins_off}, tie {ties}")
        print()

    # Cases where ON and OFF diverge (interesting)
    diverging = [r for r in rows if r["on_s2"] != r["off_s2"]]
    if diverging:
        print(f"{'='*80}\nDiverging cases ({len(diverging)}):\n{'='*80}")
        for r in diverging:
            winner = "ON"  if r["wer_on"]  < r["wer_off"] else \
                     "OFF" if r["wer_off"] < r["wer_on"]  else "TIE"
            print(f"  [{winner}] {r['category']:8s} GT={r['gt']:25s}  "
                  f"ON=({r['on_frames']}fr,{r['on_clips']}c) {' '.join(r['on_s2']) or '-'}  |  "
                  f"OFF=({r['off_frames']}fr,{r['off_clips']}c) {' '.join(r['off_s2']) or '-'}")

    out = ROOT / "mobile_export" / "reports" / "trim_prediction_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
