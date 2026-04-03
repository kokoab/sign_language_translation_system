"""
Test Stage 2 CTC on phrase videos and isolated sign videos.
Uses Apple Vision extraction (same as demo/training).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/test_stage2_videos.py
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))

# Fake mediapipe to avoid import error
import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract_apple_vision import extract_frames_continuous, extract_frames_isolated
from train_stage_1 import compute_bone_features_np
from model_v14 import SLTStage1V14
from train_stage_2 import SLTStage2CTC

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1_CKPT = os.path.join(PROJECT_ROOT, "models", "output_v15_clean", "best_model.pth")
STAGE2_CKPT = os.path.join(PROJECT_ROOT, "models", "output_stage2_v15_reextracted", "stage2_best_model.pth")
DEVICE = torch.device("cpu")

PHRASE_GLOSSES = {
    "GOOD_MORNING": "GOOD MORNING",
    "HELLO_HOW_YOU": "HELLO HOW YOU",
    "PLEASE_HELP_ME": "PLEASE HELP I",
    "SORRY_I_LATE": "SORRY I LATE",
    "MY_NAME": "MY NAME",
    "YESTERDAY_TEACHER_MEET": "YESTERDAY TEACHER MEET",
    "THANKYOU_FRIEND": "THANKYOU FRIEND",
    "TOMORROW_SCHOOL_GO": "TOMORROW SCHOOL GO",
    "I_WANT_FOOD": "I WANT EAT_FOOD",
}

def load_models():
    # Stage 1
    ckpt1 = torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=False)
    idx_to_label = {str(v): k for k, v in ckpt1['label_to_idx'].items()}
    d_model = ckpt1.get('d_model', 384)
    num_classes = len(ckpt1['label_to_idx'])
    model1 = SLTStage1V14(num_classes=num_classes, d_model=d_model).to(DEVICE)
    sd1 = {k.replace('_orig_mod.', ''): v for k, v in ckpt1['model_state_dict'].items()}
    model1.load_state_dict(sd1, strict=False)
    model1.eval()
    model1.set_epoch(200)

    # Stage 2 — don't pass stage1_ckpt to avoid v14 encoder auto-detection
    # The checkpoint was trained with train_stage_2.py's own DSGCNEncoder (3 GCN layers)
    ckpt2 = torch.load(STAGE2_CKPT, map_location=DEVICE, weights_only=False)
    s2_vocab = ckpt2['vocab_size']
    s2_d_model = ckpt2.get('d_model', 384)
    model2 = SLTStage2CTC(vocab_size=s2_vocab, stage1_ckpt=None, d_model=s2_d_model).to(DEVICE)
    sd2 = {k.replace('_orig_mod.', ''): v for k, v in ckpt2['model_state_dict'].items()}
    model2.load_state_dict(sd2, strict=False)
    model2.eval()
    idx_to_gloss = ckpt2['idx_to_gloss']

    return model1, model2, idx_to_label, idx_to_gloss

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames

def add_bone_features(data):
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    return compute_bone_features_np(data)

def decode_ctc(logits, lengths, idx_to_gloss):
    """Greedy CTC decode."""
    preds = logits.argmax(dim=-1)  # [B, T]
    results = []
    for b in range(preds.shape[0]):
        seq = preds[b, :lengths[b]].tolist()
        # Remove blanks and duplicates
        decoded = []
        prev = 0  # blank
        for idx in seq:
            if idx != 0 and idx != prev:
                decoded.append(idx)
            prev = idx
        glosses = [idx_to_gloss.get(str(i), idx_to_gloss.get(i, f"UNK{i}")) for i in decoded]
        results.append(" ".join(glosses))
    return results

def test_phrases(model1, model2, idx_to_label, idx_to_gloss):
    phrase_dir = os.path.join(PROJECT_ROOT, "data", "raw_videos", "phrases")
    print("\n" + "="*60)
    print("PHRASE VIDEO TEST (Stage 2 CTC)")
    print("="*60)

    total = 0
    correct_words = 0
    total_words = 0

    for phrase_name, expected_gloss in PHRASE_GLOSSES.items():
        pdir = os.path.join(phrase_dir, phrase_name)
        if not os.path.isdir(pdir):
            print(f"  Skip {phrase_name} (not found)")
            continue

        videos = sorted(glob.glob(os.path.join(pdir, "*.mp4")))[:5]  # Test 5 per phrase
        phrase_correct = 0
        phrase_total = 0
        expected_words = expected_gloss.split()

        for vid in videos:
            frames = read_video(vid)
            if len(frames) < 8:
                continue

            data = extract_frames_continuous(frames)
            if data is None:
                continue

            data = add_bone_features(data)
            x = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)
            x_lens = torch.tensor([x.shape[1]], dtype=torch.long)

            with torch.no_grad():
                logits, out_lens = model2(x, x_lens)
                decoded = decode_ctc(logits, out_lens, idx_to_gloss)

            pred = decoded[0] if decoded else ""
            pred_words = pred.split() if pred else []

            # Word-level accuracy
            for ew in expected_words:
                total_words += 1
                if ew in pred_words:
                    correct_words += 1

            match = pred.strip() == expected_gloss.strip()
            if match:
                phrase_correct += 1
            phrase_total += 1
            total += 1

        if phrase_total > 0:
            acc = phrase_correct / phrase_total * 100
            print(f"  {phrase_name:30s} {phrase_correct}/{phrase_total} exact ({acc:.0f}%) | expected: {expected_gloss}")
        # Show last prediction as example
        if videos:
            print(f"    Last pred: '{pred}'")

    if total_words > 0:
        print(f"\n  Word-level accuracy: {correct_words}/{total_words} ({correct_words/total_words*100:.1f}%)")
    print(f"  Total exact match: tested {total} videos")

def test_isolated(model1, idx_to_label):
    sample_dir = os.path.join(PROJECT_ROOT, "sample_videos")
    print("\n" + "="*60)
    print("ISOLATED VIDEO TEST (Stage 1)")
    print("="*60)

    for vid_path in sorted(glob.glob(os.path.join(sample_dir, "*.mp4"))):
        name = os.path.basename(vid_path)
        frames = read_video(vid_path)
        if len(frames) < 8:
            print(f"  {name}: too short ({len(frames)} frames)")
            continue

        data = extract_frames_isolated(frames)
        if data is None:
            print(f"  {name}: extraction failed")
            continue

        data = add_bone_features(data)
        x = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model1(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=-1)
            top5 = torch.topk(probs, 5, dim=-1)

        print(f"  {name}:")
        for i in range(5):
            idx = top5.indices[0][i].item()
            prob = top5.values[0][i].item()
            label = idx_to_label.get(str(idx), f"UNK{idx}")
            marker = " <--" if i == 0 else ""
            print(f"    {label:20s} {prob*100:5.1f}%{marker}")

def test_isolated_from_training(model1, idx_to_label):
    """Test a few random training videos to verify extraction match."""
    asl_dir = os.path.join(PROJECT_ROOT, "data", "raw_videos", "ASL VIDEOS")
    if not os.path.isdir(asl_dir):
        return

    print("\n" + "="*60)
    print("ISOLATED TRAINING VIDEO TEST (random 10)")
    print("="*60)

    # Pick 10 random class folders
    classes = sorted(os.listdir(asl_dir))
    import random
    random.seed(42)
    selected = random.sample(classes, min(10, len(classes)))

    correct = 0
    total = 0
    for cls_name in selected:
        cls_dir = os.path.join(asl_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        vids = glob.glob(os.path.join(cls_dir, "*.mp4"))
        if not vids:
            continue
        vid = vids[0]
        frames = read_video(vid)
        if len(frames) < 8:
            continue

        data = extract_frames_isolated(frames)
        if data is None:
            continue

        data = add_bone_features(data)
        x = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model1(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_idx = logits.argmax(dim=-1).item()
            pred_label = idx_to_label.get(str(pred_idx), f"UNK{pred_idx}")

        match = "OK" if pred_label == cls_name else "MISS"
        if pred_label == cls_name:
            correct += 1
        total += 1
        print(f"  {cls_name:20s} -> {pred_label:20s} [{match}]")

    if total > 0:
        print(f"\n  Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")


if __name__ == "__main__":
    print("Loading models...")
    model1, model2, idx_to_label, idx_to_gloss = load_models()
    print(f"  Stage 1: {STAGE1_CKPT}")
    print(f"  Stage 2: {STAGE2_CKPT}")
    print(f"  Labels: {len(idx_to_label)} classes")
    print(f"  Glosses: {len(idx_to_gloss)} vocab")

    test_isolated(model1, idx_to_label)
    test_isolated_from_training(model1, idx_to_label)
    test_phrases(model1, model2, idx_to_label, idx_to_gloss)
