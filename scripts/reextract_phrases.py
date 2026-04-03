"""
Re-extract phrase videos using extract_frames_continuous.
This ensures training .npy matches inference format exactly.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python scripts/reextract_phrases.py
"""
import os, sys, glob, json, time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import types
_fake = types.ModuleType('mediapipe')
_fake.solutions = types.ModuleType('mediapipe.solutions')
sys.modules['mediapipe'] = _fake
sys.modules['mediapipe.solutions'] = _fake.solutions

from extract_apple_vision import extract_frames_continuous

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

if __name__ == "__main__":
    phrase_dir = "data/raw_videos/phrases"
    output_dir = "ASL_phrases_reextracted"
    os.makedirs(output_dir, exist_ok=True)

    manifest = {}
    count = 0
    failed = 0
    t0 = time.time()

    for phrase_name, gloss_str in PHRASE_GLOSSES.items():
        pdir = os.path.join(phrase_dir, phrase_name)
        if not os.path.isdir(pdir):
            print(f"  Skip {phrase_name} (not found)")
            continue

        videos = sorted(glob.glob(os.path.join(pdir, "*.mp4")))
        print(f"  {phrase_name}: {len(videos)} videos")

        for vi, vid in enumerate(videos):
            cap = cv2.VideoCapture(vid)
            frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(f)
            cap.release()

            if len(frames) < 8:
                failed += 1
                continue

            data = extract_frames_continuous(frames)
            if data is None:
                failed += 1
                continue

            vid_hash = os.path.splitext(os.path.basename(vid))[0]
            out_name = f"{phrase_name}_{vi:04d}_{vid_hash[:8]}.npy"
            np.save(os.path.join(output_dir, out_name), data)
            manifest[out_name] = gloss_str
            count += 1

            if (count % 50) == 0:
                elapsed = time.time() - t0
                rate = count / elapsed
                remaining = (780 - count) / rate if rate > 0 else 0
                print(f"    {count} done | {rate:.1f} vid/s | ETA {remaining/60:.0f}m")

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone: {count} extracted, {failed} failed")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Output: {output_dir}/manifest.json")
