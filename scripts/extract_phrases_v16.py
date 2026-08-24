"""
Extract phrase videos using v16 Apple Vision pipeline.
Outputs variable-length [T, 61, 5] .npy files for Stage 2 CTC.

Uses the SAME extraction pipeline as isolated signs (extract_v16.py)
to ensure NO extraction mismatch between training data and phrases.

Phrase structure:
    data/raw_videos/PHRASES/PHRASE_NAME/hash.mp4
    → ASL_phrases_v16/PHRASE_NAME_hash.npy

Usage:
    python scripts/extract_phrases_v16.py --workers 8
    python scripts/extract_phrases_v16.py --workers 8 --resume
"""
import os, sys, time, argparse, gc
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src_v16'))
from extract_v16 import extract_continuous_v16


PHRASE_DIR = "data/raw_videos/PHRASES"
OUTPUT_DIR = "ASL_phrases_v16"

# Phrase → gloss sequence mapping
PHRASE_GLOSSES = {
    "GOOD_MORNING":           ["GOOD", "MORNING"],
    "HELLO_HOW_YOU":          ["HELLO", "HOW", "YOU"],
    "I_WANT_FOOD":            ["I", "WANT", "EAT_FOOD"],
    "MY_NAME":                ["MY", "NAME"],
    "PLEASE_HELP_ME":         ["PLEASE", "HELP", "I"],  # ME = I in ASL
    "SORRY_I_LATE":           ["SORRY", "I", "LATE"],
    "THANKYOU_FRIEND":        ["THANKYOU", "FRIEND"],
    "TOMORROW_SCHOOL_GO":     ["TOMORROW", "SCHOOL", "GO"],
    "YESTERDAY_TEACHER_MEET": ["YESTERDAY", "TEACHER", "MEET"],
}


def extract_one_phrase(video_path):
    """Extract a phrase video → variable-length [T, 61, 5] float16."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 8:
        return None

    return extract_continuous_v16(frames, segment_len=28)


def main():
    parser = argparse.ArgumentParser(description="Extract phrase videos with v16 pipeline")
    parser.add_argument("--input", default=PHRASE_DIR, help="Phrase videos directory")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output .npy directory")
    parser.add_argument("--resume", action="store_true", help="Skip already extracted")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    video_exts = {'.mp4', '.mov', '.avi', '.mkv'}

    # Build job list
    jobs = []
    skipped = 0

    for phrase_name in sorted(os.listdir(args.input)):
        phrase_dir = os.path.join(args.input, phrase_name)
        if not os.path.isdir(phrase_dir):
            continue

        if phrase_name not in PHRASE_GLOSSES:
            print(f"  WARNING: Unknown phrase '{phrase_name}', skipping")
            continue

        for vid_file in sorted(os.listdir(phrase_dir)):
            ext = os.path.splitext(vid_file)[1].lower()
            if ext not in video_exts:
                continue

            stem = os.path.splitext(vid_file)[0]
            out_name = f"{phrase_name}_{stem}.npy"
            out_path = os.path.join(args.output, out_name)

            if args.resume and os.path.exists(out_path):
                skipped += 1
                continue

            vid_path = os.path.join(phrase_dir, vid_file)
            jobs.append((vid_path, out_path, phrase_name))

    print(f"Phrases: {len(PHRASE_GLOSSES)}")
    print(f"Jobs: {len(jobs)} to extract, {skipped} skipped (resume={args.resume})")
    print(f"Output: {args.output}")

    if not jobs:
        print("Nothing to extract.")
        return

    t0 = time.time()
    ok = fail = no_hands = 0

    for i, (vid_path, out_path, phrase_name) in enumerate(jobs):
        try:
            result = extract_one_phrase(vid_path)
            if result is not None:
                np.save(out_path, result)
                ok += 1
            else:
                no_hands += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  FAIL: {vid_path}: {e}")

        if (i + 1) % 10 == 0:
            gc.collect()

        if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1)
            eta = (len(jobs) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(jobs)}] ok={ok} no_hands={no_hands} fail={fail} "
                  f"rate={rate:.1f}/s eta={eta/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Extracted: {ok}")
    print(f"  No hands:  {no_hands}")
    print(f"  Failed:    {fail}")

    # Save gloss annotations
    annotations = {}
    for f in sorted(os.listdir(args.output)):
        if not f.endswith('.npy'):
            continue
        # Parse phrase name from filename: PHRASE_NAME_hash.npy
        # Find which phrase this belongs to
        for phrase_name in PHRASE_GLOSSES:
            if f.startswith(phrase_name + "_"):
                annotations[f] = PHRASE_GLOSSES[phrase_name]
                break

    ann_path = os.path.join(args.output, "phrase_annotations.json")
    import json
    with open(ann_path, 'w') as fh:
        json.dump(annotations, fh, indent=2)
    print(f"  Annotations: {ann_path} ({len(annotations)} entries)")


if __name__ == "__main__":
    main()
