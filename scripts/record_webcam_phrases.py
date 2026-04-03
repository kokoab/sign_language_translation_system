"""
Record phrase videos from Mac webcam for Stage 2 training.
Records each phrase multiple times, saves to ASL_phrases_webcam/.

Usage:
    python scripts/record_webcam_phrases.py
"""
import cv2
import os
import time
import hashlib

PHRASES = {
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

REPS_PER_PHRASE = 10
OUTPUT_DIR = "data/raw_videos/webcam_phrases"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("="*60)
    print("WEBCAM PHRASE RECORDER")
    print("="*60)
    print(f"Recording {REPS_PER_PHRASE} reps x {len(PHRASES)} phrases = {REPS_PER_PHRASE * len(PHRASES)} videos")
    print()
    print("Controls:")
    print("  SPACE = start/stop recording")
    print("  S     = skip to next phrase")
    print("  Q     = quit")
    print("="*60)

    total_recorded = 0

    for phrase_name, gloss_str in PHRASES.items():
        phrase_dir = os.path.join(OUTPUT_DIR, phrase_name)
        os.makedirs(phrase_dir, exist_ok=True)

        existing = len([f for f in os.listdir(phrase_dir) if f.endswith('.mp4')])
        if existing >= REPS_PER_PHRASE:
            print(f"\n{phrase_name}: already have {existing} videos, skipping")
            continue

        reps_needed = REPS_PER_PHRASE - existing
        print(f"\n{'='*60}")
        print(f"PHRASE: {phrase_name} ({gloss_str})")
        print(f"  Need {reps_needed} more recordings (have {existing})")
        print(f"  Press SPACE to start recording, SPACE again to stop")
        print(f"{'='*60}")

        rep = 0
        recording = False
        frames = []
        record_start = 0

        while rep < reps_needed:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            display = cv2.flip(display, 1)

            # Status bar
            status = f"{phrase_name} ({gloss_str}) | Rep {existing + rep + 1}/{REPS_PER_PHRASE}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if recording:
                frames.append(frame.copy())  # Save unflipped for extraction
                elapsed = time.time() - record_start
                cv2.putText(display, f"REC {elapsed:.1f}s ({len(frames)}f)", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(display, (2, 2), (638, 478), (0, 0, 255), 3)
            else:
                cv2.putText(display, "Press SPACE to record", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Phrase Recorder", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if not recording:
                    recording = True
                    frames = []
                    record_start = time.time()
                else:
                    recording = False
                    elapsed = time.time() - record_start
                    actual_fps = len(frames) / elapsed if elapsed > 0 else 30

                    if len(frames) < 15:
                        print(f"    Too short ({len(frames)} frames), discarded")
                        continue

                    # Save video
                    vid_hash = hashlib.md5(f"{phrase_name}_{time.time()}".encode()).hexdigest()[:8]
                    out_path = os.path.join(phrase_dir, f"{vid_hash}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_path, fourcc, actual_fps, (640, 480))
                    for f in frames:
                        out.write(f)
                    out.release()

                    rep += 1
                    total_recorded += 1
                    print(f"    Saved: {out_path} ({len(frames)} frames, {actual_fps:.0f}fps, {elapsed:.1f}s)")

            elif key == ord('s'):
                print(f"  Skipping {phrase_name}")
                break

            elif key == ord('q'):
                print(f"\nDone! Recorded {total_recorded} videos total.")
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Recorded {total_recorded} videos total.")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"\nNext: run extract + retrain:")
    print(f"  KMP_DUPLICATE_LIB_OK=TRUE python scripts/reextract_phrases.py --webcam")


if __name__ == "__main__":
    main()
