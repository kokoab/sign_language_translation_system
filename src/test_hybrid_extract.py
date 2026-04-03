"""
Hybrid extraction test: MediaPipe first (fast), rtmlib fills missing frames.
Compares three approaches on the same video:
  1. MediaPipe only
  2. rtmlib only
  3. Hybrid (MediaPipe + rtmlib fallback)

Usage:
    python src/test_hybrid_extract.py
    python src/test_hybrid_extract.py data/raw_videos/PHRASES/HELLO_HOW_YOU/video1.mp4
"""
import sys, os, glob, time
import numpy as np
import cv2


def find_video():
    for p in ["data/raw_videos/PHRASES/*/*.mp4", "data/raw_videos/PHRASES/*/*.mov",
              "data/raw_videos/PHRASES/*/*.webm"]:
        files = glob.glob(p)
        if files:
            return files[0]
    return None


def extract_mediapipe(frames):
    """Fast extraction with MediaPipe."""
    import mediapipe as mp
    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    all_kps = []    # [T, 133, 2] to match rtmlib format
    all_scores = [] # [T, 133]

    for frame in frames:
        h, w = frame.shape[:2]
        kps = np.zeros((133, 2))
        scores = np.zeros(133)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_lm, handed in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handed.classification[0].label
                pts = np.array([[lm.x * w, lm.y * h] for lm in hand_lm.landmark])
                conf = handed.classification[0].score

                if label == "Right":  # Camera-mirrored = left hand
                    kps[91:112] = pts
                    scores[91:112] = conf
                else:
                    kps[112:133] = pts
                    scores[112:133] = conf

        all_kps.append(kps)
        all_scores.append(scores)

    hands.close()
    return np.array(all_kps), np.array(all_scores)


def extract_rtmlib_frames(frames, indices=None):
    """Extract specific frames with rtmlib. If indices=None, extract all."""
    from rtmlib import Wholebody
    wholebody = Wholebody(mode="lightweight", backend="onnxruntime", device="cpu")

    if indices is None:
        indices = list(range(len(frames)))

    results = {}
    for idx in indices:
        kps_out, scores_out = wholebody(frames[idx])
        if kps_out is not None and len(kps_out) > 0:
            results[idx] = (kps_out[0], scores_out[0])
        else:
            results[idx] = (np.zeros((133, 2)), np.zeros(133))

    return results


def hybrid_extract(frames):
    """MediaPipe first, rtmlib fills missing."""
    # Step 1: MediaPipe (fast)
    t0 = time.time()
    mp_kps, mp_scores = extract_mediapipe(frames)
    t_mp = time.time() - t0

    # Step 2: Find frames where MediaPipe missed hands
    missing_frames = []
    for i in range(len(frames)):
        l_detected = mp_scores[i, 91:112].mean() > 0.2
        r_detected = mp_scores[i, 112:133].mean() > 0.2
        if not l_detected and not r_detected:
            missing_frames.append(i)
        elif not l_detected or not r_detected:
            missing_frames.append(i)  # Partial detection — fill missing hand

    n_missing = len(missing_frames)

    # Step 3: rtmlib on missing frames only
    t0 = time.time()
    if missing_frames:
        rtm_results = extract_rtmlib_frames(frames, missing_frames)
    else:
        rtm_results = {}
    t_rtm = time.time() - t0

    # Step 4: Merge — use rtmlib for missing, MediaPipe for detected
    merged_kps = mp_kps.copy()
    merged_scores = mp_scores.copy()

    for idx, (rtm_kps, rtm_scores) in rtm_results.items():
        # Fill left hand if MediaPipe missed it
        if mp_scores[idx, 91:112].mean() < 0.2 and rtm_scores[91:112].mean() > 0.2:
            merged_kps[idx, 91:112] = rtm_kps[91:112]
            merged_scores[idx, 91:112] = rtm_scores[91:112]
        # Fill right hand if MediaPipe missed it
        if mp_scores[idx, 112:133].mean() < 0.2 and rtm_scores[112:133].mean() > 0.2:
            merged_kps[idx, 112:133] = rtm_kps[112:133]
            merged_scores[idx, 112:133] = rtm_scores[112:133]
        # Fill face if needed
        face_indices = [53, 31, 50, 23, 39]  # nose, chin, forehead, L_ear, R_ear
        for fi in face_indices:
            if mp_scores[idx, fi] < 0.2 and rtm_scores[fi] > 0.2:
                merged_kps[idx, fi] = rtm_kps[fi]
                merged_scores[idx, fi] = rtm_scores[fi]

    return merged_kps, merged_scores, t_mp, t_rtm, n_missing, len(frames)


def count_detections(scores, n_frames):
    """Count frames with hand detections."""
    l_count = sum(1 for i in range(n_frames) if scores[i, 91:112].mean() > 0.2)
    r_count = sum(1 for i in range(n_frames) if scores[i, 112:133].mean() > 0.2)
    both = sum(1 for i in range(n_frames)
               if scores[i, 91:112].mean() > 0.2 and scores[i, 112:133].mean() > 0.2)
    return l_count, r_count, both


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else find_video()
    if video_path is None:
        print("No video found.")
        return

    print(f"Video: {video_path}")

    # Read frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames ({len(frames)/30:.1f}s)\n")

    # === Method 1: MediaPipe only ===
    print("=" * 50)
    print("1. MediaPipe only")
    t0 = time.time()
    mp_kps, mp_scores = extract_mediapipe(frames)
    t_mp = time.time() - t0
    l, r, both = count_detections(mp_scores, len(frames))
    print(f"   Time: {t_mp:.1f}s")
    print(f"   Left hand: {l}/{len(frames)} frames")
    print(f"   Right hand: {r}/{len(frames)} frames")
    print(f"   Both hands: {both}/{len(frames)} frames")

    # === Method 2: rtmlib only ===
    print("\n" + "=" * 50)
    print("2. rtmlib only")
    t0 = time.time()
    rtm_results = extract_rtmlib_frames(frames)
    t_rtm = time.time() - t0
    rtm_kps = np.array([rtm_results[i][0] for i in range(len(frames))])
    rtm_scores = np.array([rtm_results[i][1] for i in range(len(frames))])
    l, r, both = count_detections(rtm_scores, len(frames))
    print(f"   Time: {t_rtm:.1f}s")
    print(f"   Left hand: {l}/{len(frames)} frames")
    print(f"   Right hand: {r}/{len(frames)} frames")
    print(f"   Both hands: {both}/{len(frames)} frames")

    # === Method 3: Hybrid ===
    print("\n" + "=" * 50)
    print("3. Hybrid (MediaPipe + rtmlib fallback)")
    hybrid_kps, hybrid_scores, ht_mp, ht_rtm, n_missing, n_total = hybrid_extract(frames)
    l, r, both = count_detections(hybrid_scores, len(frames))
    total_time = ht_mp + ht_rtm
    print(f"   MediaPipe: {ht_mp:.1f}s | rtmlib fallback: {ht_rtm:.1f}s | Total: {total_time:.1f}s")
    print(f"   Frames needing rtmlib: {n_missing}/{n_total}")
    print(f"   Left hand: {l}/{len(frames)} frames")
    print(f"   Right hand: {r}/{len(frames)} frames")
    print(f"   Both hands: {both}/{len(frames)} frames")

    # === Summary ===
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"   MediaPipe only:  {t_mp:.1f}s")
    print(f"   rtmlib only:     {t_rtm:.1f}s")
    print(f"   Hybrid:          {total_time:.1f}s  (saved {t_rtm - total_time:.1f}s vs rtmlib)")
    print(f"   Hybrid speedup:  {t_rtm/max(total_time, 0.1):.1f}x faster than pure rtmlib")


if __name__ == "__main__":
    main()
