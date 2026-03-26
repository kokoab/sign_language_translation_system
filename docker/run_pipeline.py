"""
SLT Docker Pipeline Entrypoint
Video -> Extract (MediaPipe) -> Stage 2 (CTC) -> Stage 3 (T5) -> English

Usage:
    python run_pipeline.py /app/input/video.mp4
    python run_pipeline.py /app/input/            # all .mp4/.mov files
    python run_pipeline.py /app/input/ --save-npy  # also save .npy landmarks
"""
import sys, os, glob, json, argparse, warnings, time
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, '/app/src')

import numpy as np
import torch

# Import extraction pipeline (same code used in training)
from extract import (
    PipelineConfig, NUM_NODES,
    interpolate_hand, interpolate_face, temporal_resample,
    normalize_sequence, compute_kinematics_batch,
    one_euro_filter, stabilize_bones, reject_temporal_outliers,
    FACE_LANDMARK_INDICES,
)

# Import model architecture from test_video_pipeline
from test_video_pipeline import (
    SLTStage2CTC, DSGCNEncoder,
    compute_bone_features, ctc_greedy_decode,
)

import cv2
import subprocess
import tempfile
import mediapipe as mp
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =====================================================================
# PATHS (inside container — models mounted at /app/models)
# =====================================================================
STAGE2_CKPT = "/app/models/output/stage2_best_model.pth"
STAGE3_DIR = "/app/models/output/slt_conversational_t5_model"
MANIFEST = "/app/models/manifest.json"
OUTPUT_DIR = "/app/output"

DEVICE = torch.device("cpu")  # CPU-only container


def reencode_to_cfr(video_path, fps=30):
    """Re-encode video to constant frame rate for consistent extraction."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-r", str(fps),
             "-vsync", "cfr", "-an", "-c:v", "libx264",
             "-preset", "ultrafast", "-crf", "18", tmp.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return tmp.name
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        return None


def extract_from_video(video_path):
    """Extract landmarks using MediaPipe (exact same pipeline as training).
    Returns [32, 47, 16] numpy array (float32)."""
    print(f"\n[EXTRACT] {os.path.basename(video_path)}")

    cfg = PipelineConfig()

    # Re-encode to CFR
    cfr_path = reencode_to_cfr(video_path)
    read_path = cfr_path or video_path

    cap = cv2.VideoCapture(read_path)
    if not cap.isOpened():
        if cfr_path and os.path.exists(cfr_path):
            os.unlink(cfr_path)
        raise ValueError(f"Cannot open: {video_path}")

    frames_rgb = []
    frame_indices = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        frame_indices.append(idx)
        idx += 1
    cap.release()
    if cfr_path and os.path.exists(cfr_path):
        os.unlink(cfr_path)

    total_frames = len(frames_rgb)
    if total_frames < cfg.min_raw_frames:
        raise ValueError(f"Too few frames: {total_frames}")
    print(f"  Frames: {total_frames}")

    # Subsample if too many
    max_process = cfg.target_frames * 3
    if total_frames > max_process:
        step = total_frames / max_process
        selected = [int(i * step) for i in range(max_process)]
        frames_rgb = [frames_rgb[i] for i in selected]
        frame_indices = [frame_indices[i] for i in selected]

    processed_count = len(frames_rgb)

    # ── Pass 1: MediaPipe video mode (tracking) ──
    l_seq, r_seq, l_valid, r_valid = [], [], [], []
    face_seq, face_valid = [], []

    hands_v = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=cfg.min_detection_conf,
        min_tracking_confidence=cfg.min_tracking_conf,
        model_complexity=cfg.model_complexity
    )
    face_v = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        min_detection_confidence=cfg.min_detection_conf,
        min_tracking_confidence=cfg.min_tracking_conf,
        refine_landmarks=False,
    )

    for i, rgb in enumerate(frames_rgb):
        fi = frame_indices[i]
        res = hands_v.process(rgb)
        if res.multi_hand_landmarks:
            for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handedness.classification[0].label
                score = handedness.classification[0].score
                if score >= cfg.min_detection_conf:
                    coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                    if label == "Left" and (not l_valid or l_valid[-1] != fi):
                        l_seq.append(coords); l_valid.append(fi)
                    elif label == "Right" and (not r_valid or r_valid[-1] != fi):
                        r_seq.append(coords); r_valid.append(fi)
        face_res = face_v.process(rgb)
        if face_res.multi_face_landmarks:
            fl = face_res.multi_face_landmarks[0]
            face_pts = [[fl.landmark[idx].x, fl.landmark[idx].y, fl.landmark[idx].z]
                        for idx in FACE_LANDMARK_INDICES]
            face_seq.append(face_pts)
            face_valid.append(fi)
    hands_v.close(); face_v.close()

    # ── Pass 2: Static mode if coverage < 80% ──
    dom_coverage = max(len(l_valid), len(r_valid)) / max(processed_count, 1)
    if dom_coverage < 0.80:
        print(f"  Video pass coverage: {dom_coverage:.0%}, running static pass...")
        s_l_seq, s_r_seq, s_l_valid, s_r_valid = [], [], [], []
        s_face_seq, s_face_valid = [], []

        hands_s = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=cfg.min_detection_conf,
            min_tracking_confidence=cfg.min_tracking_conf,
            model_complexity=cfg.model_complexity
        )
        face_s = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            min_detection_confidence=cfg.min_detection_conf,
            min_tracking_confidence=cfg.min_tracking_conf,
            refine_landmarks=False,
        )

        for i, rgb in enumerate(frames_rgb):
            fi = frame_indices[i]
            res = hands_s.process(rgb)
            if res.multi_hand_landmarks:
                for hand_lm, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handedness.classification[0].label
                    score = handedness.classification[0].score
                    if score >= cfg.min_detection_conf:
                        coords = [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark]
                        if label == "Left" and (not s_l_valid or s_l_valid[-1] != fi):
                            s_l_seq.append(coords); s_l_valid.append(fi)
                        elif label == "Right" and (not s_r_valid or s_r_valid[-1] != fi):
                            s_r_seq.append(coords); s_r_valid.append(fi)
            face_res = face_s.process(rgb)
            if face_res.multi_face_landmarks:
                fl = face_res.multi_face_landmarks[0]
                face_pts = [[fl.landmark[idx].x, fl.landmark[idx].y, fl.landmark[idx].z]
                            for idx in FACE_LANDMARK_INDICES]
                s_face_seq.append(face_pts)
                s_face_valid.append(fi)
        hands_s.close(); face_s.close()

        if len(s_l_valid) > len(l_valid):
            l_seq, l_valid = s_l_seq, s_l_valid
        if len(s_r_valid) > len(r_valid):
            r_seq, r_valid = s_r_seq, s_r_valid
        if len(s_face_valid) > len(face_valid):
            face_seq, face_valid = s_face_seq, s_face_valid

    del frames_rgb  # Free memory

    l_ever, r_ever = bool(l_valid), bool(r_valid)
    face_ever = bool(face_valid)
    print(f"  Detections: L={len(l_valid)} R={len(r_valid)} Face={len(face_valid)}")

    if not l_valid and not r_valid:
        raise ValueError("No hands detected")

    # Temporal coherence rejection
    if l_valid:
        l_seq, l_valid = reject_temporal_outliers(l_seq, l_valid)
    if r_valid:
        r_seq, r_valid = reject_temporal_outliers(r_seq, r_valid)
    if not l_valid and not r_valid:
        raise ValueError("All detections rejected as outliers")

    # Interpolation
    l_full = interpolate_hand(
        np.array(l_seq) if l_seq else np.zeros((0, 21, 3)),
        l_valid, total_frames)
    r_full = interpolate_hand(
        np.array(r_seq) if r_seq else np.zeros((0, 21, 3)),
        r_valid, total_frames)
    face_full = interpolate_face(
        np.array(face_seq) if face_seq else np.zeros((0, 5, 3)),
        face_valid, total_frames)

    combined = np.concatenate([l_full, r_full, face_full], axis=1)  # [T, 47, 3]

    # Temporal resample to 32 frames
    resampled = temporal_resample(combined, cfg.target_frames)

    # 1-Euro filter
    smoothed_xyz = one_euro_filter(resampled[:, :, :3])
    resampled[:, :, :3] = smoothed_xyz

    # Bone stabilization
    if l_ever:
        resampled = stabilize_bones(resampled, 0, 21)
    if r_ever:
        resampled = stabilize_bones(resampled, 21, 42)

    # Normalize
    normalized = normalize_sequence(resampled, l_ever, r_ever)

    # Per-frame confidence mask
    T = cfg.target_frames
    per_frame_mask = np.zeros((1, T, NUM_NODES, 1), dtype=np.float32)
    if l_valid:
        l_cov = np.interp(np.linspace(0, total_frames - 1, T),
                          sorted(l_valid), np.ones(len(l_valid)))
        for t in range(T):
            per_frame_mask[0, t, 0:21, 0] = l_cov[t]
    if r_valid:
        r_cov = np.interp(np.linspace(0, total_frames - 1, T),
                          sorted(r_valid), np.ones(len(r_valid)))
        for t in range(T):
            per_frame_mask[0, t, 21:42, 0] = r_cov[t]
    if face_valid:
        per_frame_mask[0, :, 42:47, 0] = 1.0

    # Kinematics (Savitzky-Golay) -> [32, 47, 10]
    features_10ch = compute_kinematics_batch(
        normalized[np.newaxis, ...], l_ever, r_ever, face_ever,
        per_frame_mask=per_frame_mask
    ).squeeze(0)

    # Bone features -> [32, 47, 6]
    bone = compute_bone_features(normalized)

    # Full 16-channel -> [32, 47, 16]
    features_16ch = np.concatenate([features_10ch, bone], axis=-1).astype(np.float32)

    print(f"  Output: {features_16ch.shape}")
    return features_16ch


def load_models():
    """Load Stage 2 + Stage 3 models."""
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    # Stage 2
    print(f"\nStage 2: {STAGE2_CKPT}")
    if not os.path.exists(STAGE2_CKPT):
        print(f"ERROR: Not found: {STAGE2_CKPT}")
        print("Mount your models directory: -v ./models:/app/models")
        sys.exit(1)

    ckpt = torch.load(STAGE2_CKPT, map_location=DEVICE, weights_only=False)
    idx_to_gloss = ckpt["idx_to_gloss"]
    vocab_size = ckpt["vocab_size"]

    s2_model = SLTStage2CTC(vocab_size=vocab_size, d_model=384).to(DEVICE)
    s2_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    s2_model.eval()
    print(f"  Loaded ({vocab_size} classes)")

    # Stage 3
    print(f"\nStage 3: {STAGE3_DIR}")
    if not os.path.exists(STAGE3_DIR):
        print(f"ERROR: Not found: {STAGE3_DIR}")
        sys.exit(1)

    s3_tokenizer = AutoTokenizer.from_pretrained(STAGE3_DIR)
    s3_model = AutoModelForSeq2SeqLM.from_pretrained(STAGE3_DIR).to(DEVICE)
    s3_model.eval()
    param_count = sum(p.numel() for p in s3_model.parameters()) / 1e6
    print(f"  Loaded Flan-T5 ({param_count:.1f}M params)")

    return s2_model, s3_model, s3_tokenizer, idx_to_gloss


def run_pipeline(video_path, s2_model, s3_model, s3_tokenizer, idx_to_gloss, save_npy=False):
    """Full pipeline: extract -> Stage 2 CTC -> Stage 3 T5."""
    basename = os.path.splitext(os.path.basename(video_path))[0]

    # Stage 0: Extract
    features = extract_from_video(video_path)

    if save_npy:
        npy_path = os.path.join(OUTPUT_DIR, f"{basename}.npy")
        np.save(npy_path, features.astype(np.float16))
        print(f"  Saved: {npy_path}")

    # Stage 2: CTC Recognition
    print("\n[STAGE 2] CTC Recognition...")
    x = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)
    lens = torch.tensor([x.shape[1]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits, out_lens = s2_model(x, lens)
        log_probs = torch.log_softmax(logits[0], dim=-1).cpu().numpy()
        n_tokens = out_lens[0].item()

    glosses = ctc_greedy_decode(log_probs[:n_tokens], idx_to_gloss)
    gloss_str = " ".join(glosses) if glosses else "(no signs detected)"
    print(f"  Glosses: {gloss_str}")

    # Stage 3: Translation
    print("\n[STAGE 3] Translation...")
    if not glosses:
        translation = "[No signs detected]"
    else:
        prompt = f"Translate this ASL gloss to natural conversational English: {' '.join(glosses)}"
        inputs = s3_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = s3_model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
        translation = s3_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print(f"  Video   : {os.path.basename(video_path)}")
    print(f"  Glosses : {gloss_str}")
    print(f"  English : {translation}")
    print("=" * 60)

    return {"video": os.path.basename(video_path), "glosses": glosses, "english": translation}


def main():
    parser = argparse.ArgumentParser(description="SLT Docker Pipeline")
    parser.add_argument("input", help="Video file or folder of videos")
    parser.add_argument("--save-npy", action="store_true", help="Save extracted .npy files")
    parser.add_argument("--output", default="/app/output", help="Output directory")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find videos
    if os.path.isdir(args.input):
        videos = sorted(
            glob.glob(os.path.join(args.input, "*.mp4")) +
            glob.glob(os.path.join(args.input, "*.mov")) +
            glob.glob(os.path.join(args.input, "*.avi"))
        )
        print(f"Found {len(videos)} videos in {args.input}")
    else:
        videos = [args.input]

    if not videos:
        print("No videos found.")
        sys.exit(1)

    # Load models once
    s2_model, s3_model, s3_tokenizer, idx_to_gloss = load_models()

    # Process each video
    results = []
    for video_path in videos:
        if not os.path.exists(video_path):
            print(f"\nERROR: Not found: {video_path}")
            continue
        try:
            t0 = time.time()
            result = run_pipeline(video_path, s2_model, s3_model, s3_tokenizer,
                                  idx_to_gloss, save_npy=args.save_npy)
            result["time_sec"] = round(time.time() - t0, 1)
            results.append(result)
        except Exception as e:
            print(f"\nERROR processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"video": os.path.basename(video_path), "error": str(e)})

    # Save results JSON
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  FAIL  {r['video']}: {r['error']}")
        else:
            print(f"  OK    {r['video']}: {r['english']} ({r['time_sec']}s)")


if __name__ == "__main__":
    main()
