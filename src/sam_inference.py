"""
SAM-SLR-v2 inference for live/video sign recognition.
Replaces Stage 1 in the SLT pipeline.

Usage:
    # Test on a video
    python src/sam_inference.py --video sample_videos/hello.mp4

    # Test on extracted .npy
    python src/sam_inference.py --npy ASL_landmarks_sam27/HELLO_xxx.npy
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import cv2

# SAM's exact keypoint indices from COCO-WholeBody 133
SAM_COCO_INDICES = [0, 5, 6, 7, 8, 9, 10,
                    91, 95, 96, 99, 100, 103, 104, 107, 108, 111,
                    112, 116, 117, 120, 121, 124, 125, 128, 129, 132]

SAM_BONE_PAIRS = [
    (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),
    (7, 8), (7, 9), (7, 11), (7, 13), (7, 15),
    (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (17, 19), (17, 21), (17, 23), (17, 25),
    (19, 20), (21, 22), (23, 24), (25, 26),
    (5, 7), (6, 17),
]

MAX_FRAME = 150


def extract_sam27_from_video(video_path, wholebody=None):
    """Extract SAM-27 keypoints from video using rtmlib.
    Returns: [T, 27, 3] raw pixel coordinates."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()

    if len(frames) < 4:
        return None

    if wholebody is None:
        from rtmlib import Wholebody
        wholebody = Wholebody(to_openpose=False, mode='performance',
                             backend='onnxruntime', device='cpu')

    # Extract keypoints per frame
    sam_data = []
    for frame in frames:
        kps, scs = wholebody(frame)
        if kps is None or len(kps) == 0:
            continue
        if kps.ndim == 3:
            kps = kps[0]
        if len(kps) < 133:
            continue
        # Select SAM's 27 nodes
        sam_frame = np.zeros((27, 3), dtype=np.float32)
        for i, coco_idx in enumerate(SAM_COCO_INDICES):
            sam_frame[i, 0] = kps[coco_idx, 0]
            sam_frame[i, 1] = kps[coco_idx, 1]
        sam_data.append(sam_frame)

    if len(sam_data) < 4:
        return None

    return np.array(sam_data, dtype=np.float32)  # [T, 27, 3]


def prepare_sam_input(joint_data, stream='joint', window_size=120):
    """Prepare SAM input tensor from raw joint data.
    joint_data: [T, 27, 3] -> tensor [1, 3, T, 27, 1]"""
    is_vector = stream in ('bone', 'joint_motion', 'bone_motion')

    # Generate stream data
    if stream == 'joint':
        data = joint_data.copy()
    elif stream == 'bone':
        data = np.zeros_like(joint_data)
        for v1, v2 in SAM_BONE_PAIRS:
            data[:, v2, :] = joint_data[:, v2, :] - joint_data[:, v1, :]
    elif stream == 'joint_motion':
        data = np.zeros_like(joint_data)
        data[:-1] = joint_data[1:] - joint_data[:-1]
    elif stream == 'bone_motion':
        bone = np.zeros_like(joint_data)
        for v1, v2 in SAM_BONE_PAIRS:
            bone[:, v2, :] = joint_data[:, v2, :] - joint_data[:, v1, :]
        data = np.zeros_like(bone)
        data[:-1] = bone[1:] - bone[:-1]

    # Pad to MAX_FRAME
    T = data.shape[0]
    if T < MAX_FRAME:
        rest = MAX_FRAME - T
        num_repeats = int(np.ceil(rest / T))
        pad = np.concatenate([data for _ in range(num_repeats)], axis=0)[:rest]
        data = np.concatenate([data, pad], axis=0)
    else:
        data = data[:MAX_FRAME]

    # Transpose to [3, T, 27]
    data = data.transpose(2, 0, 1)

    # Crop to window_size
    if data.shape[1] > window_size:
        data = data[:, :window_size, :]

    # Add M dimension: [3, T, 27, 1]
    data = np.expand_dims(data, axis=-1)

    # Normalize (center on nose)
    if is_vector:
        data[0, :, 0, :] -= data[0, :, 0, 0].mean()
        data[1, :, 0, :] -= data[1, :, 0, 0].mean()
    else:
        data[0, :, :, :] -= data[0, :, 0, 0].mean()
        data[1, :, :, :] -= data[1, :, 0, 0].mean()

    return torch.from_numpy(data).float().unsqueeze(0)  # [1, 3, T, 27, 1]


def load_sam_model(checkpoint_path, device='cpu'):
    """Load a trained SAM model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Need SAM code in path
    for p in ['SAM-SLR-v2/SL-GCN', '/workspace/SLT/SAM-SLR-v2/SL-GCN']:
        if os.path.isdir(p):
            sys.path.insert(0, p)
            break

    from model.decouple_gcn_attn import Model as SLGCN

    num_classes = ckpt['num_classes']
    model = SLGCN(num_class=num_classes, num_point=27, num_person=1,
                  graph='graph.sign_27.Graph',
                  graph_args={'labeling_mode': 'spatial'},
                  groups=16, block_size=41, in_channels=3).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model, ckpt.get('idx_to_label', {}), ckpt.get('label_to_idx', {})


def ensemble_predict(joint_data, models, idx_to_label, device='cpu'):
    """Run 4-stream ensemble prediction.
    models: dict of {stream_name: model}
    Returns: (predicted_label, confidence, top5)"""
    streams = {
        'joint': {'window': 120, 'weight': 1.0},
        'bone': {'window': 100, 'weight': 0.9},
        'joint_motion': {'window': 100, 'weight': 0.5},
        'bone_motion': {'window': 100, 'weight': 0.5},
    }

    total_logits = None
    total_weight = 0

    for stream_name, model in models.items():
        cfg = streams[stream_name]
        x = prepare_sam_input(joint_data, stream=stream_name,
                             window_size=cfg['window']).to(device)
        with torch.no_grad():
            logits = model(x, keep_prob=1.0)  # no dropout at inference

        if total_logits is None:
            total_logits = logits * cfg['weight']
        else:
            total_logits += logits * cfg['weight']
        total_weight += cfg['weight']

    total_logits /= total_weight
    probs = torch.softmax(total_logits, dim=-1)
    top5 = probs.topk(5)

    results = []
    for p, i in zip(top5.values[0], top5.indices[0]):
        label = idx_to_label.get(str(i.item()), idx_to_label.get(i.item(), f"UNK_{i.item()}"))
        results.append((label, p.item()))

    return results[0][0], results[0][1], results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--video', default=None)
    p.add_argument('--npy', default=None)
    p.add_argument('--checkpoint', default='/workspace/sam_joint/best_model.pth')
    p.add_argument('--stream', default='joint')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.npy:
        joint_data = np.load(args.npy).astype(np.float32)
    elif args.video:
        joint_data = extract_sam27_from_video(args.video)
    else:
        print("Provide --video or --npy")
        sys.exit(1)

    if joint_data is None:
        print("Failed to extract keypoints")
        sys.exit(1)

    model, idx_to_label, _ = load_sam_model(args.checkpoint, device)
    x = prepare_sam_input(joint_data, stream=args.stream,
                         window_size=120 if args.stream == 'joint' else 100).to(device)

    with torch.no_grad():
        logits = model(x, keep_prob=1.0)
        probs = torch.softmax(logits, dim=-1)
        top5 = probs.topk(5)

    print("Top 5 predictions:")
    for p_val, i in zip(top5.values[0], top5.indices[0]):
        label = idx_to_label.get(str(i.item()), f"Class_{i.item()}")
        print(f"  {label:20s} {p_val.item()*100:.1f}%")
