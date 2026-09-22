"""Bounded published-weight MPS smoke on one approved video; not an accuracy test."""
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / 'artifacts/vendor/shubert'
WEIGHTS = ROOT / 'artifacts/models/shubert_pretrained'
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'artifacts/vendor/shubert_runtime'), str(VENDOR / 'fairseq')]
import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from examples.shubert.models.shubert import SHubertConfig, SHubertModel


def upstream_functions(path):
    # Execute only published function definitions, avoiding unused decord/Slurm drivers.
    tree = ast.parse((VENDOR / path).read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {'np': np, 'cv2': cv2}
    exec(compile(tree, str(VENDOR / path), 'exec'), namespace)
    return namespace


def sync():
    torch.mps.synchronize()


def main(video=None, output=OUT, signer_crop=False):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    assert torch.backends.mps.is_available()
    manifest = json.loads((ROOT / 'data/local/combined_dataset_v17_20260922/manifest.json').read_text())
    row = next(r for r in manifest['records'] if (r['video_path'] == video if video is not None else r['source'] == 'asllrp_contiguous' and r['role'] == 'validation'))
    path = ROOT / row['video_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['video_sha256']
    crop_seconds = 0.
    if signer_crop:
        from ultralytics import YOLO
        helpers = upstream_functions('dataset/clips_bbox.py')
        helpers['YOLO'] = lambda p: YOLO(p).to('mps')
        target = output / 'signer_crop.mp4'
        start = time.perf_counter()
        helpers['crop_clip'](str(path), str(output / 'crop_errors.txt'), str(target), str(WEIGHTS / 'yolov8n.pt'))
        crop_seconds = time.perf_counter() - start
        assert target.exists() and target.stat().st_size > 0
        path = target
    face = upstream_functions('dataset/crop_face.py')
    hand = upstream_functions('dataset/crop_hands.py')
    body = upstream_functions('features/body_features.py')
    kpe = upstream_functions('dataset/kpe_mediapipe.py')
    kpe['mp'] = mp
    kpe['face_detector'] = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(WEIGHTS / 'face_landmarker.task')),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6))
    kpe['hand_detector'] = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(WEIGHTS / 'hand_landmarker.task')),
        num_hands=6, min_hand_detection_confidence=0.05))
    kpe['mp_holistic'] = mp.solutions.holistic.Holistic(min_detection_confidence=0.1)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    crops = {key: [] for key in ['face', 'left_hand', 'right_hand']}
    poses, present = [], {'face': 0, 'left_hand': 0, 'right_hand': 0, 'pose': 0}
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, detected = kpe['detect_holistic'](rgb)
            pose = detected['pose_landmarks']
            current = {}
            if pose:
                present['pose'] += 1
                normalized = np.asarray(body['normalize_pose_keypoints'](pose[0][:25]))
                poses.append(normalized[[0, 11, 12, 13, 14, 15, 16]].reshape(14))
                if detected['face_landmarks']:
                    selected = face['select_face'](pose[0], detected['face_landmarks'])
                    current['face'] = face['resize_frame'](face['cues_on_grey_background'](rgb, selected), (224, 224))
                selected = hand['select_hands'](pose[0], detected['hand_landmarks'], rgb.shape)
                for key, landmarks in zip(['left_hand', 'right_hand'], selected):
                    if landmarks is not None:
                        box = hand['adjust_bounding_box'](hand['get_bounding_box'](landmarks, rgb.shape, 1.5), rgb.shape)
                        current[key] = hand['resize_frame'](hand['crop_frame'](rgb, box), (224, 224))
            else:
                poses.append(poses[-1].copy() if poses else np.full(14, -9999.))
            for key in crops:
                crop = current.get(key)
                if crop is not None:
                    present[key] += 1
                else:
                    crop = crops[key][-1].copy() if crops[key] else np.zeros((224, 224, 3), np.uint8)
                crops[key].append(crop)
    finally:
        cap.release()
        for key in ['face_detector', 'hand_detector', 'mp_holistic']:
            kpe[key].close()
    frontend_seconds = time.perf_counter() - start
    assert poses and present['pose'] > 0
    source = {'body_posture': torch.tensor(np.asarray(poses), dtype=torch.float32, device='mps')}
    start = time.perf_counter()
    for group, keys in [('face', ['face']), ('hands', ['left_hand', 'right_hand'])]:
        model = torch.hub.load(str(ROOT / 'artifacts/vendor/shubert_dinov2'), 'dinov2_vits14_reg', source='local', pretrained=False)
        checkpoint = torch.load(WEIGHTS / f'{group}_dinov2_checkpoint.pth', map_location='cpu', weights_only=False)
        state = {key.replace('backbone.', ''): value for key, value in checkpoint['teacher'].items() if 'dino_head' not in key}
        model.pos_embed = torch.nn.Parameter(torch.zeros(1, 257, 384))
        model.load_state_dict(state, strict=True)
        model.eval().to('mps')
        with torch.inference_mode():
            for key in keys:
                array = torch.tensor(np.stack(crops[key]), device='mps', dtype=torch.float32).permute(0, 3, 1, 2) / 255
                array = (array - array.new_tensor([.485, .456, .406])[None, :, None, None]) / array.new_tensor([.229, .224, .225])[None, :, None, None]
                source[key] = torch.cat([model(batch) for batch in array.split(8)])
        del model, checkpoint, state
    sync()
    dino_seconds = time.perf_counter() - start
    np.savez_compressed(output / 'streams.npz', **{key: value.cpu().numpy() for key, value in source.items()})
    checkpoint = torch.load(WEIGHTS / 'checkpoint_836_400000.pt', map_location='cpu', weights_only=False)
    model = SHubertModel(SHubertConfig())
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    model.eval().to('mps')
    del checkpoint
    for key in list(source):
        source['label_' + key] = torch.zeros((len(poses), 1), device='mps')
    sync()
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.extract_features([source], padding_mask=None, kmeans_labels=None, mask=False)
    sync()
    encoder_seconds = time.perf_counter() - start
    features = result['x'].cpu().numpy()
    assert np.isfinite(features).all() and features.shape[1] == len(poses)
    np.save(output / 'features.npy', features)
    report = dict(source=row['video_path'], video_sha256=row['video_sha256'], frames=len(poses), fps=fps,
                  source_seconds=len(poses)/fps, device='mps', strict_weights=True, presence=present,
                  signer_crop_seconds=crop_seconds, frontend_seconds=frontend_seconds, dino_seconds_including_load=dino_seconds,
                  encoder_seconds=encoder_seconds, feature_shape=list(features.shape), finite=True,
                  limitations=['Single approved validation video; no accuracy/WER test',
                               'Published YOLO crop' if signer_crop else 'Already single-signer video; no YOLO signer crop',
                               'OpenCV RGB decode and in-memory crops replace lossy intermediate MP4s',
                               'Full-clip noncausal encoder; not streaming latency'])
    (output / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main(signer_crop="--signer-crop" in sys.argv)
