import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.interpolate import interp1d

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                       min_detection_confidence=0.7, model_complexity=1)

def get_augmentation(data):
    """Applies complex 3D transformations to a sequence of [60, 21, 3]"""
    # 1. Random Rotation (All 3 axes)
    # Corrected unpacking: np.cos returns an array, we index it [0, 1, 2]
    angles = np.radians(np.random.uniform(-10, 10, 3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    # Rotation matrices for X, Y, Z axes
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    # Combined Rotation
    R = Rx @ Ry @ Rz
    
    # Apply to data
    data = data @ R
    
    # 2. Random Scaling (Depth simulation)
    scale = np.random.uniform(0.85, 1.15)
    data *= scale
    
    # 3. Random Jitter (Subtle)
    data += np.random.normal(0, 0.002, data.shape)
    
    return data

def interpolate_landmarks(sequence, target_frames=60):
    curr = len(sequence)
    x = np.linspace(0, curr - 1, num=curr)
    x_new = np.linspace(0, curr - 1, num=target_frames)
    f = interp1d(x, sequence, axis=0, kind='linear')
    return f(x_new)

def process_and_augment(base_video_dir, output_dir, variations_per_vid=30):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    print(f"🚀 Starting Extraction & Augmentation...")
    
    for root, _, files in os.walk(base_video_dir):
        # Sort files to keep processing consistent
        for video_name in sorted(files):
            if not video_name.lower().endswith(('.mp4', '.mov')): 
                continue
            
            label = os.path.basename(root)
            video_path = os.path.join(root, video_name)
            cap = cv2.VideoCapture(video_path)
            sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Convert BGR to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(image)
                
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0]
                    sequence.append([[l.x, l.y, l.z] for l in lm.landmark])
                elif len(sequence) > 0:
                    # Fill gaps with last known position for continuity
                    sequence.append(sequence[-1])
            
            cap.release()

            if len(sequence) > 15:
                # 1. Process and save Original
                base_data = interpolate_landmarks(np.array(sequence))
                orig_filename = f"{label}_{video_name.split('.')[0]}_orig.npy"
                np.save(os.path.join(output_dir, orig_filename), base_data.astype(np.float32))
                
                # 2. Generate and save Augmentations
                for v in range(variations_per_vid):
                    aug_data = get_augmentation(base_data.copy())
                    aug_filename = f"{label}_{video_name.split('.')[0]}_aug{v}.npy"
                    np.save(os.path.join(output_dir, aug_filename), aug_data.astype(np.float32))
                
                print(f"✅ {label}: Generated {variations_per_vid + 1} samples from {video_name}")
            else:
                print(f"⚠️ {video_name} skipped: Not enough frames with hand detected.")

if __name__ == "__main__":
    # Ensure these paths exist on your Mac
    # variations_per_vid=35 will give you 36 total samples per video
    process_and_augment('data/raw_videos', 'data/landmarks', variations_per_vid=35)