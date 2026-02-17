import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.interpolate import interp1d

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    model_complexity=1 
)

def interpolate_landmarks(sequence, target_frames=60):
    current_frames = len(sequence)
    if current_frames == target_frames:
        return sequence
    x = np.linspace(0, current_frames - 1, num=current_frames)
    x_new = np.linspace(0, current_frames - 1, num=target_frames)
    f = interp1d(x, sequence, axis=0, kind='linear')
    return f(x_new)

def generate_idle_samples(output_dir, num_samples=50):
    """
    Simulates 'Background/Idle' by creating sequences of a hand in a 
    relaxed neutral position with natural micro-movements (noise).
    """
    print(f"Generating {num_samples} synthetic 'Idle' samples...")
    for i in range(num_samples):
        # Start with a base 'neutral' hand pose (scaled 0 to 1)
        # Using a simple flattened array representing a relaxed palm
        base_hand = np.array([
            0.5, 0.9, 0.0, # Wrist
            0.4, 0.8, 0.0, 0.3, 0.7, 0.0, 0.2, 0.6, 0.0, 0.1, 0.5, 0.0, # Thumb
            0.4, 0.4, 0.0, 0.4, 0.3, 0.0, 0.4, 0.2, 0.0, 0.4, 0.1, 0.0, # Index
            0.5, 0.4, 0.0, 0.5, 0.3, 0.0, 0.5, 0.2, 0.0, 0.5, 0.1, 0.0, # Middle
            0.6, 0.4, 0.0, 0.6, 0.3, 0.0, 0.6, 0.2, 0.0, 0.6, 0.1, 0.0, # Ring
            0.7, 0.5, 0.0, 0.7, 0.4, 0.0, 0.7, 0.3, 0.0, 0.7, 0.2, 0.0  # Pinky
        ])
        
        # Create 60 frames of this hand with brownian-style motion (drift)
        sequence = []
        current_pose = base_hand.copy()
        for _ in range(60):
            noise = np.random.normal(0, 0.002, 63) # Subtle jitter
            current_pose += noise
            sequence.append(current_pose.copy())
            
        save_filename = f"Idle_simulated_{i}.npy"
        np.save(os.path.join(output_dir, save_filename), np.array(sequence))

def process_and_save(base_video_dir, output_dir):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    # --- INSERT SIMULATION HERE ---
    generate_idle_samples(output_dir, num_samples=60)
    
    for root, dirs, files in os.walk(base_video_dir):
        for video_name in files:
            if not video_name.endswith(('.mp4', '.mov')): continue
            
            label = os.path.basename(root)
            path = os.path.join(root, video_name)
            cap = cv2.VideoCapture(path)
            sequence = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    points = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
                    sequence.append(points)
                # CRITICAL: If hand is lost mid-video, don't use zeros. 
                # Use the last known position to keep the sequence fluid.
                elif len(sequence) > 0:
                    sequence.append(sequence[-1])
            
            cap.release()
            
            # Minimum frame check to ensure quality
            if len(sequence) > 15:
                final_seq = interpolate_landmarks(np.array(sequence), target_frames=60)
                save_filename = f"{label}_{video_name[:-4]}.npy"
                np.save(os.path.join(output_dir, save_filename), final_seq)
                print(f"✅ Processed: {label} -> {video_name}")

if __name__ == "__main__":
    # Ensure paths are correct for your M4 Air
    process_and_save('data/raw_videos', 'data/landmarks')