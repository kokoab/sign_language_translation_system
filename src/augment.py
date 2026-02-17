import numpy as np
import os

def rotate_landmarks(data, max_angle=15):
    """Rotates landmarks slightly to simulate different hand angles."""
    # Pick a random angle
    angle = np.radians(np.random.uniform(-max_angle, max_angle))
    # Rotation Matrix for Y-axis (side to side tilt)
    ry = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    # Reshape to [60, 21, 3], rotate, then flatten back to [60, 63]
    reshaped = data.reshape(60, 21, 3)
    rotated = np.matmul(reshaped, ry)
    return rotated.reshape(60, 63)

def add_noise(data, sigma=0.003):
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise

def run_augmentation(input_dir):
    # Only augment original files (not previous augmentations)
    files = [f for f in os.listdir(input_dir) if f.endswith('.npy') and '_aug' not in f]
    print(f"🔄 Augmenting {len(files)} original samples...")

    for file in files:
        data = np.load(os.path.join(input_dir, file))
        
        # Create 10 variations for every 1 original (helps hit that 40+ mark)
        for i in range(10):
            # Apply rotation THEN noise
            aug_data = rotate_landmarks(data)
            aug_data = add_noise(aug_data)
            
            new_name = f"{file[:-4]}_aug_v{i}.npy"
            np.save(os.path.join(input_dir, new_name), aug_data)
            
    print("✅ High-fidelity augmentation complete.")

if __name__ == "__main__":
    run_augmentation('data/landmarks')