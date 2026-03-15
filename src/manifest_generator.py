import os
import json
from pathlib import Path

def generate_manifest(npy_dir):
    out_path = Path(npy_dir).expanduser() # .expanduser() resolves the '~' on Mac
    manifest = {}
    
    if not out_path.exists():
        print(f"❌ Error: Directory not found at {out_path}")
        return

    # Get all .npy files in the directory
    npy_files = [f for f in os.listdir(out_path) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"⚠️ No .npy files found in {out_path}")
        return

    print(f"🔍 Scanning {len(npy_files)} files...")

    # Parse labels from the filenames 
    # Assumes format: {label}_video_{hash}.npy OR {label}_{stem}_{hash}.npy
    for f in npy_files:
        if '_video_' in f:
            label = f.split('_video_')[0]
        else:
            # Fallback if your naming convention uses a different separator
            label = f.split('_')[0] 
            
        manifest[f] = label

    # Save to disk right next to the .npy files
    manifest_path = out_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    unique_labels = set(manifest.values())
    print(f"✅ Success! Generated manifest.json with {len(manifest)} entries.")
    print(f"📊 Found {len(unique_labels)} unique sign language classes.")
    print(f"💾 Saved to: {manifest_path}")

# --- Execute ---
# UPDATE THIS PATH to where your .npy files are stored on your Mac.
# You can use '~' to represent your user home directory (e.g., '~/Desktop/ALPHABETS_landmarks')
MAC_NPY_DIR = '/Users/frnzlo/Documents/machine_learning/SLT/ASL_landmarks_float16' 

if __name__ == "__main__":
    generate_manifest(MAC_NPY_DIR)