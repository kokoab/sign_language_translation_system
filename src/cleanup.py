import os
import shutil

def clear_landmarks(directory):
    if os.path.exists(directory):
        print(f"⚠️ Deleting all files in {directory}...")
        # Option A: Delete the whole folder and recreate it
        shutil.rmtree(directory)
        os.makedirs(directory)
        print("✅ Directory cleared.")
    else:
        os.makedirs(directory)
        print("📂 Created new directory.")

if __name__ == "__main__":
    # Update this path to match your local setup
    clear_landmarks('data/landmarks')