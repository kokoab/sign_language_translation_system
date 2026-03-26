"""
Download RTMPose-WholeBody model for improved hand detection in SLT extraction.

This model detects 133 keypoints (body + hands + face) and is much better than
MediaPipe at detecting fists, back-of-hand, and occluded hands.

Usage:
    python3 models/download_rtmpose.py
"""

import os
import urllib.request
import sys

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# RTMPose-WholeBody-m (medium) — matches MMPoseInferencer(pose2d='wholebody') config
# IMPORTANT: 'wholebody' alias loads RTMPose-M architecture, NOT RTMPose-L.
# Using RTMPose-L checkpoint with RTMPose-M config causes size mismatch and silent failures.
FILES = {
    "rtmpose_wholebody.pth": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth",
        "size_mb": 69,
        "description": "RTMPose-WholeBody-m checkpoint (133 keypoints)",
    },
}


def download_file(url, dest_path, description=""):
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Already exists: {os.path.basename(dest_path)} ({size_mb:.1f} MB)")
        return True

    print(f"  Downloading: {description or os.path.basename(dest_path)}")
    print(f"  URL: {url}")

    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  {mb_down:.1f}/{mb_total:.1f} MB ({pct:.0f}%)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
        print()

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Saved: {dest_path} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n  FAILED: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    print("RTMPose-WholeBody Model Download")
    print("=" * 50)

    os.makedirs(MODELS_DIR, exist_ok=True)

    success = 0
    for filename, info in FILES.items():
        dest = os.path.join(MODELS_DIR, filename)
        if download_file(info["url"], dest, info["description"]):
            success += 1

    print(f"\n{'='*50}")
    print(f"Downloaded {success}/{len(FILES)} files to {MODELS_DIR}")

    if success == len(FILES):
        print("RTMPose-WholeBody is ready for extraction!")
    else:
        print("Some downloads failed. Re-run this script to retry.")


if __name__ == "__main__":
    main()
