"""
Download RTMW-l WholeBody model for SLT extraction.

RTMW-l is the primary pose extractor — trained on 14 datasets (Cocktail14),
distilled from RTMW-x teacher into RTMW-l architecture.
66.3% hand AP vs RTMPose-m's 47.5% — the single biggest quality improvement.

Usage:
    python3 models/download_rtmw.py
"""

import os
import urllib.request
import sys

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# RTMW-l WholeBody (Cocktail14, 133 keypoints, 384x288)
# Naming: "dw-x-l" = DWPose distillation, teacher=X, student=L
# Same keypoint layout: body 0-16, feet 17-22, face 23-90, lhand 91-111, rhand 112-132
# Config: rtmw-l_8xb320-270e_cocktail14-384x288.py
FILES = {
    "rtmw_l_wholebody.pth": {
        "url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth",
        "size_mb": 230,
        "description": "RTMW-l WholeBody checkpoint (133 keypoints, Cocktail14, 384x288)",
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
    print("RTMW-l WholeBody Model Download")
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
        print("RTMW-l WholeBody is ready for extraction!")
    else:
        print("Some downloads failed. Re-run this script to retry.")


if __name__ == "__main__":
    main()
