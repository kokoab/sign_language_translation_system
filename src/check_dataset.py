import os
from collections import Counter
from pathlib import Path

# Path to your processed data
OUTPUT_DIR = "ASL_landmarks_float16"

# Match extract.py: hash is last 6 hex chars before .npy; label is first segment
def _label_from_npy_basename(basename: str) -> str:
    """Extract sign label from npy basename. Format: {label}_{stem}_{hash}.npy or {...}_fast|_slow|_mirror_{hash}.npy"""
    for suffix in ["_fast", "_slow", "_mirror"]:
        if suffix in basename:
            basename = basename.replace(suffix, "")
            break
    # Hash is always last 6 hex chars
    if len(basename) >= 7 and basename[-7] == "_":
        basename = basename[:-7]  # remove _XXXXXX
    # Label is first underscore-separated segment
    return basename.split("_", 1)[0] if "_" in basename else basename


def audit_dataset():
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Error: Folder '{OUTPUT_DIR}' not found.")
        return

    # Get all .npy files
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".npy")]

    if not files:
        print("Empty folder. No processed videos found.")
        return

    # Count unique videos per class (base names only - augmentations count as 1 video)
    seen_bases = set()
    labels = []
    for f in files:
        base = f.replace(".npy", "")
        for suffix in ["_fast", "_slow", "_mirror"]:
            if suffix in base:
                base = base.replace(suffix, "")
                break
        if base in seen_bases:
            continue
        seen_bases.add(base)
        labels.append(_label_from_npy_basename(base))

    counts = Counter(labels)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print("-" * 40)
    print(f"📊 DATASET AUDIT: {OUTPUT_DIR}")
    print(f"Total .npy files:      {len(files)}")
    print(f"Unique videos (bases): {len(seen_bases)}")
    print(f"Unique Classes/Signs:  {len(counts)}")
    print("-" * 40)

    # Display Top 10
    print("\n✅ TOP 10 CLASSES (Most Data):")
    for label, count in sorted_counts[:10]:
        print(f"  {label:<20} : {count} vids")

    # Display Bottom 10
    print("\n⚠️ BOTTOM 10 CLASSES (Least Data):")
    for label, count in sorted_counts[-10:]:
        print(f"  {label:<20} : {count} vids")

    # Optional: Alert you to very low data
    very_low = [label for label, count in counts.items() if count < 5]
    if very_low:
        print(f"\n🚨 ALERT: {len(very_low)} classes have fewer than 5 samples!")
    
    print("-" * 40)

if __name__ == "__main__":
    audit_dataset()