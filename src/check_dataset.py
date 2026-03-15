import os
from collections import Counter
from pathlib import Path

# Path to your processed data
OUTPUT_DIR = "ASL_landmarks_float16"

def audit_dataset():
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Error: Folder '{OUTPUT_DIR}' not found.")
        return

    # Get all .npy files
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    
    if not files:
        print("Empty folder. No processed videos found.")
        return

    # Extract labels
    # Filename format: {label}_{stem}_{hash}.npy
    # We use rsplit(_, 2) to handle labels that might have underscores in them
    labels = []
    for f in files:
        label = f.rsplit('_', 2)[0]
        labels.append(label)

    counts = Counter(labels)
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print("-" * 40)
    print(f"📊 DATASET AUDIT: {OUTPUT_DIR}")
    print(f"Total Processed Videos: {len(files)}")
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