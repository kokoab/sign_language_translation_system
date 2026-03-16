import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

# =====================================================================
# 🛑 STEP 1: CHANGE THIS TO MATCH YOUR ACTUAL MODEL NAME
# Open your original train_stage_1.py to see what your model class is called!
# =====================================================================
from train_stage_1 import SLTStage1 

# =====================================================================
# Self-Contained Dataset (So we don't rely on your original script)
# =====================================================================
class SignDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(list(Path(data_dir).glob('*.npy')))
        raw_labels = [f.stem.split('_')[0] for f in self.files]
        self.classes = sorted(list(set(raw_labels)))
        self.num_classes = len(self.classes)
        self.label_to_idx = {l: i for i, l in enumerate(self.classes)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}
        self.labels = [self.label_to_idx[l] for l in raw_labels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx]).astype(np.float32)
        return torch.tensor(x), self.labels[idx]

def main():
    parser = argparse.ArgumentParser(description="Stage 1 confusion matrix")
    parser.add_argument("--full", action="store_true",
                        help="Use full dataset for matrix (default: 15%% validation split only)")
    parser.add_argument("--val-fraction", type=float, default=0.15,
                        help="Validation fraction when not using --full (default: 0.15)")
    args = parser.parse_args()

    print("⏳ Setting up...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    DATA_DIR = Path('ASL_landmarks_float16')
    BEST_CKPT = Path('weights/best_model.pth')

    # 1. Load Data
    full_ds = SignDataset(DATA_DIR)
    if args.full:
        print("📂 Using full dataset for confusion matrix...")
        eval_ds = full_ds
        val_loader = DataLoader(eval_ds, batch_size=128, shuffle=False)
    else:
        print(f"📂 Using {args.val_fraction*100:.0f}% validation split (see --full to use all data)...")
        train_idx, val_idx = train_test_split(
            list(range(len(full_ds))), test_size=args.val_fraction,
            stratify=full_ds.labels, random_state=42
        )
        val_ds = Subset(full_ds, val_idx)
        val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    idx_to_label = full_ds.idx_to_label

    # 2. Initialize Model
    print("🧠 Loading pre-trained model weights...")
    try:
        # =====================================================================
        # 🛑 STEP 2: MAKE SURE THIS MATCHES YOUR MODEL INITIALIZATION
        # =====================================================================
        model = SLTStage1(num_classes=full_ds.num_classes).to(device)
        
        # NOTE: If your model needs more arguments, change it to match your train_stage_1.py:
        # model = YOUR_MODEL_CLASS_NAME(in_channels=10, num_classes=full_ds.num_classes).to(device)
    except TypeError as e:
        print(f"\n❌ Initialization Error: {e}")
        print("Your model requires different arguments. Check how 'model = ...' is initialized in your train_stage_1.py and update Line 72!")
        return
    
    # Load weights
    checkpoint = torch.load(BEST_CKPT, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint) # Fallback if saved differently
        
    model.eval()

    # 3. Run Inference
    print("🚀 Running inference to generate Confusion Matrix...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    # 4. Filter for Top 15 Classes and Plot
    print("📊 Drawing the chart...")
    top_classes = [item[0] for item in Counter(all_labels).most_common(15)]
    
    filtered_labels = []
    filtered_preds = []
    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label in top_classes:
            filtered_labels.append(true_label)
            filtered_preds.append(pred_label) 
            
    final_true = [t for t, p in zip(filtered_labels, filtered_preds) if p in top_classes]
    final_pred = [p for t, p in zip(filtered_labels, filtered_preds) if p in top_classes]
    
    target_names = [idx_to_label[i] for i in top_classes]
    cm = confusion_matrix(final_true, final_pred, labels=top_classes)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=target_names, yticklabels=target_names,
                     cbar_kws={'label': 'Number of Predictions'},
                     linewidths=.5, square=True)
                     
    data_note = "Full dataset" if args.full else f"Validation set ({args.val_fraction*100:.0f}% of data)"
    plt.title(f'Stage 1 (DS-GCN) Confusion Matrix\n(Top 15 Most Common ASL Signs — {data_note})', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Actual ASL Sign (Ground Truth)', fontsize=14, fontweight='bold')
    plt.xlabel('Model Prediction', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig('stage1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ DONE! Saved as 'stage1_confusion_matrix.png' in your current folder.")

if __name__ == "__main__":
    main()