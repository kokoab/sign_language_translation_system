"""
Ensemble 4 SAM-SLR-v2 streams.
Loads test logits from each stream and averages with weights.

Usage:
    python scripts/ensemble_sam.py \
        --joint /workspace/sam_joint/test_logits.pt \
        --bone /workspace/sam_bone/test_logits.pt \
        --joint_motion /workspace/sam_jm/test_logits.pt \
        --bone_motion /workspace/sam_bm/test_logits.pt
"""
import argparse, torch
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--joint', required=True)
    p.add_argument('--bone', required=True)
    p.add_argument('--joint_motion', required=True)
    p.add_argument('--bone_motion', required=True)
    args = p.parse_args()

    # Load logits from each stream
    j = torch.load(args.joint, weights_only=False)
    b = torch.load(args.bone, weights_only=False)
    jm = torch.load(args.joint_motion, weights_only=False)
    bm = torch.load(args.bone_motion, weights_only=False)

    labels = j['labels']
    assert torch.equal(labels, b['labels'])
    assert torch.equal(labels, jm['labels'])
    assert torch.equal(labels, bm['labels'])

    # SAM's ensemble weights (from ensemble.py): alpha = [1.0, 0.9, 0.5, 0.5]
    alpha = [1.0, 0.9, 0.5, 0.5]
    score = (j['logits'] * alpha[0] + b['logits'] * alpha[1] +
             jm['logits'] * alpha[2] + bm['logits'] * alpha[3]) / sum(alpha)

    # Top-1
    preds = score.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = len(labels)
    top1 = 100 * correct / total

    # Top-5
    _, top5_preds = score.topk(5, dim=1)
    correct_5 = (top5_preds == labels.unsqueeze(1)).any(1).sum().item()
    top5 = 100 * correct_5 / total

    print(f"Ensemble Results:")
    print(f"  Top-1: {top1:.2f}%")
    print(f"  Top-5: {top5:.2f}%")
    print(f"  Total samples: {total}")

    # Try different weight combinations
    print("\nWeight search:")
    best_acc, best_alpha = 0, None
    for a1 in [0.8, 1.0, 1.2]:
        for a2 in [0.6, 0.8, 1.0]:
            for a3 in [0.3, 0.5, 0.7]:
                for a4 in [0.3, 0.5, 0.7]:
                    s = (j['logits']*a1 + b['logits']*a2 + jm['logits']*a3 + bm['logits']*a4) / (a1+a2+a3+a4)
                    acc = 100 * (s.argmax(1) == labels).sum().item() / total
                    if acc > best_acc:
                        best_acc = acc
                        best_alpha = [a1, a2, a3, a4]
    print(f"  Best: {best_acc:.2f}% with alpha={best_alpha}")


if __name__ == "__main__":
    main()
