"""
Generate training charts for paper/presentation.
3-panel layout: Loss Convergence, Validation Accuracy, LR Schedule.

Usage:
    python scripts/plot_training.py --history models/output_v15/history_round1.json --output v15_charts.png
    python scripts/plot_training.py --history models/output_v15/history_round1.json --eval models/output_v15/eval_results_round1.json --output v15_charts.png
"""
import json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', required=True, help='Path to history.json')
    parser.add_argument('--eval', default=None, help='Path to eval_results.json (optional)')
    parser.add_argument('--output', default='v15_charts.png')
    parser.add_argument('--title', default='Stage 1 Training (v15 — Apple Vision, d_model=384)')
    args = parser.parse_args()

    with open(args.history) as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    val_top5 = [h['val_top5'] for h in history]
    lr = [h['lr'] for h in history]

    # Train loss
    if 'train_loss' in history[0]:
        train_loss = [h['train_loss'] for h in history]
    else:
        train_loss = None

    # Find best epoch
    best_idx = np.argmax(val_acc)
    best_epoch = epochs[best_idx]
    best_acc = val_acc[best_idx]

    # Load eval results if available
    eval_data = None
    if args.eval:
        with open(args.eval) as f:
            eval_data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(args.title, fontsize=14, fontweight='bold')

    # ---- Panel 1: Loss Convergence ----
    ax1 = axes[0]
    ax1.set_title('Training and Validation Loss\nwith Convergence Metrics')
    if train_loss:
        ax1.plot(epochs, train_loss, color='blue', linewidth=1.2, label='Training Loss')
    ax1.plot(epochs, val_loss, color='orange', linewidth=1.5, label='Validation Loss')
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Epoch of Convergence')

    # Mark convergence point
    converged_loss = val_loss[best_idx]
    ax1.plot(best_epoch, converged_loss, 'ro', markersize=8, zorder=5,
             label=f'Converged Validation Loss: {converged_loss:.4f}')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ---- Panel 2: Validation Accuracy ----
    ax2 = axes[1]
    ax2.set_title('Validation Accuracy')
    ax2.bar(epochs, val_acc, color='green', alpha=0.6, width=0.8, label='Top-1 Accuracy')
    ax2.plot(epochs, val_top5, color='purple', linewidth=1.5, label='Top-5 Accuracy')
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=best_acc, color='red', linestyle=':', alpha=0.3)
    ax2.text(best_epoch + 2, best_acc - 3, f'{best_acc:.2f}%\n(Ep {best_epoch})',
             fontsize=9, color='red', fontweight='bold')

    # Add eval results
    if eval_data:
        test_acc = eval_data['test']['accuracy']
        ax2.text(epochs[-1] * 0.5, 10,
                f"Test: {test_acc:.2f}%\n"
                f"Precision: {eval_data['test']['precision_weighted']:.1f}%\n"
                f"Recall: {eval_data['test']['recall_weighted']:.1f}%\n"
                f"F1: {eval_data['test']['f1_weighted']:.1f}%",
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- Panel 3: Learning Rate Schedule ----
    ax3 = axes[2]
    ax3.set_title('Learning Rate Schedule')
    ax3.plot(epochs, lr, color='purple', linewidth=2)
    ax3.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")

    # Print summary
    print(f"\nBest epoch: {best_epoch}")
    print(f"Val accuracy: {best_acc:.2f}%")
    print(f"Val top-5: {val_top5[best_idx]:.2f}%")
    if eval_data:
        print(f"\nFinal eval (no ArcFace, no dropout):")
        for split_key, split_name in [('train_augmented', 'Train (aug)'), ('train_clean', 'Train (clean)'),
                                       ('train', 'Train'), ('val', 'Val'), ('test', 'Test')]:
            if split_key in eval_data:
                m = eval_data[split_key]
                print(f"  {split_name:15s}: Acc={m['accuracy']:.2f}% | F1={m['f1_weighted']:.2f}% | Loss={m['loss']:.4f}")


if __name__ == '__main__':
    main()
