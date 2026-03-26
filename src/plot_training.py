"""Plot Stage 1 training curves from history.json"""
import json, sys, matplotlib.pyplot as plt

def plot(history_path, title="Stage 1 Training"):
    with open(history_path) as f:
        history = json.load(f)

    epochs, train_loss, val_loss, val_acc, val_top5 = [], [], [], [], []
    raw_train_loss = []

    for entry in history:
        if "epoch" not in entry:
            continue
        ep = entry["epoch"]
        epochs.append(ep)
        train_loss.append(entry.get("train_loss"))
        raw_train_loss.append(entry.get("raw_train_loss"))
        val_loss.append(entry.get("val_loss"))
        val_acc.append(entry.get("val_top1"))
        val_top5.append(entry.get("val_top5"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot - use raw_train_loss if available for fair comparison
    has_raw = any(x is not None for x in raw_train_loss)
    if has_raw:
        ax1.plot(epochs, raw_train_loss, label="Train Loss (raw)", alpha=0.8)
    ax1.plot(epochs, train_loss, label="Train Loss (regularized)", alpha=0.5 if has_raw else 0.8)
    vl = [v for v in val_loss if v is not None]
    vl_ep = [e for e, v in zip(epochs, val_loss) if v is not None]
    ax1.plot(vl_ep, vl, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    va = [v for v in val_acc if v is not None]
    va_ep = [e for e, v in zip(epochs, val_acc) if v is not None]
    vt5 = [v for v in val_top5 if v is not None]
    vt5_ep = [e for e, v in zip(epochs, val_top5) if v is not None]
    ax2.plot(va_ep, va, label="Val Top-1", color="tab:green")
    ax2.plot(vt5_ep, vt5, label="Val Top-5", color="tab:blue", alpha=0.6)
    best_acc = max(va) if va else 0
    best_ep = va_ep[va.index(best_acc)] if va else 0
    ax2.axhline(y=best_acc, color="tab:green", linestyle="--", alpha=0.3)
    ax2.annotate(f"Best: {best_acc:.2f}% (ep {best_ep})",
                 xy=(best_ep, best_acc), fontsize=9,
                 xytext=(best_ep - len(epochs)*0.3, best_acc - 3),
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{title} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = history_path.replace(".json", "_chart.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "history_joint.json"
    title = sys.argv[2] if len(sys.argv) > 2 else "Stage 1 Joint Stream"
    plot(path, title)
