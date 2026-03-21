import matplotlib.pyplot as plt
import json
import os

def create_dashboard(save_path='unified_dashboard.png'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('SLT Pipeline Training Summary', fontsize=16, fontweight='bold')

    # Stage 1: Accuracy
    if os.path.exists('history.json'):
        with open('history.json', 'r') as f:
            s1_data = json.load(f)
        epochs = [d['epoch'] for d in s1_data]
        acc = [d['val_acc'] for d in s1_data]
        axes[0].plot(epochs, acc, color='tab:green', marker='o', markersize=4, linewidth=2)
        axes[0].set_title('Stage 1: Feature Extraction\n(Recognizing Isolated Signs)', fontsize=12)
        axes[0].set_ylabel('Accuracy %')
        axes[0].set_xlabel('Epochs')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Stage 1 Logs Missing', ha='center')

    # Stage 2: WER
    if os.path.exists('stage2_history.json'):
        with open('stage2_history.json', 'r') as f:
            s2_data = json.load(f)
        epochs = [d['epoch'] for d in s2_data]
        wer = [d['val_wer'] for d in s2_data]
        axes[1].plot(epochs, wer, color='tab:red', marker='s', markersize=4, linewidth=2)
        axes[1].set_title('Stage 2: Sequence Alignment\n(Word Error Rate - Lower is Better)', fontsize=12)
        axes[1].set_ylabel('WER %')
        axes[1].set_xlabel('Epochs')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Stage 2 Logs Missing', ha='center')

    # Stage 3: Translation Loss
    hf_log_path = 'asl_flan_t5_results/stage3_history.json'
    fallback_log_path = 'asl_flan_t5_results/trainer_state.json'
    
    history = []
    if os.path.exists(hf_log_path):
        with open(hf_log_path, 'r') as f:
            history = json.load(f)
    elif os.path.exists(fallback_log_path):
        with open(fallback_log_path, 'r') as f:
            s3_data = json.load(f)
            history = s3_data.get('log_history', [])
            
    if history:
        epochs = [h['epoch'] for h in history if 'eval_loss' in h]
        loss = [h['eval_loss'] for h in history if 'eval_loss' in h]
        axes[2].plot(epochs, loss, color='tab:blue', marker='v', markersize=4, linewidth=2)
        axes[2].set_title('Stage 3: Language Modeling\n(Translation Eval Loss)', fontsize=12)
        axes[2].set_ylabel('Loss')
        axes[2].set_xlabel('Epochs')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Stage 3 Logs Missing\n(Run Stage 3 first)', ha='center')

    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    plt.savefig(save_path)
    print(f"✅ Unified Dashboard saved to {save_path}")

if __name__ == "__main__":
    create_dashboard()