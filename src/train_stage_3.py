"""
╔══════════════════════════════════════════════════════════════════╗
║  SLT Stage 3 — Gloss-to-English Translation (Conversational)    ║
║  Flan-T5-Base with optional dialogue context support            ║
║  Input : ASL gloss sequences                                     ║
║  Output: Natural conversational English                          ║
╚══════════════════════════════════════════════════════════════════╝

Improvements over v1:
- Uses Flan-T5-Base (250M params) instead of T5-small (60M)
- Supports dialogue context for conversational translations
- Better prompt engineering for natural output
- Improved training configuration
"""

import pandas as pd
import os
import torch
import json
import random
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
try:
    import evaluate as hf_evaluate
    _bleu_metric = hf_evaluate.load("sacrebleu")
    _rouge_metric = hf_evaluate.load("rouge")
    HAS_METRICS = True
except Exception:
    HAS_METRICS = False
    print("WARNING: 'evaluate' package not found. Install with: pip install evaluate sacrebleu rouge-score")

print("=" * 60)
print("SLT Stage 3 — Conversational Gloss-to-English Training")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# Model selection - Flan-T5-Base is instruction-tuned and better for this task
MODEL_CHECKPOINT = "google/flan-t5-base"  # 250M params (was t5-small 60M)

# Training configuration
MAX_INPUT_LENGTH = 96   # Longer to accommodate context
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 32
NUM_EPOCHS = 25         # Increased from 10 (was underfitting)
LEARNING_RATE = 2e-4
WARMUP_STEPS = 200
NOISY_GLOSS_PROB = 0.3  # Probability of corrupting gloss input (robustness to Stage 2 errors)

# File paths — auto-detect environment (Vast.ai / Kaggle / local)
_DATA_SEARCH = [
    "/workspace/slt_stage3_dataset_v2.csv",
    "/workspace/slt_stage3_dataset_final.csv",
    "/kaggle/input/datasets/kokoab/rosetta-stone5/slt_stage3_dataset_v2.csv",
    "slt_stage3_dataset_v2.csv",
    "slt_stage3_dataset_final.csv",
]
_DIALOGUE_SEARCH = [
    "/workspace/slt_dialogue_dataset.csv",
    "/kaggle/input/datasets/kokoab/rosetta-stone5/slt_dialogue_dataset.csv",
    "slt_dialogue_dataset.csv",
]

if os.path.isdir("/workspace"):
    OUTPUT_DIR = "/workspace/output/asl_flan_t5_results"
    SAVE_PATH = "/workspace/output/slt_conversational_t5_model"
elif os.path.isdir("/kaggle/working"):
    OUTPUT_DIR = "/kaggle/working/asl_flan_t5_results"
    SAVE_PATH = "/kaggle/working/slt_conversational_t5_model"
else:
    OUTPUT_DIR = "./asl_flan_t5_results"
    SAVE_PATH = "./slt_conversational_t5_model"

# ══════════════════════════════════════════════════════════════════
#  NOISY GLOSS AUGMENTATION (robustness to Stage 2 CTC errors)
# ══════════════════════════════════════════════════════════════════

def augment_noisy_gloss(gloss_str, all_glosses, prob=0.3):
    """Simulate realistic Stage 2 CTC errors: deletion, insertion, substitution,
    repetition, and multi-error sequences. Based on real CTC error patterns:
    - Deletion: CTC merges signs that are too similar (most common, ~40%)
    - Substitution: Confused sign classes from Stage 1 (~30%)
    - Insertion: CTC spurious activations at boundaries (~15%)
    - Repetition: CTC stutter on confident frames (~15%)"""
    if random.random() > prob:
        return gloss_str
    tokens = gloss_str.strip().split()
    if len(tokens) == 0:
        return gloss_str
    # Apply 1-3 errors depending on sequence length (longer = more errors)
    num_errors = 1 if len(tokens) <= 2 else random.choices([1, 2, 3], weights=[50, 35, 15])[0]
    for _ in range(num_errors):
        if len(tokens) == 0:
            break
        error_type = random.choices(
            ['delete', 'substitute', 'insert', 'repeat'],
            weights=[40, 30, 15, 15]
        )[0]
        if error_type == 'delete' and len(tokens) > 1:
            idx = random.randint(0, len(tokens) - 1)
            tokens.pop(idx)
        elif error_type == 'substitute':
            idx = random.randint(0, len(tokens) - 1)
            tokens[idx] = random.choice(all_glosses)
        elif error_type == 'insert':
            idx = random.randint(0, len(tokens))
            tokens.insert(idx, random.choice(all_glosses))
        elif error_type == 'repeat' and len(tokens) > 0:
            idx = random.randint(0, len(tokens) - 1)
            tokens.insert(idx, tokens[idx])
    return " ".join(tokens)

# ══════════════════════════════════════════════════════════════════
#  1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════
print("\n1. Loading ASL Dataset...")

# Try different paths
for path in _DATA_SEARCH:
    if os.path.exists(path):
        file_path = path
        break
else:
    raise FileNotFoundError("No dataset file found! Generate with generate_stage3_data_v2.py first.")

df = pd.read_csv(file_path)
df = df.dropna(subset=["gloss", "text"]).reset_index(drop=True)
print(f"   Loaded {len(df)} rows from {file_path}")

# Try to load dialogue dataset for context-aware samples
dialogue_df = None
for dpath in _DIALOGUE_SEARCH:
    if os.path.exists(dpath):
        dialogue_df = pd.read_csv(dpath)
        print(f"   Loaded {len(dialogue_df)} dialogue turns from {dpath}")
        break

# ══════════════════════════════════════════════════════════════════
#  2. PREPARE CONTEXT-AWARE DATA
# ══════════════════════════════════════════════════════════════════
print("\n2. Preparing training data...")

# Combine standard data with context-aware dialogue data
all_samples = []

# Add standard samples (no context)
for _, row in df.iterrows():
    all_samples.append({
        "gloss": row["gloss"],
        "text": row["text"],
        "context": ""
    })

# Add dialogue samples with context (if available)
if dialogue_df is not None and "context" in dialogue_df.columns:
    for _, row in dialogue_df.iterrows():
        all_samples.append({
            "gloss": row["gloss"],
            "text": row["text"],
            "context": row.get("context", "")
        })
    print(f"   Added {len(dialogue_df)} context-aware dialogue samples")

combined_df = pd.DataFrame(all_samples)
combined_df = combined_df.drop_duplicates(subset=["gloss", "context"]).reset_index(drop=True)
print(f"   Base samples: {len(combined_df)}")

# Collect all unique glosses for augmentation
all_unique_glosses = list(set(
    token for gloss_str in combined_df["gloss"] for token in gloss_str.split()
))

# Generate noisy gloss augmented copies (doubles effective dataset size)
noisy_samples = []
for _, row in combined_df.iterrows():
    noisy_gloss = augment_noisy_gloss(row["gloss"], all_unique_glosses, prob=0.8)
    if noisy_gloss != row["gloss"]:
        noisy_samples.append({
            "gloss": noisy_gloss,
            "text": row["text"],  # Same target — model learns to handle errors
            "context": row.get("context", "")
        })

if noisy_samples:
    noisy_df = pd.DataFrame(noisy_samples)
    combined_df = pd.concat([combined_df, noisy_df], ignore_index=True)
    print(f"   + {len(noisy_samples)} noisy gloss augmented samples")

print(f"   Total training samples: {len(combined_df)}")

# ══════════════════════════════════════════════════════════════════
#  3. CREATE HF DATASET
# ══════════════════════════════════════════════════════════════════
print("\n3. Creating HuggingFace Dataset...")

dataset = Dataset.from_pandas(combined_df)
# 70/15/15 split: first split 30% off, then split that 50/50 into val/test
train_temp = dataset.train_test_split(test_size=0.30, seed=42)
val_test = train_temp["test"].train_test_split(test_size=0.50, seed=42)
dataset = DatasetDict({
    "train": train_temp["train"],
    "validation": val_test["train"],
    "test": val_test["test"],
})
print(f"   Train: {len(dataset['train'])}  |  Val: {len(dataset['validation'])}  |  Test: {len(dataset['test'])}")

# ══════════════════════════════════════════════════════════════════
#  4. MODEL & TOKENIZER
# ══════════════════════════════════════════════════════════════════
print("\n4. Initializing Model & Tokenizer...")
print(f"   Using: {MODEL_CHECKPOINT}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"   Model parameters: {param_count:.1f}M")

# ══════════════════════════════════════════════════════════════════
#  5. TOKENIZATION WITH CONTEXT
# ══════════════════════════════════════════════════════════════════
print("\n5. Preprocessing & Tokenizing Data...")

# Prompt format optimized for Flan-T5
MAX_CONTEXT_TURNS = 4  # Expanded from 2 for better disambiguation

def create_prompt(gloss, context=""):
    """
    Create instruction prompt for Flan-T5.
    Flan-T5 is instruction-tuned so it responds well to clear instructions.
    Context supports up to MAX_CONTEXT_TURNS previous dialogue turns.
    """
    if context and len(context.strip()) > 0:
        # Limit context to MAX_CONTEXT_TURNS turns to avoid input overflow
        turns = [t.strip() for t in context.split("|") if t.strip()]
        turns = turns[-MAX_CONTEXT_TURNS:]
        context = " | ".join(turns)
        return f"[Previous: {context}] Translate this ASL gloss to natural conversational English: {gloss}"
    else:
        return f"Translate this ASL gloss to natural conversational English: {gloss}"


def preprocess_function(examples):
    """Tokenize inputs and targets with context support."""
    inputs = []
    for gloss, context in zip(examples["gloss"], examples["context"]):
        prompt = create_prompt(gloss, context if context else "")
        inputs.append(prompt)

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    )

    labels = tokenizer(
        text_target=examples["text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# ══════════════════════════════════════════════════════════════════
#  6. TRAINING ARGUMENTS
# ══════════════════════════════════════════════════════════════════
print("\n6. Setting up Training Arguments...")

# BLEU/ROUGE compute_metrics (Section 3.4 of plan)
def compute_metrics(eval_preds):
    if not HAS_METRICS:
        return {}
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100 (padding) with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Strip whitespace
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    # BLEU
    bleu_result = _bleu_metric.compute(predictions=decoded_preds,
                                        references=[[l] for l in decoded_labels])
    # ROUGE
    rouge_result = _rouge_metric.compute(predictions=decoded_preds,
                                          references=decoded_labels)
    return {
        "bleu": round(bleu_result["score"], 2),
        "rouge1": round(rouge_result["rouge1"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
    }

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # Evaluation strategy
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=False,  # Disabled: early epochs generate garbage causing OverflowError

    # Learning rate & schedule
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,

    # Batch & epochs (increased from 10 to 25)
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,

    # Regularization
    weight_decay=0.01,
    label_smoothing_factor=0.1,

    # Generation settings
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,

    # Memory & speed
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,

    # Misc
    save_total_limit=3,
    report_to="none",
    logging_steps=50,
)

# ══════════════════════════════════════════════════════════════════
#  7. TRAINER
# ══════════════════════════════════════════════════════════════════
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    pad_to_multiple_of=8
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=None,  # Disabled: predict_with_generate=False means logits not tokens
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],  # Increased from 3
)

# ══════════════════════════════════════════════════════════════════
#  8. TRAIN
# ══════════════════════════════════════════════════════════════════
print("\n7. STARTING TRAINING! 🚀")
print(f"   Model: {MODEL_CHECKPOINT}")
print(f"   Batch size: {BATCH_SIZE} x 2 (grad accum) = {BATCH_SIZE * 2}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE}")

trainer.train()

# ══════════════════════════════════════════════════════════════════
#  8b. SAVE HISTORY
# ══════════════════════════════════════════════════════════════════
print("\n8b. Saving Training History...")
history_path = os.path.join(OUTPUT_DIR, "stage3_history.json")
with open(history_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

# ══════════════════════════════════════════════════════════════════
#  9. SAVE MODEL
# ══════════════════════════════════════════════════════════════════
print("\n8. Saving the final Model...")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"   ✅ Model saved to {SAVE_PATH}")

# ══════════════════════════════════════════════════════════════════
#  9b. FINAL TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════
print("\n8b. Evaluating on held-out test set...")
test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
print(f"   Test Loss: {test_results['test_loss']:.4f}")
if 'test_bleu' in test_results:
    print(f"   Test BLEU: {test_results['test_bleu']:.2f}")
if 'test_rougeL' in test_results:
    print(f"   Test ROUGE-L: {test_results['test_rougeL']:.4f}")
# Save all test results
with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
    json.dump({k: float(v) if isinstance(v, (int, float)) else v for k, v in test_results.items()}, f, indent=2)

# ══════════════════════════════════════════════════════════════════
#  10. INFERENCE TESTS
# ══════════════════════════════════════════════════════════════════
print("\n9. Running inference tests...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_cases = [
    # Single words
    ("HELLO", ""),
    ("THANK-YOU", ""),

    # Standard sentences
    ("I GO STORE", ""),
    ("NAME WHAT YOU", ""),
    ("HELP ME PLEASE", ""),
    ("YESTERDAY I BUY FOOD", ""),
    ("TOMORROW WE GO SCHOOL", ""),

    # With context
    ("I GOOD THANK-YOU", "A: Hello, how are you?"),
    ("NICE MEET YOU", "A: My name is John."),
    ("OK SEE YOU LATER", "A: I have to go now."),
]

print("\n" + "-" * 60)
for gloss, context in test_cases:
    prompt = create_prompt(gloss, context)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if context:
        print(f"  Context: [{context}]")
    print(f"  [{gloss}] → {result}")
    print()

# ══════════════════════════════════════════════════════════════════
#  11. PACKAGE FOR DOWNLOAD
# ══════════════════════════════════════════════════════════════════
print("\n10. Packaging model for download...")
import shutil

try:
    shutil.make_archive(SAVE_PATH, 'zip', SAVE_PATH)
    print(f"   Model zipped to {SAVE_PATH}.zip")
except Exception as e:
    print(f"   ⚠️ Could not zip: {e}")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
