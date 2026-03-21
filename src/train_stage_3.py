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
BATCH_SIZE = 32         # Smaller batch for larger model
NUM_EPOCHS = 10
LEARNING_RATE = 2e-4
WARMUP_STEPS = 200

# File paths
KAGGLE_DATA_PATH = "/kaggle/input/datasets/kokoab/rosetta-stone5/slt_stage3_dataset_v2.csv"
LOCAL_DATA_PATH = "slt_stage3_dataset_v2.csv"
FALLBACK_PATH = "slt_stage3_dataset_final.csv"

DIALOGUE_PATH = "slt_dialogue_dataset.csv"
KAGGLE_DIALOGUE_PATH = "/kaggle/input/datasets/kokoab/rosetta-stone5/slt_dialogue_dataset.csv"

OUTPUT_DIR = "/kaggle/working/asl_flan_t5_results"
SAVE_PATH = "/kaggle/working/slt_conversational_t5_model"

# ══════════════════════════════════════════════════════════════════
#  1. LOAD DATASET
# ══════════════════════════════════════════════════════════════════
print("\n1. Loading ASL Dataset...")

# Try different paths
for path in [KAGGLE_DATA_PATH, LOCAL_DATA_PATH, FALLBACK_PATH]:
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
for dpath in [KAGGLE_DIALOGUE_PATH, DIALOGUE_PATH]:
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

# Adjust paths for local vs Kaggle
if not os.path.exists("/kaggle"):
    OUTPUT_DIR = "./asl_flan_t5_results"
    SAVE_PATH = "./slt_conversational_t5_model"

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # Evaluation strategy
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Learning rate & schedule (tuned for Flan-T5-Base)
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,

    # Batch & epochs
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,

    # Regularization
    weight_decay=0.01,
    label_smoothing_factor=0.1,  # Helps with diverse outputs

    # Generation settings
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,

    # Memory & speed
    fp16=torch.cuda.is_available(),
    gradient_accumulation_steps=2,  # Effective batch size = 64

    # Misc
    save_total_limit=2,
    report_to="none",
    logging_steps=100,
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
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
print(f"   🧪 Test Loss: {test_results['test_loss']:.4f}")

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
    print(f"   ✅ Model zipped to {SAVE_PATH}.zip")
    print("   → Open the Kaggle Output tab and click the download icon")
except Exception as e:
    print(f"   ⚠️ Could not zip: {e}")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
