import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import os

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("1. Loading ASL Dataset...")
file_path = "/kaggle/input/datasets/kokoab/rosetta-stone5/slt_stage3_dataset_final.csv"
if not os.path.exists(file_path):
    file_path = "slt_stage3_dataset_final.csv"

df = pd.read_csv(file_path)

# Safety-net rows are now clean (generate_stage3_data.py R4+: noun-classes only).
# Drop any rows with NaN from CSV read errors, if any.
df = df.dropna(subset=["gloss", "text"]).reset_index(drop=True)
print(f"   Loaded {len(df)} rows.")

# ── 2. HF Dataset split 90/10 ─────────────────────────────────────────────────
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"   Train: {len(dataset['train'])}  |  Eval: {len(dataset['test'])}")

# ── 3. Model & Tokenizer ──────────────────────────────────────────────────────
print("2. Initializing Model & Tokenizer...")
# t5-small is English-only and well-suited for this synthetic English gloss->English task.
# Switch to "google/mt5-small" only if you add non-English target sentences later.
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# ── 4. Tokenization ───────────────────────────────────────────────────────────
print("3. Preprocessing & Tokenizing Data...")
PREFIX = "translate ASL gloss to English: "

# Max lengths — prefix is ~10 tokens, longest gloss ~12, longest target ~25
MAX_INPUT = 32
MAX_TARGET = 48

def preprocess_function(examples):
    inputs = [PREFIX + g for g in examples["gloss"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT, truncation=True, padding=False)
    labels = tokenizer(text_target=examples["text"], max_length=MAX_TARGET, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# ── 5. Training Arguments ─────────────────────────────────────────────────────
print("4. Setting up Training Arguments...")
args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/asl_t5_results",

    # Evaluation
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,       # keeps the best checkpoint, not just last
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # LR & schedule
    learning_rate=3e-4,                # slightly higher than before — fine for t5-small
    lr_scheduler_type="cosine",        # smoother decay than linear

    # Batch & epochs
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=10,               # was 4 — dataset is small, needs more passes
    warmup_steps=100,

    # Regularisation
    weight_decay=0.01,

    # Generation — only used at inference, not during eval
    # (predict_with_generate removed: no compute_metrics uses it, saves ~10x eval time)
    generation_max_length=MAX_TARGET,
    generation_num_beams=4,

    # Misc
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # stop if no improvement for 3 epochs
)

# ── 6. Train ──────────────────────────────────────────────────────────────────
print("5. STARTING TRAINING! 🚀")
trainer.train()

# ── 7. Save ───────────────────────────────────────────────────────────────────
print("6. Saving the final Model...")
SAVE_PATH = "/kaggle/working/slt_final_t5_model"
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"✅ Training Complete! Model saved to {SAVE_PATH}")

# ── 8. Quick Inference Test ───────────────────────────────────────────────────
print("\n7. Running inference tests...")
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_cases = [
    "STORE GO I",
    "NAME WHAT YOU",
    "HELP ME PLEASE",
    "YESTERDAY I BUY FOOD",
    "TOMORROW WE GO SCHOOL",
    "NOW HE FEEL HAPPY",
]

for gloss in test_cases:
    input_ids = tokenizer(PREFIX + gloss, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=MAX_TARGET,
            num_beams=4,
            early_stopping=True,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  [{gloss}] → {result}")

# ── 9. Package & auto-download model ─────────────────────────────────────────
print("\n8. Packaging model for download...")
import shutil

ZIP_PATH = "/kaggle/working/slt_final_t5_model"
shutil.make_archive(ZIP_PATH, 'zip', ZIP_PATH)
print(f"✅ Model zipped to {ZIP_PATH}.zip")
print("   → Open the Kaggle Output tab and click the download icon next to slt_final_t5_model.zip")