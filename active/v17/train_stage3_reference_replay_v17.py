#!/usr/bin/env python3
"""Warm-start Stage 3 on genuine gloss/English pairs with bounded replay.

Train data:
* all current 2M-Flores ``dev`` rows except the 155 previously selected rows;
* public NCSLGR gloss/English annotations;
* a deterministic, equal-mass replay sample from the legacy synthetic CSV.

Validation is exactly the 155 previously selected 2M-Flores ``dev`` rows.  The
reserved 2M-Flores ``devtest`` split and all project test splits remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WARM_START = ROOT / "weights/slt_final_t5_model"
DEFAULT_ALL_FLORES = ROOT / "data/local/dataset_metadata/2m_flores_asl/dev_all_metadata_v17.json"
DEFAULT_SELECTED_FLORES = ROOT / "data/local/dataset_metadata/2m_flores_asl/dev_selected_v17.json"
DEFAULT_NCSLGR = ROOT / "data/local/ncslgr_continuous_v17_source/manifest.json"
DEFAULT_SYNTHETIC = ROOT / "artifacts/reports/slt_stage3_dataset_final.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/models/stage3_v17_reference_replay_v1"
PROMPT = "Translate this ASL gloss to natural conversational English: {gloss}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(row: dict[str, str], seed: int) -> str:
    value = f'{seed}\0{row["gloss"]}\0{row["text"]}'.encode()
    return hashlib.sha256(value).hexdigest()


def load_data(
    all_flores_path: Path,
    selected_flores_path: Path,
    ncslgr_path: Path,
    synthetic_path: Path,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    all_flores = json.loads(all_flores_path.read_text())
    selected_flores = json.loads(selected_flores_path.read_text())
    ncslgr = json.loads(ncslgr_path.read_text())
    if all_flores.get("source_split") != "dev" or all_flores.get("reserved_devtest_accessed") is not False:
        raise ValueError("full Flores metadata violates the dev-only contract")
    if selected_flores.get("source_split") != "dev" or selected_flores.get("reserved_devtest_accessed") is not False:
        raise ValueError("selected Flores metadata violates the dev-only contract")

    def current_key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row["id"]), str(row["signer"])

    def selected_key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row["id"]), str(row["signer_local_id"])

    selected_by_key = {selected_key(row): row for row in selected_flores["rows"]}
    all_by_key = {current_key(row): row for row in all_flores["rows"]}
    if len(all_by_key) != len(all_flores["rows"]):
        raise ValueError("duplicate Flores (id, signer) row keys")
    missing = sorted(set(selected_by_key) - set(all_by_key))
    if missing:
        raise ValueError(f"selected Flores rows missing from current metadata: {missing[:5]}")
    for row_key, selected in selected_by_key.items():
        current = all_by_key[row_key]
        if selected["gloss"] != current["gloss"] or selected["sentence"] != current["sentence"]:
            raise ValueError(f"selected Flores row {row_key} changed")

    genuine_train = [
        {
            "source": "2m_flores_dev_train",
            "id": f'{row["id"]}:{row["signer"]}',
            "gloss": str(row["gloss"]).strip(),
            "text": str(row["sentence"]).strip(),
        }
        for row_key, row in sorted(all_by_key.items())
        if row_key not in selected_by_key
    ]
    for item in ncslgr["items"]:
        gloss = " ".join(str(token) for token in item["main_glosses"]).strip()
        text = str(item["english_translation"]).strip()
        if gloss and text:
            genuine_train.append(
                {
                    "source": "ncslgr_train",
                    "id": f'{item["collection"]}:{item["source_id"]}',
                    "gloss": gloss,
                    "text": text,
                }
            )

    synthetic_rows = []
    with synthetic_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            gloss = str(row.get("gloss", "")).strip()
            text = str(row.get("text", "")).strip()
            if gloss and text:
                synthetic_rows.append({"source": "legacy_synthetic_replay", "id": "", "gloss": gloss, "text": text})
    synthetic_rows.sort(key=lambda row: stable_key(row, seed))
    replay = synthetic_rows[: len(genuine_train)]
    train_rows = genuine_train + replay

    validation_rows = [
        {
            "source": "2m_flores_dev_validation",
            "id": f"{row_key[0]}:{row_key[1]}",
            "gloss": str(all_by_key[row_key]["gloss"]).strip(),
            "text": str(all_by_key[row_key]["sentence"]).strip(),
        }
        for row_key in sorted(selected_by_key)
    ]
    train_ids = {row["id"] for row in genuine_train if row["source"] == "2m_flores_dev_train"}
    validation_ids = {row["id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("Flores train/validation ID leakage")
    plan = {
        "genuine_train_rows": len(genuine_train),
        "flores_train_rows": sum(row["source"] == "2m_flores_dev_train" for row in genuine_train),
        "ncslgr_train_rows": sum(row["source"] == "ncslgr_train" for row in genuine_train),
        "synthetic_replay_rows": len(replay),
        "validation_rows": len(validation_rows),
        "flores_train_validation_overlap": 0,
    }
    return train_rows, validation_rows, plan


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def make_collator(tokenizer: Any, max_input: int, max_target: int):
    def collate(rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompts = [PROMPT.format(gloss=row["gloss"]) for row in rows]
        encoded_inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_input,
            return_tensors="pt",
        )
        encoded_targets = tokenizer(
            [row["text"] for row in rows],
            padding="max_length",
            truncation=True,
            max_length=max_target,
            return_tensors="pt",
        )
        labels = encoded_targets["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": encoded_inputs["input_ids"],
            "attention_mask": encoded_inputs["attention_mask"],
            "labels": labels,
        }

    return collate


def generate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    device: torch.device,
    batch_size: int,
    num_beams: int,
    max_input: int,
    max_target: int,
) -> list[str]:
    model.eval()
    predictions: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            prompts = [PROMPT.format(gloss=row["gloss"]) for row in rows[start : start + batch_size]]
            encoded = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=max_input,
                return_tensors="pt",
            )
            ids = model.generate(
                input_ids=encoded["input_ids"].to(device),
                attention_mask=encoded["attention_mask"].to(device),
                max_new_tokens=max_target,
                num_beams=num_beams,
                do_sample=False,
            )
            predictions.extend(tokenizer.batch_decode(ids.cpu(), skip_special_tokens=True))
    return [value.strip() for value in predictions]


def translation_metrics(predictions: list[str], rows: list[dict[str, str]]) -> dict[str, float]:
    references = [row["text"] for row in rows]
    return {
        "sacrebleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf2_plus_plus": sacrebleu.corpus_chrf(predictions, [references], word_order=2).score,
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_START)
    parser.add_argument("--all-flores", type=Path, default=DEFAULT_ALL_FLORES)
    parser.add_argument("--selected-flores", type=Path, default=DEFAULT_SELECTED_FLORES)
    parser.add_argument("--ncslgr", type=Path, default=DEFAULT_NCSLGR)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-input-length", type=int, default=192)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17031)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows, validation_rows, plan = load_data(
        args.all_flores, args.selected_flores, args.ncslgr, args.synthetic, args.seed
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(args.warm_start / "tokenizer.json"),
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(str(args.warm_start), local_files_only=True)
    device = resolve_device(args.device)
    loader = DataLoader(
        PairDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        collate_fn=make_collator(tokenizer, args.max_input_length, args.max_target_length),
    )
    total_steps = args.epochs * len(loader)
    warmup = max(1, int(0.05 * total_steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []
    best_chrf = float("-inf")
    best_epoch = -1
    global_step = 0
    nonfinite_batches = 0

    baseline_predictions = generate(
        model, tokenizer, validation_rows, torch.device("cpu"), args.eval_batch_size,
        args.num_beams, args.max_input_length, args.max_target_length,
    )
    baseline = translation_metrics(baseline_predictions, validation_rows)
    print(f"baseline {baseline}", flush=True)
    model.to(device)
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                if nonfinite_batches > 3:
                    raise RuntimeError("too many nonfinite training batches")
                continue
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                nonfinite_batches += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_batches > 3:
                    raise RuntimeError("too many nonfinite gradient batches")
                continue
            optimizer.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach().cpu()))
            if batch_index % 25 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} batch {batch_index}/{len(loader)} "
                    f"loss {losses[-1]:.4f}",
                    flush=True,
                )
                if device.type == "mps":
                    torch.mps.empty_cache()

        evaluation_device = device
        if device.type == "mps":
            model.to("cpu")
            torch.mps.empty_cache()
            evaluation_device = torch.device("cpu")
        predictions = generate(
            model, tokenizer, validation_rows, evaluation_device, args.eval_batch_size,
            args.num_beams, args.max_input_length, args.max_target_length,
        )
        metrics = translation_metrics(predictions, validation_rows)
        entry = {
            "epoch": epoch + 1,
            "mean_train_loss": sum(losses) / len(losses),
            **metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["chrf2_plus_plus"] > best_chrf:
            best_chrf = metrics["chrf2_plus_plus"]
            best_epoch = epoch + 1
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            with (args.output_dir / "validation_predictions.jsonl").open("w") as handle:
                for row, prediction in zip(validation_rows, predictions):
                    handle.write(json.dumps({**row, "prediction": prediction}, ensure_ascii=False) + "\n")
        if device.type == "mps" and epoch + 1 < args.epochs:
            model.to(device)
            torch.mps.empty_cache()

    elapsed = time.perf_counter() - started
    result = {
        "format": "stage3_v17_reference_replay_training",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "warm_start_model_sha256": sha256_file(args.warm_start / "model.safetensors"),
        "warm_start_tokenizer_sha256": sha256_file(args.warm_start / "tokenizer.json"),
        "selected_model_sha256": sha256_file(args.output_dir / "model.safetensors"),
        "selected_tokenizer_sha256": sha256_file(args.output_dir / "tokenizer.json"),
        "device": str(device),
        "configuration": vars(args) | {"output_dir": str(args.output_dir), "warm_start": str(args.warm_start), "all_flores": str(args.all_flores), "selected_flores": str(args.selected_flores), "ncslgr": str(args.ncslgr), "synthetic": str(args.synthetic)},
        "data_plan": plan,
        "source_sha256": {
            "all_flores": sha256_file(args.all_flores),
            "selected_flores": sha256_file(args.selected_flores),
            "ncslgr": sha256_file(args.ncslgr),
            "synthetic": sha256_file(args.synthetic),
        },
        "baseline_validation": baseline,
        "history": history,
        "selected_epoch": best_epoch,
        "selected_chrf2_plus_plus": best_chrf,
        "elapsed_seconds": elapsed,
        "nonfinite_batches_discarded": nonfinite_batches,
        "reserved_devtest_accessed": False,
        "test_split_accessed": False,
        "claim_scope": "reference-gloss-to-English validation only; not end-to-end translation",
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"selected epoch {best_epoch}, chrF++ {best_chrf:.4f}")
    print(f"result: {args.output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
