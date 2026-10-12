#!/usr/bin/env python3
"""Fine-tune the 15.6M T5-efficient-tiny model for locked-100 gloss rendering.

The legacy CSV supplies broad synthetic grammar. Reviewed templates and deterministic
locked-vocabulary compositions receive extra weight. Selection uses only validation;
the fixed test split is opened once after the best checkpoint is selected.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any

import sacrebleu
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "visheratin/t5-efficient-tiny-grammar-correction"
BASE_REVISION = "a98f126664317cf2e68d33234c7072fdd6b289f3"
LEGACY_CSV = ROOT / "artifacts/reports/slt_stage3_dataset_final.csv"
DIALOGUE_CSV = ROOT / "artifacts/reports/slt_dialogue_dataset.csv"
CONTRACT = ROOT / "active/v17/stage2_to_stage3_contract_v17.json"
NATURALIZER = ROOT / "active/v17/stage3_mobile_naturalizer_manifest_v17.json"
OUTPUT = ROOT / "artifacts/models/stage3_v17_t5_efficient_tiny_locked100_v1"
REPORT = ROOT / "artifacts/reports/stage3_v17_t5_efficient_tiny_locked100_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def stable_bucket(value: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()[:8], 16) % 10


@dataclass(frozen=True)
class Pair:
    gloss: str
    text: str
    source: str
    split: str
    family: str

    @property
    def length(self) -> int:
        return len(self.gloss.split())


def place_english(place: str) -> str:
    return {"HOME": "home", "SCHOOL": "to school", "HOSPITAL": "to the hospital"}[place]


def subject_english(subject: str) -> str:
    return {"I": "I", "YOU": "you", "WE": "we", "THEY": "they", "HE": "he"}[subject]


def past_action(action: str) -> str:
    return {"READ": "read", "LEARN": "learned", "FIND": "found", "SEE": "saw"}[action]


def controlled_long_rows(seed: int) -> list[Pair]:
    """Produce auditable 5-12 gloss compositions using only the frozen vocabulary."""
    raw: list[tuple[str, str, str]] = []
    actions = (
        ("READ", "SIGN", "read the sign"),
        ("READ", "NAME", "read the name"),
        ("LEARN", "LANGUAGE", "learn the language"),
        ("LEARN", "SIGN", "learn the sign"),
        ("FIND", "DOCTOR", "find the doctor"),
        ("FIND", "FRIEND", "find my friend"),
        ("SEE", "DOCTOR", "see the doctor"),
        ("SEE", "FRIEND", "see my friend"),
    )
    for subject in ("I", "YOU", "WE"):
        subject_text = subject_english(subject)
        for place in ("HOME", "SCHOOL", "HOSPITAL"):
            destination = place_english(place)
            for action, obj, phrase in actions:
                gloss = f"TOMORROW MORNING {subject} WANT GO {place} {action} {obj}"
                text = f"Tomorrow morning, {subject_text} want to go {destination} and {phrase}."
                raw.append((gloss, text, "future_plan"))

    past_actions = (
        ("READ", "SIGN"), ("READ", "NAME"), ("LEARN", "LANGUAGE"),
        ("FIND", "TIME"), ("FIND", "DOCTOR"), ("SEE", "DOCTOR"),
    )
    for subject in ("I", "YOU", "WE"):
        subject_text = subject_english(subject)
        for place in ("SCHOOL", "HOSPITAL"):
            for action, obj in past_actions:
                gloss = f"YESTERDAY {subject} GO {place} {action} {obj} COME HOME"
                text = (
                    f"Yesterday, {subject_text} went {place_english(place)}, "
                    f"{past_action(action)} the {obj.lower()}, and came home."
                )
                raw.append((gloss, text, "past_trip"))

    needs = (
        ("SICK", "DOCTOR", "a doctor"),
        ("SICK", "HELP", "help"),
        ("HUNGRY", "WATER", "water"),
        ("TIRED", "SLEEP", "sleep"),
        ("COLD", "HOME", "to go home"),
        ("HOT", "WATER", "water"),
    )
    for subject in ("I", "YOU", "WE"):
        subject_text = subject_english(subject)
        verb = "need" if subject != "HE" else "needs"
        for feeling, need, need_text in needs:
            gloss = f"NOW {subject} FEEL {feeling} {subject} NEED {need}"
            raw.append((
                gloss,
                f"{subject_text.capitalize()} feel {feeling.lower()} now and {verb} {need_text}.",
                "feeling_need",
            ))

    for thing, phrase in (
        ("WATER", "water"), ("HELP", "help"), ("DOCTOR", "a doctor"),
        ("HOME", "to go home"), ("SLEEP", "sleep"),
    ):
        raw.append((
            f"PLEASE HELP I NEED {thing} NOW",
            f"Please help me; I need {phrase} now.",
            "urgent_request",
        ))

    for relative in ("MOTHER", "FATHER", "CHILD", "FRIEND"):
        owner = "my" if relative != "CHILD" else "my"
        raw.append((
            f"MY {relative} FEEL SICK NEED DOCTOR NOW",
            f"{owner.capitalize()} {relative.lower()} feels sick and needs a doctor now.",
            "family_health",
        ))

    # Longer endpoint buffers: explicit time markers make clause boundaries auditable.
    for place, action, obj, action_text in (
        ("SCHOOL", "READ", "SIGN", "read the sign"),
        ("SCHOOL", "LEARN", "LANGUAGE", "learn the language"),
        ("HOSPITAL", "SEE", "DOCTOR", "see the doctor"),
        ("HOSPITAL", "FIND", "DOCTOR", "find the doctor"),
    ):
        raw.append((
            f"HELLO TOMORROW MORNING I WANT GO {place} {action} {obj}",
            f"Hello. Tomorrow morning, I want to go {place_english(place)} and {action_text}.",
            "greeting_plan",
        ))
        raw.append((
            f"YESTERDAY I GO {place} {action} {obj} COME HOME NOW I FEEL TIRED",
            f"Yesterday, I went {place_english(place)}, {past_action(action)} the {obj.lower()}, "
            "and came home. Now I feel tired.",
            "past_trip_followup",
        ))

    output = []
    for gloss, text, family in raw:
        bucket = stable_bucket(gloss, seed)
        split = "validation" if bucket < 2 else "test" if bucket < 4 else "train"
        output.append(Pair(gloss, text, "controlled_locked100", split, family))
    return output


def load_pairs(seed: int) -> tuple[list[Pair], dict[str, Any]]:
    contract = json.loads(CONTRACT.read_text())
    vocabulary = set(contract["vocabulary"]["labels"])
    manifest = json.loads(NATURALIZER.read_text())
    reviewed = {
        " ".join(row["glosses"]): str(row["english"])
        for row in manifest["reviewed_templates"]
    }
    with LEGACY_CSV.open(newline="", encoding="utf-8-sig") as handle:
        legacy_raw = list(csv.DictReader(handle))
    legacy_locked_long = {
        " ".join(row["gloss"].split())
        for row in legacy_raw
        if len(row["gloss"].split()) >= 5
        and all(token in vocabulary for token in row["gloss"].split())
    }
    controlled = [
        Pair(row.gloss, row.text, row.source,
             "test" if row.gloss in legacy_locked_long else row.split, row.family)
        for row in controlled_long_rows(seed)
    ]
    controlled_by_gloss = {row.gloss: row for row in controlled}
    if len(controlled_by_gloss) != len(controlled):
        raise ValueError("duplicate controlled long gloss sequence")
    if any(row.length < 5 for row in controlled):
        raise ValueError("controlled long set contains a sequence below five glosses")
    if any(token not in vocabulary for row in controlled for token in row.gloss.split()):
        raise ValueError("controlled long set escaped the locked vocabulary")

    legacy: list[Pair] = []
    for raw in legacy_raw:
        gloss, text = " ".join(raw["gloss"].split()), raw["text"].strip()
        if not gloss or not text or gloss in reviewed or gloss in controlled_by_gloss:
            continue
        tokens = gloss.split()
        locked = all(token in vocabulary for token in tokens)
        if locked and len(tokens) >= 5:
            split, family = "test", "legacy_locked100_long"
        else:
            bucket = stable_bucket(gloss, seed)
            split = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
            family = "legacy_synthetic"
        legacy.append(Pair(gloss, text, "legacy_stage3_csv", split, family))

    reviewed_rows = [
        Pair(gloss, text, "reviewed_template", "train", "reviewed_template")
        for gloss, text in sorted(reviewed.items())
    ]
    rows = legacy + controlled + reviewed_rows
    by_gloss: dict[str, set[str]] = {}
    for row in rows:
        by_gloss.setdefault(row.gloss, set()).add(row.split)
    leakage = sorted(gloss for gloss, splits in by_gloss.items() if len(splits) != 1)
    if leakage:
        raise ValueError(f"gloss split leakage: {leakage[:5]}")

    with DIALOGUE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        dialogue = list(csv.DictReader(handle))
    dialogue_pairs = {(" ".join(row["gloss"].split()), row["text"].strip()) for row in dialogue}
    plan = {
        "rows": len(rows),
        "unique_glosses": len(by_gloss),
        "legacy_csv_rows_loaded": len(legacy),
        "reviewed_template_rows": len(reviewed_rows),
        "controlled_long_rows": len(controlled),
        "controlled_long_by_split": {
            split: sum(row.split == split for row in controlled)
            for split in ("train", "validation", "test")
        },
        "locked100_legacy_long_test_rows": sum(
            row.family == "legacy_locked100_long" for row in rows
        ),
        "dialogue_csv_rows_audited": len(dialogue),
        "dialogue_csv_unique_pairs": len(dialogue_pairs),
        "dialogue_csv_used": False,
        "dialogue_csv_exclusion_reason": "duplicate-heavy and contains no 5+ gloss rows",
        "split_gloss_overlap": 0,
    }
    return rows, plan


class PairDataset(Dataset):
    def __init__(self, rows: list[Pair]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Pair:
        return self.rows[index]


def training_rows(rows: list[Pair]) -> list[Pair]:
    output: list[Pair] = []
    for row in rows:
        if row.split != "train":
            continue
        repeats = 16 if row.source in ("reviewed_template", "controlled_locked100") else 1
        output.extend([row] * repeats)
    return output


def collator(tokenizer: Any, max_input: int, max_target: int):
    def collate(rows: list[Pair]) -> dict[str, torch.Tensor]:
        inputs = tokenizer(
            [row.gloss.lower() for row in rows], padding="max_length", truncation=True,
            max_length=max_input, return_tensors="pt",
        )
        targets = tokenizer(
            [row.text for row in rows], padding="max_length", truncation=True,
            max_length=max_target, return_tensors="pt",
        )["input_ids"]
        targets[targets == tokenizer.pad_token_id] = -100
        return {**inputs, "labels": targets}
    return collate


def generate(
    model: Any, tokenizer: Any, rows: list[Pair], device: torch.device,
    batch_size: int, max_input: int, max_target: int,
) -> tuple[list[str], list[float]]:
    predictions: list[str] = []
    latencies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [row.gloss.lower() for row in batch], padding=True, truncation=True,
                max_length=max_input, return_tensors="pt",
            ).to(device)
            began = time.perf_counter()
            output = model.generate(
                **encoded, max_new_tokens=max_target, num_beams=1, do_sample=False,
            )
            elapsed = 1000 * (time.perf_counter() - began)
            latencies.extend([elapsed / len(batch)] * len(batch))
            predictions.extend(tokenizer.batch_decode(output.cpu(), skip_special_tokens=True))
    return [value.strip() for value in predictions], latencies


def metrics(rows: list[Pair], predictions: list[str]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    references = [row.text for row in rows]
    exact = [normalized(a) == normalized(b) for a, b in zip(predictions, references)]
    return {
        "rows": len(rows),
        "normalized_exact": sum(exact),
        "normalized_exact_rate": sum(exact) / len(exact),
        "chrf2_plus_plus": sacrebleu.corpus_chrf(
            predictions, [references], word_order=2
        ).score,
    }


def evaluate(
    model: Any, tokenizer: Any, rows: list[Pair], device: torch.device,
    batch_size: int, max_input: int, max_target: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions, latencies = generate(
        model, tokenizer, rows, device, batch_size, max_input, max_target
    )
    payload: dict[str, Any] = {"overall": metrics(rows, predictions)}
    locked_vocabulary = set(json.loads(CONTRACT.read_text())["vocabulary"]["labels"])
    for name, selected in (
        ("locked100", [
            i for i, row in enumerate(rows)
            if all(token in locked_vocabulary for token in row.gloss.split())
        ]),
        ("five_or_more", [i for i, row in enumerate(rows) if row.length >= 5]),
        ("controlled_five_or_more", [i for i, row in enumerate(rows) if row.source == "controlled_locked100"]),
        ("nine_or_more", [i for i, row in enumerate(rows) if row.length >= 9]),
    ):
        payload[name] = metrics(
            [rows[i] for i in selected], [predictions[i] for i in selected]
        )
    payload["latency_ms_per_sentence"] = {
        "median": statistics.median(latencies),
        "p90": sorted(latencies)[int(0.9 * (len(latencies) - 1))],
    }
    details = [
        {
            "gloss": row.gloss, "reference": row.text, "prediction": prediction,
            "source": row.source, "family": row.family, "split": row.split,
            "gloss_count": row.length, "normalized_exact": normalized(row.text) == normalized(prediction),
        }
        for row, prediction in zip(rows, predictions)
    ]
    return payload, details


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=REPORT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-input-length", type=int, default=48)
    parser.add_argument("--max-target-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=17032)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, data_plan = load_pairs(args.seed)
    train = training_rows(rows)
    validation = [row for row in rows if row.split == "validation"]
    test = [row for row in rows if row.split == "test"]
    if not any(row.source == "controlled_locked100" for row in validation):
        raise ValueError("validation lacks controlled long rows")
    if not any(row.source == "controlled_locked100" for row in test):
        raise ValueError("test lacks controlled long rows")

    snapshot = Path(snapshot_download(
        BASE_MODEL, revision=BASE_REVISION, local_files_only=args.local_files_only
    ))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(snapshot, local_files_only=True)
    device = resolve_device(args.device)
    loader = DataLoader(
        PairDataset(train), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
        collate_fn=collator(tokenizer, args.max_input_length, args.max_target_length),
    )
    optimizer = Adafactor(
        model.parameters(), lr=args.learning_rate, scale_parameter=False,
        relative_step=False, warmup_init=False, weight_decay=0.01,
    )
    total_steps = args.epochs * len(loader)
    warmup = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup else 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))
        ),
    )
    args.output_dir.mkdir(parents=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    with (args.report_dir / "data_manifest.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

    baseline, _ = evaluate(
        model, tokenizer, validation, torch.device("cpu"), args.eval_batch_size,
        args.max_input_length, args.max_target_length,
    )
    print("baseline", json.dumps(baseline), flush=True)
    model.to(device)
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = -1
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if index % 100 == 0:
                print(f"epoch {epoch} batch {index}/{len(loader)} loss {losses[-1]:.4f}", flush=True)

        if device.type == "mps":
            model.to("cpu")
            torch.mps.empty_cache()
        validation_metrics, validation_details = evaluate(
            model, tokenizer, validation, torch.device("cpu"), args.eval_batch_size,
            args.max_input_length, args.max_target_length,
        )
        long_metrics = validation_metrics["controlled_five_or_more"]
        score = 100 * float(long_metrics["normalized_exact_rate"]) + float(
            validation_metrics["overall"]["chrf2_plus_plus"]
        )
        entry = {
            "epoch": epoch,
            "mean_train_loss": sum(losses) / len(losses),
            "selection_score": score,
            "validation": validation_metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)
            with (args.report_dir / "validation_predictions.jsonl").open("w") as handle:
                for row in validation_details:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if device.type == "mps" and epoch < args.epochs:
            model.to(device)

    # The fixed test split is first opened after validation-only checkpoint selection.
    selected = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir, local_files_only=True)
    test_metrics, test_details = evaluate(
        selected, tokenizer, test, torch.device("cpu"), args.eval_batch_size,
        args.max_input_length, args.max_target_length,
    )
    with (args.report_dir / "test_predictions.jsonl").open("w") as handle:
        for row in test_details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "format": "slt_stage3_t5_efficient_tiny_locked100_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "base_weight_sha256": sha256_file(snapshot / "pytorch_model.bin"),
        "selected_model_sha256": sha256_file(args.output_dir / "model.safetensors"),
        "selected_tokenizer_sha256": sha256_file(args.output_dir / "tokenizer.json"),
        "source_sha256": {
            "legacy_csv": sha256_file(LEGACY_CSV),
            "dialogue_csv": sha256_file(DIALOGUE_CSV),
            "contract": sha256_file(CONTRACT),
            "naturalizer": sha256_file(NATURALIZER),
        },
        "configuration": {**vars(args), "output_dir": str(args.output_dir), "report_dir": str(args.report_dir)},
        "data_plan": {**data_plan, "weighted_train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
        "baseline_validation": baseline,
        "history": history,
        "selected_epoch": best_epoch,
        "selected_score": best_score,
        "test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_used_test": False,
        "test_split_accessed_once_after_selection": True,
        "citizen_test_accessed": False,
        "two_m_flores_devtest_accessed": False,
        "claim_scope": "synthetic/reviewed locked-vocabulary naturalization; not general ASL translation",
    }
    (args.report_dir / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({"selected_epoch": best_epoch, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
