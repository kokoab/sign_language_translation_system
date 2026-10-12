#!/usr/bin/env python3
"""Retrain Stage 3 as an evidence-conditioned ASL-to-English renderer.

The deployed checkpoint `stage3_v17_t5_efficient_tiny_locked100_v1` was fine-tuned on a
corpus that almost never reorders a gloss (95 reordering rows out of 11,840 matchable)
and that places only 742 of 15,843 rows inside the locked 100. It learned to insert
function words in the order it received them, which is wrong for every ASL
topic-comment, object-fronted or wh-final utterance, and it renders recognizer noise
faithfully because its contract forbade omission.

This trainer replaces both behaviours. Initialization is the original grammar-correction
base rather than the deployed checkpoint, so the monotone bias is not inherited.

Selection uses validation BLEU only. The test split is opened once, after selection, and
the deployed checkpoint is scored on the same rows as a paired control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import sacrebleu
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from active.v17.stage3_asl_encoding_v17 import encode  # noqa: E402

BASE_MODEL = "visheratin/t5-efficient-tiny-grammar-correction"
BASE_REVISION = "a98f126664317cf2e68d33234c7072fdd6b289f3"
DEPLOYED = ROOT / "artifacts/models/stage3_v17_t5_efficient_tiny_locked100_v1"
CORPUS = ROOT / "data/local/stage3_asl_corpus_v17/corpus.jsonl"
OUTPUT = ROOT / "artifacts/models/stage3_v17_asl_order_v1"
REPORT = ROOT / "artifacts/reports/stage3_v17_asl_order_v1"

MAX_INPUT = 64
MAX_TARGET = 64


@dataclass(frozen=True)
class Row:
    glosses: tuple[str, ...]
    confidences: tuple[float, ...]
    noise_indices: tuple[int, ...]
    structure: str
    english: str
    split: str

    @property
    def source(self) -> str:
        return encode(self.glosses, self.confidences)

    @property
    def plain(self) -> str:
        """Untagged input, for scoring the deployed checkpoint on the same rows."""
        return " ".join(g.lower() for g in self.glosses)

    @property
    def family(self) -> str:
        return self.structure.split("+")[0].split("(")[0]

    @property
    def noisy(self) -> bool:
        return bool(self.noise_indices)


def relative_path(path: Path) -> str:
    """Repo-relative when possible; an out-of-tree pilot corpus stays absolute."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus(path: Path) -> list[Row]:
    rows: list[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(Row(
            glosses=tuple(payload["glosses"]),
            confidences=tuple(payload["confidences"]),
            noise_indices=tuple(payload["noise_indices"]),
            structure=payload["structure"],
            english=payload["english"],
            split=payload["split"],
        ))
    return rows


class RowDataset(Dataset):
    def __init__(self, rows: list[Row]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Row:
        return self.rows[index]


def collator(tokenizer: Any):
    def collate(rows: list[Row]) -> dict[str, torch.Tensor]:
        source = tokenizer(
            [r.source for r in rows], padding="max_length", truncation=True,
            max_length=MAX_INPUT, return_tensors="pt",
        )
        target = tokenizer(
            [r.english for r in rows], padding="max_length", truncation=True,
            max_length=MAX_TARGET, return_tensors="pt",
        )
        labels = target.input_ids.clone()
        # Padding must not contribute to the loss.
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": source.input_ids,
            "attention_mask": source.attention_mask,
            "labels": labels,
        }
    return collate


@torch.inference_mode()
def generate(
    model: Any, tokenizer: Any, texts: list[str], device: torch.device, batch_size: int,
) -> list[str]:
    out: list[str] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        encoded = tokenizer(
            chunk, padding=True, truncation=True, max_length=MAX_INPUT,
            return_tensors="pt",
        ).to(device)
        produced = model.generate(
            **encoded, max_new_tokens=MAX_TARGET, num_beams=1, do_sample=False,
        )
        out.extend(
            " ".join(tokenizer.decode(row, skip_special_tokens=True).split())
            for row in produced.cpu()
        )
    return out


def normalized(text: str) -> str:
    return " ".join(
        "".join(c.lower() for c in text if c.isalnum() or c.isspace()).split()
    )


def score(rows: list[Row], predictions: list[str]) -> dict[str, Any]:
    """BLEU plus the slices that say whether the actual defects were repaired."""
    references = [r.english for r in rows]
    exact = sum(
        1 for r, p in zip(rows, predictions) if normalized(r.english) == normalized(p)
    )
    result: dict[str, Any] = {
        "rows": len(rows),
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf2_plus_plus": sacrebleu.corpus_chrf(
            predictions, [references], word_order=2
        ).score,
        "normalized_exact": exact,
        "normalized_exact_rate": exact / len(rows) if rows else 0.0,
    }
    slices = {
        "reordering": lambda r: r.family in {"osv_topic", "osv_place", "wh_final"},
        "noisy": lambda r: r.noisy,
        "clean": lambda r: not r.noisy,
        "time_fronted": lambda r: r.family == "time_fronted",
        "negation": lambda r: r.family == "negation",
        "svo_regression": lambda r: r.family in {"svo", "motion", "state_predicate"},
    }
    result["slices"] = {}
    for name, keep in slices.items():
        pairs = [(r, p) for r, p in zip(rows, predictions) if keep(r)]
        if not pairs:
            continue
        subset_rows = [r for r, _ in pairs]
        subset_pred = [p for _, p in pairs]
        subset_ref = [r.english for r in subset_rows]
        hits = sum(
            1 for r, p in pairs if normalized(r.english) == normalized(p)
        )
        result["slices"][name] = {
            "rows": len(pairs),
            "bleu": sacrebleu.corpus_bleu(subset_pred, [subset_ref]).score,
            "normalized_exact_rate": hits / len(pairs),
        }
    # Negation is the one error that inverts meaning rather than degrading it, so it
    # is measured on its own: a dropped NO turns "I do not like water" into
    # "I like the water". A high BLEU can hide this.
    negation_rows = [
        (r, p) for r, p in zip(rows, predictions)
        if "NO" in r.glosses
        and r.glosses.index("NO") not in set(r.noise_indices)
    ]
    if negation_rows:
        NEGATIVE_WORDS = {
            "not", "no", "never", "cannot", "nothing",
            "dont", "doesnt", "didnt", "wont", "isnt", "arent",
            "wasnt", "werent", "hasnt", "havent", "couldnt", "wouldnt", "shouldnt",
        }
        preserved = sum(
            1 for _, prediction in negation_rows
            if NEGATIVE_WORDS & set(normalized(prediction).split())
        )
        result["negation_rows"] = len(negation_rows)
        result["negation_preserved_rate"] = preserved / len(negation_rows)

    # On noisy rows the question is simply whether the error word was suppressed.
    noisy = [(r, p) for r, p in zip(rows, predictions) if r.noisy]
    if noisy:
        suppressed = 0
        for row, prediction in noisy:
            words = set(normalized(prediction).split())
            dropped = {row.glosses[i].lower() for i in row.noise_indices}
            kept = {g.lower() for i, g in enumerate(row.glosses)
                    if i not in set(row.noise_indices)}
            # Only count a leak when no retained gloss could explain the word.
            if not any(d in words and d not in kept for d in dropped):
                suppressed += 1
        result["noise_suppression_rate"] = suppressed / len(noisy)
        result["noise_rows"] = len(noisy)
    return result


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--seed", type=int, default=17701)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--skip-control", action="store_true",
        help="skip scoring the deployed checkpoint on the test split",
    )
    parser.add_argument("--init", type=Path, default=None,
                        help="fine-tune from this local checkpoint instead of the base model")
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    rows = load_corpus(args.corpus)
    train = [r for r in rows if r.split == "train"]
    validation = [r for r in rows if r.split == "validation"]
    test = [r for r in rows if r.split == "test"]
    if not train or not validation or not test:
        raise SystemExit("corpus is missing one of the train/validation/test splits")

    # A sequence appearing in two splits would make held-out BLEU meaningless.
    keys = {name: {" ".join(r.glosses) for r in group}
            for name, group in
            (("train", train), ("validation", validation), ("test", test))}
    overlap = (
        len(keys["train"] & keys["validation"])
        + len(keys["train"] & keys["test"])
        + len(keys["validation"] & keys["test"])
    )
    if overlap:
        raise SystemExit(f"{overlap} gloss sequences appear in more than one split")

    print(f"train {len(train)}  validation {len(validation)}  test {len(test)}")
    print(f"device {device}  seed {args.seed}")

    if args.init is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.init, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.init, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = model.to(device)
    optimizer = Adafactor(
        model.parameters(), lr=args.learning_rate or 1e-3, relative_step=False,
        scale_parameter=False, warmup_init=False,
    )
    loader = DataLoader(
        RowDataset(train), batch_size=args.batch_size, shuffle=True,
        collate_fn=collator(tokenizer), num_workers=0,
    )

    args.report.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best = {"epoch": -1, "bleu": -1.0}
    best_state: dict[str, Any] | None = None
    started = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))

        model.eval()
        predictions = generate(
            model, tokenizer, [r.source for r in validation], device,
            args.eval_batch_size,
        )
        measured = score(validation, predictions)
        entry = {
            "epoch": epoch,
            "train_loss": statistics.fmean(losses),
            "validation": measured,
        }
        history.append(entry)
        print(
            f"epoch {epoch}  loss {entry['train_loss']:.4f}  "
            f"val BLEU {measured['bleu']:.2f}  exact {measured['normalized_exact_rate']:.3f}  "
            f"noise-suppressed {measured.get('noise_suppression_rate', 0):.3f}  "
            f"negation-kept {measured.get('negation_preserved_rate', 0):.3f}",
            flush=True,
        )
        if measured["bleu"] > best["bleu"]:
            best = {"epoch": epoch, "bleu": measured["bleu"]}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise SystemExit("no epoch was selected")
    model.load_state_dict(best_state)
    model.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    # Declare the input format beside the weights so a runtime that loads this
    # checkpoint cannot feed it the plain gloss text the legacy model expected.
    (args.output / "stage3_input_contract.json").write_text(json.dumps({
        "format": "slt_stage3_input_contract_v17",
        "version": 1,
        "encoding": "evidence",
        "description": (
            "Each gloss is preceded by a confidence bucket token (hi/mid/lo). A "
            "checkpoint without this file expects plain lowercased gloss text."
        ),
        "buckets": {"hi": ">=0.60", "mid": "0.40-0.60", "lo": "<0.40"},
        "missing_confidence": "hi",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"selected epoch {best['epoch']} at validation BLEU {best['bleu']:.2f}")

    # The test split is opened exactly once, after selection.
    test_predictions = generate(
        model, tokenizer, [r.source for r in test], device, args.eval_batch_size
    )
    test_scores = score(test, test_predictions)
    print(f"test BLEU {test_scores['bleu']:.2f}")

    control: dict[str, Any] | None = None
    control_predictions: list[str] = []
    if not args.skip_control and DEPLOYED.exists():
        print("scoring the deployed checkpoint on the same test rows")
        control_tokenizer = AutoTokenizer.from_pretrained(DEPLOYED, local_files_only=True)
        control_model = AutoModelForSeq2SeqLM.from_pretrained(
            DEPLOYED, local_files_only=True
        ).to(device).eval()
        # The deployed model never saw evidence tokens, so it gets the plain input
        # it was trained on. This is the fairest available paired comparison.
        control_predictions = generate(
            control_model, control_tokenizer, [r.plain for r in test], device,
            args.eval_batch_size,
        )
        control = score(test, control_predictions)
        print(f"deployed checkpoint test BLEU {control['bleu']:.2f}")

    with (args.report / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, (row, prediction) in enumerate(zip(test, test_predictions)):
            handle.write(json.dumps({
                "glosses": list(row.glosses),
                "confidences": list(row.confidences),
                "noise_indices": list(row.noise_indices),
                "structure": row.structure,
                "reference": row.english,
                "prediction": prediction,
                "deployed_prediction": (
                    control_predictions[index] if control_predictions else None
                ),
            }) + "\n")

    result = {
        "format": "slt_stage3_asl_order_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "initialization": "fresh_from_base",
        "corpus": relative_path(args.corpus),
        "corpus_sha256": sha256_file(args.corpus),
        "input_encoding": "per-gloss confidence bucket token (hi/mid/lo) before each gloss",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "split_overlap": overlap,
        "selected_epoch": best["epoch"],
        "selected_validation_bleu": best["bleu"],
        "history": history,
        "test": test_scores,
        "deployed_control": control,
        "training_seconds": time.perf_counter() - started,
        "selection_used_test": False,
        "claim_scope": (
            "rule-generated ASL word order with model-written English references; "
            "held-out BLEU measures agreement with that generated English, "
            "not human-judged translation quality on real signing"
        ),
    }
    (args.report / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.report / 'result.json'}")


if __name__ == "__main__":
    main()
