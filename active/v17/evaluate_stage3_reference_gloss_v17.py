#!/usr/bin/env python3
"""Evaluate the legacy Stage-3 translator on genuine reference gloss/English pairs.

This is a Stage-3-only evaluation: the model receives human reference gloss, not
Stage-2 predictions.  It deliberately uses only the already-acquired 2M-Flores
``dev`` split and the public NCSLGR annotations.  Reserved ``devtest`` and test
splits are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "weights/slt_final_t5_model"
DEFAULT_NCSLGR = ROOT / "data/local/ncslgr_continuous_v17_source/manifest.json"
DEFAULT_FLORES = ROOT / "data/local/dataset_metadata/2m_flores_asl/dev_selected_v17.json"
DEFAULT_OUTPUT = ROOT / "artifacts/reports/stage3_v17_reference_gloss_baseline"
PROMPT = "Translate this ASL gloss to natural conversational English: {gloss}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(ncslgr_manifest: Path, flores_manifest: Path) -> list[dict[str, str]]:
    ncslgr = json.loads(ncslgr_manifest.read_text())
    flores = json.loads(flores_manifest.read_text())
    if flores.get("source_split") != "dev" or not flores.get("reserved_devtest_accessed") is False:
        raise ValueError("2M-Flores contract violation: expected dev with reserved devtest untouched")

    rows: list[dict[str, str]] = []
    for item in ncslgr["items"]:
        gloss = " ".join(str(token) for token in item["main_glosses"]).strip()
        text = str(item["english_translation"]).strip()
        if gloss and text:
            rows.append(
                {
                    "source": "ncslgr",
                    "id": f'{item["collection"]}:{item["source_id"]}',
                    "gloss": gloss,
                    "reference": text,
                }
            )
    for item in flores["rows"]:
        if item.get("split") != "dev":
            raise ValueError("2M-Flores manifest unexpectedly contains a non-dev row")
        gloss = str(item["gloss"]).strip()
        text = str(item["sentence"]).strip()
        if gloss and text:
            rows.append(
                {
                    "source": "2m_flores_dev",
                    "id": str(item["id"]),
                    "gloss": gloss,
                    "reference": text,
                }
            )
    return rows


def load_legacy_model(model_dir: Path) -> tuple[Any, Any]:
    """Load the old fast-tokenizer asset without mutating its legacy config."""
    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), local_files_only=True)
    return tokenizer, model


def generate_predictions(
    rows: list[dict[str, str]],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    num_beams: int,
    max_input_length: int,
    max_new_tokens: int,
) -> list[str]:
    predictions: list[str] = []
    model.eval()
    torch.set_grad_enabled(False)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [PROMPT.format(gloss=row["gloss"]) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        output_ids = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        predictions.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
        print(f"evaluated {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return [prediction.strip() for prediction in predictions]


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def score_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [
        rouge.score(reference, prediction)["rougeL"].fmeasure
        for prediction, reference in zip(predictions, references)
    ]
    exact = sum(normalized_text(p) == normalized_text(r) for p, r in zip(predictions, references))
    return {
        "rows": len(rows),
        "sacrebleu": bleu.score,
        "chrf2_plus_plus": chrf.score,
        "mean_rouge_l_f1": sum(rouge_l) / len(rouge_l),
        "normalized_exact_match": exact,
        "normalized_exact_match_rate": exact / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--ncslgr-manifest", type=Path, default=DEFAULT_NCSLGR)
    parser.add_argument("--flores-manifest", type=Path, default=DEFAULT_FLORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ncslgr-role", default="evaluation")
    parser.add_argument("--flores-role", default="evaluation")
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_beams < 1:
        raise ValueError("batch size and beam count must be positive")
    rows = load_rows(args.ncslgr_manifest, args.flores_manifest)
    tokenizer, model = load_legacy_model(args.model)
    started = time.perf_counter()
    predictions = generate_predictions(
        rows,
        tokenizer,
        model,
        args.batch_size,
        args.num_beams,
        args.max_input_length,
        args.max_new_tokens,
    )
    elapsed = time.perf_counter() - started
    for row, prediction in zip(rows, predictions):
        row["prediction"] = prediction

    by_source = {
        source: score_rows([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    report = {
        "format": "stage3_v17_reference_gloss_baseline",
        "version": 1,
        "scope": "Stage-3-only reference-gloss-to-English evaluation",
        "model_dir": str(args.model),
        "model_safetensors_sha256": sha256_file(args.model / "model.safetensors"),
        "tokenizer_json_sha256": sha256_file(args.model / "tokenizer.json"),
        "prompt": PROMPT,
        "generation": {
            "num_beams": args.num_beams,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
        },
        "elapsed_seconds": elapsed,
        "overall": score_rows(rows),
        "by_source": by_source,
        "source_manifests": {
            "ncslgr": {
                "path": str(args.ncslgr_manifest),
                "sha256": sha256_file(args.ncslgr_manifest),
                "role": args.ncslgr_role,
            },
            "2m_flores_dev": {
                "path": str(args.flores_manifest),
                "sha256": sha256_file(args.flores_manifest),
                "reserved_devtest_accessed": False,
                "role": args.flores_role,
            },
        },
        "test_split_accessed": False,
        "limitations": [
            "Inputs are human reference glosses, not Stage-2 predictions.",
            "Both sources have a single English reference, so valid paraphrases are under-rewarded.",
            "The checkpoint has no trustworthy retained training history; its legacy history file is simulated.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report["by_source"], indent=2))
    print(f"report: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
