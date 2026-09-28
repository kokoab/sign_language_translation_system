#!/usr/bin/env python3
"""Train a multi-sentence Stage 3 renderer under a wall-clock budget (user decisions 2026-09-29).

Bake-off order, smallest first: the deployed tiny T5 (continued from composition v2), then
flan-t5-small, then flan-t5-base, each only if the smaller one misses the bar on the held-out
multi-sentence set. The user capped every training run at one hour.

Input: the existing evidence encoding (confidence bucket before each gloss). Target: "N: sentence"
per sentence, N = input glosses that sentence consumed, so the app can lock finished sentences
during incremental translation. Corpus: scripts/build_stage3_multisentence_corpus_v17.py.

Budget: speed is measured over the first steps and re-measured every 250 steps; the total step count
is set (and only ever shrunk) so training ends inside
the budget, the learning rate warms up then decays linearly to 10% over those steps, and a hard
wall-clock stop saves whatever exists. Three fixed length buckets (24/40/64) keep the MPS graph cache
bounded. The final step is kept; validation loss is reported, never used for selection. The
300-session held-out evaluation set never enters training (asserted).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data/local/stage3_multisentence_corpus_v17/corpus.jsonl"
EVAL_SET = ROOT / "artifacts/reports/stage3_multisentence_eval_v17_20260929/eval_set.jsonl"
REPORT = ROOT / "artifacts/reports/stage3_multisentence_bakeoff_v17_20260929"
MODELS = {
    "tiny": str(ROOT / "artifacts/models/stage3_composition_v17_20260929_v2"),
    "small": "google/flan-t5-small",
    "base": "google/flan-t5-base",
}
LENGTH = 64
BUCKETS = (24, 40, 64)
SEED = 29092026
WARMUP = 100
MEASURE_STEPS = 60
RESERVE_SECONDS = 150  # final validation + save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--minutes", type=float, required=True, help="wall-clock budget (user cap: 60)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    assert args.minutes <= 60, "user cap: training runs last at most one hour"

    import torch
    from transformers import Adafactor, AutoModelForSeq2SeqLM, AutoTokenizer
    from active.v17.stage3_asl_encoding_v17 import encode

    base = MODELS[args.model]
    out_dir = ROOT / f"artifacts/models/stage3_multisentence_{args.model}_v17_20260929"
    report = REPORT / args.model
    report.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    held = {tuple(json.loads(line)["glosses"]) for line in EVAL_SET.open(encoding="utf-8")}
    assert not any(tuple(r["glosses"]) in held for r in rows), "evaluation session in training corpus"
    train = [r for r in rows if r["split"] == "train"]
    valid = [r for r in rows if r["split"] == "validation"]

    torch.manual_seed(SEED)
    random.seed(SEED)
    assert torch.backends.mps.is_available(), "MPS required"
    torch.mps.set_per_process_memory_fraction(0.6)
    tok = AutoTokenizer.from_pretrained(base)

    def tensors(part):
        buckets = {}
        for r in part:
            src = tok(encode(r["glosses"], r["confidences"])).input_ids
            tgt = tok(r["english"]).input_ids
            assert len(src) <= LENGTH and len(tgt) <= LENGTH, "refuse truncation"
            size = next(b for b in BUCKETS if b >= max(len(src), len(tgt)))
            buckets.setdefault(size, []).append((src, tgt))
        out = {}
        for size, pairs in buckets.items():
            ids = torch.full((len(pairs), size), tok.pad_token_id)
            mask = torch.zeros((len(pairs), size), dtype=torch.long)
            labels = torch.full((len(pairs), size), -100)
            for i, (src, tgt) in enumerate(pairs):
                ids[i, :len(src)] = torch.tensor(src)
                mask[i, :len(src)] = 1
                labels[i, :len(tgt)] = torch.tensor(tgt)
            out[size] = (ids, mask, labels)
        return out

    def batches(sets, size, shuffle):
        order = []
        for ids, mask, labels in sets.values():
            index = torch.randperm(len(ids)) if shuffle else torch.arange(len(ids))
            for start in range(0, len(ids), size):
                chunk = index[start:start + size]
                if shuffle and len(chunk) < size:
                    continue
                order.append((ids[chunk], mask[chunk], labels[chunk]))
        if shuffle:
            random.shuffle(order)
        return order

    train_set, valid_set = tensors(train), tensors(valid)
    model = AutoModelForSeq2SeqLM.from_pretrained(base).to("mps")
    optimizer = Adafactor(model.parameters(), lr=args.lr, relative_step=False, scale_parameter=False,
                          warmup_init=False)
    recipe = {"model": args.model, "init": base, "minutes": args.minutes, "batch_size": args.batch_size,
              "peak_lr": args.lr, "schedule": f"linear warmup {WARMUP} steps, linear decay to 10% at the budgeted step",
              "optimizer": "Adafactor", "device": "mps", "precision": "fp32", "seed": SEED,
              "buckets": BUCKETS, "bucket_rows": {k: len(v[0]) for k, v in sorted(train_set.items())},
              "train_rows": len(train), "validation_rows": len(valid), "corpus_sha256": digest(CORPUS),
              "eval_set_sha256": digest(EVAL_SET),
              "selection": "final step kept; validation loss reported only"}
    (report / "recipe.json").write_text(json.dumps(recipe, indent=2))
    print(json.dumps(recipe), flush=True)

    def validation_loss():
        model.eval()
        losses = []
        with torch.no_grad():
            for ids, mask, labels in batches(valid_set, 64, False):
                losses.append(float(model(input_ids=ids.to("mps"), attention_mask=mask.to("mps"),
                                          labels=labels.to("mps")).loss))
        model.train()
        return sum(losses) / max(len(losses), 1)

    budget = args.minutes * 60
    start = time.monotonic()
    total_steps = None
    step = 0
    epoch = 0
    history = []
    losses = []
    stop_reason = "budgeted steps reached"
    model.train()
    while total_steps is None or step < total_steps:
        epoch += 1
        for ids, mask, labels in batches(train_set, args.batch_size, True):
            if total_steps is not None and step >= total_steps:
                break
            if time.monotonic() - start > budget - RESERVE_SECONDS:
                stop_reason = "wall-clock stop"
                total_steps = step
                break
            if total_steps is None and step == MEASURE_STEPS:
                rate = (MEASURE_STEPS - 5) / (time.monotonic() - measure_start)
                total_steps = int(rate * (budget - RESERVE_SECONDS - (time.monotonic() - start)) * 0.95) + step
                print(json.dumps({"measured_steps_per_second": round(rate, 3), "total_steps": total_steps,
                                  "planned_epochs": round(total_steps * args.batch_size / len(train), 2)}), flush=True)
            if step == 5:
                measure_start = time.monotonic()
            if total_steps is not None and step > MEASURE_STEPS and step % 250 == 0:
                # Speed drifts (thermal, bucket mix): shrink the plan so the decay completes in time.
                rate = 250 / (time.monotonic() - window_start)
                remaining = budget - RESERVE_SECONDS - (time.monotonic() - start)
                total_steps = min(total_steps, step + int(rate * remaining * 0.95))
            if step % 250 == 0:
                window_start = time.monotonic()
            horizon = total_steps or 10 ** 9
            scale = min(1.0, (step + 1) / WARMUP) * max(0.1, 1 - 0.9 * step / horizon)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids.to("mps"), attention_mask=mask.to("mps"), labels=labels.to("mps")).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 250 == 0:
                print(json.dumps({"step": step, "epoch": epoch, "loss": round(sum(losses[-250:]) / 250, 4),
                                  "lr": args.lr * scale, "elapsed": round(time.monotonic() - start)}), flush=True)
            if total_steps and step % max(1, total_steps // 4) == 0:
                entry = {"step": step, "epoch": epoch, "train_loss": round(sum(losses[-200:]) / len(losses[-200:]), 4),
                         "validation_loss": round(validation_loss(), 4), "elapsed": round(time.monotonic() - start)}
                history.append(entry)
                print(json.dumps(entry), flush=True)
        if stop_reason == "wall-clock stop":
            break

    final = {"step": step, "epoch": epoch, "train_loss": round(sum(losses[-200:]) / len(losses[-200:]), 4),
             "validation_loss": round(validation_loss(), 4), "elapsed": round(time.monotonic() - start)}
    history.append(final)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu").save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    contract = {"format": "slt_stage3_input_contract_v17", "version": 2, "encoding": "evidence",
                "buckets": {"hi": ">=0.60", "mid": "0.40-0.60", "lo": "<0.40"}, "missing_confidence": "hi",
                "reviewed_templates_enabled": False, "utterance_segmentation": "model",
                "output": "counted_sentences",
                "output_description": "'N: sentence' per sentence; N = input glosses consumed, for incremental locking",
                "recipe_sha256": digest(report / "recipe.json"), "training_corpus_sha256": digest(CORPUS)}
    (out_dir / "stage3_input_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    result = {"recipe": recipe, "history": history, "steps": step,
              "epochs_seen": round(step * args.batch_size / len(train), 2), "stop_reason": stop_reason,
              "seconds": round(time.monotonic() - start), "checkpoint": str(out_dir.relative_to(ROOT)),
              "weights_sha256": digest(out_dir / "model.safetensors")}
    (report / "training_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"done": True, **{k: result[k] for k in ("steps", "epochs_seen", "stop_reason", "seconds")},
                      "final": final}), flush=True)


if __name__ == "__main__":
    main()
