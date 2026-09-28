#!/usr/bin/env python3
"""Training corpus for the flan-t5-base multi-sentence Stage 3 renderer (user decision 2026-09-29).

Same English-first recipe as the held-out evaluation (`scripts/eval_stage3_multisentence_v17.py`):
DeepSeek writes everyday sessions with their ASL gloss, each sentence must pass the two-way lemma
check, DeepSeek then fixes grammar only. Differences from the evaluation build:

- Admission is per sentence: contiguous runs of passing sentences are kept as natural sessions,
  so one bad sentence no longer discards its neighbours (56% of sentences pass vs 25% of sessions).
- The evaluation set is held out strictly: any sentence whose gloss sequence occurs in the
  evaluation set, and any session equal to an evaluation session, is removed.
- Training sessions are natural runs, single sentences, and random 2-4 sentence concatenations.
- Targets carry a gloss count before each sentence ("3: Hello, my friend. 2: How are you?") so the
  app can lock a finished sentence and drop exactly its glosses from the open buffer.
- Per-gloss confidences follow the live bucket contract. In a minority of sessions one spurious
  gloss is inserted with a low (<0.40) score and must not be rendered; genuine glosses are never
  dropped, whatever their score.
- A sample of the previous corpus's clean single-sentence rows adds rule-generated ASL orders.

Spending stops at --budget dollars (user cap: $2).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.asl_corpus_v17 import FUNCTION_WORDS, locked_vocabulary  # noqa: E402
from scripts.build_stage3_asl_corpus_v17 import resolve_api_key  # noqa: E402
from scripts.eval_stage3_multisentence_v17 import (  # noqa: E402
    CACHE, GEN_SYSTEM, OUT as EVAL_DIR, SLOT, STRUCTURES, TOPICS, USAGE, VET_SYSTEM, Cache, chat,
    coverage, json_lines,
)

OUT = ROOT / "data/local/stage3_multisentence_corpus_v17"
REPORT = ROOT / "artifacts/reports/stage3_multisentence_bakeoff_v17_20260929"
OLD_TRAIN = ROOT / "artifacts/reports/stage3_composition_v17_20260929/train.jsonl"
SEED = 29092026
LOCK = threading.Lock()
# Spend across runs. The per-run counter alone let a rerun exceed the user's $2 cap (2026-09-29:
# $1.852 + $0.500); every budget check now includes what earlier runs already spent.
LEDGER = OUT / "spend_ledger.json"


def prior_spend():
    return sum(r["cost"] for r in json.loads(LEDGER.read_text())) if LEDGER.exists() else 0.0


def spent():
    return PRIOR + USAGE["cost"]


PRIOR = prior_spend()


def target_text(sentences, counts):
    return " ".join(f"{n}: {s['en']}" for s, n in zip(sentences, counts))


def parse_target(text):
    """Inverse of target_text: [(gloss_count, sentence)]. Used by the app-side logic and tests."""
    return [(int(m.group(1)), m.group(2).strip())
            for m in re.finditer(r"(\d+):\s*(.*?)(?=\s+\d+:\s|$)", text.strip())]


def clean_sentence(raw, vocab):
    gloss = [str(g).upper().strip() for g in raw.get("gloss", [])]
    gloss = [g for g in gloss if any(c.isalnum() for c in g)]
    gloss = [g for g in gloss if g in vocab or g == SLOT or g.lower() not in FUNCTION_WORDS]
    english = " ".join(str(raw.get("en", "")).split())
    if not gloss or not english or any(g not in vocab and g != SLOT for g in gloss):
        return None
    check = coverage(gloss, english)
    if check["missing"] or check["invented"] or not check["slot_ok"]:
        return None
    return {"gloss": gloss, "en": english, "type": str(raw.get("type", ""))}


def generate(args, api_key, cache, vocab, held_sentences, held_sessions):
    system = GEN_SYSTEM.format(vocab=" ".join(locked_vocabulary()))
    rng = random.Random(SEED)
    requests = []
    for index in range(args.calls):
        counts = [rng.choice((1, 2, 2, 3, 3, 3, 4, 4, 5)) for _ in range(8)]
        lines = ["Write 8 sessions (items 1-8).",
                 f"Topics to draw from: {'; '.join(rng.sample(TOPICS, 3))}.",
                 f"Across the sessions, include these structures: {'; '.join(rng.sample(STRUCTURES, 3))}.",
                 "Sentences per session: " + ", ".join(f"item {i + 1}: {c}" for i, c in enumerate(counts)) + ".",
                 "Vary the sentences; do not repeat stock phrases across items."]
        requests.append((index, "\n".join(lines)))

    def fetch(item):
        index, user = item
        with LOCK:
            if spent() >= args.gen_budget:
                return ""
        return chat(api_key, cache, system, user, 0.9, 4000, f"train{index}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        replies = list(pool.map(fetch, requests))
    runs, stats = [], {"calls_answered": sum(bool(r) for r in replies), "sentences": 0, "passed": 0,
                       "held_out_removed": 0}
    for content in replies:
        for row in json_lines(content):
            sentences = row.get("sentences")
            if not isinstance(sentences, list):
                continue
            run = []
            for raw in sentences + [None]:
                s = clean_sentence(raw, vocab) if isinstance(raw, dict) else None
                if raw is not None:
                    stats["sentences"] += 1
                if s is not None and tuple(s["gloss"]) in held_sentences:
                    stats["held_out_removed"] += 1
                    s = None
                if s is not None:
                    stats["passed"] += 1
                    run.append(s)
                    continue
                if run and tuple(g for x in run for g in x["gloss"]) not in held_sessions:
                    runs.append(run)
                run = []
    return runs, stats


def vet(api_key, cache, runs, args):
    batches = [runs[i:i + 10] for i in range(0, len(runs), 10)]

    def run(batch):
        with LOCK:
            if spent() >= args.budget:
                return batch, None
        lines = []
        for n, session in enumerate(batch, 1):
            lines.append(f"{n}.")
            for s in session:
                lines += [f"   GLOSS: {' '.join(s['gloss'])}", f"   ENGLISH: {s['en']}"]
        content = chat(api_key, cache, VET_SYSTEM, "\n".join(lines), 0.0, 120 * len(batch) + 200, "vet_train")
        return batch, {r.get("n"): r for r in json_lines(content)}

    kept, stats = [], {"vet_not_ok": 0, "vet_broke_coverage": 0, "unvetted_budget": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch, verdicts in pool.map(run, batches):
            if verdicts is None:
                stats["unvetted_budget"] += len(batch)
                continue
            for n, session in enumerate(batch, 1):
                verdict = verdicts.get(n) or {}
                fixed = verdict.get("en")
                if verdict.get("ok") is not True or not isinstance(fixed, list) or len(fixed) != len(session):
                    stats["vet_not_ok"] += 1
                    continue
                out = []
                for s, english in zip(session, fixed):
                    english = " ".join(str(english).split())
                    check = coverage(s["gloss"], english)
                    if check["missing"] or check["invented"] or not check["slot_ok"]:
                        break
                    out.append({**s, "en": english})
                if len(out) != len(session):
                    stats["vet_broke_coverage"] += 1
                    continue
                kept.append(out)
    return kept, stats


def confidences_for(n, rng):
    """Genuine glosses: mostly high, some mid/low. Low never means 'drop' on its own."""
    out = []
    for _ in range(n):
        r = rng.random()
        out.append(round(rng.uniform(.6, 1.0) if r < .8 else rng.uniform(.4, .6) if r < .95 else rng.uniform(.2, .4), 3))
    return out


def make_row(sentences, source, rng, vocab_list, noise_rate):
    glosses = [g for s in sentences for g in s["gloss"]]
    confidences = confidences_for(len(glosses), rng)
    counts = [len(s["gloss"]) for s in sentences]
    noise = None
    if rng.random() < noise_rate:
        # One spurious low-score gloss inside a sentence; it counts toward that sentence's span.
        k = rng.randrange(len(sentences))
        start = sum(counts[:k])
        pos = start + rng.randrange(counts[k] + 1)
        spurious = rng.choice([g for g in vocab_list if g not in glosses])
        glosses.insert(pos, spurious)
        confidences.insert(pos, round(rng.uniform(.1, .39), 3))
        counts[k] += 1
        noise = {"index": pos, "gloss": spurious}
    return {"glosses": glosses, "confidences": confidences, "english": target_text(sentences, counts),
            "sentence_counts": counts, "noise": noise, "source": source}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=3200)
    parser.add_argument("--gen-budget", type=float, default=1.45)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--composed", type=int, default=24000, help="random 2-4 sentence concatenations")
    parser.add_argument("--old-rows", type=int, default=8000)
    parser.add_argument("--noise-rate", type=float, default=.08)
    parser.add_argument("--validation", type=float, default=.02)
    args = parser.parse_args()
    api_key = resolve_api_key(None)
    cache = Cache(CACHE)
    vocab = set(locked_vocabulary())
    vocab_list = sorted(vocab)

    evaluation = [json.loads(l) for l in (EVAL_DIR / "eval_set.jsonl").open(encoding="utf-8")]
    held_sentences = {tuple(s["gloss"]) for r in evaluation if r["sentences"] for s in r["sentences"]}
    held_sessions = {tuple(r["glosses"]) for r in evaluation}

    runs, gen_stats = generate(args, api_key, cache, vocab, held_sentences, held_sessions)
    print(json.dumps({"generation": gen_stats, "runs": len(runs), "cost": USAGE["cost"]}), flush=True)
    runs, vet_stats = vet(api_key, cache, runs, args)
    print(json.dumps({"vetting": vet_stats, "kept_runs": len(runs), "cost": USAGE["cost"]}), flush=True)

    # Unique sentences by gloss sequence (first English kept); natural runs deduplicated as wholes.
    pool, seen_runs, natural = {}, set(), []
    for session in runs:
        for s in session:
            pool.setdefault(tuple(s["gloss"]), s)
        key = tuple(tuple(s["gloss"]) for s in session)
        if key not in seen_runs:
            seen_runs.add(key)
            natural.append(session)
    sentences = list(pool.values())

    rng = random.Random(SEED)
    rows = [make_row(s, "natural", rng, vocab_list, args.noise_rate) for s in natural if len(s) > 1]
    rows += [make_row([s], "single", rng, vocab_list, args.noise_rate) for s in sentences]
    for _ in range(args.composed):
        picked = rng.sample(sentences, rng.choice((2, 2, 3, 3, 4)))
        if tuple(g for s in picked for g in s["gloss"]) in held_sessions:
            continue
        rows.append(make_row(picked, "composed", rng, vocab_list, args.noise_rate))

    old = []
    for line in OLD_TRAIN.open(encoding="utf-8"):
        r = json.loads(line)
        english = r["english"].strip()
        if (r.get("noise_indices") or len(re.findall(r"[.?!](\s|$)", english)) != 1
                or any(g not in vocab and g != SLOT for g in r["glosses"])
                or tuple(r["glosses"]) in held_sentences):
            continue
        old.append({"gloss": r["glosses"], "en": english})
    rng.shuffle(old)
    rows += [make_row([s], "previous_corpus", rng, vocab_list, 0.0) for s in old[:args.old_rows]]

    # Model context: evidence encoding and target must both fit 64 tokens (no truncation).
    from transformers import AutoTokenizer
    from active.v17.stage3_asl_encoding_v17 import encode
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    # Held-out phone inputs have no sentence breakdown, so a generated row can equal one exactly.
    before = len(rows)
    rows = [r for r in rows if tuple(r["glosses"]) not in held_sessions]
    held_session_rows_removed = before - len(rows)
    fitted = [r for r in rows
              if len(tok(encode(r["glosses"], r["confidences"])).input_ids) <= 64
              and len(tok(r["english"]).input_ids) <= 64]
    rng.shuffle(fitted)
    for r in fitted:
        bucket = int(hashlib.sha256(" ".join(r["glosses"]).encode()).hexdigest()[:8], 16) % 1000
        r["split"] = "validation" if bucket < args.validation * 1000 else "train"
    assert not any(tuple(r["glosses"]) in held_sessions for r in fitted)

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    ledger.append({"run": len(ledger) + 1, "cost": USAGE["cost"]})
    LEDGER.write_text(json.dumps(ledger, indent=2))
    path = OUT / "corpus.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fitted), encoding="utf-8")
    sources = {}
    for r in fitted:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
    summary = {"generation": gen_stats, "vetting": vet_stats, "unique_sentences": len(sentences),
               "natural_multi_sentence_runs": sum(len(s) > 1 for s in natural), "rows": len(fitted), "held_session_rows_removed": held_session_rows_removed,
               "dropped_over_context": len(rows) - len(fitted), "by_source": sources,
               "train": sum(r["split"] == "train" for r in fitted),
               "validation": sum(r["split"] == "validation" for r in fitted),
               "noise_rows": sum(r["noise"] is not None for r in fitted),
               "held_out_sentences": len(held_sentences), "usage": USAGE, "cumulative_spend": spent(),
               "corpus": str(path.relative_to(ROOT)),
               "corpus_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (REPORT / "corpus_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
