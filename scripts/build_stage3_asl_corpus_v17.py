#!/usr/bin/env python3
"""Build the ASL-order gloss/English corpus for retraining Stage 3.

Gloss sequences come from `active/v17/asl_corpus_v17.py`, which generates genuine ASL
word order. English targets come from DeepSeek V4 Flash over OpenRouter, batched and
cached so a rerun costs nothing and the corpus is reproducible from the cache alone.

Every target is checked against the glosses that produced it before admission. A row
that invents content, leaks an injected noise gloss or loses the subject is sent back
once with its failure reason; if it fails again it is dropped, not repaired by hand.

The deployed renderer was trained on 15,843 rows of which only 95 reorder a gloss and
742 are fully inside the locked 100. This corpus exists to replace both deficits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.asl_corpus_v17 import (  # noqa: E402
    Utterance,
    corpus_signature,
    generate_corpus,
    locked_vocabulary,
    validate_english,
)

CONTINUE_CONFIG = Path.home() / ".continue/config.yaml"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"
OUT_DIR = ROOT / "data/local/stage3_asl_corpus_v17"

SYSTEM_PROMPT = """You convert American Sign Language gloss sequences into natural English.

ASL glosses carry no copula, no articles, no tense marking and no word NOT. The English
renderer must supply those. ASL also uses topic-comment and object-first order, puts time
first, and can put a question sign last.

For each item you receive the gloss sequence and an explicit statement of its intended
meaning. Write the English sentence that expresses that meaning.

Rules:
- Reorder freely into natural English word order. Do not preserve gloss order.
- Add articles, copulas, auxiliaries, prepositions and tense as English requires.
- Never add content, objects, places or facts that are not in the meaning statement.
- If the item says a gloss is a recognition error, that word must not appear at all.
- Keep it to one sentence unless the meaning states two clauses.
- Use ordinary contractions where natural.

Reply with one JSON object per line: {"n": <item number>, "en": "<sentence>"}
No preamble, no code fences, no other text."""


def resolve_api_key(explicit: str | None) -> str:
    """Find the OpenRouter key: flag, environment, then the Continue config."""
    if explicit:
        return explicit
    import os

    for name in ("OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    if CONTINUE_CONFIG.exists():
        for line in CONTINUE_CONFIG.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("#"):
                continue
            match = re.search(r'apiKey:\s*"([^"]+)"', line)
            if match and match.group(1).startswith("sk-or"):
                return match.group(1)
    raise SystemExit(
        "No OpenRouter key found. Set OPENROUTER_API_KEY or pass --api-key."
    )


def cache_key(utterance: Utterance, repair: str = "") -> str:
    payload = f"{utterance.key}\0{utterance.meaning}\0{repair}"
    return hashlib.sha256(payload.encode()).hexdigest()


class Cache:
    """Append-only JSONL cache of gloss/meaning -> English."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.entries: dict[str, str] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                self.entries[row["key"]] = row["en"]
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def get(self, key: str) -> str | None:
        return self.entries.get(key)

    def put(self, key: str, english: str) -> None:
        with self.lock:
            if key in self.entries:
                return
            self.entries[key] = english
            self.handle.write(json.dumps({"key": key, "en": english}) + "\n")
            self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def call_model(key: str, user_message: str, max_tokens: int, timeout: int) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        # Reasoning tokens are pure cost here; the task is mechanical.
        "reasoning": {"enabled": False},
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if "choices" not in payload:
        raise RuntimeError(f"unexpected response: {str(payload)[:200]}")
    return payload["choices"][0]["message"]["content"], payload.get("usage", {})


def render_item(index: int, utterance: Utterance, note: str = "") -> str:
    lines = [f"{index}. GLOSS: {utterance.key}", f"   MEANING: {utterance.meaning}"]
    if note:
        lines.append(f"   PREVIOUS ATTEMPT WAS REJECTED: {note}")
    return "\n".join(lines)


def parse_reply(content: str, expected: list[int]) -> dict[int, str]:
    """Read one JSON object per line, tolerating stray prose or fences."""
    out: dict[int, str] = {}
    for line in content.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        number, english = row.get("n"), row.get("en")
        if isinstance(number, int) and isinstance(english, str) and number in expected:
            out[number] = " ".join(english.split())
    return out


def translate_batch(
    key: str,
    batch: list[tuple[int, Utterance, str]],
    cache: Cache,
    timeout: int,
    usage: dict,
    usage_lock: threading.Lock,
) -> None:
    pending = [(n, u, note) for n, u, note in batch if cache.get(cache_key(u, note)) is None]
    if not pending:
        return
    message = "\n".join(render_item(n, u, note) for n, u, note in pending)
    expected = [n for n, _, _ in pending]
    budget = 90 * len(pending) + 200
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            content, stats = call_model(key, message, budget, timeout)
            with usage_lock:
                usage["calls"] += 1
                usage["prompt_tokens"] += stats.get("prompt_tokens", 0)
                usage["completion_tokens"] += stats.get("completion_tokens", 0)
                usage["cost"] += float(stats.get("cost", 0.0))
            replies = parse_reply(content, expected)
            for number, utterance, note in pending:
                english = replies.get(number)
                if english:
                    cache.put(cache_key(utterance, note), english)
            return
        except (urllib.error.URLError, RuntimeError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"  batch failed after retries: {last_error}", file=sys.stderr)


def translate_all(
    key: str,
    items: list[tuple[int, Utterance, str]],
    cache: Cache,
    batch_size: int,
    workers: int,
    timeout: int,
    usage: dict,
) -> None:
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    todo = [b for b in batches if any(cache.get(cache_key(u, n)) is None for _, u, n in b)]
    if not todo:
        print(f"  all {len(items)} rows already cached")
        return
    print(f"  {len(todo)} batches to fetch ({len(batches) - len(todo)} fully cached)")
    usage_lock = threading.Lock()
    done = 0
    done_lock = threading.Lock()

    def run(batch: list[tuple[int, Utterance, str]]) -> None:
        nonlocal done
        translate_batch(key, batch, cache, timeout, usage, usage_lock)
        with done_lock:
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} batches  ${usage['cost']:.4f}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, todo))


def split_for(utterance: Utterance, seed: int) -> str:
    """Assign a split by the clean gloss core.

    Keying on the clean core keeps a sequence and its noise variant on the same side,
    so a held-out BLEU score cannot be inflated by having seen the clean twin.
    """
    core = " ".join(utterance.clean_glosses)
    bucket = int(hashlib.sha256(f"{seed}\0{core}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17601)
    parser.add_argument("--noise-rate", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--dry-run", type=int, default=0,
        help="generate and cost-estimate this many rows without calling the API",
    )
    args = parser.parse_args()

    vocabulary = set(locked_vocabulary())
    rows = generate_corpus(args.count, args.seed, args.noise_rate)
    stray = [r for r in rows if any(g not in vocabulary for g in r.glosses)]
    if stray:
        raise SystemExit(f"{len(stray)} generated rows leave the locked vocabulary")
    print(f"generated {len(rows)} sequences, signature {corpus_signature(rows)[:16]}")

    if args.dry_run:
        for row in rows[:args.dry_run]:
            print(f"  {row.key}\n     {row.meaning[:110]}")
        estimate = len(rows) * 9.0e-6
        print(f"dry run only; estimated cost for {len(rows)} rows ~= ${estimate:.3f}")
        return

    key = resolve_api_key(args.api_key)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(args.out_dir / "english_cache.jsonl")
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}

    print("pass 1: generating English")
    translate_all(
        key, [(i, u, "") for i, u in enumerate(rows, start=1)], cache,
        args.batch_size, args.workers, args.timeout, usage,
    )

    # Validate, then give every rejected row exactly one corrective retry.
    admitted: list[dict[str, Any]] = []
    repairs: list[tuple[int, Utterance, str]] = []
    rejected: list[dict[str, Any]] = []
    for index, utterance in enumerate(rows, start=1):
        english = cache.get(cache_key(utterance))
        if english is None:
            rejected.append({"gloss": utterance.key, "reasons": ["no_response"]})
            continue
        ok, reasons = validate_english(utterance, english)
        if ok:
            admitted.append({"utterance": utterance, "english": english})
        else:
            repairs.append((index, utterance, "; ".join(reasons)))
    print(f"pass 1 admitted {len(admitted)}, repairing {len(repairs)}")

    if repairs:
        print("pass 2: repairing rejected rows")
        translate_all(
            key, repairs, cache, args.batch_size, args.workers, args.timeout, usage,
        )
        for _, utterance, note in repairs:
            english = cache.get(cache_key(utterance, note))
            if english is None:
                rejected.append({"gloss": utterance.key, "reasons": ["no_repair_response"]})
                continue
            ok, reasons = validate_english(utterance, english)
            if ok:
                admitted.append({"utterance": utterance, "english": english})
            else:
                rejected.append({"gloss": utterance.key, "reasons": reasons, "english": english})

    corpus_path = args.out_dir / "corpus.jsonl"
    counts = {"train": 0, "validation": 0, "test": 0}
    structures: dict[str, int] = {}
    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in admitted:
            utterance: Utterance = row["utterance"]
            split = split_for(utterance, args.seed)
            counts[split] += 1
            family = utterance.structure.split("+")[0].split("(")[0]
            structures[family] = structures.get(family, 0) + 1
            handle.write(json.dumps({
                "glosses": list(utterance.glosses),
                "confidences": list(utterance.confidences),
                "noise_indices": list(utterance.noise_indices),
                "structure": utterance.structure,
                "meaning": utterance.meaning,
                "english": row["english"],
                "split": split,
            }) + "\n")

    (args.out_dir / "rejected.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rejected), encoding="utf-8"
    )
    manifest = {
        "format": "slt_stage3_asl_corpus_v17",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "active/v17/asl_corpus_v17.py",
        "english_model": MODEL,
        "english_provider": "openrouter",
        "seed": args.seed,
        "requested": args.count,
        "generated": len(rows),
        "sequence_signature": corpus_signature(rows),
        "admitted": len(admitted),
        "rejected": len(rejected),
        "noise_rate": args.noise_rate,
        "noisy_rows": sum(1 for r in admitted if r["utterance"].noise_indices),
        "splits": counts,
        "structures": dict(sorted(structures.items(), key=lambda kv: -kv[1])),
        "api_usage": usage,
        "claim_scope": (
            "rule-generated ASL word order with model-written English; "
            "not human-reviewed translation and not a genuine ASL corpus"
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    cache.close()
    print(json.dumps({k: manifest[k] for k in
                      ("admitted", "rejected", "splits", "noisy_rows", "api_usage")}, indent=2))
    print(f"wrote {corpus_path}")


if __name__ == "__main__":
    main()
