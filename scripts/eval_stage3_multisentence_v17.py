#!/usr/bin/env python3
"""Multi-sentence Stage 3 evaluation set, renderer runs and meaning judgement.

Stage 3's existing scores are BLEU against model-written English for rule-generated
sequences drawn from the same ~40 templates the renderer trained on. They cannot say
whether a real Finish buffer holding several sentences comes out right. This builds a
held-out set the other way round: DeepSeek writes everyday English first (2-4 related
sentences per session, locked-100 signs only) together with its ASL gloss, and every
sentence must pass the corpus lemma check in both directions (no gloss unrealized, no
English content word without a gloss). Sessions whose gloss sequence occurs in either
composition training corpus are removed. Saved phone Finish inputs are added as a
reference-free slice.

References and the judge are both DeepSeek. No fluent signer reviewed anything here;
this measures agreement with an LLM on unseen compositions, not ASL translation accuracy.

  build   generate and validate the set
  render  run renderers over the set
  judge   DeepSeek meaning scores for each rendering (plus reference/literal controls)
  report  metrics and REPORT.md
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
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active.v17.asl_corpus_v17 import (  # noqa: E402
    FUNCTION_WORDS, GLOSS_LEMMAS, OBLIGATORY_GLOSSES, _covers, _tokens, locked_vocabulary,
)
from scripts.build_stage3_asl_corpus_v17 import ENDPOINT, MODEL, resolve_api_key  # noqa: E402

OUT = ROOT / "artifacts/reports/stage3_multisentence_eval_v17_20260929"
CACHE = ROOT / "data/local/stage3_multisentence_eval_v17/api_cache.jsonl"
TRAINING = (
    ROOT / "artifacts/reports/stage3_composition_v17_20260929/train.jsonl",
    ROOT / "artifacts/reports/stage3_composition_v17_20260929/request_continuation/train.jsonl",
)
PHONE_PAIRS = ROOT / "artifacts/reports/phone_all_history_review_20260929/translation_pairs.json"
V2 = ROOT / "artifacts/models/stage3_composition_v17_20260929_v2"
SLOT = "FS0"
SEED = 20260929

TOPICS = (
    "meeting someone new and introducing yourself", "a sick family member and the hospital",
    "plans for tomorrow morning", "learning sign language at school", "being hungry and eating",
    "asking a friend for help", "a doctor visit yesterday", "feeling tired after work",
    "a child who is sick at night", "asking where someone's family is", "saying goodbye and meeting again",
    "not understanding and asking again", "wanting water or a drink", "a friend coming home",
    "what time something happens", "feelings: happy, sad, angry, excited", "work and school week",
    "apologizing for being late", "asking someone's name", "thanking someone for help",
    "parents (mother, father) and their work", "hot or cold weather feelings", "reading and writing at school",
    "telling someone important news", "trying something new and whether it is easy",
    "waiting for a friend", "sleeping badly last night", "asking why someone is sad",
    "a man and a woman at the hospital", "comparing things: same, different, more, less",
)
STRUCTURES = (
    "two verbs in a row (e.g. WANT LEARN, LIKE READ, NEED SLEEP, TRY EAT)",
    "an embedded clause (e.g. I THINK ..., I KNOW ..., YOU KNOW WHERE ...)",
    "a time sign first, then the event (tense must come from the time sign)",
    "a wh-question with the wh-sign at the end (ASL style)",
    "a negation using NO",
    "a request with PLEASE and a recipient",
    "topic-comment order (object or topic first, then comment)",
    "a greeting or address followed by a question",
    "a yes/no question (ASL marks it with the face only; the gloss has no question sign)",
    "MAYBE or a feeling verb (FEEL, THINK)",
    "a fingerspelled name, written as FS0 in the gloss and in the English",
    "possessives (MY, YOUR, OUR) with family members",
)

GEN_SYSTEM = """You write evaluation data for a translator from American Sign Language gloss to English.

The signer's vocabulary is exactly these 100 signs, nothing else:
{vocab}

Each item is one SESSION: 2 to 4 related sentences a Deaf person might sign in one go in
everyday conversation. For each sentence give natural English and its ASL gloss.

English rules:
- Every content word must come from a sign in the vocabulary. Only these may be added
  freely: articles, forms of be/do/have as auxiliaries, will/can, prepositions,
  and/but/or/so, it/there/this/that, him/her/them/us/me. No other nouns, verbs,
  adjectives or adverbs. No "because", "very", "really", "also", "some", "all".
- Ordinary, natural, grammatical English. Contractions allowed.
- A fingerspelled name is written literally as FS0 in the English (e.g. "My name is FS0.").

Gloss rules (real ASL grammar, not English word order):
- Only vocabulary signs, UPPERCASE, one sign per token. The name slot is FS0.
- No articles, no copula, no tense endings; time signs usually come first.
- Negation with NO. Possessives MY/YOUR/OUR. Pronouns I/YOU/HE/WE/THEY (HE also covers she).
- Wh-signs may come at the end. Topic-comment order is fine.
- Every sign in the gloss must appear in the English meaning, and vice versa.
- English function words have NO sign: never gloss BUT, AND, FOR, TO, THE, IS, AT, OR, SO.
- There is no sign for anything outside the list (no TODAY, BROTHER, FOOD, STAY, NEWS,
  BECAUSE, AGAIN, ME, HER, SAY). If you cannot say it with the list, say something else.
- Before replying, check every gloss token against the vocabulary list.

Types: statement, wh_question, yn_question, request, greeting, negation.

Reply with one JSON object per line, nothing else:
{{"n": <item>, "sentences": [{{"en": "...", "gloss": ["..."], "type": "..."}}, ...]}}"""

VET_SYSTEM = """You check evaluation items for a translator from American Sign Language gloss to English.

Each item is a session of sentences, each with an ASL GLOSS and its ENGLISH translation.
For each item decide:
- ok=false if any English sentence is nonsense or unnatural in meaning, or a gloss does not
  plausibly express its English sentence in ASL.
- otherwise return the English sentences with grammar corrected: fix agreement, missing
  words like "to", and tense (time signs such as YESTERDAY or TOMORROW set the tense).
  Keep the meaning and the content words. Do not add new nouns, verbs, adjectives or
  adverbs. Keep FS0 exactly as written. Return the same number of sentences.

Reply with one JSON object per line, nothing else:
{"n": <item>, "ok": true|false, "en": ["<sentence 1>", "<sentence 2>", ...]}"""

JUDGE_SYSTEM = """You grade English renderings produced by a translator from ASL gloss.

For each item you get the ASL GLOSS the signer produced, usually a REFERENCE English
translation, and a CANDIDATE English. Grade the CANDIDATE on meaning and grammar:

2 = same meaning as the reference (paraphrase is fine), grammatical, nothing important
    missing or added, sentence breaks sensible.
1 = mostly right: a reader gets the intended meaning, but there is a minor error
    (slightly wrong tense or article, awkward wording, one minor word missing).
0 = wrong: meaning changed, a clause or key content word missing, content invented,
    wrong person doing the action, or so ungrammatical the meaning is unclear.

ASL marks yes/no questions only with the face, so when the gloss has no question sign a
statement rendering of a yes/no question is acceptable. With no reference, grade against
the most plausible meaning of the gloss. Fingerspelled names appear as FS0.

Reply with one JSON object per line, nothing else:
{"n": <item>, "score": <0|1|2>, "why": "<at most 12 words>"}"""


# --------------------------------------------------------------------------- API

class Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.entries[row["key"]] = row["content"]
        self.handle = path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, content):
        with self.lock:
            self.entries[key] = content
            self.handle.write(json.dumps({"key": key, "content": content}) + "\n")
            self.handle.flush()


USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
USAGE_LOCK = threading.Lock()


def chat(api_key, cache, system, user, temperature, max_tokens, tag):
    key = hashlib.sha256(f"{MODEL}\0{tag}\0{temperature}\0{system}\0{user}".encode()).hexdigest()
    hit = cache.get(key)
    if hit is not None:
        return hit
    body = {"model": MODEL, "reasoning": {"enabled": False}, "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    last = None
    for attempt in range(5):
        try:
            request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            content = payload["choices"][0]["message"]["content"]
            if not content:
                raise KeyError("empty content")
            stats = payload.get("usage", {})
            with USAGE_LOCK:
                USAGE["calls"] += 1
                USAGE["prompt_tokens"] += stats.get("prompt_tokens", 0)
                USAGE["completion_tokens"] += stats.get("completion_tokens", 0)
                USAGE["cost"] += float(stats.get("cost", 0.0))
            cache.put(key, content)
            return content
        except (urllib.error.URLError, KeyError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    print(f"API failed after retries ({tag}): {last}", file=sys.stderr)
    return ""


def json_lines(content):
    for line in content.splitlines():
        line = line.strip().strip("`")
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# --------------------------------------------------------------------------- coverage

# The corpus table lacks these: HE is the sign for she too, and "differ" + "ent" is not an
# allowed inflection. Local to this evaluation; the training validator is unchanged.
LEMMAS = {g: tuple(v) for g, v in GLOSS_LEMMAS.items()}
LEMMAS["HE"] += ("she", "her", "hers")
LEMMAS["DIFFERENT"] += ("different",)


def coverage(glosses, english):
    """Lemma coverage in both directions: unrealized glosses and unexplained content words."""
    tokens = [t for t in _tokens(english) if t != "fs"]
    lemmas = {lemma for g in glosses for lemma in LEMMAS.get(g, ())}
    invented = sorted({t for t in tokens if t not in FUNCTION_WORDS and not _covers(t, lemmas)})
    missing = [g for g in dict.fromkeys(glosses)
               if LEMMAS.get(g) and not any(_covers(t, LEMMAS[g]) for t in tokens)]
    slot_ok = (SLOT in glosses) == (SLOT in english) and english.count(SLOT) <= glosses.count(SLOT)
    return {"missing": missing, "invented": invented, "slot_ok": slot_ok,
            "missing_obligatory": [g for g in missing if g in OBLIGATORY_GLOSSES]}


def training_sequences():
    seen = set()
    for path in TRAINING:
        for line in path.open(encoding="utf-8"):
            seen.add(tuple(json.loads(line)["glosses"]))
    return seen


# --------------------------------------------------------------------------- build

def build(args):
    api_key = resolve_api_key(None)
    cache = Cache(CACHE)
    vocab = set(locked_vocabulary())
    system = GEN_SYSTEM.format(vocab=" ".join(locked_vocabulary()))
    rng = random.Random(SEED)
    requests = []
    for index in range(args.calls):
        topics = rng.sample(TOPICS, 3)
        structures = rng.sample(STRUCTURES, 3)
        counts = [rng.choice((2, 2, 3, 3, 3, 4)) for _ in range(args.per_call)]
        lines = [f"Write {args.per_call} sessions (items 1-{args.per_call}).",
                 f"Topics to draw from: {'; '.join(topics)}.",
                 f"Across the sessions, include these structures: {'; '.join(structures)}.",
                 "Sentences per session: " + ", ".join(f"item {i + 1}: {c}" for i, c in enumerate(counts)) + ".",
                 "Vary the sentences; do not repeat stock phrases across items."]
        requests.append((index, "\n".join(lines)))

    def fetch(item):
        index, user = item
        return index, chat(api_key, cache, system, user, 0.9, 4000, f"gen{index}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        replies = list(pool.map(fetch, requests))

    train = training_sequences()
    sessions, rejected, seen = [], [], set()
    for index, content in replies:
        for row in json_lines(content):
            sentences = row.get("sentences")
            if not isinstance(sentences, list) or not 2 <= len(sentences) <= 4:
                rejected.append({"call": index, "reason": "sentence_count", "row": row})
                continue
            clean, reasons = [], []
            for s in sentences:
                gloss = [str(g).upper().strip() for g in s.get("gloss", [])]
                gloss = [g for g in gloss if any(c.isalnum() for c in g)]
                english = " ".join(str(s.get("en", "")).split())
                if not gloss or not english:
                    reasons.append("empty")
                    continue
                # Function words carry no sign, so a glossed BUT or FOR is dropped, not rejected.
                gloss = [g for g in gloss if g in vocab or g == SLOT or g.lower() not in FUNCTION_WORDS]
                oov = [g for g in gloss if g not in vocab and g != SLOT]
                if oov:
                    reasons.append(f"oov:{','.join(oov)}")
                check = coverage(gloss, english)
                if check["missing"]:
                    reasons.append(f"unrealized:{','.join(check['missing'])}")
                if check["invented"]:
                    reasons.append(f"invented:{','.join(check['invented'])}")
                if not check["slot_ok"]:
                    reasons.append("slot")
                clean.append({"en": english, "gloss": gloss, "type": str(s.get("type", ""))})
            if reasons:
                rejected.append({"call": index, "reason": ";".join(reasons), "row": row})
                continue
            glosses = [g for s in clean for g in s["gloss"]]
            key = tuple(glosses)
            if key in seen:
                rejected.append({"call": index, "reason": "duplicate", "row": row})
                continue
            if key in train:
                rejected.append({"call": index, "reason": "in_training", "row": row})
                continue
            seen.add(key)
            sessions.append({
                "glosses": glosses, "sentences": clean,
                "reference": " ".join(s["en"] for s in clean),
                "sentence_in_training": [tuple(s["gloss"]) in train for s in clean],
                "source": "generated",
            })
    pool = len(sessions)
    sessions, vet_rejected = vet(api_key, cache, sessions, args.workers)
    rejected += vet_rejected
    rng.shuffle(sessions)
    sessions = sessions[:args.size]
    for n, row in enumerate(sessions):
        row["id"] = f"gen{n:03d}"

    phone = []
    pairs = json.loads(PHONE_PAIRS.read_text(encoding="utf-8"))
    extra = [["I", "HUNGRY", "HAVE", "YOU", "EAT"], ["HELLO", "FRIEND"]]  # 2026-09-29 sessions after the audit
    inputs = [[("FS0" if g.startswith("fs-") else g) for g in p["glosses"]] for p in pairs] + extra
    for glosses in inputs:
        if glosses.count(SLOT) > 1 or tuple(glosses) in {tuple(p["glosses"]) for p in phone}:
            continue
        phone.append({"id": f"phone{len(phone):02d}", "glosses": glosses, "sentences": None,
                      "reference": None, "source": "phone",
                      "sentence_in_training": [tuple(glosses) in train]})
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sessions + phone
    with (OUT / "eval_set.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    reasons = {}
    for row in rejected:
        head = row["reason"].split(":")[0].split(";")[0]
        reasons[head] = reasons.get(head, 0) + 1
    summary = {"parsed_sessions": pool + len(rejected) - len(vet_rejected), "passed_rule_checks": pool,
               "passed_vetting": pool - len(vet_rejected), "admitted": len(sessions),
               "phone": len(phone), "rejected": len(rejected), "reject_reasons": reasons,
               "usage": USAGE, "eval_set_sha256": hashlib.sha256((OUT / "eval_set.jsonl").read_bytes()).hexdigest()}
    (OUT / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def vet(api_key, cache, sessions, workers):
    """DeepSeek grammar/sense review; corrected English must still pass the lemma check."""
    batches = [sessions[i:i + 10] for i in range(0, len(sessions), 10)]

    def run(batch):
        lines = []
        for n, row in enumerate(batch, 1):
            lines.append(f"{n}.")
            for s in row["sentences"]:
                lines.append(f"   GLOSS: {' '.join(s['gloss'])}")
                lines.append(f"   ENGLISH: {s['en']}")
        content = chat(api_key, cache, VET_SYSTEM, "\n".join(lines), 0.0, 120 * len(batch) + 200, "vet")
        return batch, {r.get("n"): r for r in json_lines(content)}

    kept, dropped = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch, verdicts in pool.map(run, batches):
            for n, row in enumerate(batch, 1):
                verdict = verdicts.get(n)
                if not verdict or verdict.get("ok") is not True:
                    dropped.append({"reason": "vet_not_ok" if verdict else "vet_unparsed", "row": row})
                    continue
                fixed = verdict.get("en")
                if not isinstance(fixed, list) or len(fixed) != len(row["sentences"]):
                    dropped.append({"reason": "vet_shape", "row": row})
                    continue
                reasons = []
                for s, english in zip(row["sentences"], fixed):
                    english = " ".join(str(english).split())
                    check = coverage(s["gloss"], english)
                    if check["missing"] or check["invented"] or not check["slot_ok"]:
                        reasons.append(f"vet_broke_coverage:{english}")
                    s["original_en"], s["en"] = s["en"], english
                if reasons:
                    dropped.append({"reason": ";".join(reasons), "row": row})
                    continue
                row["reference"] = " ".join(s["en"] for s in row["sentences"])
                kept.append(row)
    return kept, dropped


# --------------------------------------------------------------------------- render

def load_set():
    return [json.loads(line) for line in (OUT / "eval_set.jsonl").open(encoding="utf-8")]


def render(args):
    from scripts.live_isolated_v17 import TinyStage3Naturalizer, literal_render
    naturalizer = TinyStage3Naturalizer(argparse.Namespace(
        stage3_checkpoint=Path(args.checkpoint), stage3_device="cpu", stage3_encoding="auto"))
    rows = load_set()
    out = []
    for row in rows:
        glosses = row["glosses"]
        started = time.perf_counter()
        full = naturalizer.rephrase(glosses, [0.9] * len(glosses))
        full_ms = 1000 * (time.perf_counter() - started)
        record = {"id": row["id"], "v2_full": full["sentence"], "v2_full_mode": full["rendering_mode"],
                  "v2_full_ms": full_ms,
                  "literal": literal_render(glosses, naturalizer.manifest)}
        if row["sentences"]:
            parts = [naturalizer.rephrase(s["gloss"], [0.9] * len(s["gloss"])) for s in row["sentences"]]
            record["v2_split"] = " ".join(p["sentence"] for p in parts)
            record["v2_split_modes"] = [p["rendering_mode"] for p in parts]
        out.append(record)
    with (OUT / "renders.jsonl").open("w", encoding="utf-8") as handle:
        for record in out:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"rendered {len(out)} rows")


COUNTED = re.compile(r"(\d+):\s*(.*?)(?=\s+\d+:\s|$)")


def parse_counted(text, glosses):
    """'3: Hello, my friend. 2: How are you?' -> [(3, 'Hello, my friend.'), (2, 'How are you?')].

    Output without counts becomes one sentence covering the whole buffer."""
    parts = [(int(m.group(1)), m.group(2).strip()) for m in COUNTED.finditer(text.strip())]
    return parts or [(len(glosses), text.strip())]


def verified_split(buffer, first, render):
    """Counts can be off by one. Accept the split n in {c, c-1, c+1} whose glosses, rendered alone,
    give back exactly the first sentence; None when no split does."""
    count, text = first
    for n in (count, count - 1, count + 1):
        if 0 < n < len(buffer):
            alone = parse_counted(render(buffer[:n]), buffer[:n])
            if len(alone) == 1 and alone[0][1] == text:
                return n
    return None


def incremental(glosses, render, verify=False):
    """Reference for the app: lock a finished sentence while signing, render only the tail at Finish.

    After each committed sign the open buffer is rendered. The first sentence is locked, and its
    glosses dropped from the buffer, once two consecutive renders agree on it, the output holds at
    least two sentences, and the counts cover the buffer exactly. With verify, the split point is
    confirmed by rendering the candidate sentence alone (extra renders happen while signing)."""
    locked, buffer, previous, renders = [], [], None, 0
    for gloss in glosses:
        buffer.append(gloss)
        parts = parse_counted(render(buffer), buffer)
        renders += 1
        if len(parts) >= 2 and sum(n for n, _ in parts) == len(buffer) and 0 < parts[0][0] < len(buffer):
            if parts[0] == previous:
                n = verified_split(buffer, parts[0], render) if verify else parts[0][0]
                if n is not None:
                    locked.append(parts[0][1])
                    buffer = buffer[n:]
                    previous = None
                    continue
            previous = parts[0]
        else:
            previous = None
    tail = [t for _, t in parse_counted(render(buffer), buffer)] if buffer else []
    return " ".join(locked + tail), {"locked": len(locked), "finish_glosses": len(buffer), "renders": renders + 1}


def render_model(args):
    """Counted-output checkpoints: whole buffer at Finish, and simulated incremental translation."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from active.v17.stage3_asl_encoding_v17 import encode
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).eval()
    cache = {}

    def render_one(glosses):
        key = tuple(glosses)
        if key not in cache:
            ids = tok(encode(list(glosses), [0.9] * len(glosses)), return_tensors="pt")
            with torch.inference_mode():
                out = model.generate(**ids, max_new_tokens=64, num_beams=1, do_sample=False)
            cache[key] = tok.decode(out[0], skip_special_tokens=True).strip()
        return cache[key]

    path = OUT / "renders.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    rows = {row["id"]: row for row in load_set()}
    ms = []
    for record in records:
        glosses = rows[record["id"]]["glosses"]
        started = time.perf_counter()
        raw = render_one(glosses)
        ms.append(1000 * (time.perf_counter() - started))
        parts = parse_counted(raw, glosses)
        record[f"{args.name}_full"] = " ".join(t for _, t in parts)
        record[f"{args.name}_full_raw"] = raw
        record[f"{args.name}_full_counts_ok"] = sum(n for n, _ in parts) == len(glosses)
        text, meta = incremental(glosses, render_one)
        record[f"{args.name}_incr"] = text
        record[f"{args.name}_incr_meta"] = meta
        text, meta = incremental(glosses, render_one, verify=True)
        record[f"{args.name}_incrv"] = text
        record[f"{args.name}_incrv_meta"] = meta
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    ms.sort()
    print(json.dumps({"name": args.name, "rows": len(records), "cpu_ms_median": round(ms[len(ms) // 2], 1),
                      "counts_ok": sum(r[f"{args.name}_full_counts_ok"] for r in records)}))


# --------------------------------------------------------------------------- judge

META_SUFFIXES = ("_mode", "_ms", "_modes", "_raw", "_meta", "_counts_ok")


def candidates(renders):
    names = {k for r in renders for k in r if k != "id" and not k.endswith(META_SUFFIXES)}
    return ("reference",) + tuple(sorted(names))


def judge(args):
    api_key = resolve_api_key(None)
    cache = Cache(CACHE)
    rows = {row["id"]: row for row in load_set()}
    renders = [json.loads(line) for line in (OUT / "renders.jsonl").open(encoding="utf-8")]
    path = OUT / "judgements.jsonl"
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            j = json.loads(line)
            if j["score"] in (0, 1, 2):
                done[(j["id"], j["candidate"])] = j
    items = []
    for record in renders:
        row = rows[record["id"]]
        for name in candidates(renders):
            text = row["reference"] if name == "reference" else record.get(name)
            if text is None or (record["id"], name) in done:
                continue
            items.append((record["id"], name, row, text))
    batches = [items[i:i + 10] for i in range(0, len(items), 10)]

    def run(batch):
        lines = []
        for n, (_, _, row, text) in enumerate(batch, 1):
            lines.append(f"{n}. GLOSS: {' '.join(row['glosses'])}")
            lines.append(f"   REFERENCE: {row['reference'] or '(none)'}")
            lines.append(f"   CANDIDATE: {text}")
        user = "\n".join(lines)
        content = chat(api_key, cache, JUDGE_SYSTEM, user, 0.0, 60 * len(batch) + 200, "judge")
        got = {r.get("n"): r for r in json_lines(content)}
        return [(ident, name, got.get(n, {}).get("score"), got.get(n, {}).get("why"))
                for n, (ident, name, _, _) in enumerate(batch, 1)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for chunk in pool.map(run, batches) for r in chunk]
    for ident, name, score, why in results:
        done[(ident, name)] = {"id": ident, "candidate": name, "score": score, "why": why}
    path.write_text("".join(json.dumps(j) + "\n" for j in done.values()), encoding="utf-8")
    missing = sum(1 for r in results if r[2] not in (0, 1, 2))
    print(f"judged {len(results)} ({missing} unparsed)  usage {USAGE}")


# --------------------------------------------------------------------------- report

def report(args):
    rows = {row["id"]: row for row in load_set()}
    renders = {r["id"]: r for r in (json.loads(l) for l in (OUT / "renders.jsonl").open(encoding="utf-8"))}
    scores = {}
    for line in (OUT / "judgements.jsonl").open(encoding="utf-8"):
        j = json.loads(line)
        scores[(j["id"], j["candidate"])] = j
    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None

    def metrics(ids, name):
        texts, refs, judged, faithful, missing, invented, n = [], [], [], 0, 0, 0, 0
        for ident in ids:
            row = rows[ident]
            text = row["reference"] if name == "reference" else renders[ident].get(name)
            if text is None:
                continue
            n += 1
            check = coverage(row["glosses"], text)
            faithful += not check["missing"] and not check["invented"]
            missing += bool(check["missing"])
            invented += bool(check["invented"])
            s = scores.get((ident, name), {}).get("score")
            if s in (0, 1, 2):
                judged.append(s)
            if row["reference"]:
                texts.append(text)
                refs.append(row["reference"])
        with_no = [i for i in ids if "NO" in rows[i]["glosses"]
                   and (rows[i]["reference"] if name == "reference" else renders[i].get(name)) is not None]
        no_dropped = sum("NO" in coverage(rows[i]["glosses"], rows[i]["reference"] if name == "reference"
                                          else renders[i][name])["missing"] for i in with_no)
        out = {"n": n, "faithful_pct": 100 * faithful / max(n, 1),
               "no_dropped": f"{no_dropped}/{len(with_no)}",
               "drops_gloss_pct": 100 * missing / max(n, 1), "invents_pct": 100 * invented / max(n, 1),
               "judge_mean": sum(judged) / max(len(judged), 1),
               "judge_2_pct": 100 * judged.count(2) / max(len(judged), 1),
               "judge_0_pct": 100 * judged.count(0) / max(len(judged), 1)}
        if sacrebleu and texts:
            out["bleu"] = sacrebleu.corpus_bleu(texts, [refs]).score
            out["chrf"] = sacrebleu.corpus_chrf(texts, [refs]).score
        return out

    gen = [i for i, r in rows.items() if r["source"] == "generated"]
    phone = [i for i, r in rows.items() if r["source"] == "phone"]
    slices = {"generated_all": gen,
              "generated_2_sentences": [i for i in gen if len(rows[i]["sentences"]) == 2],
              "generated_3_sentences": [i for i in gen if len(rows[i]["sentences"]) == 3],
              "generated_4_sentences": [i for i in gen if len(rows[i]["sentences"]) == 4],
              "generated_no_sentence_in_training": [i for i in gen if not any(rows[i]["sentence_in_training"])],
              "generated_with_yn_question": [i for i in gen if any(s["type"] == "yn_question" for s in rows[i]["sentences"])],
              "generated_with_name_slot": [i for i in gen if SLOT in rows[i]["glosses"]],
              "phone_history": phone}
    names = candidates(list(renders.values()))
    table = {s: {c: metrics(ids, c) for c in names} for s, ids in slices.items()}
    over_context = sum(1 for i in gen if renders[i]["v2_full_mode"] != "t5_efficient_tiny")
    lengths = sorted(len(rows[i]["glosses"]) for i in gen)
    ms = sorted(renders[i]["v2_full_ms"] for i in gen)
    summary = {"slices": table, "v2_full_non_neural": over_context,
               "gloss_length": {"min": lengths[0], "median": lengths[len(lengths) // 2], "max": lengths[-1]},
               "v2_full_desktop_cpu_ms": {"median": ms[len(ms) // 2], "p90": ms[int(.9 * len(ms))]}}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--calls", type=int, default=50)
    b.add_argument("--per-call", type=int, default=8)
    b.add_argument("--size", type=int, default=300)
    b.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("render")
    r.add_argument("--checkpoint", default=str(V2))
    m = sub.add_parser("render-model", help="counted-output checkpoint: whole buffer + incremental")
    m.add_argument("--name", required=True)
    m.add_argument("--checkpoint", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--workers", type=int, default=8)
    sub.add_parser("report")
    args = parser.parse_args()
    {"build": build, "render": render, "render-model": render_model, "judge": judge, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
