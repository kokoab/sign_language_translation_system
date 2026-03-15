"""
eval_stage3.py — Batch evaluation for the Stage 3 gloss-to-English translator.

Mirrors the structure of the Stage 2 WER evaluator.
Instead of WER on gloss tokens, we measure:
  - BLEU-1  : unigram overlap (vocabulary coverage)
  - BLEU-2  : bigram overlap  (phrase fluency)
  - Exact Match : % of sentences translated perfectly

Run from the root SLT folder:
    python eval_stage3.py
"""

import os
import random
import time
import math
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════════════

MODEL_PATH      = "weights/slt_final_t5_model"   # path to your downloaded model folder
NUM_EVAL        = 200                              # number of random gloss sentences to test
PREFIX          = "translate ASL gloss to English: "
MAX_NEW_TOKENS  = 80
NUM_BEAMS       = 4

# ── Vocabulary (must match generate_stage3_data.py) ──────────────
pronouns    = ["I", "YOU", "HE", "THEY", "WE", "SHE"]
people      = ["FRIEND", "FATHER", "BOSS", "WOMAN", "SISTER", "MAN",
               "BROTHER", "DOCTOR", "CHILD", "TEACHER", "MOTHER", "WORKER", "STUDENT"]
places      = ["BANK", "HOSPITAL", "CITY", "OFFICE", "SCHOOL", "HOUSE",
               "MARKET", "STORE", "LIBRARY", "PARK", "RESTAURANT", "ROOM"]
tech        = ["PHONE", "COMPUTER", "DATA", "CODE", "APP", "LAPTOP", "INTERNET"]
consumables = ["FOOD", "WATER", "COFFEE", "APPLE"]
vehicles    = ["CAR", "BUS", "BIKE", "TRAIN"]
things      = ["BOOK", "MONEY", "MEETING", "PROJECT", "IDEA", "PROBLEM",
               "QUESTION", "LETTER", "BAG", "KEY", "LESSON"]
feelings    = ["HAPPY", "SAD", "ANGRY", "TIRED", "EXCITED", "SCARED", "BORED", "SICK"]
times       = ["TODAY", "YESTERDAY", "TOMORROW", "NOW", "MORNING", "NIGHT", "AFTERNOON"]

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — GLOSS SENTENCE GENERATOR (evaluation distribution)
# ══════════════════════════════════════════════════════════════════

def random_gloss():
    """
    Generates a random gloss sentence drawn from the same distribution
    the model was trained on. Returns (gloss_string, expected_hint).
    expected_hint is a loose description for display only — not used in scoring.
    """
    pattern = random.choice([
        "time_svo", "no_time_svo", "sov_osv", "sov_ovs",
        "feeling", "wh_what", "wh_where", "imperative",
    ])

    # ── Verb-object rules mirror the training generator exactly ──────
    # visit verbs → places only
    # tech verbs  → tech/things only
    # study verbs → things/tech only
    # eat         → food/apple only
    # drink       → water/coffee only
    # buy_sell    → buyable (tech+things+vehicles) only — no places
    buyable = tech + things + vehicles

    if pattern == "time_svo":
        t = random.choice(times)
        s = random.choice(pronouns)
        cat = random.choice(["visit", "tech", "study", "mental", "eat", "drink", "buy"])
        if cat == "visit":
            v = random.choice(["GO", "WALK", "COME", "VISIT", "LEAVE", "RETURN"])
            o = random.choice(places)
        elif cat == "tech":
            v = random.choice(["USE", "FIX", "CREATE", "SEND", "DEVELOP"])
            o = random.choice(tech + things)
        elif cat == "study":
            v = random.choice(["READ", "LEARN", "STUDY", "WRITE", "PRACTICE"])
            o = random.choice(things + tech)
        elif cat == "mental":
            v = random.choice(["NEED", "WANT", "FIND", "CALL", "KNOW", "LIKE"])
            o = random.choice(things + tech + places)
        elif cat == "eat":
            v = "EAT"
            o = random.choice(["FOOD", "APPLE"])
        elif cat == "drink":
            v = "DRINK"
            o = random.choice(["WATER", "COFFEE"])
        else:  # buy
            v = random.choice(["BUY", "GIVE", "TAKE", "RECEIVE"])
            o = random.choice(buyable)
        return f"{t} {s} {v} {o}", "time + subject + verb + object"

    if pattern == "no_time_svo":
        s = random.choice(pronouns + people)
        cat = random.choice(["visit", "tech", "study", "mental", "eat", "drink", "buy"])
        if cat == "visit":
            v = random.choice(["GO", "WALK", "COME", "VISIT"])
            o = random.choice(places)
        elif cat == "tech":
            v = random.choice(["USE", "FIX", "CREATE", "SEND", "DEVELOP"])
            o = random.choice(tech + things)
        elif cat == "study":
            v = random.choice(["READ", "LEARN", "STUDY", "WRITE"])
            o = random.choice(things + tech)
        elif cat == "mental":
            v = random.choice(["NEED", "WANT", "FIND", "CALL", "KNOW", "LIKE"])
            o = random.choice(things + tech + places)
        elif cat == "eat":
            v = "EAT"
            o = random.choice(["FOOD", "APPLE"])
        elif cat == "drink":
            v = "DRINK"
            o = random.choice(["WATER", "COFFEE"])
        else:  # buy
            v = random.choice(["BUY", "GIVE", "TAKE", "RECEIVE"])
            o = random.choice(buyable)
        return f"{s} {v} {o}", "subject + verb + object (no time)"

    if pattern == "sov_osv":
        s = random.choice(pronouns)
        cat = random.choice(["visit", "tech", "study", "buy"])
        if cat == "visit":
            v = random.choice(["GO", "WALK", "COME"])
            o = random.choice(places)
        elif cat == "tech":
            v = random.choice(["USE", "FIX", "CREATE"])
            o = random.choice(tech + things)
        elif cat == "study":
            v = random.choice(["READ", "STUDY"])
            o = random.choice(things + tech)
        else:  # buy
            v = random.choice(["BUY", "GIVE"])
            o = random.choice(buyable)
        return f"{o} {s} {v}", "OSV topicalization"

    if pattern == "sov_ovs":
        s = random.choice(pronouns)
        cat = random.choice(["visit", "tech", "study", "buy"])
        if cat == "visit":
            v = random.choice(["GO", "WALK", "COME"])
            o = random.choice(places)
        elif cat == "tech":
            v = random.choice(["USE", "FIX", "CREATE"])
            o = random.choice(tech + things)
        elif cat == "study":
            v = random.choice(["READ", "STUDY"])
            o = random.choice(things + tech)
        else:  # buy
            v = random.choice(["BUY", "GIVE"])
            o = random.choice(buyable)
        return f"{o} {v} {s}", "OVS topicalization"

    if pattern == "feeling":
        t = random.choice(times)
        s = random.choice(pronouns)
        f = random.choice(feelings)
        return f"{t} {s} FEEL {f}", "feeling sentence"

    if pattern == "wh_what":
        s = random.choice(pronouns)
        return f"{s} WANT WHAT", "what-question"

    if pattern == "wh_where":
        o = random.choice(places + things)
        art = random.choice(["MY", "THE"])
        return f"{art} {o} WHERE", "where-question"

    if pattern == "imperative":
        return random.choice([
            "HELP ME PLEASE",
            f"PLEASE GO",
            f"PLEASE WAIT",
            f"COME {random.choice(places)}",
            f"STOP TALK",
        ]), "imperative"


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — BLEU METRICS
# ══════════════════════════════════════════════════════════════════

def tokenize(sentence):
    """Lowercase, strip punctuation, split into words."""
    for ch in ".,?!;:\"'":
        sentence = sentence.replace(ch, "")
    return sentence.lower().split()

def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def bleu_n(references, hypotheses, n):
    """Corpus-level BLEU-n (no brevity penalty for simplicity)."""
    clipped_matches = 0
    total_hyp_ngrams = 0
    for ref, hyp in zip(references, hypotheses):
        ref_tok = tokenize(ref)
        hyp_tok = tokenize(hyp)
        ref_counts = ngram_counts(ref_tok, n)
        hyp_counts = ngram_counts(hyp_tok, n)
        for gram, cnt in hyp_counts.items():
            clipped_matches += min(cnt, ref_counts.get(gram, 0))
        total_hyp_ngrams += max(0, len(hyp_tok) - n + 1)
    if total_hyp_ngrams == 0:
        return 0.0
    return clipped_matches / total_hyp_ngrams

# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — MAIN EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════

def run_evaluation():
    # ── Load model ────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at '{MODEL_PATH}'")
        print("   Update MODEL_PATH at the top of this script.")
        return

    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"✅ Model loaded on {device}!\n" + "=" * 55)

    references    = []   # ground-truth English (not available — we store the gloss instead)
    hypotheses    = []   # model output
    exact_matches = 0

    # ── Known ground-truth pairs for exact-match scoring ─────────
    # 100 manually verified gloss → English pairs covering all sentence types.
    known_pairs = [
        # ── Time + SVO ─────────────────────────────────────────────
        ("YESTERDAY I BUY FOOD",        "Yesterday, I bought some food."),
        ("TOMORROW WE GO SCHOOL",       "Tomorrow, we will go to the school."),
        ("NOW HE FEEL HAPPY",           "Now, he is feeling happy."),
        ("TODAY SHE DRINK WATER",       "Today, she is drinking some water."),
        ("MORNING I GO HOSPITAL",       "In the morning, I am going to the hospital."),
        ("NIGHT YOU EAT FOOD",          "At night, you are eating some food."),
        ("AFTERNOON WE STUDY CODE",     "In the afternoon, we are studying the code."),
        ("YESTERDAY HE GO SCHOOL",      "Yesterday, he went to the school."),
        ("TOMORROW I DRINK COFFEE",     "Tomorrow, I will drink some coffee."),
        ("TODAY THEY GO MARKET",        "Today, they are going to the market."),
        ("MORNING SHE EAT FOOD",        "In the morning, she is eating some food."),
        ("NIGHT WE GO HOSPITAL",        "At night, we are going to the hospital."),
        ("YESTERDAY YOU BUY PHONE",     "Yesterday, you bought the phone."),
        ("TOMORROW HE STUDY BOOK",      "Tomorrow, he will study the book."),
        ("TODAY I GO STORE",            "Today, I am going to the store."),
        ("AFTERNOON THEY DRINK WATER",  "In the afternoon, they are drinking some water."),
        ("MORNING WE GO OFFICE",        "In the morning, we are going to the office."),
        ("YESTERDAY SHE DRINK COFFEE",  "Yesterday, she drank some coffee."),
        ("NOW THEY GO HOSPITAL",        "Now, they are going to the hospital."),
        ("TODAY HE USE PHONE",          "Today, he is using the phone."),
        # ── Feelings ───────────────────────────────────────────────
        ("TODAY I FEEL HAPPY",          "Today, I am feeling happy."),
        ("YESTERDAY SHE FEEL SAD",      "Yesterday, she felt sad."),
        ("TOMORROW YOU FEEL EXCITED",   "Tomorrow, you will feel excited."),
        ("NOW WE FEEL TIRED",           "Now, we are feeling tired."),
        ("MORNING HE FEEL ANGRY",       "In the morning, he is feeling angry."),
        ("NIGHT THEY FEEL SCARED",      "At night, they are feeling scared."),
        ("TODAY SHE FEEL SICK",         "Today, she is feeling sick."),
        ("YESTERDAY I FEEL BORED",      "Yesterday, I felt bored."),
        ("NOW YOU FEEL SAD",            "Now, you are feeling sad."),
        ("AFTERNOON WE FEEL TIRED",     "In the afternoon, we are feeling tired."),
        # ── No-time SVO ────────────────────────────────────────────
        ("I DRINK WATER",               "I am drinking some water."),
        ("SHE NEED BOOK",               "She needs the book."),
        ("HE NEED BOOK",                "He needs the book."),
        ("NOW THEY GO HOSPITAL",        "Now, they are going to the hospital."),
        ("WE STUDY CODE",               "We are studying the code."),
        ("YOU USE PHONE",               "You are using the phone."),
        ("DOCTOR EAT FOOD",             "The doctor is eating some food."),
        ("TEACHER GO SCHOOL",           "The teacher is going to the school."),
        ("STUDENT READ BOOK",           "The student is reading the book."),
        ("FATHER DRINK COFFEE",         "The father is drinking some coffee."),
        ("MOTHER GO MARKET",            "The mother is going to the market."),
        ("BOSS USE COMPUTER",           "The boss is using the computer."),
        ("FRIEND GO STORE",             "The friend is going to the store."),
        ("WORKER USE PHONE",            "The worker is using the phone."),
        ("CHILD EAT FOOD",              "The child is eating some food."),
        # ── OSV topicalization (OBJECT SUBJECT VERB) ───────────────
        ("STORE GO I",                  "I am going to the store."),
        ("SCHOOL WE GO",                "We are going to the school."),
        ("HOSPITAL HE GO",              "He is going to the hospital."),
        ("BOOK I READ",                 "As for the book, I am reading it."),
        ("CODE SHE USE",                "As for the code, she is using it."),
        ("PHONE YOU USE",               "As for the phone, you are using it."),
        ("MARKET THEY GO",              "They are going to the market."),
        ("OFFICE I GO",                 "I am going to the office."),
        ("LAPTOP HE USE",               "As for the laptop, he is using it."),
        ("PROJECT WE FIX",              "We are fixing the project."),
        # ── OVS topicalization (OBJECT VERB SUBJECT) ───────────────
        ("STORE GO I",                  "I am going to the store."),
        ("SCHOOL GO WE",                "We are going to the school."),
        ("BOOK READ I",                 "I am reading the book."),
        ("HOSPITAL GO HE",              "He is going to the hospital."),
        ("CODE USE SHE",                "She is using the code."),
        ("MARKET GO THEY",              "They are going to the market."),
        ("LAPTOP USE HE",               "He is using the laptop."),
        ("OFFICE GO I",                 "I am going to the office."),
        ("PROJECT FIX SHE",             "She is fixing the project."),
        ("PHONE USE YOU",               "You are using the phone."),
        # ── WH-questions ───────────────────────────────────────────
        ("NAME WHAT YOU",               "What is your name?"),
        ("NAME WHAT HE",                "What is his name?"),
        ("NAME WHAT SHE",               "What is her name?"),
        ("I WANT WHAT",                 "What do I want?"),
        ("YOU WANT WHAT",               "What do you want?"),
        ("HE WANT WHAT",                "What does he want?"),
        ("SHE WANT WHAT",               "What does she want?"),
        ("THEY WANT WHAT",              "What do they want?"),
        ("WE WANT WHAT",                "What do we want?"),
        ("MY BOOK WHERE",               "Where is my book?"),
        ("MY PHONE WHERE",              "Where is my phone?"),
        ("MY SCHOOL WHERE",             "Where is my school?"),
        ("THE STORE WHERE",             "Where is the store?"),
        ("THE HOSPITAL WHERE",          "Where is the hospital?"),
        ("WHO DOCTOR",                  "Who is the doctor?"),
        ("WHO TEACHER",                 "Who is the teacher?"),
        # ── Imperatives ────────────────────────────────────────────
        ("HELP ME PLEASE",              "Please help me."),
        ("PLEASE WAIT",                 "Please wait."),
        ("PLEASE GO",                   "Please go."),
        ("PLEASE COME",                 "Please come."),
        ("PLEASE STOP",                 "Please stop."),
        ("PLEASE LISTEN",               "Please listen."),
        ("PLEASE CALL",                 "Please call."),
        ("PLEASE READ",                 "Please read."),
        ("PLEASE WRITE",                "Please write."),
        ("STOP TALK",                   "Stop talking."),
        ("COME HOSPITAL",               "Come to the hospital."),
        ("COME SCHOOL",                 "Come to the school."),
        ("COME STORE",                  "Come to the store."),
        ("COME OFFICE",                 "Come to the office."),
        # ── Mental / opinion ───────────────────────────────────────
        ("I NEED PHONE",                "I need the phone."),
        ("SHE LIKE BOOK",               "She likes the book."),
        ("HE KNOW CODE",                "He knows the code."),
        ("THEY WANT LAPTOP",            "They want the laptop."),
        ("WE NEED WATER",               "We need some water."),
    ]

    print(f"{'GLOSS':<35} {'→ OUTPUT'}")
    print("-" * 75)

    start_time  = time.time()
    all_glosses = []
    all_outputs = []

    # ── Run known pairs first ──────────────────────────────────────
    print("\n── KNOWN TEST CASES ──────────────────────────────────────")
    known_correct = 0
    for gloss, expected in known_pairs:
        ids    = tokenizer(PREFIX + gloss, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                                 num_beams=NUM_BEAMS, early_stopping=True)
        result = tokenizer.decode(out[0], skip_special_tokens=True)
        match  = "✅" if result.strip().lower() == expected.strip().lower() else "❌"
        if result.strip().lower() == expected.strip().lower():
            known_correct += 1
        print(f"  {match} [{gloss}]")
        print(f"       Got     : {result}")
        print(f"       Expected: {expected}\n")
        all_glosses.append(gloss)
        all_outputs.append(result)

    print(f"Known pairs: {known_correct}/{len(known_pairs)} exact matches\n")

    # ── Run random gloss sentences ─────────────────────────────────
    print(f"── RANDOM GLOSS EVALUATION ({NUM_EVAL} sentences) ────────────")
    random_refs   = []
    random_hyps   = []

    for i in range(1, NUM_EVAL + 1):
        result_pair = random_gloss()
        if result_pair is None:
            continue
        gloss, pattern_hint = result_pair

        ids = tokenizer(PREFIX + gloss, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                                 num_beams=NUM_BEAMS, early_stopping=True)
        result = tokenizer.decode(out[0], skip_special_tokens=True)

        random_refs.append(gloss)   # gloss used as proxy reference for BLEU
        random_hyps.append(result)
        all_glosses.append(gloss)
        all_outputs.append(result)

    # ── BLEU on random outputs ─────────────────────────────────────
    # Note: we measure self-BLEU against gloss tokens as a fluency proxy,
    # not true BLEU (which needs human references). This shows the model
    # is producing diverse, non-degenerate output.
    b1 = bleu_n(random_refs, random_hyps, 1)
    b2 = bleu_n(random_refs, random_hyps, 2)

    # Diversity: unique outputs / total
    unique_ratio = len(set(random_hyps)) / len(random_hyps) * 100

    # Avg output length
    avg_len = sum(len(tokenize(h)) for h in random_hyps) / len(random_hyps)

    elapsed = time.time() - start_time

    # ── All random outputs table ───────────────────────────────────
    print(f"\n── ALL RANDOM OUTPUTS ({NUM_EVAL} sentences) ──────────────────")
    print(f"  {'GLOSS':<35} {'OUTPUT'}")
    print("  " + "-" * 70)
    for g, o in zip(random_refs, random_hyps):
        print(f"  {g:<35} {o}")

    print("\n" + "═" * 55)
    print("  📊 STAGE 3 EVALUATION RESULTS")
    print("═" * 55)
    print(f"  Known pair exact match  : {known_correct}/{len(known_pairs)}"
          f"  ({known_correct/len(known_pairs)*100:.0f}%)")
    print(f"  Random sentences tested : {NUM_EVAL}")
    print(f"  Unique outputs          : {unique_ratio:.1f}%  (higher = less repetition)")
    print(f"  Avg output length       : {avg_len:.1f} words")
    print(f"  Self-BLEU-1 (unigram)   : {b1*100:.1f}%  (overlap with gloss tokens)")
    print(f"  Self-BLEU-2 (bigram)    : {b2*100:.1f}%")
    print(f"  Time taken              : {elapsed:.1f}s")
    print("═" * 55)

if __name__ == "__main__":
    run_evaluation()