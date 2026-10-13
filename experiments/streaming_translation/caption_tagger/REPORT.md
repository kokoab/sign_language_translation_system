# Phase 2: subtitle-style captions from the gloss stream — 2026-10-13 PHT

## Decision

Your hypothesis holds for this vocabulary. English is mostly each recognized sign
in order, plus a small closed set of inserted words. A 1.6M-parameter learned tagger
captions while the person signs. It never erases text, keeps every sign in order,
never shows a word for a sign that wasn't signed, and kept negation in every
held-out case. It is not ready to replace Finish-time translation everywhere. Three
ASL constructions still need a bounded, learned revision: a time sign after the
verb, a WH sign at the end, and NO at the end. No ATLAS file, app or deployment
changed. Training used 19.3 of the 30 authorized minutes.

Literature and ASL-grammar background: `RESEARCH.md`. Frozen plan: `PLAN.md`.

## A. Corpus evidence (54K-row Stage3 corpus, train split, 105,578 sentences)

Each English sentence was aligned to its gloss span with a monotone matcher over
a fixed surface-form dictionary (`forms.py`, e.g. GO→go/goes/went/going).

| Sentence kind | Share |
|---|---:|
| Glosses in sign order, insertions only | 77.2% |
| Same, plus a deliberate noise gloss dropped | 3.0% |
| English moved a WH sign (YOU GO WHERE → Where do you go?) | 8.3% |
| English moved NO (WE READY NO → We are not ready) | 2.9% |
| English moved a time sign (stylistic; either order is fine English) | 2.7% |
| Other moves (mostly optional, e.g. "Maybe you need…") | 5.9% |

The inserted material is small and closed:
- 64% of glosses need nothing inserted before them.
- 40% of English tokens, counting punctuation, are insertions.
- Only 72 distinct inserted word types occur. The top 20 cover 92.7% of insertions
  and the top 30 cover 97.3%: to, the, is, at, and, do, will, for, a, are, can, am,
  with, but, this, it, me, about, was, that…
- 170 insertion phrases cover 99.2% of insertion slots; 55 sentence-end patterns
  cover 99.8%.

Lookahead needed (count-based backoff predictor, train → validation): exact
per-sign decision accuracy is 77.5% with no lookahead and 85.1% with one sign.
Insertion and word form are mostly decidable from signs already seen (89% / 96% at
k=0). Sentence punctuation gains most from one more sign (88% → 94%).

## B. Model

A causal transformer: 3 layers, d=192, 1.58M parameters. Gloss i's insertion,
form and punctuation are read from hidden states at i and at i+k. A sign's rendering
therefore depends only on signs up to i+k, which is exactly what is available when
sign i+k is recognized. 3,476 trained-weight checks confirm this: changing future
signs never changed an earlier rendering. Insertion and form are decoded jointly
over pairs seen ≥5 times in training, so "going" always comes with am/is/are.
Corpus train split only; epochs chosen on corpus validation. Exact per-sign tag
accuracy on validation:

| k=0 | k=1 | k=2 | full context (3 epochs only) |
|---:|---:|---:|---:|
| 75.7% | 83.0% | 84.2% | 81.8% |

Training wall clock: k1 223 s, k0 466 s, k2 305 s, full 109 s, plus a discarded
57 s NaN epoch (mask bug fixed) = 1,160 s. The full-context model had only 3 epochs,
so it is not a clean upper bound.

## C. Held-out English (300 generated sessions, never in training)

References are DeepSeek English, which often moves time and WH words. Monotone
captions are penalized for word order that is still acceptable English.

| System | BLEU | chrF++ | Exact | All signs present | All in sign order | NO kept | Outputs with an inserted pronoun |
|---|---:|---:|---:|---:|---:|---:|---:|
| Literal words | 11.5 | 49.4 | 0 | 295 | 295 | 42/42 | 0 |
| Tagger k=0 | 54.7 | 74.4 | 11 | 300 | 300 | 42/42 | 16 |
| **Tagger k=1** | 63.4 | 78.8 | 32 | **300** | **300** | 42/42 | 21 |
| Tagger k=2 | 65.3 | 79.7 | 37 | 300 | 300 | 42/42 | 29 |
| Tagger k=1 + sentence-end T5 rewrite | 73.7 | 84.0 | 77 | 300 | 210 | 42/42 | 44 |
| Installed tiny T5, whole input at Finish | 75.1 | 85.1 | 88 | 290 | 178 | 42/42 | 34 |

The 237-session subset with no sentence in training gives the same picture (k=1
64.3/79.3; T5 75.1/85.0). chrF++ by sentence type, k=1 / k=1+rewrite / T5:
statement 85.7 / 87.0 / 87.5, request 86.4 / 88.2 / 89.8, yes-no 85.0 / 85.9 / 86.7,
greeting 84.1 / 93.4 / 92.9, negation 67.2 / 83.3 / 84.8, **WH question 54.9 / 88.9 / 90.6**.
The gap to T5 is concentrated in WH-final and late-NO sentences.

"NO kept" is lexical, not semantic. NO I TIRED (reference "No, I am tired.") becomes
"I am not tired" under both T5 and the rewrite. I KNOW YOU LIKE NO becomes "I do not
know you like it" under T5. A word-level check cannot catch these meaning flips.

## D. Live captions on the 20 recorded phone traces

Original recognition arrival times; desktop CPU; Finish timestamps were not logged.

| Policy | Normalized erasure | Median first text | Median sign→final caption | Max |
|---|---:|---:|---:|---:|
| Phase 1 tiny T5, revisable re-translation | 0.91 | 0.012 s | — | — |
| Phase 1, 1.5 s pause timer | changed 12/20 outputs | | | |
| Commit k=1 (never revised) | **0** | 1.17 s | 1.04 s | 6.5 s |
| Commit k=2 | 0 | 2.18 s | 1.81 s | 8.5 s |
| **Draft newest sign with k=0, commit with k=1** | **0.10** | 0.001 s | 1.04 s | 6.5 s |
| Draft k=0, commit k=2 | 0.21 | 0.001 s | 1.81 s | 8.5 s |
| Commit k=1 + sentence-end rewrite | 0.16 | 1.17 s | 1.04 s | 6.5 s |

Draft + commit k=1 stays below the 0.2 low-revision regime from Arivazhagan et al.
2020. Only the newest word can change, and only once. The maximum lag happens when
the next sign is slow to arrive; it is flushed at the (oracle) end of the utterance.
A tagger update costs about 1–3 ms on CPU.

Your example, commit k=1:
`I → I go → I go to the hospital → I go to the hospital tomorrow → I go to the hospital tomorrow morning.`
(The sentence-end rewrite gives "I am going to the hospital tomorrow morning.")
TOMORROW MORNING I GO HOSPITAL becomes "Tomorrow morning I will go to the hospital."
YESTERDAY I GO HOSPITAL becomes "Yesterday I went to the hospital."
Phase-1 failures now captioned correctly: HOW YOU (was "How. You."),
HELLO MY FRIEND HOW YOU (was "Hello, how are my friend?"), I → "I" (was "Is it
here?"), I GIVE YOU WATER → "I will give you water." (Qwen reversed the roles).

## E. The full 100-class space, not only recorded sessions

| Space | Possible | Seen in training |
|---|---:|---:|
| Classes (incl. FS0 slot) | 101 | all (rarest LISTEN, 309 occurrences) |
| Ordered sign pairs | 10,201 | 7,625 (74.7%) |
| Ordered sign triples | 1,030,301 | 59,369 (5.8%) |
| 5-sign sequences | 1.05×10¹⁰ | 25,734 distinct training sentences in total |

The held-out set hides this. Only 5 of its 784 sentences contain an unseen sign
pair (chrF++ there: k=1 69.9, T5 81.1, n=5, too few to conclude).

Stress test: 3,200 new combinations (`coverage.py`). 2,400 come from 16 slot
templates (time / subject / verb / object / WH / negation / discourse) drawn uniformly
across all 101 classes, and 800 are random orders. Every class appears at least 47
times; 2,965 sequences never occur in training and 912 contain an unseen pair.
There are no references, so only fidelity is scored:

| | Tagger k=1 | k=1 + rewrite | Tiny T5 |
|---|---:|---:|---:|
| All signs present | 3,179 / 3,200 | 3,188 | 2,693 |
| … on random orders | 779 / 800 | 788 | 444 |
| Words for signs not in input | 0 | 0 | 70 outputs |
| NO kept | 539 / 539 | 539 | 528 |
| Classes ever dropped | 11 / 101 | 6 | 97 / 101 |
| Worst class drop rate | EAT 7.1% | EAT 5.1% | DAY 19.8% |
| Outputs with an inserted pronoun | 223 | 486 | 445 |

The tagger's structure (each sign → its own form plus local insertions) carries
over to unseen combinations. T5 increasingly drops or substitutes signs. The
tagger's drops are a learned "noise sign" behavior from the corpus; EAT was already
a known corpus defect on 2026-10-11. The gate rejected 440 of 3,279 rewritten
sentences (13%) on the stress set, against 7 of 699 on held-out. Samples, including
ungrammatical random orders, are in `stress_samples.csv`.

Tense against time signs (`tense_check.py`, conflict = past form with TOMORROW, or
"will" with YESTERDAY):

| | k=0 | k=1 | k=2 | k=1 + rewrite | T5 |
|---|---:|---:|---:|---:|---:|
| Time sign before the verb (97) | 0 | 0 | 0 | 0 | 1 |
| Time sign after the verb (49) | 4 | **10** | 1 | 0 | 0 |

A conflict undercounts errors: "They go to school yesterday" is not flagged. The
recorded phone trace I GO DOCTOR TOMORROW MORNING gives "I went to the doctor tomorrow
morning" at k=1; k=2 gives "I will go…".

## Failures to fix next

1. **Late time sign.** The verb form is committed before TOMORROW/YESTERDAY arrives.
2. **WH-final.** Some echo forms read well ("You go to the hospital when?"); others
   don't ("They are hungry, who", "You where").
3. **Late NO.** "I want water, no." / "We write the time. No." are ambiguous. The
   live caption is wrong until NO arrives, an inherent property of the sign order.
4. **Inserted pronouns.** 7% of stress outputs, e.g. HELLO FRIEND I HELP →
   "I will help her." This is the remaining invention channel.
5. **Learned sign drops.** EAT, USE, TALK.
6. **Split fingerspelling.** fs-GE fs-LO → "Ge is Lo".
7. **The T5 sentence-end rewrite raises scores but flips meaning.** NO I TIRED →
   "not tired"; SORRY THEY → "Sorry, they are sorry". Its gate is lexical and known
   to accept role/scope corruption. Do not adopt it as is.

## Recommended next experiment (not started)

Keep the append-only k=1 tagger with a k=0 draft of the newest sign as the live
caption. Add one learned, bounded revision label instead of a free rewrite. When a
clause closes, allow (a) a verb-form change when a later time sign arrives, (b)
WH-fronting, and (c) negation placement. Each revision must stay within the closed
insertion/form inventory, so no sign can be dropped or invented. Targets come from
the 19.8% of corpus sentences now masked. Evaluate on fresh combinations covering
all 100 classes with independent references, ideally Deaf/fluent review; the
generated references and the lexical checks used here are not ASL ground truth.
Speak only committed sentences. Mobile size/latency remain unmeasured; the model is
1.6M parameters, about 6 MB FP32.

## Scope and limits

Text-level replay of recognized glosses, not camera accuracy or a signer study.
Corpus and held-out references are LLM-written. The 24 declared diagnostics and the
stress set are development evidence. Two bugs were found after the first evaluation
run and fixed without changing any model decision: duplicated text in the rewrite
policy when the last sentence had no end punctuation, and doubled punctuation in
the renderer. All tables above come from the corrected rerun. No API spending,
dataset acquisition, protected test access or ATLAS writes.

## Files

`forms.py` form dictionary/tokenizer · `align.py` monotone alignment ·
`build_data.py` labels + `corpus_stats.json` · `lookahead_counts.py` ·
`tagger.py` model/training/decoding · `evaluate.py` → `eval_*.json`, `heldout_*.csv` ·
`coverage.py` → `coverage.json`, `stress_samples.csv` · `tense_check.py` ·
`verify.py` → `integrity.json` (corpus hash, causal checks, file hashes) ·
`training_log.json`, `budget.log`. `data.json` and `tagger_k*.pt` are git-ignored;
regenerate them with `build_data.py`, `tagger.py prepare` and `run_training.sh`
(then `tagger.py full 3`).
