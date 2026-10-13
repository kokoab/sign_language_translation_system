# Phase 2: subtitle-style gloss captions — 2026-10-13 PHT

Continues `../REPORT.md`. User direction: English should behave like a live
subtitle, mostly the exact recognized word plus a small closed set of insertions
(I GO HOSPITAL TOMORROW MORNING → "I'm going to the hospital tomorrow morning"),
not a sentence that must wait for Finish. Do not touch `ATLAS/`.

Scope: research + desktop replay only. No ATLAS writes, app changes, deployment,
API spending, sign-data acquisition or protected visual test access. Existing
artifacts and corpora are read-only. Training uses the remaining 30-minute
adaptation allowance from phase 1 (total wall-clock across all runs, logged).

## Hypothesis

For this 100-gloss vocabulary, English output is close to monotone with the gloss
stream. Each gloss can be rendered as: an insertion before it from a closed phrase
inventory (articles, copula/auxiliaries, infinitive "to", prepositions, "will"),
a surface form of the gloss (go/goes/went/going, I/me), and optional punctuation
after it. If so, a learned tagger with small lookahead can caption while the user
signs, every recognized sign appears exactly once and in order (no dropped NO, no
participant swap, no invented content words beyond the closed inventory).

This is a learned model, not phrase triggers (binding 2026-09-29 decision). The
surface-form inventory is a fixed morphological dictionary like GECToR's verb-form
table; choices are learned from data.

## A. Corpus measurement (train split only)

Align each English sentence to its gloss span (corpus sentence counts) with a
monotone DP over a per-gloss surface-form dictionary. Report: fraction of
sentences fully monotone-alignable; English tokens that are gloss forms versus
insertions; insertion inventory size and coverage of top-N phrases; glosses left
unrealized; types of non-monotone sentences (WH-final, topicalization).
Lookahead need: accuracy of a count-based tag predictor using left context and
right context k = 0, 1, 2, 3, full — trained on train, measured on validation.

## B. Learned caption tagger

Tiny transformer encoder over gloss tokens with a lookahead-k attention mask
(k = 0, 1, 2 and full context as upper bound), predicting per gloss: insertion
before, surface form, and punctuation after. Trained on corpus train split; only
the corpus validation split selects epochs. Non-monotone sentences are excluded from
training labels and counted, not hidden.

Frozen evaluation (run once after training finishes):
1. Held-out multi-sentence set (`artifacts/reports/stage3_multisentence_eval_v17_20260929/eval_set.jsonl`,
   300 generated + 23 phone sessions; DeepSeek references, not ASL ground truth).
   Compare final captions to the installed tiny T5 run here on identical inputs:
   chrF++, BLEU, exact sentence match; fidelity checks (every gloss realized in order,
   negation retained, content words not in the input inserted).
   Report subset whose sentences are all absent from training separately.
2. Phase-1 diagnostics and 20 timed phone traces (`../cases.json`): caption timeline,
   normalized erasure (NE), latency in glosses and seconds, compare with phase-1 T5
   policies.

Caption policies: (i) commit-k: gloss i rendered once gloss i+k arrives, never
revised (NE = 0); end-of-utterance flushes with no-future context. (ii) draft-k0:
k=0 rendering shown immediately, the last ≤k words may be revised.

Limits fixed in advance: corpus references are LLM-written English, not fluent signer
review; these are text-level results over recognized glosses, not camera accuracy.
No threshold tuning on the evaluation sets.
