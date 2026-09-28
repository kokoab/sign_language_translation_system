# Stage 3 multi-sentence evaluation — v2 baseline — 2026-09-29

User direction (2026-09-29): Stage 3 must handle **multiple sentences in one Finish**, respond in
**under one second**, and be evaluated without a human reviewer, using **DeepSeek via OpenRouter**
for data. This report covers step 1: build the held-out set and score the installed
`stage3_composition_v17_20260929_v2` renderer. No training, no model or app change.

Runner: `scripts/eval_stage3_multisentence_v17.py` (`build`, `render`, `judge`, `report`).
API cache (replays at zero cost): `data/local/stage3_multisentence_eval_v17/api_cache.jsonl`.

## Evaluation set

- **English-first generation.** DeepSeek V4 Flash wrote everyday sessions of 2–4 related sentences,
  each with an ASL gloss using only the locked 100 signs (+ `FS0` name slot). 30 topics × 12 structure
  prompts (verb chains, embedded clauses, time-first, wh-final, NO, requests, topic-comment,
  greetings+questions, yes/no questions, MAYBE/FEEL/THINK, names, possessives).
- **Rule checks, both directions** (corpus lemma table, `active/v17/asl_corpus_v17.py`): every gloss must
  be realized in the English and every English content word must come from a gloss. Glossed function
  words (BUT, FOR) are dropped; anything else outside the vocabulary rejects the session.
  Two local lemma additions: HE also covers she/her; DIFFERENT covers "different".
- **DeepSeek vetting**: nonsense sessions dropped; grammar and tense fixed only, and the fixed English
  must pass the rule check again.
- **Held-out from training**: any session whose gloss sequence occurs in either composition training
  corpus is removed.

| Stage | Sessions |
|---|---:|
| Parsed from 300 generation calls | 2,385 |
| Passed rule checks | 610 |
| Passed vetting | 582 |
| Sampled for the set | **300** |

300 sessions: 150 with 2 sentences, 116 with 3, 34 with 4. Mean 10.7 signs (max 24).
237 contain no sentence whose gloss sequence appears in training; 111 contain a yes/no question;
35 contain a name slot. Plus 23 unique saved phone Finish inputs (reference-free; 11 of them are in
training, so that slice is inflated).
Longest session is 53 input tokens, under the 64-token context; no session needed the literal fallback.

Cost: about $0.15 for generation and vetting, $0.024 for judging.

## Scoring

- **Faithful**: every sign realized, no content word invented (same lemma check).
- **Judge**: DeepSeek, temperature 0, 2 = same meaning and grammatical, 1 = understandable with a minor
  error, 0 = wrong meaning, missing clause/content, invented content or unclear. A statement
  rendering of a yes/no question is accepted when the gloss has no question sign.
- Controls: the reference scores 2.00 on all 300 (judge agrees with its own references; self-grading bias
  is possible). Word-by-word literal rendering is the floor.
- BLEU/chrF with sacrebleu, secondary.

`v2_full` is the deployed path (whole Finish buffer, one call). `v2_split` feeds each reference sentence
separately (oracle sentence boundaries), to separate composition failures from sentence splitting.

## Results (300 generated sessions)

| Renderer | Faithful | Drops ≥1 sign | Invents | Judge mean | Judged 2 | Judged 0 | BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2_full (deployed) | 31% | 68% | 8% | 0.74 | **6%** | **31%** | 31 |
| v2_split (oracle boundaries) | 51% | 46% | 8% | 0.87 | 10% | 23% | 41 |
| literal word-by-word | 100% | 0% | 0% | 0.77 | 0% | 23% | 8 |
| reference (control) | 100% | 0% | 0% | 2.00 | 100% | 0% | 100 |

By length (v2_full): 2 sentences 10% judged 2 / 26% judged 0; 3 sentences 2% / 33%;
4 sentences **0% / 47%**, 91% drop a sign. Unseen-only slice (237): 5% / 34%.
Yes/no-question slice: 1% / 32%. Name slice: 0% / 40%.

Most-dropped signs (dropped / sessions containing it): SIGN 26/35, MAYBE 23/28, **NO 21/42**,
YOUR 19/79, LANGUAGE 17/33, SCHOOL 16/56, HELP 15/84, HOME 15/40, SEE 14/30, YOU 13/164.
Dropping NO reverses meaning.

Phone history slice (23, reference-free, half in training): 83% faithful, 83% judged 2.

Desktop CPU (PyTorch) v2_full median 28 ms, p90 37 ms per session. Not phone latency.

## Findings

1. On unseen multi-sentence input the installed renderer is judged no better than reading the signs
   word by word (0.74 vs 0.77); it produces fluent English for only 6% of sessions.
2. Failure grows with the number of sentences, and oracle sentence boundaries only partly help
   (6%→10% fully right), so the model fails at composition inside sentences, not only at splitting.
3. It drops negation in half the sessions that contain NO.
4. The phone-history slice looks good because it overlaps training; it is not evidence of generalization.

## Limits

References, vetting and judge are all DeepSeek; no fluent signer reviewed any gloss or sentence, and the
gloss order is an LLM's idea of ASL. Clean input only (all confidences 0.9, no recognition noise).
Scores measure agreement with DeepSeek on unseen compositions, not ASL translation accuracy.

Files: `eval_set.jsonl` (set, sha256 in `build_summary.json`), `rejected.jsonl`, `renders.jsonl`,
`judgements.jsonl`, `metrics.json`.
