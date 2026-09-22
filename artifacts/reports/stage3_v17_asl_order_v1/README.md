# Stage 3 ASL-order renderer v1

Retrains Stage 3 so it reads ASL word order and ignores low-confidence recognizer
noise. It replaces `stage3_v17_t5_efficient_tiny_locked100_v1` as a candidate; nothing
is promoted and the live default is unchanged.

## Why the deployed renderer was wrong

The deployed checkpoint is a monotone function-word inserter, not a translator. Its
training corpus (`artifacts/reports/slt_stage3_dataset_final.csv`, 15,843 rows) is
almost entirely order-preserving: of 11,840 rows whose glosses can all be located in
the target, 11,745 keep gloss order and 95 reorder. Only 742 rows are fully inside the
locked 100, and those are template-generated `TIME SUBJ VERB OBJ` already in English
order. Its contract also forbade omission, so it rendered every recognition error.

Measured consequences on held-out rows: it drops 3.03% of genuine glosses, keeps 57.50%
of injected noise glosses, and 231 of 958 of its outputs fail a content check that
matches each word back to the glosses that produced it — 211 inventing content and 20
losing the sentence subject. That last class is the reported live failure where
`I TIRED` renders as a question about someone else.

## What replaced it

`active/v17/asl_corpus_v17.py` generates gloss sequences in genuine ASL order:
topic-comment and object-fronting, time-fronting, wh-final and wh-initial questions,
`NO` negation, copula-less predicates, coordinated states, verb complements, greetings
and conjoined clauses. The locked vocabulary has no copula, no articles, no tense
morphology and no NOT, so supplying those is the renderer's job.

Noise is injected into 34% of rows as either a spurious gloss or an adjacent duplicate,
and the English target omits it. Confidences are drawn from bands that deliberately
overlap: genuine signs really are accepted down to 0.250 in `artifacts/app_sessions/`
(2,006 accepted predictions, median 0.615, p10 0.315), so a disjoint split would teach
"low score means drop" as a perfect rule and the renderer would delete real content
whenever the recognizer was merely unsure. A sub-0.40 gloss is genuine about 60% of the
time, which forces the model to weigh plausibility as well as evidence.

English targets come from DeepSeek V4 Flash over OpenRouter, batched 40 per call and
cached. Every target is checked against its glosses before admission, and a rejected
row gets exactly one corrective retry. 13,029 of 13,096 rows were admitted; all 67
rejections were invented content. Total API cost was $0.31 across all builds.

Stage 3 now also receives evidence. Each gloss is preceded by a confidence bucket token
(`hi`/`mid`/`lo`, `active/v17/stage3_asl_encoding_v17.py`). Per-gloss word tags cost 11
tokens for a five-gloss buffer against 6 untagged; inline punctuation markers fragment
the gloss itself and a tilde suffix tokenizes to `<unk>`. A missing score buckets to
`hi`, so a caller without evidence never has its glosses discarded.

## Results

Initialization is the original grammar-correction base, not the deployed checkpoint, so
the monotone bias is not inherited. Selection used validation BLEU only; the test split
was opened once after selection, and the deployed checkpoint was scored on the same rows
with the plain input it was trained on.

| Test, 1,261 rows | retrained | deployed |
| --- | ---: | ---: |
| BLEU | 92.74 | 43.58 |
| chrF2++ | 96.60 | 69.91 |
| normalized exact | 0.905 | 0.260 |
| noise suppressed | 0.992 | 0.589 |
| negation preserved | 1.000 | 0.990 |
| genuine glosses dropped | 10/4,784 (0.21%) | 148/4,784 (3.09%) |
| noise glosses kept | 6/494 (1.21%) | 253/494 (51.21%) |
| content-check failures | 7/1,261 (0.56%) | 320/1,261 (25.4%) |
| words invented from no gloss at all | 1/1,261 | 57/1,261 |

| BLEU by slice | rows | retrained | deployed |
| --- | ---: | ---: | ---: |
| reordering | 251 | 89.43 | 7.53 |
| noisy | 494 | 92.95 | 36.63 |
| clean | 767 | 92.60 | 48.54 |
| time-fronted | 190 | 82.87 | 52.02 |
| negation | 76 | 83.42 | 68.05 |
| SVO regression guard | 178 | 96.42 | 55.38 |

The SVO row is the guard against trading one failure for another: the cases the deployed
model already handled did not regress.

Negation is reported separately because it is the one error that inverts meaning rather
than degrading it. An earlier 300-row pilot scored BLEU 81.58 and still turned
`I NO LIKE WATER` into "I like the water"; that disappeared at full corpus size, but the
metric stays because BLEU hides it.

## Two defects found by running it live, and fixed

Making this the live default surfaced two failures that held-out BLEU never showed.

**A confidently-recognized gloss can still have to be dropped.** Session
`20260922_122517_239185` produced `I SICK MY HUNGRY` at confidences
0.97/0.89/0.91/0.52 and the renderer answered "I am sick, and my family is hungry." MY
scored 0.91, so no confidence threshold could have caught it: the gloss was stranded,
a possessive with no noun to possess, and the model supplied the commonest possessed
noun. The corpus now carries stranded determiners as a distinct artifact class whose
confidence is drawn from the genuine band, so omission cannot be learned as a function
of the score. A possessive before a noun stays untouched — `MY MOTHER SICK` is
unchanged. The generated rows are restricted to a possessive before an adjective,
because "less time" and "more water" are ordinary English and treating those as
artifacts would teach the renderer to delete real modifiers.

**Verbless wh-questions were missing entirely.** `WHO YOU` became "Who do you see?" The
locked vocabulary has no copula, and the user's real sessions are full of wh plus a
noun phrase with no verb — `WHO YOU`, `WHERE YOUR FAMILY`, `WHAT TIME TOMORROW`,
`HOW YOUR DAY`. With no such rows the model invented a verb instead of supplying "is".

Both are now in the probe set, so neither can return silently.

## Limits

BLEU here measures agreement with model-written English on rule-generated gloss
sequences. It is not human-judged translation quality, the corpus is not a genuine ASL
corpus, and no Deaf or fluent signer reviewed these targets. No Citizen, SemLex,
2M-Flores or other protected split was accessed, and no recognition model changed.

Two probe inputs remain weak, both long multi-clause buffers with injected noise:
`I UNDERSTAND SIGN *TIME LANGUAGE WORK TIME` and
`WHO YOU HAPPY I UNDERSTAND *YEAR SIGN LANGUAGE` -> "Who do you be happy?". Those inputs
are themselves garbled recognizer output, but the second is ungrammatical and drops a
whole clause, which is a real weakness on long noisy buffers.

Twenty ASL-structure probes read correctly except `WHAT WORK TIME`, which drops TIME,
and 9 of 11 noise probes are correct. Those probes have no automatic score and are for reading.

Single-gloss buffers are covered because the recognizer really emits them — a saved
session produced a bare `I I`, and the 2026-09-16 review recorded the deployed renderer
turning `I` into "Is that?". The deployed checkpoint also gives `I SICK` -> "Is I sick?"
and `YOU` -> "Thank you."; the retrained one gives "I.", "I am sick." and "You." An
earlier build of this model instead answered `I` with "I am happy.", inventing a
predicate, which is why 96 single-gloss rows were added.

Greedy generation measures 26.1 ms per sentence on CPU for a five-gloss buffer.

## Files

- model: `../../models/stage3_v17_asl_order_v1/`
- corpus: `data/local/stage3_asl_corpus_v17/` (corpus, English cache, rejects, manifest)
- `result.json` — full provenance, per-epoch history, slices
- `test_predictions.jsonl` — every test row with both models' output
- `probe.jsonl` — structure, noise and real-session probes

## Live opt-in

```
venv/bin/python scripts/app_shell_v17.py \
  --stage3-checkpoint artifacts/models/stage3_v17_asl_order_v1 \
  --stage3-encoding evidence
```

`--stage3-encoding` defaults to `plain`, which is the deployed behaviour byte for byte:
same input string and the same 48-token window, since widening it changes deployed
output once an input passes about 41 tokens.
