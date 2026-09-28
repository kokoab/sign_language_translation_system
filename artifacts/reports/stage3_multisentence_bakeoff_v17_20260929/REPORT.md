# Stage 3 multi-sentence renderer — corpus, training and held-out results — 2026-09-29

User decisions this session: multiple sentences per Finish, under one second on the iPhone 13, DeepSeek
via OpenRouter for all data (no human reviewer), incremental translation, each training run under one
hour, corpus spend capped at $2. The user then asked to try the smallest model first and move up in size
only if the held-out scores require it. Pass bar agreed in advance: at least 60% judged fully right,
under 10% judged wrong, and NO never dropped.

**Result: the current tiny T5 size (15.6M), retrained on the new corpus, passes the bar.** No larger model
was trained. After the user reviewed these results it was integrated and installed on the iPhone 13 (see the last section).

## Corpus (`scripts/build_stage3_multisentence_corpus_v17.py`)

English-first DeepSeek generation, same rule checks and vetting as the held-out set, admitted per
sentence (contiguous passing runs kept). Every sentence whose gloss sequence occurs in the held-out set
(9,432 generated sentences) and every row equal to a held-out session (20, phone inputs) was removed.

| | |
|---|---:|
| Generated sentences / passed rule checks | 75,809 / 35,962 |
| Multi-sentence runs kept after vetting | 21,910 |
| Distinct sentences | 17,560 |
| Rows: single / natural runs / composed 2–4 / previous corpus | 17,546 / 7,525 / 23,996 / 4,994 |
| Train / validation rows | 52,961 / 1,100 |
| Rows with one spurious low-score (<0.40) gloss to drop | 3,937 |

Targets are `N: sentence` per sentence (N = input glosses consumed) for incremental locking.
Corpus: `data/local/stage3_multisentence_corpus_v17/corpus.jsonl` (sha256 in `corpus_summary.json`).

**Spend: $2.37, over the user's $2 cap.** The first run stopped at a held-out-overlap assertion after
$1.85 (the check worked; phone inputs have no sentence breakdown). The rerun's budget guard counted only
its own spend and used another $0.50 on 258 skipped generation calls and on re-vetting reshuffled
batches. The guard now reads a persistent ledger
(`data/local/stage3_multisentence_corpus_v17/spend_ledger.json`). The held-out evaluation and judging
spent about $0.23 separately.

## Training (`scripts/train_stage3_multisentence_v17.py`)

Continued from the deployed `stage3_composition_v17_20260929_v2` weights; Adafactor, peak LR 3e-4, 100-step
warmup, linear decay to 10%; batch 32; three fixed length buckets; FP32 on MPS; wall-clock budget
20 minutes; final step kept (validation loss reported only).

| Run | Steps | Epochs | Stop | Final validation loss | Status |
|---|---:|---:|---|---:|---|
| run 1 | 4,902 | 2.96 | wall-clock at 68% of plan (speed estimate too high; LR never decayed) | 0.227 | preserved as `stage3_multisentence_tiny_run1_v17_20260929` |
| **run 2** | 4,539 | 2.74 | budgeted steps reached (speed re-measured every 250 steps) | 0.252 | **candidate**, `stage3_multisentence_tiny_v17_20260929` |

Run 2 was designated the candidate before it was scored, to avoid choosing between runs by held-out
score. A third attempt was killed after one minute by a session interruption (log kept).

## Held-out results (300 generated sessions of 2–4 sentences; DeepSeek judge)

| Renderer | Fully right | Wrong | Judge mean | Any sign dropped | Invented | NO dropped | BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 (installed) | 6% | 31% | 0.74 | 68% | 8% | 21/42 | 31 |
| **run 2, whole buffer** | **60%** | **6%** | 1.53 | 3% | 0% | **0/42** | 75 |
| run 2, incremental (count-only locking) | 58% | 8% | 1.50 | 4% | 0% | 1/42 | 74 |
| **run 2, incremental, verified locking** | **60%** | **8%** | 1.52 | 4% | 0% | **0/42** | 74 |
| run 1, whole buffer | 64% | 7% | 1.57 | 2% | 0% | 0/42 | 75 |
| run 1, incremental, verified | 61% | 7% | 1.53 | 3% | 0% | 0/42 | 74 |

Run 2 by length (whole buffer): 2 sentences 66% / 7%, 3 sentences 56% / 5%, 4 sentences 44% / 8%.
Sessions with no sentence seen in any training: 62% / 6%. Yes/no questions: 52% / 10%. Names: 54% / 14%.
Run 1 and run 2 differ by 0–8 points per slice; with 300 sessions the 95% interval on 60% is about ±5.5,
so they are not distinguishable.

Incremental locking: after each committed sign, the open buffer is rendered; the first sentence is
locked when two consecutive renders agree, there are at least two sentences, and counts cover the buffer.
Counts are right in only 202 of 323 whole-buffer outputs (off by one), so the verified variant renders the
candidate sentence alone at N, N−1, N+1 and locks only the split that reproduces it exactly. Those extra
renders happen while the user signs, not at Finish. Reference implementation:
`incremental()` / `verified_split()` in `scripts/eval_stage3_multisentence_v17.py`.

## Latency (same architecture as installed v2; phone numbers from the step 2 benchmark)

Run 2 output tokens (counts included): whole buffer median 24, p90 33, max 51; incremental Finish tail
median 17, p90 27, max 51. At the measured 3.9 ms + 8.5 ms per token (idle iPhone 13, current no-cache
export): whole buffer ~208 / 284 / 437 ms, incremental Finish ~148 / 233 / 437 ms (median / p90 / max).
Live Finish latency has run 1–3× the idle figure, so the longest sessions could approach or pass 1 s
at 3×. A KV-cache export (6.1 ms per token for this size) and a warm-up call after model load would give
more margin; neither is implemented yet.

## Core ML

`artifacts/coreml/stage3_multisentence_tiny_v17_20260929/` via the existing exporter (FP32, no cache):
0 tokenization mismatches on 1,100 validation rows; Core ML greedy output equals PyTorch on 200/200.
No FP16 issue: this size already runs FP32. (flan-t5-base would have needed an encoder rescale; its
residual stream peaks at 332,589 on these inputs.)

## Earlier failure probes (run 2, whole buffer)

| Input | Output | In training |
|---|---|---|
| I WANT LEARN SIGN LANGUAGE | 5: I want to learn sign language. | no |
| WHERE YOUR MOTHER WORK | 4: Where does your mother work? | no |
| MY FATHER WORK HOSPITAL HE DOCTOR | 6: My father works at the hospital, and he is a doctor. | no |
| MY FRIEND COME TOMORROW NIGHT WE EAT | 6: My friend is coming tomorrow night, and we will eat. | no |
| YOU KNOW WHERE HOSPITAL | 4: Do you know where the hospital is? | no |
| MAYBE TOMORROW WE GO DOCTOR | 5: Maybe tomorrow we go to the doctor. | no |
| HELLO GOOD MORNING HOW YOU FRIEND | 6: Hello, good morning, how are you, friend? | no |
| I HUNGRY HAVE YOU EAT | 5: I am hungry, have you eat? | no (grammar error) |
| I HELLO FRIEND I NEED HELP | 6: Hello friend, I need help. | no (drops a leading I) |
| MY NAME FS0 TAKE | 4: My name is FS0. | no (drops TAKE at high confidence) |

The last two drop a confidently scored sign. That is arguably the right English for these noisy phone
buffers, but it shows that the model can drop a high-score gloss, which training only intended for
low-score spurious glosses.

## Limits

References, vetting and judge are all DeepSeek, and the gloss order is an LLM's idea of ASL. No fluent
signer reviewed anything. Evaluation inputs are clean (all confidences 0.9). Scores measure agreement
with DeepSeek on unseen compositions, not ASL translation accuracy. Phone latency is estimated from the
idle benchmark, not measured with this model live.

## Not done (needs the user)

App integration: counted-output parsing, incremental locking in Swift, contract/token metadata for the
new output format, a warm-up call, optionally a KV-cache export, then a device test and install.

## App integration — installed on the iPhone 13 (user approved after the results above)

User chose incremental translation with run 2. Changes (pre-edit copies in `app_backup_before/`):

- `LiveReelStage3.swift`: `countedOutput` from the token table; `parseCounted`, `renderParts`, gloss
  groups for the log; `LiveIncrementalTranslator` (port of `incremental(verify=True)`); `warm()`; each
  generation and each decoder step now drain an autorelease pool (a 323-session native replay exhausted
  IOSurface memory without it).
- `LiveReelViewController.swift`: committed words go to a lock-protected pending list; one drain on the
  language queue takes everything pending (no render backlog before Finish); Finish renders only the
  open tail; reset clears the translator; a word-count consistency check falls back to the previous
  whole-buffer path. New history fields: `stage3_lock` events, and `locked` / `tail_words` on each
  sentence event. Older models keep their previous behaviour.
- `LiveReelApp.swift`, `ReelCamera.swift`: warm-up render after Stage 3 loads.
- Bundled Stage 3 packages and `stage3_tokens.json` replaced with the run 2 export (hashes verified);
  v2 copies in `app_backup_before/models/`. The exporter now writes `output` into the token table.
- `RunnerTests.swift`: the v2 composition test is replaced by `testStage3MultiSentenceModel`
  (greetings, request, time, negation, three sentences, spelled name restored, incremental run locking
  2 sentences with a 7-word tail).

Verification: native macOS Swift replay of all 323 held-out sessions equals the Python reference for
whole-buffer text, incremental text, and lock/tail counts (323/323 each; `swift_probe/`). On the
physical iPhone 13, all 10 RunnerTests pass, including the new Stage 3 test. Release rebuilt without
testability, codesign verified, installed; saved sessions intact. 67 of 68 Python Stage 3 tests pass;
`test_stage3_mobile_naturalizer_v17` fails in setUpClass on a Stage-2 recognizer hash check that the
log already records as pre-existing and unrelated.

Not verified: live camera signing with the new model, and live Finish latency (estimated only). The
desktop app still defaults to v2; switching it needs the counted-output parsing in Python.
Rollback: restore the three files in `app_backup_before/models/` and the Swift files in
`app_backup_before/`, then rebuild.
