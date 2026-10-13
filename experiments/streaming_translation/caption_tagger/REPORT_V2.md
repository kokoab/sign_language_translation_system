# Fixed live caption engine (v2) — 2026-10-13 PHT

You asked me to fix the first tagger's failures and make the captions work well and
fast. Translation needs to convey the meaning, not be word-for-word. Builds on
`REPORT.md` (first tagger) and `RESEARCH.md`. No ATLAS file, app or deployment was
changed. Total training is 26.5 of the 30 authorized minutes (log: `training_log.json`).

## Result

The captions now carry the meaning as often as the installed translator does at
Finish: 706 vs 708 of 784 held-out sentences. They appear the moment a sign is
recognized, rarely rewrite themselves, keep every sign, and never add a word for an
unsigned gloss. Each update costs about 5 ms of one CPU thread.

| | Phase 1: T5 re-translated live | First tagger | **v2 engine** | T5 at Finish |
|---|---:|---:|---:|---:|
| Held-out sentences with the thought there (of 784) | — | 665 | **706** | 708 |
| … every sign represented | — | 784 | **784** | 769 |
| … WH questions right (of 75) | — | 53 | **67** | 69 |
| … negation sentences right (of 29) | — | 15 | **24** | 24 |
| 3,200 new combos, all 101 classes: every sign shown | — | 3,179 | **3,200** | 2,693 |
| … words for unsigned glosses | — | 0 | **0** | 70 outputs |
| … NO kept (of 539) | — | 539 | **539** | 528 |
| … outputs inserting a pronoun not signed | — | 223 | **74** | 445 |
| … tense contradicting its time sign (of 192) | — | 10 | **5** | 1 |
| 20 phone traces: normalized erasure | 0.91 | 0 / 0.10 with draft | **0.16** | n/a (shown once) |
| … time to first text | 0.012 s | 1.17 s / immediate | **immediate** | after Finish |
| … final text settled after last sign (median) | — | — | **0.003 s** | needs Finish |
| Compute per update, 1 CPU thread, p50 / p95 / max | 18 ms model call | — | **4.7 / 8.1 / 15.8 ms** | 38 ms per request |

"Thought there" is an automatic proxy over held-out sentences. It requires the
output to match the reference on negation, question vs statement, tense and WH
word, and to represent every sign. References are DeepSeek English, not signer
judgement. Sessions overall: v2 BLEU 67.2 / chrF++ 81.1; T5 at Finish 75.1 / 85.1.
The gap is mostly wording; on meaning features they are nearly level.

## What was fixed and how

1. **Clause-final WH and late NO.** A learned clause finalizer runs when a clause
   closes. A bidirectional mover chooses at most one sign to move, which covers
   97% of corpus sentences. A bidirectional labeler (91.6% exact per-sign tags) then
   renders the clause. Both are trained on the corpus English with the evaluation
   sets held out. Examples:
   - YOU GO SCHOOL WHERE: `You go to school where` → **Where do you go to school?**
   - I WANT WATER NO: `I want water, not` → **I do not want water**
   - I GO HOSPITAL NO → **I did not go to the hospital**
2. **Live tagger closed sentences too early** (before a final WH). It had never seen
   reordered sentences. Retrained with sentence-boundary labels from every corpus
   sentence (123 s), then averaged with the first model. This was chosen on corpus
   validation: boundary accuracy 93.0% → 93.8% with exact per-sign accuracy kept
   (82.6%).
3. **Late time sign.** A clause is rewritten when its verb tense contradicts the
   time sign governing it, e.g. THEY GO HOSPITAL TOMORROW: `They went to the
   hospital tomorrow` → **They are going to the hospital tomorrow**. Present tense
   with a future time word ("I go to the doctor tomorrow morning") is left alone.
4. **Too much rewriting.** Phone signs arrive a median 1.3 s apart, and 10% of gaps
   exceed 3.4 s, so pause timers can't tell hesitation from the end of a thought.
   Following the live-caption stability study (viewers prefer stability), the engine:
   - rewrites a closed clause only to fix meaning (moved WH/NO, missing negation,
     contradicting tense), not to polish wording;
   - shows the newest sign at once, but without guessed punctuation;
   - finalizes the tail only after 3.5 s of no signing (end of turn), with no period
     that would be erased if signing resumes.

   Erasure on phone traces went 0.57 (first v2 run) → 0.38 → **0.16**. With no draft
   at all it would be 0.06, but each sign would wait about 1.2 s for the next one.
5. **Invented referents.** "Hello, friend, I will help her" came from inserting
   "her". Pronoun and here/there insertions now require at least 50% model
   probability, cutting such outputs from 223 to 74 of 3,200.
6. **Dropped signs.** A sign recognized with confidence ≥ 0.6 can no longer be
   silently dropped. EAT had been dropped 7% of the time; now 3,200/3,200 outputs
   keep every sign.
7. **Split fingerspelling.** fs-GE fs-LO → **Gelo**. Updates now happen on every
   fragment, so fingerspelling no longer looks like a pause.
8. **Long conversations.** Closed clauses are frozen. The live model sees only the
   open clause plus 16 earlier signs, so cost stays flat: 1.9 ms at 5 signs, 5.3 ms
   at 80. The first tagger would have failed past 48 signs.

## Your examples (live sequence, one sign per second)

- I GO HOSPITAL TOMORROW MORNING: `I → I go → I go to the hospital → … tomorrow → … tomorrow morning`
- TOMORROW MORNING I GO HOSPITAL: `… → Tomorrow morning I will go to the hospital`
- YOU GO HOSPITAL WHEN: `… → You go to the hospital when → When will you go to the hospital?`
- HELLO MY FRIEND HOW YOU: `… → Hello, my friend, how are you?` (phase 1: "Hello, how are my friend?")
- NO I TIRED: `No, I am tired` (T5 at Finish: "No, I am not tired" — flips the meaning)

Recorded phone traces settle on, for example, "How are you?", "Hello, my name is Gec",
"Yesterday I went to the doctor", "I go to the doctor tomorrow morning",
"Hello, my friend, how are you?". 15 of 20 show no rewrite at all.

## Still wrong (known)

- **YOU used for "your".** YOU NAME WHAT → "What are you name?"; YOU DAY HOW →
  "How are you day?". The corpus realizes YOU as "your" in only 23 of 105,578
  sentences, so the model can't learn it. It needs recognizer/corpus attention
  (YOU vs YOUR), not a caption rule.
- **NO scope is ambiguous after several verbs.** I KNOW YOU LIKE NO → "I do not know
  if you like it" (the reference reads "I know you do not like this").
- **Some unusual inputs stay awkward.** PLEASE YOU GIVE I WATER → "Please you give
  me water?"; THEY HUNGRY WHO → "Who are they hungry?". Random orders are often
  ungrammatical, though every sign is kept.
- **Tense contradictions:** 5 of 192 remain (T5 has 1).
- **74 of 3,200 stress outputs still insert a pronoun**, e.g. "please help me".
  Some are right, some invent a referent.

## How this was tuned, and limits

- **Phone traces are tuning data.** The draft display mode, 3.5 s end of turn and
  meaning-only rewrite rule were chosen on the 20 phone traces; treat those stream
  numbers as development figures. Held-out and stress sets were not used for any
  choice. Two scoring checks were corrected after inspection, and all tables come
  from the final run.
- **Speed is desktop only.** Measured on an Apple M4 CPU, one thread. An earlier run
  under heavy outside load (load average 18–24) measured p50 14–19 ms; the engine
  stayed about 7× cheaper than T5. No iPhone measurement was made. The engine is five
  small encoders, 7.3M parameters in total (about 29 MB FP32). A normal update runs
  three encoder passes and a clause close adds two; there is no autoregressive decoding.
- **Not a signer study.** Text-level replay of recognized glosses; not camera
  accuracy. No API spending, data acquisition or protected test access.

## Files

`live_caption.py` (engine) · `finalizer.py` (mover/labeler, data, training) ·
`add_boundaries.py` · `evaluate_v2.py` → `eval_v2.json`, `eval_v2_streaming_cases.json`,
`v2_heldout_sentences.csv`, `v2_stress_samples.csv` · `speed_rerun.json` ·
`verify.py` → `integrity.json` (adds 577 causal checks on the retrained model and
150 frozen-clause stability sequences) · `README.md` (usage and rebuild).

## Desktop app integration — 2026-10-14

The engine is now the default English path of the desktop Live screen
(`scripts/app_shell_v17.py` → `scripts/live_segmental_v17.py`). Nothing in `ATLAS/`
or the phone apps changed.

- Runtime: `scripts/live_captions_v17.py`. It is self-contained, loads
  `artifacts/models/live_captions_v17_20261014/` (five state dicts, a 25 KB
  vocab.json and a manifest with SHA-256), and needs no training data or
  experiment imports.
- Screen: captions fill the existing sentence panel while signing, showing the
  newest three lines. The sign rail, preview and controls are unchanged.
- Voice: speaks each English sentence once it settles (clause close or 3.5 s pause)
  instead of every gloss. Fingerspelled names are spoken with their own sentence.
- Finish is optional. It finalizes the caption and records the utterance
  (`rendering_mode: live_captions`); the T5 naturalizer is not loaded in this mode.
- History: every caption change is logged as a `live_caption` event (text, kind,
  spoken part), and the model package name and hashes go into the session
  provenance.
- Options: `--no-live-captions` restores the old path (gloss speech, T5 sentence at
  Finish); `--caption-end-seconds` changes the 3.5 s pause.

**Context size changed for the app.** The live tagger sees only the open clause
(context 0, previously 16 signs). This was chosen on the corpus validation split:
across 1,100 sessions, sentence-count agreement was 75.0% vs 70.3% with the same
chrF++ (`context_select.py`). Prompted by the recorded session below, where
16-sign context split "HELLO HOW YOUR DAY". Effect on the frozen evaluations, with
context 0 vs 16 (both kept: `eval_v2.json`, `eval_v2_context16.json`):

| Measure | Context 16 | Context 0 (app) |
|---|---:|---:|
| Held-out "thought there" (784) | 706 | 697 |
| Held-out session BLEU / chrF++ | 67.2 / 81.1 | 67.6 / 81.2 |
| 3,200 stress: every sign / invented / NO kept | 3,200 / 0 / 539 | 3,200 / 0 / 539 |
| Phone traces: erasure / rewrites | 0.16 / 5 of 20 | 0.16 / 5 of 20 |
| Compute per update p50 / p95 / max | 4.7 / 8.1 / 15.8 ms | **2.1 / 4.2 / 9.9 ms** |

The 9-sentence held-out drop is reported, not re-tuned. The test set was not used
to choose. Cost no longer grows with conversation length: 1.95 ms at 5 signs and
2.11 ms at 80.

Checks:
- **Parity.** `test/test_live_captions_v17.py` (9 tests) shows the app runtime
  matches the evaluated engine on all 46 recorded/diagnostic streams and 60 random
  ones. The only deliberate display difference is that orphan punctuation is
  hidden when every sign in a clause was dropped as low-confidence noise.
- **Other tests.** It also covers frozen-text stability, no repeated speech,
  spoken names, end-of-turn resume, unknown glosses and reset. A new display test
  checks that the newest caption lines are shown.
- **Test suites.** 195 tests pass across the caption, display, shell, pages, app
  integration, Live Reel, segmental runtime, continuous and Renz suites.
- **End to end.** The real desktop pipeline ran headless on a recorded 58-sign
  desktop session (`artifacts/app_sessions/20260928_095748_940053`). Recognition
  was identical with and without captions (53 committed words), and per-frame
  compute was unchanged (23.4 vs 24.4 ms median). The finish step took 9 ms.
  - Captions read, for example: "Hello, I feel sick, tired and hungry. Hello,
    friend, do you go to the doctor tomorrow morning? Hello, how is your day?
    Hello, how is your morning? I will go tomorrow."
  - The old path's T5 exceeded its context on this 53-sign utterance and fell back
    to the literal gloss list.
  - Misrecognized bursts (WE I WE I …) are rendered faithfully as noise.
- **Not covered.** No interactive camera session was run in this check.
