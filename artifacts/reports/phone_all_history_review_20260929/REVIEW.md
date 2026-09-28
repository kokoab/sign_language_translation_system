# Mobile translation history review — 2026-09-29

Reviewed all 12 saved Live session files currently in the connected iPhone app's Documents/live_reel_sessions directory (September 28–29 PHT), containing 26 saved translations. This is all persisted Live history at copy time, not deleted/unsaved sessions or older diagnostic recordings. Seven files are incomplete autosaves. History inspection was followed by six text-only Core ML probes using the current mobile source model packages on macOS. No training, app edits or deployment was performed.

## Exact reported failure

Session `20260929_115725`, second translation after restarting capture around 12:19 PHT:

- Recognized: `HELLO MY FRIEND HOW YOU`.
- Saved English: **Hello, how are my friend?**
- Intended English from the user's stated phrase: **Hello, my friend. How are you?**
- All five words reached Stage 3 as one clause. This failure is downstream of word recognition and is not caused by the 1.5-second splitter.
- Logged scores: HELLO 0.9164, MY 0.9833, FRIEND 0.8686, HOW 0.9028, YOU 0.9070. These are model scores, not calibrated correctness probabilities.
- Final inter-word gaps: 0.333, 0.333, 0.367, 0.267 seconds. Translation took 175.01 ms. The immediately preceding finished attempt, HELLO MY FRIEND, rendered correctly as Hello, my friend.

## Session inventory

| Session (PHT filename) | Complete | Word events | Saved glosses | Previews | Translations |
|---|---|---:|---:|---:|---:|
| 20260928_224024 | True | 11 | 11 | 0 | 0 |
| 20260929_001047 | True | 0 | 0 | 1 | 0 |
| 20260929_001418 | False | 6 | 7 | 26 | 2 |
| 20260929_004215 | False | 73 | 74 | 350 | 6 |
| 20260929_013353 | True | 72 | 76 | 250 | 12 |
| 20260929_015257 | False | 16 | 16 | 91 | 0 |
| 20260929_092837 | False | 2 | 2 | 15 | 0 |
| 20260929_095006 | True | 0 | 0 | 0 | 0 |
| 20260929_095705 | False | 1 | 1 | 4 | 1 |
| 20260929_102654 | False | 42 | 42 | 155 | 3 |
| 20260929_115505 | True | 25 | 25 | 92 | 0 |
| 20260929_115725 | False | 38 | 38 | 110 | 2 |

Saved-gloss totals may exceed word-event totals because Finish can flush pending words. Resets and capture restarts are retained in HISTORY.md; concatenating an entire session would mix separate attempts. The newest session includes a long stop interval, so its duration is not active inference time.

## Every saved translation

| Session / row | Logged clauses | Saved English | Review |
|---|---|---|---|
| 20260929_001418 / 1 | HOW YOU | How are you? | Faithful to logged input. |
| 20260929_001418 / 2 | MY NAME fs-GELO | My name is Gelo. | Faithful to logged spelling; intended name not independently verified. |
| 20260929_004215 / 1 | HELLO MY NAME fs-GEC | Hello, my name is Gec. | Faithful to logged spelling; intended name not independently verified. |
| 20260929_004215 / 2 | HOW / YOU / HOW YOUR DAY | How. You. How is your day? | HOW / YOU split apart; fragments instead of a question. |
| 20260929_004215 / 3 | HELLO HOW / MORNING / GOOD DAY | Hello, how is he? In the morning, Friend. Good. | Adds HE and FRIEND; drops DAY. |
| 20260929_004215 / 4 | I / HELLO FRIEND I / NEED / HELP | I. Hello, I am a friend. Need. Help. | Splits I / NEED / HELP; greeting becomes an identity statement. |
| 20260929_004215 / 5 | HELLO FRIEND I / HELP | Hello, I am a friend. Help. | Splits HELP from the preceding clause; greeting becomes an identity statement. |
| 20260929_004215 / 6 | USE / FEEL HAPPY / PLEASE / SORRY / HUNGRY MY NAME fs-GELO | Use. Feel happy. Please. Sorry. hungry is Gelo. | HUNGRY joined to MY NAME; name introduction is corrupted. |
| 20260929_013353 / 1 | HELLO HOW YOU | Hello, how are you? | Faithful. |
| 20260929_013353 / 2 | MY NAME fs-GEL TAKE | Gel takes the name. | MY NAME becomes a taking-the-name statement; TAKE was already present upstream. |
| 20260929_013353 / 3 | HELLO HOW YOU | Hello, how are you? | Faithful. |
| 20260929_013353 / 4 | MY NAME fs-GE fs-LO | My name Ge Lo. | Two separate spelling tokens retained as Ge Lo; incomplete English, not evidence that the translator should guess Gelo. |
| 20260929_013353 / 5 | HELLO MY NAME fs-GELO | Hello, my name is Gelo. | Faithful to logged spelling. |
| 20260929_013353 / 6 | YESTERDAY / I GO DOCTOR | Yesterday. I go to the doctor. | YESTERDAY separated from I GO DOCTOR; time context fails to guide tense. |
| 20260929_013353 / 7 | I FEEL HUNGRY | I feel hungry. | Faithful. |
| 20260929_013353 / 8 | HELLO / MY NAME fs-GELO | Hello. My name is Gelo. | Meaning retained despite a greeting split. |
| 20260929_013353 / 9 | MY NAME fs-GELCO | My name is Gelco. | Faithful to logged spelling; GELCO originates upstream. |
| 20260929_013353 / 10 | I GO SCHOOL TOMORROW MORNING | I go to school tomorrow morning. | Faithful. |
| 20260929_013353 / 11 | GOOD DAY | Good. | Drops DAY. |
| 20260929_013353 / 12 | YOU DAY / HOW | You are a day. How. | Malformed YOU DAY becomes You are a day; HOW is split off. Intended input not known. |
| 20260929_095705 / 1 | HELLO | Hello. | Faithful. |
| 20260929_102654 / 1 | I LOVE YOU | I love you. | Faithful. |
| 20260929_102654 / 2 | WHAT TIME | What time is it? | Faithful. |
| 20260929_102654 / 3 | I GO DOCTOR TOMORROW MORNING | I go to the doctor, and tomorrow I go to the morning. | Treats MORNING as a destination and invents a second trip instead of attaching TOMORROW MORNING to the doctor visit. |
| 20260929_115725 / 1 | HELLO MY FRIEND | Hello, my friend. | Faithful. |
| 20260929_115725 / 2 | HELLO MY FRIEND HOW YOU | Hello, how are my friend? | Exact user-reported sequence recognized; drops YOU as addressee and produces malformed question. No pause split. |

## Requested phrase and model probes

Executed the current mobile source encoder/decoder packages, token table and mobile manifest through the equivalent Python Core ML renderer on macOS (CPU_AND_GPU). All inputs were one clause; new probes used score 0.9 (the high bucket), and the historical greeting used its recorded scores. This is a translation-only check, not physical signing or a new iPhone run. Package hashes are preserved in model_probes.json.

| Input | Actual output | Path |
|---|---|---|
| HELLO GOOD MORNING HOW YOU FRIEND | Hello, how are you a friend? | t5_efficient_tiny |
| HELLO MY FRIEND HOW YOU | Hello, how are my friend? | t5_efficient_tiny |
| HELLO MY FRIEND | Hello, my friend. | t5_efficient_tiny |
| HELLO HOW YOU | Hello, how are you? | reviewed_template |
| HELLO GOOD MORNING | Hello, good morning. | t5_efficient_tiny |
| HOW YOU FRIEND | How are you? | t5_efficient_tiny |

**HELLO GOOD MORNING HOW YOU FRIEND** produces **Hello, how are you a friend?** It omits GOOD MORNING and changes the friend address into a malformed question. A faithful intended rendering is **Hello, good morning! How are you, friend?** The exact historical failure is reproduced, while the shorter greeting components work individually (HOW YOU FRIEND still loses FRIEND). This is evidence of a composition failure; it does not prove the training-level cause.

Validation: 12 copied sessions match the 12-file device inventory; all 26 saved sentence-list entries match sentence events and every flattened clause list matches its preceding Finish words. Six model probes completed without fallback; the exact historical failure matches byte-for-byte. These are development diagnostics, not held-out accuracy.

Execution note: an initial model-loading probe was restarted with optional heavy imports disabled; its duplicate process was terminated. Only the completed model_probes.json run is used as evidence.

## Findings and next safe action

1. The current renderer can corrupt a correctly recognized five-word greeting. The latest example isolates Stage 3; raising recognition confidence or changing pause thresholds would not address it.
2. Other saved failures include unsupported HE/FRIEND, dropped DAY, MY NAME role corruption, and MORNING treated as a destination. These are broader than one missing greeting template.
3. Independent pause segmentation failures also persist in the historical sessions: HOW / YOU, YESTERDAY / I GO DOCTOR, and detached HELP. The splitter uses a hard 1.5-second cutoff.
4. Recognition/spelling issues are separately visible (e.g. fs-CB, fs-GEC, fs-GELCO), but history has no matching signing video or ground-truth transcript for most attempts. Do not assign recognition accuracy or assume intended words from those outputs.
5. Current source LiveReelStage3.swift:95 accepts a reviewed template, otherwise any nonempty generated output <=300 characters. Slot checks protect slot occurrence only. The sentence history omits per-clause rendering mode and model/build fingerprint, so exact historical template/model provenance is not established from the logs.
6. Next: preserve all 26 pairs as development regression cases; implement and verify meaning-preserving translation acceptance/fallback plus phrase-aware segmentation on additional reviewed compositions. Include names, pronoun roles, negation and time attachment. Do not claim a one-template patch solves this class of errors. No repair was applied during this inspection.

Sources: copied sessions/*.json; summary.json includes hashes; translation_pairs.json preserves all Finish inputs; HISTORY.md contains every non-preview event. Code inspected: `/Volumes/secret/SLT/mobile_app/slt_mobile_app/ios/Runner/LiveReel/LiveReelStage3.swift` (95–152), `LiveReelDecoder.swift` (731–744), `LiveReelViewController.swift` (560–595).
