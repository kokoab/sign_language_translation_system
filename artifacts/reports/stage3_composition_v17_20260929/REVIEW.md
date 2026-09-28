# Stage 3 learned composition repair — 2026-09-29

User request: repair model weights without phrase triggers. Final model: `artifacts/models/stage3_composition_v17_20260929_v2`; Core ML: `artifacts/coreml/stage3_composition_v17_20260929_v2`. Original checkpoints and prior mobile resources are preserved. Physical iPhone 13 model test passes. Final signed Release is installed and launched on the connected iPhone 13; existing app data was retained.

## Result

Both user-requested inputs were excluded from both fine-tuning corpora and now produce the intended English directly from the model:

| Input | Previous model | Final model |
|---|---|---|
| HELLO GOOD MORNING HOW YOU FRIEND | Hello, how are you a friend? | Hello, good morning. How are you, friend? |
| HELLO MY FRIEND HOW YOU | Hello, how are my friend? | Hello, my friend. How are you? |

The follow-up also repairs PLEASE GIVE MY CHILD WATER -> Please give my child water. and HELLO FRIEND I HELP -> Hello, friend. I help. The recipient request was explicitly included as training data during the continuation, so it is a training regression, not an unseen-example result.

## Implementation

- Fixed five-epoch fine-tune of the existing slot T5: 16,070 old TRAIN rows + 17,225 generated composition rows, seed 17929, Adafactor LR 0.0002, batch 32, MPS; 1326.80 seconds.
- Bounded three-epoch continuation: 4,194 recipient/object/time/message/address examples + 4,000 balanced prior TRAIN replay rows, LR 0.0001, batch 32, MPS; 96.12 seconds.
- Both runs use the predetermined final epoch. Generated data is training-only; no generated validation/test selection or independent ASL accuracy claim. Old non-train gloss sequences remain reserved. No Stage 1/2, visual phrase dataset, acquisition, or protected Citizen test use.
- Per-checkpoint metadata disables existing reviewed-phrase overrides. No greeting lookup, semantic output rewrite or phrase-specific inference trigger was added.
- The composition model receives the complete Finish buffer and learns punctuation; old models retain their pause behavior. Over-context inputs take a literal fallback instead of silent truncation. Existing spelling-slot restoration protects recognized names.
- Updated Python/Torch and Core ML defaults and the current segmental config to the new weights. Mobile sources/resources use the same model, and future phone session histories record its checkpoint identifier.

## Validation

- 54 focused and regression Python tests pass after default changes.
- Final Core ML export: 8,194 tokenization rows, 0 mismatches; 200/200 greedy outputs and piece decoding match PyTorch. These are numerical export checks on training rows, not translation accuracy.
- Native macOS Swift renderer: 44/44 outputs match PyTorch, all use neural generation, artificial long pauses retain one complete utterance; over-context literal input preserves every word.
- Actual default desktop render_utterance smoke passes both user phrases, recipient request, single spelled name and the two-slot literal fallback.
- Signed iOS Release build succeeds; codesign verification passes; final install succeeds for com.kokoab.sltMobileApp. Initial physical test attempt failed because the phone disconnected (xcodebuild destination unavailable), before tests ran. After reconnection, RunnerTests.testStage3ModelCompositionWithoutPhraseOverrides passed on physical iPhone 13 in 7.162 seconds. It checks both user phrases, recipient request, time attachment, a formerly overridden greeting, a restored spelled name and negation; direct generation and whole-utterance behavior are asserted. This is device text inference, not new live-camera accuracy.
- Initial v2 export invocation exposed an existing relative-path bug. CLI paths are now resolved before computing metadata; the unchanged weights exported successfully on retry.

## Remaining limits

Some historical noisy or ambiguous buffers still produce bad English. Examples: I HELLO FRIEND I NEED HELP -> I am Hello, friend. I need help.; MY NAME FS0 TAKE -> FS0 takes the name.; USE FEEL HAPPY PLEASE SORRY HUNGRY MY NAME FS0 produces malformed English. YOU DAY HOW still drops DAY. The two-slot name sequence still requires the existing literal fallback to retain both recognized spelling fragments. No claimed repair of wrong recognized signs/letters, independent signer accuracy, or every possible composition.

## All direct neural development probes

The table is diagnostic, not a validation score. `Training input` covers either fine-tuning pass; slots are intentionally displayed as FS0/FS1 here, before runtime restoration.

| Origin | Input | Previous neural output | Final neural output | Training input |
|---|---|---|---|---|
| user_requested | HELLO GOOD MORNING HOW YOU FRIEND | Hello, how are you a friend? | Hello, good morning. How are you, friend? | False |
| user_requested | HELLO MY FRIEND HOW YOU | Hello, how are my friend? | Hello, my friend. How are you? | False |
| phone_history | HOW YOU | How are you? | How are you? | True |
| phone_history | MY NAME FS0 | My name is FS0. | My name is FS0. | False |
| phone_history | HELLO MY NAME FS0 | Hello, my name is FS0. | Hello. My name is FS0. | False |
| phone_history | HOW YOU HOW YOUR DAY | How are you? | How are you. How is your day? | False |
| phone_history | HELLO HOW MORNING GOOD DAY | Hello, how is the good? | Hello, how is the morning, good day? | False |
| phone_history | I HELLO FRIEND I NEED HELP | I am Hello, and I need the friend. | I am Hello, friend. I need help. | False |
| phone_history | HELLO FRIEND I HELP | Hello, I help the friend. | Hello, friend. I help. | True |
| phone_history | USE FEEL HAPPY PLEASE SORRY HUNGRY MY NAME FS0 | The use feels happy, sorry, and hungry is FS0. | Feel happy. Please't sorry, hungry. My name is FS0. | False |
| phone_history | HELLO HOW YOU | Hello, how are you? | Hello. How are you? | True |
| phone_history | MY NAME FS0 TAKE | FS0 takes the name. | FS0 takes the name. | False |
| phone_history | HELLO HOW YOU | Hello, how are you? | Hello. How are you? | True |
| phone_history | MY NAME FS0 FS1 | My name is FS1. | My name is FS0. | False |
| phone_history | HELLO MY NAME FS0 | Hello, my name is FS0. | Hello. My name is FS0. | False |
| phone_history | YESTERDAY I GO DOCTOR | Yesterday I went to the doctor. | I went to the doctor yesterday. | True |
| phone_history | I FEEL HUNGRY | I feel hungry. | I feel hungry. | False |
| phone_history | HELLO MY NAME FS0 | Hello, my name is FS0. | Hello. My name is FS0. | False |
| phone_history | MY NAME FS0 | My name is FS0. | My name is FS0. | False |
| phone_history | I GO SCHOOL TOMORROW MORNING | I go to school tomorrow morning. | I am going to school tomorrow morning. | True |
| phone_history | GOOD DAY | Good. | Good day. | True |
| phone_history | YOU DAY HOW | How are you? | How are you? | False |
| phone_history | HELLO | Hello. | Hello. | False |
| phone_history | I LOVE YOU | I love you. | I love you. | True |
| phone_history | WHAT TIME | What time is it? | What time is it? | True |
| phone_history | I GO DOCTOR TOMORROW MORNING | I go to the doctor, and tomorrow I go to the morning. | I am going to the doctor tomorrow morning. | True |
| phone_history | HELLO MY FRIEND | Hello, my friend. | Hello, my friend. | True |
| phone_history | HELLO MY FRIEND HOW YOU | Hello, how are my friend? | Hello, my friend. How are you? | False |
| generated_diagnostic_only | HELLO GOOD MORNING HOW YOUR FAMILY | Hello, how is your family? | Hello, good morning. How is your family? | True |
| generated_diagnostic_only | GOOD DAY MY FRIEND HOW YOU | Good, how is my friend? | Good day, my friend. How are you? | True |
| generated_diagnostic_only | HELLO DOCTOR I NEED HELP | Hello, I need the doctor. | Hello, Doctor. I need help. | True |
| generated_diagnostic_only | HELLO MY FRIEND I NEED WATER | Hello, I need the water. | Hello, my friend. I need water. | True |
| generated_diagnostic_only | GOOD MORNING HOW YOU FEEL FRIEND | Good morning, how are you feeling? | Good morning. How do you feel, friend? | True |
| generated_diagnostic_only | I GO DOCTOR TOMORROW MORNING | I go to the doctor, and tomorrow I go to the morning. | I am going to the doctor tomorrow morning. | True |
| generated_diagnostic_only | I SICK MY NAME FS0 | I am sick, and FS0. | I am sick. My name is FS0. | True |
| generated_diagnostic_only | MY NAME FS0 I HUNGRY | I am hungry. | My name is FS0. I am hungry. | True |
| generated_diagnostic_only | I NO WANT WATER | I do not want the water. | I don't want the water. | False |
| generated_diagnostic_only | YOU NO NEED HELP | You do not need to help. | You don't need the help. | False |
| generated_diagnostic_only | WE LOVE OUR FAMILY | We love our family. | We love our family. | False |
| generated_diagnostic_only | PLEASE GIVE MY CHILD WATER | Please give my child. | Please give my child water. | True |
| generated_diagnostic_only | HOW YOU FRIEND | How are you? | How are you, friend? | True |
| generated_diagnostic_only | MY FRIEND SICK | My friend is sick. | My friend is sick. | True |
| generated_diagnostic_only | HELLO FS0 HOW YOU | Hello, how are you? | Hello, FS0. How are you? | True |
| generated_diagnostic_only | HELLO GOOD MORNING MY NAME FS0 HOW YOU | Hello, how are you good? | Hello, good morning. My name is FS0. How are you? | True |

## Changed files and rollback

Code: scripts/repair_stage3_composition_v17.py; active/v17/export_stage3_t5_coreml_v17.py; active/v17/stage3_coreml_v17.py; scripts/live_isolated_v17.py; scripts/live_segmental_v17.py; test/test_stage3_composition_v17.py; test/test_stage3_asl_corpus_v17.py. Current stream config: artifacts/reports/segmental_decoder_v17_20260927/stream_config_v3_final.json (Stage 3 paths only). Mobile: LiveReelStage3.swift, LiveReelViewController.swift (checkpoint history field only), RunnerTests.swift (one new device test), and two Stage3 model packages plus stage3_tokens.json. Concurrent UI/recognition work is preserved.

Rollback resources: mobile_previous/ contains previous mobile model packages/token table; previous_stage3_config.json contains only the two prior Stage 3 config values. Do not restore an entire old stream config over unrelated changes. Prior model directories remain available. Training inputs, recipes, completion records, hashes, old/new outputs and both exports remain under this report and their versioned artifact paths.
