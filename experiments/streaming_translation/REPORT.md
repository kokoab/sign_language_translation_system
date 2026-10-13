# Streaming English replay — 2026-10-13 PHT

## Decision

Revisable captions are feasible in this desktop replay, but these translators and
commitment rules do **not** justify automatic speech or a live-camera demo yet.
None of the tested pretrained replacements improves reliability enough to promote.
The existing tiny translator remains the stronger tested T5 condition. No translator
fine-tuning, recognizer training, ATLAS modification or app installation occurred.
The authorized30-minute adaptation allowance was not consumed.

## What was actually tested

26 recorded phone utterances from six sessions;20 have complete original word-event
streams. The six others contain words available only at Finish and are excluded from
timed replay. Added24 declared gloss diagnostics and2 synthetic hesitation/dropout
cases. This is52 full-input cases and46 timed cases per model condition.
These are development diagnostics, not unseen-signer accuracy. No raw-video recognition
was rerun. Intended ASL meaning is unavailable for most historical phone attempts.

Tiny T5 uses the existing trained checkpoint from artifacts, correct confidence
encoding, deterministic greedy decoding and fingerspelling slots. Its26 phone outputs
match saved native Swift outputs exactly. Those saved native incremental runs also
match full rendering26/26 and have zero locks. This verifies the translation baseline
for these phone cases, not complete application or prefix parity.

FLAN-T5-base was tested both with an instruction alone and with four fixed examples.
The few-shot condition was added after seeing the first condition fail; all comparisons
are explicitly development screening. The same four examples are used for the
additional Qwen check. No references, future words or semantic review annotations enter
inference. Full prompts and unfiltered generated text are retained in generation JSONL
(for spelling-slot failure, the saved raw field contains the explicit literal fallback).

## Policy comparison on the20 timed phone utterances

| Tiny-model policy | Same final displayed text as full input | Entire tail committed | Median first output from first recognized word |
|---|---:|---:|---:|
| Finish, oracle endpoint proxy |20/20|20/20|4.455s|
|1.5s without a new committed sign|8/20|20/20|2.064s|
|Revisable full-context captions|20/20|0/20|0.012s|
|Repeated-sentence agreement, full context retained|20/20|3/20|0.012s|

The Finish timestamp is absent in the historical logs. The proxy starts its one
translation when the final word arrives, an optimistic endpoint lower bound, **not**
a measured button/gesture time. The fast initial outputs can be fragments or wrong;
this is not time to first faithful translation. Source timestamps begin after
recognition, so the figures exclude upstream camera/recognition delay.

Revisable policies match full-input text by construction after the last word; this
does not establish that either text is correct. Their mean normalized erasure is0.91
on these phone cases: about0.91 previously displayed tokens are removed per final
token. It can be visually disruptive. The speech policy is conservative: it commits
only a completed sentence shared by two successive distinct prefixes. This is a
simple heuristic, **not** a trained readiness estimator. It never uses an oracle
end-of-stream to force the unfinished tail into speech.

Timeout examples show why no-commit cannot stand in for a true sentence boundary:

| Recorded gloss input | Full tiny output | Timeout output |
|---|---|---|
|HOW YOU|How are you?|How. You.|
|YESTERDAY I GO DOCTOR|Yesterday I went to the doctor.|Yesterday. Is it here? Go to the doctor.|
|HELLO MY FRIEND|Hello, my friend.|Hello. My friend is here.|

12/20 timeout outputs differ; difference alone is not an error count. The examples
above are directly inspected failures. Both synthetic continuation gaps also change
the full translation. Their timings are stress-test inventions, not measured signing.

## Semantic findings

The complete greeting works: HELLO GOOD MORNING HOW YOU gives
“Hello, good morning, how are you?” The intermediate sequence is unstable:

1. HELLO → Hello.
2. HELLO GOOD → Hello, it is good.
3. HELLO GOOD MORNING → Hello, good morning.
4. HELLO GOOD MORNING HOW → Hello, how is it good morning?
5. HELLO GOOD MORNING HOW YOU → Hello, good morning, how are you?

Explicit temporal/participant evidence in HELLO YESTERDAY MORNING I FEEL GOOD HOW YOU
gives “Hello, I felt good yesterday morning. How are you?” This is a declared text
interpretation, not proof that the system distinguishes two naturally signed versions
of the user's morning examples. Identical recognized input cannot encode an otherwise
missing distinction. ABOUT need not appear as a separate input to permit appropriate
English, but this experiment does not demonstrate general contextual inference.

Important failures of the tiny model:

- I → “Is it here?”; I GIVE → “I give it to him.” Missing input becomes invented content.
- YESTERDAY I WORK TOMORROW I SLEEP → “Yesterday I worked and I slept tomorrow.”
- YOU GIVE and YOU GIVE I both produce “Can you give it to me?” The repeated-sentence
  policy commits it; adding WATER changes the full text to “Can you give me water?”
  Repetition did not prove completeness, and the declared statement interpretation
  was already changed into a request.
- I WANT WATER NO initially displays “I want water.” before revising to
  “I do not want water.” The conservative policy does not speak that early draft.

There are frozen-text conflicts in2/24 declared tiny diagnostic cases (3conflicting
prefix updates). The other case is I NEED DOCTOR I NO SLEEP, where a sentence is
subsequently joined with a conjunction. Some conflicts are formatting/segmentation
changes rather than semantic contradictions; do not report all conflicts as meaning errors.

FLAN with the instruction alone often copies glosses rather than producing English;
I NO WANT WATER loses NO. With four examples, negation is retained in that one case,
but HELLO GOOD MORNING HOW YOU becomes “ASL: HELP ME WHILE YOU ARE HERE”; YESTERDAY
I GO DOCTOR becomes “Today I go Doctor”. Neither tested FLAN prompt is a replacement.
This does not establish the maximum achievable quality of FLAN with other prompts or
adaptation.

Restricted normalized reference agreement on18 unambiguous declared diagnostics:
tiny11/18, FLAN zero-shot5/18, FLAN few-shot8/18, Qwen few-shot2/18,
Qwen direct7/18. This is **not semantic accuracy**:
acceptable paraphrases are not exhaustive and punctuation/case differences disappear.
Raw outputs and explicit counterexamples are the evidence for the recommendation.

## Additional pretrained model result: Qwen2.5-0.5B-Instruct

The official instruction-tuned checkpoint was downloaded into this experiment only.
Weights match SHA256 fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe;
revision7ae557604adf67be50417f59c2c2f167def9a775. Two fixed prompts were tested:
the same four examples used for FLAN and a translation-only system message with
gloss-only user input. Direct instructions improved several cases, but failed critical
meaning checks. No demonstration or claimed deployment benefit is justified.

| Declared input | Qwen direct output | Finding |
|---|---|---|
|I GIVE YOU WATER|You give me water.|Reverses giver and recipient|
|YOU GIVE I WATER|I give you water.|Reverses giver and recipient|
|HELLO YESTERDAY MORNING I FEEL GOOD HOW YOU|I feel good today. How are you?|Changes time; loses greeting/morning|
|I WANT WATER NO|I want water|Drops late negation|
|YESTERDAY I WORK TOMORROW I SLEEP|I work tomorrow.|Loses one event and changes time|
|MY FAMILY HAPPY I LOVE THEY|I'm so happy for you! I love you too.|Answers the input; changes participants|

The few-shot condition often copied example outputs or discussed the demonstrations;
HOW YOU became “How can I assist you today?” The direct condition correctly renders
that full question, but YOU alone asks for more context in a long message. I and
I WANT remain appropriately incomplete in the direct condition, an advantage over
tiny on those particular prefixes, but this does not offset full-input semantic errors.

On20phone traces, Qwen direct timeout retains full-input text9/20; conservative
commitment finalizes3/20tails. Revisable captions have mean normalized erasure1.2825
and maximum simulated serial-queue backlog1.973s. Few-shot Qwen finalizes1/20tails,
has frozen-text conflicts in2phone cases, and reaches7.302s maximum backlog.
These are desktop prompt/configuration results, not limits of all Qwen models or
optimized mobile implementations. No further prompt selection on this set was done.

## Runtime and limits

Apple M4,24GiB memory, PyTorch CPU FP32,4threads, warm-up before collection.
123 unique model requests per completed T5 condition. Median single-request times:
tiny18.15ms, FLAN107.55ms, few-shot FLAN98.98ms. Maximums47.69/1053.12/1039.96ms.
Qwen few-shot median444.61ms/max6914.22ms; direct median315.46ms/max2473.40ms.
All five conditions completed123unique requests each,615total excluding warm-up.
Timing is descriptive, not a controlled architecture benchmark: prompts and output
lengths differ, and there is one measured sample per cached input. Serial queue replay
includes those costs, with maximum phone backlog0/453/4ms respectively.
No mobile export, sustained camera workload, thermals or actual TTS were tested.

Prior large-artifact results, recognition model and data gates were not modified.
The experiment lives in this directory; root ground-truth documentation records its
status as required by the repository guide. Production state remains unchanged.

## Next justified experiment

Do not replace Finish with a fixed idle timer. A further isolated trial should focus
on faithful partial-input rendering and learned/validated boundary readiness, using
source context that survives caption commitment. Evaluate participant/time/negation
preservation before sentence fluency. Any adaptation must use a separately reviewed
training set and stay within the user's30-minute total cap; this diagnostic set must
not become both its training examples and claimed evaluation evidence.
New raw-video annotations are needed to validate genuine rest versus occlusion,
hesitation and fingerspelling. No new acquisition or training was performed here.

## Reproduction and evidence

See `README.md` for commands. `cases.json` freezes inputs and timing eligibility;
`identity.json` pins source hashes; model-condition generation/replay files preserve
outputs and event times. `outputs.csv`, `traces.csv` and `summary.csv` are reviewable
tables. `verify.py` checks source hashes and causal invariants, including a test that
changing a future word cannot change earlier captions. `integrity.json` pins the
final scripts, evidence and local model files. `verification.json` records native
full-output parity. All generated results remain development-only.
