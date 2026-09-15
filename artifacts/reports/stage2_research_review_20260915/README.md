# Stage 2 and Reel: evidence before another architecture change

Completed 2026-09-16 PHT. Research and local evidence exports only; no runtime,
checkpoint, annotation, or training changes. No protected Citizen test access.

Open [the nine-video evidence viewer](videos.html). It contains slow playback,
clickable annotation times, original source paths, and saved model outputs.
[Machine-readable evidence](evidence.json) preserves the selected rows and hashes.

## What the evidence establishes

The observed repetition is real, but a single explanation would be misleading.
The recorded live decoder can emit the same label twice for one visible gesture;
the implementation also has a separate context-rollover edge case. Recognition
across signers remains weak even when exact sign boundaries are supplied. Neither
finding establishes that fast data or incorrect gloss annotations caused the
specific webcam error.

The inspected webcam session is `live_reel_stage1_v17/20260915_220852_978390`.
Its saved model hash matches the current general primary/specialist selector
(`0782d052…`). It did **not** use the newly trained joint CTC epoch-42 model.
That research model also failed promotion; its training repair is not evidence
that the webcam improved.

`scripts/live_reel_continuous_v17.py`, now open in the IDE, is a wrapper around the
same Reel runner. Its default uses tentative Stage-1 preview and a Finish-time
Stage-2 review, and enables OTHER preservation. With `--revisable-transcript`,
the runner enables live sequence decoding and disables Finish-only decoding.
These modes must be distinguished. The recorded session proves the failure in
its logged revisable selector mode; this review does not assert identical output
from every wrapper/default combination.

| Evidence | What it means | Limit |
|---|---|---|
| At source 24–27.2s, one visible chest-point episode becomes `I`, then `I I` | Over-segmentation occurs inside the visual CTC result | Intended ASL meaning is not expert-annotated |
| The first `I I` has only two accepted windows, positions 6 and 10, before Finish | Neither eight-window rollover nor English rendering caused this instance | Raw logits were not saved, so the feature/model cause of the split cannot be reconstructed from this log |
| At 12–18.5s, two separate salutation gestures become `HELLO HELLO` | Repeated output can be appropriate | Do not deduplicate equal strings globally |
| At 35–39s, chest-point then hand-to-mouth yields `MY → EAT → EAT EAT` | Classification and temporal segmentation can both fail | Visual review is provisional, not a replacement for an ASL annotator |
| Saved `I` becomes “Is that?”; `I I` becomes “Is it?” | The English renderer introduces an additional unsupported meaning | This is separate from recognizing or counting signs |

The contact sheets were visually inspected at sampled frames. The viewer provides
the actual motion for human review; gesture descriptions are deliberately not
presented as authoritative linguistic labels.

## Why a held sign can repeat

The actual `collapse_ctc_path` implementation behaves correctly for the elementary
CTC rule: `A A A` collapses to `A`, while `A blank A` becomes `A A`. CTC blank is
an alignment/no-emission symbol, **not proof of a physical pause or sign ending**.
Longer duration alone does not require a duplicate. A fragmented model path can
create another emission even though the signer has not begun a second sign.

In the inspected revisable path, the display replaces the current hypothesis; it
does not simply append one label on every webcam poll. Raising Stage-1 cooldowns
or adding a display deduplicator therefore does not address this example's source.
See `scripts/live_stage2_ctc_v17.py:129` and
`scripts/live_reel_stage1_v17.py:1240`.

There is also a reproducible **independent rollover defect**: a continuous run of A
beginning in an exiting window is locked as A; the retained portion of that same
run can decode as A again, producing locked A + current A. `roll_ctc_prefix` tracks
emission start positions but does not carry run ownership across the cut. The
report's executable counterexample demonstrates this with the production helpers.
It does not explain the two-window webcam error and must not be used to claim that
one rollover patch would fix all repetition.

Only accepted visual windows enter the CTC context. This session has 130 sequence
updates, of which 76 are rejected. Missing-hand windows can disappear from the
model's timeline; two such windows separate the accepted HELLO examples. That is
a temporal representation concern, not evidence that all 76 rejections were bad
or that they caused the early `I I`. Missing observations and verified non-signing
must remain distinct. Arbitrarily inserting blank labels into unknown gaps would
not solve this safely.

## Speed, annotation, and signer coverage

The earlier [complete data audit](../stage2_data_learnability_audit_v17/DATA_AUDIT.md)
measured median annotated durations of 0.259s for O5S5 training, 0.267s for ASLLRP,
and 0.340s for held-out O5S5 signer LG. These measurements argue against O5S5 being
uniquely too fast. They do **not** prove speed invariance: no matched speed/hold
intervention was performed on this webcam example.

The viewer includes two useful counterexamples to “just slow it down”: LG's
annotated HELLO lasts 0.460s and WHEN lasts 0.880s. The research model's pooled
exact-core predictions are LESS and SIGN, respectively. These are classification
errors with annotated boundaries supplied, not continuous CTC outputs.

The stronger generalization evidence is aggregate: repaired epoch 42 gets 199/199
training O5S5 cores through both heads, versus LG pooled 15/57 and actual CTC 4/57
(42 empty). Its connected development WER is 98.24%; familiar-phrase WER is
107.72%. An insertion-heavy model can have WER above 100%. Those results do not
justify replacing the live model.

Annotation quality has several different meanings here:

- O5S5 is explicitly **partial positive supervision**. Unlabeled time cannot be
  treated as silence or blank. Original ID gloss and locked-vocabulary mapping
  are both shown in the viewer; OTHER means outside the vocabulary.
- Some ASLLRP intervals carry a verified-reference flag; others do not. The local
  HELLO HOW YOU example has only a phrase-level reference and no word timings.
  The viewer preserves these differences rather than inventing precise labels.
- No identical-feature target conflicts or global extraction failure were found
  by the numeric audit. That does not certify every linguistic label, boundary,
  or vocabulary mapping. An ASL-qualified review is still needed to call a
  particular annotation wrong.
- Historical whole-window supervision diluted O5S5 targets to 29.1% of a 1.07s
  window on average. The old replacement sampler missed many training windows.
  These are verified defects of earlier experiments; later coverage/core repairs
  must not be ignored by blaming the same historical defects for every new run.
- 23 of 49 O5S5 training classes had only one training signer. More examples from
  the same person cannot establish signer generalization.

## Videos and exact provenance

| Clip | Reference / observation | Saved output | Output source |
|---|---|---|---|
| [Webcam chest point](media/webcam_i.mp4) | One visible gesture episode, intended gloss unverified | I I | Recorded live selector |
| [Webcam repeated salutation](media/webcam_hello.mp4) | Two visible gesture episodes | HELLO HELLO | Recorded live selector; repetition control |
| [Webcam hand-to-mouth](media/webcam_eat.mp4) | Chest point then hand-to-mouth | MY → EAT → EAT EAT | Recorded live selector |
| [ASLLRP NIGHT TIME](media/asllrp_night_time.mp4) | NIGHT TIME | FRIEND | Research epoch-42 continuous CTC |
| [ASLLRP TIME FRIEND](media/asllrp_time_friend.mp4) | TIME FRIEND, with an OTHER interval | FRIEND FRIEND FRIEND | Research epoch-42 continuous CTC |
| [ASLLRP MORNING in context](media/asllrp_morning.mp4) | MORNING among OTHER signs | HOME HOME HOME WHY KNOW WHY KNOW WHY | Research epoch-42 continuous CTC |
| [Local HELLO HOW YOU](media/local_hello_how_you.mp4) | HELLO HOW YOU; no word times | KNOW HELLO SMALL STOP HOW HOW HELP YOU YOU | Research epoch-42 continuous CTC |
| [LG HELLO](media/o5s5_lg_hello.mp4) | HELLO at source 1.520–1.980s | LESS | Research pooled exact-core classifier |
| [LG WHEN](media/o5s5_lg_when.mp4) | WHEN at source 6.260–7.140s | SIGN | Research pooled exact-core classifier |

These are selected explanatory failures plus one repetition control, not an
unbiased performance sample. Corpus roles come from the frozen validation
manifest, even where an older video directory happens to contain `train_candidate`.
The webcam file is constant-rate 15fps while its source timestamps are irregular.
The viewer uses the saved per-frame timestamp mapping, not MP4 time as source
time; the two trailing frames without saved timestamps are excluded.

## What gloss-free research actually offers

Gloss-free translation removes the requirement for human gloss sequences. It
still needs paired signing video and translated text for translation training.
It can replace **both Stage 2 and Stage 3** with visual features → temporal encoder
→ text decoder. Putting another language model after incorrect glosses retains
the current information bottleneck and does not constitute that replacement.

| Primary source | Relevant result / design | Fit to this project |
|---|---|---|
| [GFSLT-VLP, ICCV 2023](https://arxiv.org/abs/2307.14768) | Visual/text pretraining followed by direct video-to-text translation | Establishes the approach; PHOENIX/CSL evidence does not establish ASL webcam behavior |
| [Sign2GPT, ICLR 2024](https://arxiv.org/abs/2405.04164) | Pretrained vision/language models, adapters, and text-derived pseudo-gloss supervision | Useful transfer-learning evidence; not a small ready-made causal Stage-2 head |
| [FLa-LLM, LREC-COLING 2024](https://arxiv.org/abs/2403.12556) | Trains visual translation with a lighter module before a strong language model; diagnoses language-model dominance | Relevant lesson: verify visual grounding before adding language capacity. Its mBART is 680M parameters |
| [Uni-Sign, ICLR 2025](https://arxiv.org/abs/2501.15187) | Regional pose/temporal encoders with mT5-Base; optional RGB fusion | Closest pose-based challenger. [Official repository](https://github.com/ZechengLi19/Uni-Sign) announces pose-only How2Sign/OpenASL checkpoints |
| [SHuBERT, ACL 2025](https://aclanthology.org/2025.acl-long.1397/) | Self-supervised temporal representations from approximately 984 hours of ASL; hand/face RGB features plus body pose | Strong ASL representation baseline worth comparing; not a drop-in replacement for Apple Vision landmarks |
| [Adaptive simultaneous SLT, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.34/) | Learns when enough input has arrived to translate more text | Streaming output policy is a separate learned/evaluated problem; ordinary sentence BLEU does not establish good live behavior |
| [Toward Real-Time Sentence-Level SLT, July 2026 preprint](https://arxiv.org/abs/2607.09611) | SHuBERT/ByT5 with sentence finalization and deployment engineering | Its limitations explicitly say output occurs after the utterance ends; it is not evidence of incremental held-sign correctness |
| [SignLlama, August 2026 preprint](https://arxiv.org/abs/2608.09006) | Proposes pseudo-gloss CTC and visual-prioritized distillation to counter language bias | Abstract screened as a recent direction, not independently reproduced or recommended as the next architecture |

The downloaded seven full papers and selected official Uni-Sign source files are
under `papers/`. Uni-Sign's inspected loader indexes a 133-keypoint schema and
its model uses regional graph/temporal modules plus mT5. Current Apple Vision v17
features are not interchangeable with those pretrained inputs. Any public
checkpoint comparison must retain its native preprocessing in an isolated
research path; changing the production extractor is a separate decision.

An especially relevant independent study is
[Sincan et al., CVIU 2025 / arXiv 2026](https://arxiv.org/abs/2603.13240): many
reported gains diminish under standardized preprocessing, backbone, and training.
Its [official implementation notes](https://github.com/ozgemercanoglu/sltbaselines)
also warn that Sign2GPT's pseudo-gloss idea was transplanted into a different
mBART framework; that result is not an exact reproduction of Sign2GPT. This
supports controlled baseline comparisons, not dismissing every larger model.
The study's abstract and official repository were inspected; repeated PDF fetches
timed out, so this report does not claim a complete reading of that paper.

## Recommended next decision

**Stop adding training losses, specialists, and string-level repeat rules until
the next comparison has a frozen behavioral target.** The previous 54-epoch repair
established that training collapse can be overcome; it did not establish that its
multi-phase recipe is a good production design.

For the current 100-sign recognizer, the simplest justified target remains one
visual sequence encoder, one shallow temporal CTC head, and one transcript state
over absolute source time. Reuse the existing modules. Preserve the isolated
classifier only as an auxiliary training objective if it demonstrably protects
retention. A causal head does not make the current bidirectional/window encoder
causal: either account for chunk lookahead or actually change and evaluate it.
This is a candidate design, not a claim that the current failed checkpoint works.

The immediate next experiment should isolate **event counting**:

1. Freeze a small signer-disjoint evaluation set containing a normal single sign,
   the same sign held longer, two separately intended repetitions, transitions,
   hands-at-rest, and ordinary out-of-vocabulary signing. Include fluent human
   annotation of intended events; internal repeated movement can belong to one sign.
2. Replay one pinned runtime with absolute timestamps and saved framewise logits.
   Compare unchunked/offline collapse against the live result and deliberately
   move chunk boundaries. This separates a model emitting two runs from a runtime
   counting the same run twice. Preserve missing-data gaps explicitly without
   treating them as verified blank.
3. Measure extra events per held sign, retention of true repeats, missed signs,
   false emissions during rest/OTHER, and latency from source time through output.
   A single “repeat accuracy” or a cooldown that deletes legitimate repeats is
   insufficient. Correct the proven rollover ownership issue separately from
   any emission-training change; then rerun this same frozen set.

That is targeted missing data, not another bulk download of isolated clips.
Synthetic speed changes can support training experiments but cannot substitute
for real held gestures, rearticulation, or unseen signers.

For the broader **ASL-to-English translator**, my first gloss-free challenger
would be a released Uni-Sign ASL checkpoint evaluated with its native preprocessing
and explicit Finish/utterance boundaries. Compare SHuBERT if a stronger RGB-based
ASL representation is needed. Establish faithful sentence translation first,
then evaluate an incremental output policy. Neither is currently validated for
this webcam, the locked Apple feature schema, or offline iPhone deployment.

This route needs ASL–English pairs, not manual gloss for every frame.
[How2Sign](https://how2sign.github.io/) provides such pairs and manually realigned
sentence timestamps; its original sentence clips may be misaligned, and its
published data license is noncommercial research. Use the realigned boundaries
for a research comparison. Do not manufacture gloss truth by treating English
word order as ASL sign order.

The existing renderer's `I → Is that?` error also warrants a literal-gloss output
option during diagnosis so that fluent English cannot conceal recognition errors.
For a future direct translator, assess semantic omissions, invented content, and
repeat meaning with fluent ASL reviewers in addition to automatic text metrics.

## Verification and remaining uncertainty

`venv/bin/python artifacts/reports/stage2_research_review_20260915/build_evidence.py`
reproduces the CTC counterexamples and verifies selected video/model hashes before
building the viewer. Media were decoded for sampled-frame inspection and probed
for valid streams/durations. Report checks do not constitute a fix or a new model
benchmark. See [verification.json](verification.json) for the final check record.

We know where the observed duplicate enters the pipeline. We do not yet know
which feature/model mechanism split that particular gesture, because its raw
logits and expert event labels were not saved. No paper or audit can honestly
guarantee a replacement architecture will solve that without the controlled
hold-versus-repeat evaluation above.
