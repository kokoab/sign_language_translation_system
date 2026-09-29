# ATLAS measurement notes

Companion evidence for the technical review. Measurement environments are recorded here
so chart labels can focus on the configurations being compared.

## Streaming comparison

The two-way comparison uses the established 72 recordings and 186 reference signs:
60 familiar-signer sequences with shared phrase templates and12unseen-signer sequences.
Boundary-guided interval classification:39.78%WER. ATLAS streaming recognition:9.68%WER.
The first result is desktop recorded-sequence processing; the second is native Swift
recorded-input replay on a Mac using the iPhone implementation and settings. These are
not measurements of live-camera recognition accuracy collected on an iPhone. The full
pipeline changes, not only the decoder, between configurations.

Sources: [boundary comparison](../../artifacts/reports/boundary_expanded_eval_v17_20260922/tcn_comparison/REPORT.md),
[Swift result](../../artifacts/reports/phone_speed_v17_20260929/score_test_C.json).
The separate [CTC audit](../../artifacts/reports/capstone_ctc_comparison_v17_20260930/REPORT.md)
retains CTC predictions, training overlap and common-subset analysis. Those numerical
results are excluded from the reader-facing review and its two-way chart.

## Architecture and extraction comparisons

Base-model family/architecture times are CPU measurements on the development Mac,
using prepared inputs, rather than an iPhone comparison of every model family.
Landmark extraction timings also use the development environment. Sample membership,
training configurations and timing protocol are in the
[comparison evidence](../../artifacts/reports/capstone1_v17_revision_checklist_v1/README.md).

## iPhone execution and English assessment

The processing-time chart measures226recorded-input frames per configuration on a
physical iPhone13. All configurations already use Core ML. It compares encoder
precision and permitted recognition processors, not conversion versus no conversion.
[Profile](../../artifacts/reports/phone_speed_v17_20260929/profile_1.log).

English quality compares the same fine-tuned model with whole-sequence and incremental
processing on300generated sessions. DeepSeek generated the references and provided
automatic judgments. The source also retains the sentence-composition baseline that
is omitted from the simplified review.
[English evidence](../../artifacts/reports/stage3_multisentence_bakeoff_v17_20260929/REPORT.md).

## HELLO HOW YOU timeline

The timeline uses record `local:HELLO_HOW_YOU:ff187c3f` from the same saved Swift
recognition results used for the streaming comparison:
[recorded outputs](../../artifacts/reports/phone_speed_v17_20260929/score_test_C.json).
The reference and hypothesis are both HELLO HOW YOU. The selected intervals are
HELLO 0.6667–1.1667 s, HOW 1.4000–1.6667 s and YOU 2.2667–2.8667 s, taken directly
from the record's `start_seconds` and `end_seconds` fields. These are estimated model
intervals, not manual temporal ground truth. This is a worked example, not an aggregate
performance result.

The recording was processed using the iPhone recognition code in the development
Mac replay described above. The axis is recording time, not physical-device wall time.
The last word has a `compute_seconds` sentinel of -1 in the saved record; no processing
latency or live acceptance delay is inferred from it. Word-commit times are deliberately
not plotted. The figure shows the selected intervals in a boundary-detector/decoder row and
corresponding recognition labels. It does not plot raw boundary probabilities or all
rejected candidates. Original video stills come from
`data/raw_videos/PHRASES FIXED/HELLO_HOW_YOU/ff187c3f.mp4` (30 fps), using the nearest
frame to each interval midpoint. Exact frame indices, timestamps and the video hash are
recorded in review_assets/metrics.json. Frames retain their original appearance and
proportions. Component roles and the English/Finish behavior are explained in the document.
No English sentence or English-generation timing is attributed to this recognition-only record.

Manual clipping in the preparation discussion reflects the researchers' reported
preparation of selected recordings. It does not assert that every training interval
was manually annotated. The sign-interval training code separately records timed-label
and forced-alignment supervision and generates boundary variations for training.

## Chapter 2 literature revision — 2026-09-30

The original Section 2.1.1 vocabulary-subset passage is retained verbatim at the author's
explicit request. This is an editorial preservation decision, not independent verification
of its standardized 100-sign subset or four-step selection attribution. The
[source publication](https://arxiv.org/html/2304.05934v2) establishes the larger
isolated-sign collection and signer separation. The local vocabulary research note
continues to record the distinction between the selected 100 labels and an official
benchmark. No manifest, label membership or evaluation split changed.

Primary-source checks used for the revised literature discussion include:

- [Segmentation paper](https://aclanthology.org/2023.findings-emnlp.846/) and
  [2026 implementation](https://github.com/sign-language-processing/segmentation):
  distinct paper/code versions; teacher source distinguished from the ATLAS student.
- [MobileCLIP](https://arxiv.org/abs/2311.17049) and
  [MobileCLIP2](https://arxiv.org/abs/2508.20691): added the missing MobileCLIP bibliography
  entry and used the existing MobileCLIP2 citation for the actual feature extractor.
- [Multiple Reservoir Computing](https://doi.org/10.1371/journal.pone.0322717): retained
  60.35% Top-1, 84.65% Top-5 and 52.7 s CPU training time; no inference-speed claim.
- [Sign2Pose](https://pubmed.ncbi.nlm.nih.gov/36905057/): retained 80.90% and 64.21%
  in their respective vocabulary conditions; no matched ATLAS ranking.
- [Gamified system, institutional publication record](https://pure.athabascau.ca/en/publications/asl-recognition-and-game-based-interaction-a-machine-learningdriv/):
  retained its skeleton Transformer and interface/backend description; removed the
  unsupported entirely-in-browser execution description and omitted unchecked timings.
- [Question-sign teaching resource](https://boisestate.pressbooks.pub/pathwaysasl/front-matter/introduction-page/),
  [distillation](https://arxiv.org/abs/1503.02531),
  [dropout](https://www.jmlr.org/papers/v15/srivastava14a.html),
  [AdamW](https://arxiv.org/abs/1711.05101),
  [G-Eval](https://aclanthology.org/2023.emnlp-main.153/), and
  [Deaf-community collaboration study](https://users.umiacs.umd.edu/~hal3/docs/daume25collaboration.pdf):
  support the stated methods and interaction considerations, not certification of local
  outcomes. Existing bibliographic entries remain preserved.

## Core ML component deployment — 2026-09-30

Deployment table checked against active/v17/export_mobileclip2_image_coreml_v17.py,
export_av_boundary_coreml_v17.py, export_span_recognizer_batched_coreml_v17.py and
export_stage3_t5_coreml_v17.py. Swift LiveReelApp selects FP16 MobileCLIP2;
LiveReelEngine selects the FP16 boundary resource and LiveReelModels defaults to the
FP16 recognition resource. T5 conversion explicitly uses FLOAT32 for encoder/decoder.
Export precision is distinct from input-array types and processor permissions.
The three phone profiles vary image-encoder precision and recognition compute units;
they do not compare an all-FP32 system with an all-FP16 system or measure total export speedup.

## Before/after conversion comparison — 2026-09-30

[Matched check](../../artifacts/reports/capstone_conversion_v17_20260930/REPORT.md)
compares the actual fixed-batch recognizer export with its original checkpoint on all
378validation inputs. Top1:95.2381%→94.9735%; Top5:98.9418%→98.9418%; agreement99.7354%.
This supersedes any assumption of perfect parity for the fixed-batch app export; the
batch1export's zero mismatches describe a different export. Boundary100%agreement is
from the saved1944frame tuning-input check and is not human-ground-truth accuracy.
Phone profiles are removed from manuscript/review; original sources/assets retained.
The28ms physical-device preparation/recognition median remains one sentence.

Matched model timing on Apple M4: recognizer 27.64→11.32ms/batch8; boundary 1.39→2.00ms/window.
PyTorch CPU one thread versus Core ML ALL; prepared inputs, warmup and interleaved repeated calls.
Full protocol and raw timing samples are in the conversion report and latency.json.

Storage comparison (decimal MB): recognizer49.62→25.51; boundary9.25→4.68.
Before is saved FP32 inference state without optimizer metadata; after is full FP16
CoreML source package. Exact definitions/bytes in conversion report sizes.json.
