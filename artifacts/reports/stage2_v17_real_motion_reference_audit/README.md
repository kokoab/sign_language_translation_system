# Genuine motion reference audit

Generation remained stopped. This report characterizes real downloaded data and the two rejected pilots; it does not create training samples.

The follow-up source-balanced result is documented in [ALL_REAL_EXPERIMENT.md](ALL_REAL_EXPERIMENT.md).

## Compatible v17 experiment

| Source | Archives | Trajectories | Frames | Both hands detected | Both hands complete | All 61 nodes | Presence-change steps | Jerk p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| asllrp_contiguous | 44 | 57 | 1824 | 97.2% | 88.3% | 64.3% | 8.4% | 0.87116 |
| asllrp_other_ctc | 879 | 3704 | 118528 | 96.7% | 85.5% | 60.0% | 11.3% | 1.10925 |
| how2sign_unlabeled_continuous | 1026 | 4938 | 158016 | 95.3% | 84.8% | 65.9% | 9.0% | 1.02248 |
| local_phrases_v17 | 390 | 1144 | 36608 | 39.5% | 33.0% | 0.0% | 17.2% | 0.84260 |
| ncslgr_public_continuous | 166 | 561 | 17952 | 61.5% | 31.7% | 23.5% | 28.9% | 1.38944 |
| rejected_generated_v1 | 30 | 30 | 2857 | 29.7% | 22.3% | 9.1% | 18.1% | 0.44921 |
| rejected_generated_v2 | 30 | 30 | 2771 | 100.0% | 100.0% | 100.0% | 0.0% | 0.35231 |
| two_m_flores_asl | 155 | 2686 | 85952 | 88.1% | 80.9% | 63.0% | 7.7% | 0.80802 |
| youtube_asl | 103 | 802 | 25664 | 74.5% | 61.7% | 31.3% | 14.9% | 0.93987 |

## Local phrase inventory

All nine raw phrase folders were inspected. The v16-era presence audit below covers all 780 clips but is reference-only because that schema cannot be mixed into v17 training.

| Phrase | Raw clips | Legacy reference clips | Both hands detected | All 61 nodes |
| --- | ---: | ---: | ---: | ---: |
| GOOD_MORNING | 60 | 60 | 59.1% | 0.0% |
| HELLO_HOW_YOU | 140 | 140 | 46.4% | 0.0% |
| I_WANT_FOOD | 60 | 60 | 39.8% | 0.0% |
| MY_NAME | 60 | 60 | 59.7% | 0.0% |
| PLEASE_HELP_ME | 140 | 140 | 56.9% | 0.0% |
| SORRY_I_LATE | 140 | 140 | 3.6% | 0.0% |
| THANKYOU_FRIEND | 60 | 60 | 58.8% | 0.0% |
| TOMORROW_SCHOOL_GO | 60 | 60 | 60.8% | 0.0% |
| YESTERDAY_TEACHER_MEET | 60 | 60 | 72.4% | 0.0% |

## Frozen transition-model transfer experiment

A deterministic 4–12 frame interval was masked in every compatible genuine train-side window. The existing frozen How2Sign+YouTube model was compared with endpoint interpolation; no phrase was generated.

| Source | Windows | Improvement over interpolation | Windows improved |
| --- | ---: | ---: | ---: |
| local_phrases | 1144 | 14.0% | 50.1% |
| asllrp_contiguous | 57 | 29.0% | 68.4% |
| asllrp_other_ctc | 3704 | 21.7% | 73.0% |
| two_m_flores_asl | 2686 | 21.0% | 66.5% |
| how2sign | 4938 | 26.0% | 77.0% |
| ncslgr | 561 | 27.7% | 69.7% |
| youtube_asl_train | 802 | 7.7% | 69.5% |

## Result

- The v2 anatomy completion is a recognizer-distribution failure: all 61 nodes are present in 100.0% of v2 frames versus 54.6% across the pooled genuine v17 train-side sources.
- A detector mask is an observation, not an anatomy rig. Missing or inactive detected hands in genuine data must not be rewritten as always observed.
- The local videos show continuous arm travel, preparation, overlap, retraction, and rest across signs; isolated medoid concatenation plus a short masked gap does not model that full phrase trajectory.
- The frozen How2Sign+YouTube inpainter transfers positively to every compatible genuine corpus, but local phrases are the weakest practical result: 14.0% aggregate improvement and only 50.1% of windows improved. It is not ready to drive local phrase generation.
- Generated v1 and v2 remain review evidence only and are prohibited from recognition training, validation, and testing.
- The next experiment should learn or retrieve full genuine phrase motion with source-balanced sampling, while keeping rendering anatomy and recognizer observation masks as separate outputs.

The contact sheet is `local_phrases_contact_sheet.png`; each row samples one of the nine genuine local phrases.
