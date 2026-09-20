# What is wrong with the current continuous supervision?

The data contains learnable visual information. The latest experiment does not establish that it is globally unlearnable, that a different encoder is required, or that adding alphabet classes would repair it. It exposes both incomplete/ambiguous training evidence and poor transfer to held-out signers. The earlier statement that all training/inference mismatches had been removed was too strong and is withdrawn.

This audit traces all 1,166 combined-supervision rows, compares ASLLRP crops with the original annotation CSV and prior completeness flags, reconstructs the current sample timings, and measures the final checkpoint on all 38,043 training and 9,380 validation context windows. No training or Citizen test access occurred. Six selected examples have original-speed excerpts, corresponding labels, and model-input excerpts in [videos.html](videos.html). Sampled video frames were visually inspected; this is not an independent ASL annotation review of every video.

## Source clips versus the inputs actually given to the model

These are different objects. A source video can contain a full sign while the trailing input window contains only its beginning or middle.

* The existing source annotations flag **21/1,160 ASLLRP crops** as cutting through at least one annotation: 20 train and 1 validation. Matching original frame boundaries confirms **22 clipped occurrences**, all mapped to OTHER. Thus source-edge truncation is real, especially in the small contiguous subset (14/44 train clips), but the evidence does not show that most ASLLRP target signs were cropped away.
* The downstream manifest assigns `all_signs_annotated=true` to these ASLLRP rows without retaining `annotation_crop_complete`. Having annotations for every visible fragment is not the same as containing complete signs. A parser/shape check did not establish visual completeness.
* **All 5,366 known training windows end before their annotated sign ends**, by construction: the trainer selects a midpoint or an endpoint minus 1/30 second. It never provides a complete-sign endpoint in this context objective. Early recognition can legitimately use partial observations, but describing these inputs as exact cropped cores was incorrect.
* **2,349/5,366 windows (43.78%)** contain fewer than half of their retained raw samples inside the target annotation. **679 (12.65%)** also start after the sign has already begun. A 32-frame tensor does not imply 32 independent observed poses; the raw cache was sampled at approximately 20 fps and is interpolated/resampled.
* **31/1,490 known train event occurrences** contribute no context training window because the required history does not fit inside the archive. Complete coverage of the admitted windows therefore does not mean coverage of every annotated sign.
* **46 O5S5 training windows and 16 validation windows** have endpoints covered by an additional annotation. The trainer does not reject this ambiguity. Overlapping annotations may represent legitimate simultaneous signing; this count alone does not prove wrong source annotations, but a single endpoint target needs an explicit policy.
* **Two FEEL training windows contain zero retained raw observations inside their target interval.** The trainer checks for a visible hand anywhere in the window, not within the target annotation. This is a reproducible supervision defect. Example: O5S5_029_RD, FEEL at 237.770–237.830 seconds; the 0.27-second input runs 237.530–237.800 seconds and contains no raw target sample.

## Why the phase targets are not established physical boundaries

The current head predicts the phase at the trailing window's endpoint. **11,428/11,731 training windows labeled TRANSITION overlap a preceding annotated sign**. That overlap is not automatically an incorrect endpoint label: earlier frames may legitimately show a sign. However, it means these are not pure transition clips, and success depends on learning endpoint timing rather than ordinary whole-window classification.

The phase label itself is inferred from annotation gaps. A lexical annotation gap has not been separately verified as active coarticulation rather than a pause, hold, rest, or annotation convention. Earlier CTC supervision explicitly treated gaps as conservative no-emission evidence, not proof of a physical transition class. The new experiment promoted that evidence into a three-way physical-state target without equivalent visual verification.

Remaining train/inference differences: training selects two moments per sign and one midpoint per gap, while inference samples every 0.067 seconds throughout the stream. Training optimizes the two durations separately; inference averages their gloss logits while using only the shorter duration's phase. Shared normalization and matching output dimensions do not eliminate these sampling and decision differences. Their individual causal contributions have not been isolated.

## What the out-of-vocabulary material contains

ASLLRP training rows have **1,291 known and 6,089 OTHER occurrences** (occurrences across crops, not unique independent signs). OTHER means outside the accepted exact mapping, including unsupported variants; it does not necessarily mean a completely unrelated lexical meaning.

| Original ASLLRP sign type within OTHER | Occurrences | Share |
| --- | ---: | ---: |
| Lexical signs | 5,070 | 83.26% |
| Fingerspelled signs | 377 | 6.19% |
| Loan signs | 191 | 3.14% |
| Classifiers | 139 | 2.28% |
| Gestures | 134 | 2.20% |
| Compound signs | 130 | 2.14% |
| Number signs | 48 | 0.79% |

Seeing signs outside the 100 is expected in natural ASL. They must not be forced into one of the 100 classes or confused with transition. The latest loss correctly omits gloss classification loss for OTHER; its phase target is UNKNOWN. But UNKNOWN is a broad rejection category, not one consistent hand motion. The raw sample distribution is mostly UNKNOWN/TRANSITION; phase loss is inverse-frequency weighted, so raw counts alone do not prove that imbalance caused the failure.

**Adding 26 letters would not cover ordinary unsupported lexical signs.** Fingerspelling recognition also requires training on moving letter sequences, segmentation and coarticulation; adding isolated alphabet classes is not equivalent. See Shi et al., [American Sign Language fingerspelling recognition in the wild](https://arxiv.org/abs/1810.11438), and the ASLLRP [sign-type annotation guide](https://www.bu.edu/asllrp/signstream/3/SS_User-guide.pdf). These support the task distinction, not a measured gain on this project.

## Is the data learnable?

Fresh evaluation of the existing final checkpoint, with no optimizer steps:

| Context source | Train known-gloss accuracy | Held-out known-gloss accuracy | Train phase accuracy | Held-out phase accuracy |
| --- | ---: | ---: | ---: | ---: |
| ASLLRP contiguous | 226/248 = 91.13% | 48/67 = 71.64% | 80.98% | 35.42% |
| ASLLRP other | 3,811/4,324 = 88.14% | 587/980 = 59.90% | 69.33% | 51.04% |
| O5S5 | 543/794 = 68.39% | 73/228 = 32.02% | 92.82% | 24.56% |

These are correlated window-level measurements, not whole-video recognition accuracy. Known-gloss accuracy bypasses the phase gate. O5S5 supplies only KNOWN phase examples, so its phase accuracy is known-sign recall, not balanced three-way discrimination.

The model fits much of the known ASLLRP visual evidence. It has not fit phase discrimination cleanly even on training data, and held-out performance is substantially worse. Thus both supervision/task difficulty and signer transfer remain problems. There were no conflicting identical float16 input tensors among the reconstructed train/validation context samples; that narrow check does not rule out semantic ambiguity or incorrect boundaries.

The training annotations cover **73/100 known classes**; **16 of those 73** occur with only one continuous training signer. The other 27 retain isolated training only. A useful bounded system does not require every clip to contain every class, but available footage is not the same as varied continuous examples per class. Current evidence cannot guarantee that transition learning transfers uniformly to all 100 signs.

Earlier physically cropped-core training reached 186/199 O5S5 train events while staying 12/57 on held-out LG (`joint_ctc_v17_20260914/README.md`). That supports learnable training signal with a real generalization gap; it does not prove sufficient data for reliable unseen-signer operation.

## Decision and next action

Keep the 100 visible classes for this problem. Do not expand to 126 merely to explain OTHER, and do not slow/repeat already extracted frames as a supposed repair for missing motion. Slowing video for human review can help; it cannot recover observations already absent from the model input.

Before another training run, construct a reviewable subset that preserves whole target signs and neighbouring context, carries original crop-completeness flags, and excludes ambiguous or evidence-free targets. Preserve unknown intervals as unknown; require visually validated boundaries before calling annotation gaps physical transitions. Confirm on the existing encoder whether complete retained sign evidence is recognized on training and held-out signers, then test boundary localization on that same subset. This is a data/supervision repair and controlled diagnosis; the failed experiment alone does not justify another architecture swap.

No new training, runtime promotion, dataset deletion, or protected test evaluation was performed.
