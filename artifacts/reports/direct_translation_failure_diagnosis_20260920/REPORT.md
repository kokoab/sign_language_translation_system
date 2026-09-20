# Direct-translation failure diagnosis

## Finding

The failed hybrids are **caption memorizers with weak visual routing**, not working ASL-to-English translators. The main failure is the experiment design: 994 fixed sentence pairs were used to jointly tune 589,390,373 mT5-hybrid parameters or 146,409,509 BART-hybrid parameters for 20 passes. The isolated Stage-1 encoder is good at its 100-sign classification task, but a linear projection and sentence loss did not turn it into an open-vocabulary continuous-language encoder.

This is not mainly a signing-speed failure. It is also not evidence that all How2Sign annotations are bad. The data has real annotation/context noise, but the dominant measured failure is memorization plus poor visual-text alignment.

[Open the side-by-side video and annotation viewer](./videos.html). It contains all 12 held-out clips, both hybrid outputs, the native Uni-Sign output, and the training videos whose annotations the hybrid outputs copied most closely. It also contains the eight strongest automatic annotation-risk cases.

## Decisive evidence

| Evidence | mT5 hybrid | BART hybrid |
|---|---:|---:|
| held-out outputs exactly equal to a 994-row training caption | 1/12 | 11/12 |
| nearest-training-caption similarity ≥0.80 | 6/12 | 12/12 |
| median nearest-training-caption similarity | 0.803 | 1.000 |
| final paired BLEU / chrF | 1.076 / 18.989 | 1.061 / 17.464 |
| correct-pair chrF percentile among 2,000 random reference permutations | 99.6% | 98.4% |

BART copied 11 training captions verbatim. Its zero-visual control produced the same training caption, “A bladder infection is diagnosed at your veterinary clinic,” for all 12 clips. mT5 copied or lightly recombined training captions; six outputs exceed 0.80 similarity and its median is 0.803. This behavior explains the fluent but unrelated sentences.

Correct visual pairing is not completely irrelevant: paired chrF lands above 99.6% of random permutations for mT5 and 98.4% for BART. That is weak evidence that the video changes which memorized caption is selected. It is not evidence of compositional translation. Paired BLEU is only around the 75th percentile, and the saved one-step mismatched controls have higher BLEU than correct pairing for both models.

The unchanged native pose-only Uni-Sign system scored BLEU 8.206 / chrF 38.059 on these same 12 rows. Its output is still unreliable, but the gap shows that the text annotations and clip speed are learnable enough for a matched continuous pose encoder. Replacing Uni-Sign's pose encoder with our isolated encoder and a linear bridge removed most of that ability.

## What failed in the architecture

Our hybrid is Stage-1 Squeezeformer applied independently to non-overlapping 32-frame windows, followed by a single linear projection into a pretrained text model. Every window restarts the Stage-1 temporal context. The text encoder can attend across the concatenated tokens, but the visual encoder was originally optimized to collapse one isolated clip into one of 100 classes. There was no contrastive video-text alignment, masked visual-language pretraining, or full continuous-pose pretraining before joint sentence tuning.

That differs from the methods we were trying to borrow from:

- [GFSLT-VLP](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.html) first aligns visual and language representations with contrastive and masked pretraining, then initializes translation from those aligned encoders.
- [Uni-Sign](https://arxiv.org/abs/2501.15187) uses generative sign-language pretraining at large scale, including a reported 1,985 hours of paired CSL-News video and text, before downstream fine-tuning. Its native pose path uses region-specific spatial-temporal graph processing over the continuous sequence.
- Our experiment retained Uni-Sign's text component but discarded the visual encoder that had learned the compatible representation. The 994-pair joint fit did not recreate that pretraining.

The smaller BART run confirms that model size was a speed and storage problem, not the alignment solution. Cutting parameters from 589.39M to 146.41M made training faster, while the model still memorized 11/12 outputs exactly.

## What the data says

| Audit | Result |
|---|---:|
| paired training rows | 994 |
| paired video duration | 1.63 hours |
| training signers | 4 |
| rows from the two largest signers | 80.4% |
| unique English word types | 2,741 |
| isolated Stage-1 labels | 100 |
| held-out rows / source videos | 12 / 4 |
| invalid extracted windows | 6 / 4,813 |
| clips uniformly downsampled to 256 frames | 72 / 994 |
| median hand presence | 98.7% |

The local paired set is only 3.19% of How2Sign's roughly 31,164-row training split and covers four adaptation signers. Two signers contribute 80.4% of its rows. The targets contain 2,741 English word types, while isolated supervision covers 100 chosen signs and ASL-to-English translation is not a word-for-word mapping. High isolated accuracy therefore does not supply the missing open-vocabulary sentence alignment.

Extraction did not globally collapse: only 6 of 4,813 windows are invalid, median hand presence is 98.7%, and just 72 clips were temporally downsampled. The median clip duration is 5.52s. Speed and missing landmarks can hurt individual rows but do not explain the system-wide caption copying.

The training clips use manually realigned timestamps, as documented by the [source mirror](https://huggingface.co/datasets/martinctl/how2sign-asl-clips). The annotations are still weak sentence-level supervision. Eight of 994 rows exceed six English words per video second and are flagged in the viewer; this is a risk proxy, not proof of a wrong label. More broadly, [a Deaf-signer human study of How2Sign](https://arxiv.org/abs/2406.11049) reports that 5% of sampled realigned clips omitted the relevant content and that discourse context was needed for key details in 33.3% of cases. Annotation/context noise is real, but it is secondary to the measured memorization failure.

## Decision for Stage 2 and Reel

Do not run another language-model swap on these 994 pairs. Do not use unconditional Stage-3 deduplication: text alone cannot distinguish a held sign from an intentional repeat.

For the current 100-sign mobile system, keep the problem bounded. Use the existing Stage-1 encoder with one shallow temporal sequence head over unpooled features, trained to emit blank plus the 100 glosses. The missing data is targeted continuous boundary supervision: held signs, the same sign intentionally repeated with a real release/re-entry boundary, transitions, rest, and ordinary non-sign movement from signer-disjoint people. CTC is still appropriate for collapsing a held emission run; the data must teach the model when a new event begins. Stage 3 should only turn the recognized gloss sequence into English.

Gloss-free translation remains a separate offline research track. A valid next test would retain a continuous pose encoder already aligned to text, or pretrain one on the full realigned How2Sign training set before sentence decoding. It should use the official validation split, checkpoint selection on validation loss/translation quality, and a copying/retrieval control like this report. It should not reuse this isolated-window-to-linear-projector recipe.

## Files and limits

- `evidence.json`: machine-readable data, feature, memorization, and pairing audits.
- `videos.html`: all held-out examples and the copied-caption training videos, plus annotation-risk rows.
- `provenance.json`: exact input hashes and protected-split statement.
- `verification.json`: consistency checks and report hashes.

The 12 held-out rows come from four source videos and remain too small for a translation benchmark. Similarity matching diagnoses copying; it is not a semantic metric. No Citizen official test or How2Sign test split was accessed, and no new model was trained.
