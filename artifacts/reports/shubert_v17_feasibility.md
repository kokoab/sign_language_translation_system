# SHuBERT feasibility for v17

**Decision:** defer weight download and full integration. SHuBERT is a promising
research teacher, but it is not the next production or mobile experiment.

## Audited upstream

- Repository: `ShesterG/SHuBERT`
- Audited commit: `cc1929326075bfbad7ad73159b2acf84356059bb`
- Upstream commit date: 2025-09-10
- Local shallow clone: `artifacts/generated/shubert_upstream/` (57 MiB; no weights)
- Repository README says the project is primarily MIT licensed. The separately
  distributed checkpoint files do not include a visible model card or weight license
  in the repository, so their terms must be confirmed before downloading or using
  them beyond a bounded research probe.

## What the released inference path requires

The official quickstart does not accept v17 Apple Vision archives. It requires:

1. YOLOv8 signer crops from every raw video.
2. MediaPipe face and hand landmark detection.
3. Separate face, left-hand, and right-hand video crops.
4. Fine-tuned DINOv2-S/14 feature extraction for the three RGB streams, producing
   384-dimensional frame embeddings.
5. A separate 14-dimensional normalized MediaPipe body-pose stream.
6. A custom Fairseq environment and CUDA-only inference script.
7. The SHuBERT base encoder: 12 Transformer layers, width 768, FFN width 3072, and
   12 attention heads. Its output is saved as every layer's contextual frame sequence.

The public inference script hard-codes `.cuda()`, assumes equal temporal lengths for
all four streams, loads checkpoints non-strictly, and provides no batching or padding
mask for variable-length samples. The README's downstream fine-tuning section is
still `TODO`.

## Compatibility conclusion

- The encoder cannot be silently attached to v17 features: its RGB and MediaPipe
  frontends define a different input distribution and schema.
- Reusing the current MobileCLIP2 hand features or Apple Vision body features would
  be an unvalidated input substitution, not SHuBERT inference.
- Running the official preprocessing over the current train/validation corpora would
  be a large, GPU-oriented extraction job and would duplicate working v17 crops.
- The complete stack is inappropriate for direct iPhone deployment. At most, it can
  become an offline teacher whose logits or representations are distilled into the
  compact v17 model.

## Safe future probe

Only after checkpoint names, sizes, hashes, and terms are known:

1. Download the three official weights to an isolated Kaggle dataset, not the laptop.
2. Reproduce official inference on a tiny Citizen-train smoke set with unmodified
   MediaPipe/DINO preprocessing.
3. Freeze SHuBERT and train a small isolated-sign head using Citizen train only.
4. Select on Citizen validation and run the unchanged SemLex validation diagnostic.
5. If it clearly improves cross-domain accuracy, use it only as an offline teacher
   for distillation. Do not access Citizen or SemLex test.

