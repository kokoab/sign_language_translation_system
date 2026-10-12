# Interval adaptation and isolated accuracy — matched plan (fixed 2026-10-13, before any new training)

Question 1. Why did interval (span-recognizer) adaptation lower pooled isolated validation accuracy in the
96.83 Squeezeformer chain (phrase 4069/4252 → interval 4062/4252)?
Question 2. Does the same happen for the flat Transformer through a full matched chain
(local adaptation → multimodal fusion/distillation → phrase adaptation → interval adaptation)?

Validation and development evidence only. No Citizen/SemLex/local protected test, no lab held-out test,
no acquisition, no deployment, manuscript, slide or production change.

## Arms

| Arm | Parent | Parent sha256 | Status |
|---|---|---|---|
| H: historical Squeezeformer | partwise/global 96.83 (Kaggle) | 5c40b133… | existing chain, reused (inputs unchanged since 2026-09-27; code changes opt-in only; rerun check R1) |
| T: flat Transformer | family benchmark `transformer` 95.50% | 5d1b4e56… | new chain |
| S: matched Squeezeformer | family benchmark `squeezeformer` (partwise/global, part depth 1) 96.30% | 95cdef72… | new chain, fresh matched control |

T and S were trained by the same family-benchmark run: same data (2,863 Citizen+SemLex), seed 1701,
optimizer, schedule, augmentation (default full roll 0.35), EMA selection. S has the same architecture as
H's parent. T versus S is therefore the primary matched comparison; H is the chain being explained.

Family checkpoints are re-wrapped as `slt_stage1_v17` containers without changing any tensor
(`prepare_parents.py`, `parents/verification.json`: logits bitwise identical to the benchmark classes,
361/378 and 364/378 reproduced). The Transformer config is tagged
`architecture=flat_transformer_family_benchmark_v17`; its 256-d embedding is the attention-pooled vector
that feeds its classifier MLP (same position as SLTStage1V17's pooled embedding).

## Stage recipe (identical for T and S; identical to H except where stated)

1. Local replay: H's selected recipe (`chain_9683_floor361_mildroll/recipe.json`): 80 epochs, patience 80,
   seed 1701, batch 64, LR 5e-5, warm-up 4, WD .03, label smoothing .1, EMA .999, class/source balanced
   .34/.33/.33, approved SemLex train, local tier owner_approved_v16_deep_clean, local mouth masking.
   **Full-roll probability 0.35** for T and S (H used 0). Reason fixed in advance: H's Variant C set roll to
   match the parent's own training augmentation; T and S were trained with roll 0.35, H's parent with none.
2. Local gates (H's rules, made relative to each parent): Citizen ≥ parent − 5 clips (H's effective
   tolerance: 366 → 361; T 356, S 359; the absolute 361 would allow H five losses and T none); SemLex ≥
   parent SemLex (measured by the same script); local masked strictly above the parent's initial local;
   eight-angle worst-angle correct ≥ parent's. Parent − 1 status is also reported. Selection: promotion-gate
   checkpoint if eligible, else Citizen-best if eligible, else promotion-gate labelled not gate-eligible.
3. Fusion/distillation: unchanged trainer defaults (hand checkpoint, fixed four-stream teacher, seeds
   1701/3407/5101, selection), fresh rebuilt cache, `--citizen-floor-correct` = selected local Citizen count.
4. Phrase-segment adaptation: identical command to H (`--selection-key reel_v2 --pair-loss-weight 0
   --pair-sampling-multiplier 1`), scoped recipe view, arm's own isolated cache.
5. Interval adaptation: identical command to H (8 epochs, seed 27927, `local_train`, exclude prefix spans,
   scoped recipe manifest, H's decoder cfg); floors = parent phrase-stage domains − 1 point (a domain below
   90 keeps the parent value), same rule as H. Selection: lowest tune-pool WER subject to floors.

Raw-pixel orientation and Core ML/phone steps are not part of this question and are not run.

## Checks and ablations (all arms; non-selecting)

- R1: rerun H's interval stage from its pinned phrase checkpoint with current code; compare every epoch.
- Interval seeds 1 and 2 (else identical): run-to-run variation of the isolated change.
- `--freeze-encoders`: only the fusion head adapts (independently trained, not a weight swap).
- `--span-share 0`: identical optimizer, KD, anchor and epochs, but no interval spans (replay only).
Ablations are reported at every epoch and at their own selected epoch; none is a deployment candidate.

## Evaluation

One evaluator scores every stage checkpoint of every arm on the same 4,252 raw validation inputs
(`artifacts/generated/unfrozen_phrase_adapt_v17/{citizen,semlex,local}_val.npz`, local mouth-masked),
keeping per-clip predictions; it must reproduce H's recorded counts. Reported: per-domain and pooled
correct, transitions (lost/gained), class-level changes, and development tune WER for interval stages.
Floors and selection rules above are applied unchanged whichever arm benefits.

## User decision (2026-10-13, after launch, before any arm result)

The paper chat noted that these arms do not match the current run_v2 lineage (revised local variant data,
full roll 0; adapted partwise 95.27% on 3,957 clips). User answer: do local adaptation from the existing
checkpoints, not from scratch, and keep full roll 0.35. The running arms already do exactly that and continue
unchanged. They remain matched to chain_9683; they are not a comparator for the run_v2 95.27% row.
