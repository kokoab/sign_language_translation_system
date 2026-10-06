# Matched chain from the pinned 96.83% checkpoint (declared before training)

User decision (2026-10-07): the paper uses the part-wise/global Squeezeformer
`5c40b133…` (96.83% Citizen validation) as the isolated landmark result, and the
downstream stages are to be retrained from it. This plan is fixed before any result.

## Step 2: raw-pixel orientation with automatic orientation correction

`active/v17/evaluate_raw_orientation_v17.py --vision-auto-orient`, 100 Citizen
validation clips (one per class), angles 0/17/37/73/90/123/180/270, MPS.
Run on 96.83 and on `a7490409` (reproduction check against the recorded 91.50%
eight-angle mean, since the evaluator code has changed since August). Same code for
both. Evaluation only; no selection.

## Step 3a: local replay from 96.83

Identical to the August local replay (`stage1_v17_local_deep_clean_mouth_masked_replay_ft_v1`),
except for the parent checkpoint and the Citizen floor. Recipe verified by a 1-batch
preflight on the August parent that reproduced 362/378 Citizen, 1,765/2,896 masked local,
13,381 local train clips and identical sampler weights.

Seed 1701, max 80 epochs, patience 20, batch 64, LR 5e-5, 4 warm-up epochs, weight
decay 0.03, label smoothing 0.1, EMA 0.999, full roll 0.35/180°/mild 12°,
Citizen/SemLex/local 0.34/0.33/0.33 class/source balanced, approved SemLex train,
local tier `owner_approved_v16_deep_clean`, four lip landmarks masked on local rows only.

Citizen floor = parent − 1 clip = 365/378 (the August rule: 361 = 362 − 1).

Gates (the analogues of the August gates, relative to the new parent):
1. Citizen validation ≥ 365/378
2. SemLex validation ≥ 853/978 (parent 87.22%)
3. Local masked validation strictly above the parent's initial local score
4. Eight-angle landmark worst-angle correct ≥ parent's worst angle (2/378). This gate
   is nearly vacuous for this parent; the orientation scores are reported in full.

Selection of the landmark branch passed to fusion: `best_promotion_gate_model.pth`
if it passes all gates; otherwise `best_model.pth` (Citizen-best) if it passes all
gates; otherwise the promotion-gate candidate is used for a fusion run that is
labelled **not gate-eligible**.

## Step 3b: multimodal fusion and distillation

`active/v17/train_unified_multimodal_student_v17.py`, defaults unchanged (hand
checkpoint, four-stream teacher incl. fixed teacher landmark model, mouth/lower-face
caches, seeds 1701/3407/5101, selection rule), with `--landmark-checkpoint` = the
selected branch, `--citizen-floor-correct` = its Citizen correct count, a fresh cache
directory and `--rebuild-cache`. A reference fusion run on the August landmark branch,
again with a fresh cache, must reproduce the recorded 364/871/2812 selection before the
new result is trusted.

## Step 3c: blocked

Phrase-activity adaptation and the interval recognizer are not run: the canonical
phrase verifier reports training_ready=false and must not be bypassed.

No Citizen/SemLex/local test access. No promotion, manuscript, slide or production change.

## Variant B (user-approved after seeing the first result, 2026-10-07)

The first local replay from 96.83 produced no gate-eligible branch (Citizen ≤363 every
trained epoch versus the 365 floor; patience stop at epoch 20 while local was rising).
The user then chose to rerun with the August absolute Citizen floor and longer patience.
This rule change was made after seeing results and must be reported as such.

Changes only: `--citizen-top1-floor-correct 361` (August absolute, 95.50%) and
`--patience 80` (= the unchanged 80-epoch maximum, so the full August budget runs).
All other recipe values, data, seed and sampling are unchanged.

Gates: Citizen ≥361/378; SemLex ≥853/978 (parent, stricter than August's 839);
local masked strictly above the parent's 64.12%; worst landmark-roll angle ≥ parent (2/378).
Selection: promotion-gate checkpoint (max local subject to the Citizen floor) if it
passes all gates, else Citizen-best if it passes, else promotion-gate labelled not
gate-eligible. Fusion and raw-pixel orientation then follow exactly as in step 3b/step 2.
Output: chain_9683_floor361/. The August fusion reference already reproduced in chain_9683/.

## Variant C (corrects an error in the first run and Variant B, 2026-10-07)

Both earlier local replays from 96.83 copied the August 35% full-circle roll. Provenance
shows the 96.83 parent was trained before full roll existed, while a7490409 is the same
recipe plus full roll 0.35; so the August recipe added a large input shift only for 96.83.
Variant C keeps every Variant B setting (floor 361, patience 80) but sets
`--full-roll-probability 0` (mild ±12° branch only), matching the parent's training
augmentation and the paired fine-tune's mild control. Orientation in the pipeline is
handled by Vision auto-orientation (step 2). Gates and selection are unchanged from
Variant B; both the 361 and 365 floors are reported. Output: chain_9683_floor361_mildroll/.
