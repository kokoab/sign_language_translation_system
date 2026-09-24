# Unfrozen-encoder phrase adaptation — controlled test (2026-09-25)

**Question.** Does unfreezing the unified Stage-1 encoders on phrase segments raise the
oracle-interval ceiling (shipped verifier: 16/24 = 66.7% on held-out ASLLRP) without
degrading isolated accuracy?

**Answer.** Directionally yes, but not proven. Against a matched frozen control, the unfrozen
arm gains a net +2 (pad .05) and +1 (pad .10) held-out events out of 24. That is inside noise
(exact McNemar p ≈ 0.5). Isolated floors held. No promotion.

## Design
- Recipe: `active/v17/unfrozen_phrase_adapt_manifest_20260925.json` (pinned hashes).
  Trainer: `active/v17/train_unfrozen_phrase_adapt_v17.py`.
- Both arms start from `stage1_v17_unified_multimodal_student_v1`, which has the same encoders
  as the shipped model but a head that never saw phrases. They share one seed (25925) and the
  same data and objective. **frozen** trains the head only; **unfrozen** also trains the
  landmark and hand temporal encoders (encoder LR 1e-5, head LR 2e-5), 30 epochs.
- Phrase data comes only from approved-v2 **train**: 276 clips (232 local + 44 ASLLRP),
  2,892 equal-width segments. Isolated Citizen/SemLex/local train clips are replayed with
  KD (T=2) to the frozen base.
- The shipped head is *not* the initialization. It was fitted on the old 390-clip split, and
  158 of those clips are in approved-v2 validation.
- Selection uses approved local-phrase validation only (199 clips / 537 segments). The 12
  approved ASLLRP validation videos are the oracle and never steer selection.
- Guard (vs the shipped model): Citizen ≥ 95.53, local ≥ 96.53, SemLex ≥ 89.16. SemLex is
  already below 90, so it gets zero tolerance.
- Oracle inputs are frozen via `scripts/extract_oracle_inputs_v17.py`. They reproduce the
  live Core ML verifier exactly at every pad (12/19, 16/24, 16/24, 12/24, 11/24). The raw
  isolated cache reproduces the recorded base and shipped isolated metrics exactly.
- The unfrozen arm hit an MPS OOM at batch 256 (2.13 GiB cap). It was rerun with exact
  gradient accumulation (micro-batch 64, same objective). The frozen arm ran unsplit, which
  is mathematically the same objective.

## Results (selected epochs)
| | Citizen | SemLex | Local | local phrase val | oracle pad .00/.05/.10/.15/.20 |
|---|---|---|---|---|---|
| shipped reel_v2 | 96.03 (363) | 89.16 (872) | 97.03 (2810) | contaminated | 12/16/16/12/11 |
| base student_v1 | 96.30 | 89.06 | 97.10 | 63.7 | 12/16/16/12/11 |
| frozen, ep23 | 96.03 (363) | 89.16 (872) | 97.06 (2811) | 66.3 | 12/16/18/12/12 |
| **unfrozen, ep9** | 95.77 (362) | 89.16 (872) | 96.69 (2800) | **76.7** | 13/18/19/11/10 |
| unfrozen ep30 (ineligible) | 95.24 | 89.06 | 96.69 | 83.2 | 13/17/18/13/12 |

The unfrozen arm was eligible only at epochs 7 and 9. After that, Citizen sat 1–3 clips below
its floor while local phrase validation kept climbing to 83%. The held-out oracle stopped
improving after about epoch 9–13. The frozen head alone met the floors at epochs 1, 3, 4
and 16–30.

Paired oracle flips, unfrozen vs frozen:
- pad .05: +TIME, +BAD, no losses.
- pad .10: +READ, +TIME, −FRIEND.

## Reading
- **Local phrase validation (+10.4 pts) is not generalization evidence.** 5 of the 6
  validation phrase templates also occur in training, and signer ids are absent from the
  archive metadata. The later-epoch climb to 83% together with a flat oracle looks like
  template memorization.
- **Held-out continuous signing moved by 1–2 events out of 24.** This is consistent with
  the coarticulation hypothesis but does not demonstrate it. 24 events cannot resolve a gain
  this size. The frozen head also gained +2 events at pad .10 over the shipped model.
- **The guardrail held.** No isolated domain fell below 90 except SemLex, which stays at
  exactly the incumbent's 89.16. The cost is −1 Citizen clip and −10 local clips.
- The shipped model's phrase-head adaptation scores identically to the pre-phrase base on
  the held-out oracle.

## Next safe action
Nothing is promoted. A real answer needs many more held-out continuous events with curated
intervals from unseen signers than these 24. With that set, rerun both arms (≥2 seeds) and
compare paired. Checkpoints: `artifacts/models/stage1_v17_unfrozen_phrase_adapt_{frozen,unfrozen}/`.
