# signing-voice — high

Dated evidence for constraints that still bind. The constraints themselves are
stated in `PROJECT_GROUND_TRUTH.md`; these are the receipts behind them.

2 entries, newest first. Archived from `PROJECT_GROUND_TRUTH.md` 2026-09-10.

---

## 2026-08-22 16:46 PST — signing-voice fold 0 passes the frozen gates

The corrected contrastive fold-0 experiment selected epoch 12 and is
`artifacts/models/signing_voice_v17_fold0_contrastive/best_model.pth`, SHA-256
`1f9429d3ff141a40532730c3cad8560e185153df62e88eaacc05fc8da61cba0e`.
Across seven entirely held-out train-only identities (901 examples), generated spatial
error improves 2.4475% over the class medoid, velocity error improves 5.1772%, and
acceleration error improves 6.1021%.  The frozen Stage-1 landmark branch recognizes
100% of generated glosses versus 94.4506% for the unstyled class medoids.  Exhaustive
signer-aware cross-gloss style verification uses 901 positive and 632,068 cross-signer
negative pairs: AUC improves from the untrained encoder's 0.7165 to 0.8184.  This is
credible held-out content/style/reconstruction evidence, not human naturalness.

The architecture, loss weights, stopping rule, and evaluation contract are now frozen.
Fold 1 is running unchanged; fold 2 will follow.  No sealed or project validation/test
split was accessed.

## 2026-08-22 10:15 PST — phrase-agnostic 63-voice Stage 2 selector promoted

The phrase-specific `FRIEND NOW` gate is no longer the best general development
design. A new accuracy-research wrapper keeps the 63-voice context-adapted primary in
control, applies a class-agnostic 90/10 primary/direct-transition logit blend plus a
+0.30 blank calibration, and permits the specialist to own a row only when its
different greedy sequence has the same length (at least two signs) and has no lower
exact full-path CTC probability under the specialist. It never inspects gloss labels,
phrase identities, signer IDs, or a phrase allowlist.

The cold-reloadable artifact is
`artifacts/models/stage2_v17_general_ctc_selector_v1/model.pth`, SHA-256
`0782d052f0500164a2433ebfee86dcce7413c6bcffca03fae379871ece86dc3d`.
Relative to its 63-voice primary, it improves every development domain: ASLLRP genuine
phrases from 11/24 to 9/24 edits (45.8333% to 37.5% WER; 4/12 exact), local phrases
from 7/259 to 6/259 (2.3166% WER; 92/97 exact), and JONATHAN contextual signs from
44/254 to 43/254 (16.9291% WER; 213/254 exact). It has the same 58 aggregate edits as
the former phrase-gated artifact but distributes them without encoding a known phrase
and improves local validation by one edit. The full report is
`artifacts/reports/stage2_v17_general_ctc_selector_v1/validation.json`, SHA-256
`4194dccb3723273bb4a76c0134de9b43968df292d41ee53b38d4c14b95efb8f7`.

A broader generic prefix-extension rule can recover 8/24 ASLLRP edits, but it was
rejected because it regressed local phrases to 13/259 and JONATHAN to 49/254; training
data also strongly rejects that behavior. The retained selector changes only
same-length multi-sign hypotheses. Its exact CTC dynamic program matches PyTorch's
reference CTC loss, saved-artifact reload reproduces train and validation metrics, and
the primary still carries all 63 train-only style voices. The Python/data-dependent
selector is not the compact Core ML graph; distillation remains required.

The 0.10 blend and +0.30 blank bias were selected during development exploration, so
these results remain development evidence. No sealed test or 2M-Flores `devtest` was
accessed. An independent signer/capture set is still required for generalization, and
recognition WER still does not prove perceptually natural human motion.
