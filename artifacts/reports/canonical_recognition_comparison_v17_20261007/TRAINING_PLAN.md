# Preserve the 96.83% checkpoint while testing current training fixes

The selected historical part-wise checkpoint reproduced 96.8254% on the canonical
validation inputs. Preserve it unchanged. User authorized current orientation-robust
training fixes and a fast comparison, not a forced favorable result.

Use the existing current `active/v17/train_stage_1_v17.py`, strict exact-checkpoint
initialization, unchanged architecture, current geometry-safe augmentation, mirrored
hand swapping, missing-value handling, schema/label-map checks, balanced approved
Citizen/SemLex replay, gradient clipping and EMA selection. No new data, local-domain
supplement, test access, teacher training or production replacement in this first step.

Two paired fine-tuning arms, seed1701, same starting weights and training budget:

- Control: current trainer with mild roll only (full-roll probability0).
- Orientation treatment: full-circle roll probability0.35, maximum180degrees,
  mild branch12degrees, as recorded for the selected orientation-robust run.

Both use20epochs, patience20, batch64, LR0.00005, two warm-upepochs,
weight decay0.03, label smoothing0.1, EMA0.999, source probabilities0.5/0.5.
This is a bounded warm-start screening experiment, not a from-scratch or multi-seed study.
Retain the original at epoch0; do not promote an inferior upright-validation checkpoint.
Evaluate saved selections at the existing eight landmark-roll angles. Landmark rotation
is a robustness diagnostic, not raw-camera rotation or independent phone generalization.

Mouth masking belongs to later local-domain replay, fusion distillation to multimodal
training, and interval adaptation to sequence inputs. They are separate stages and
must not be silently mixed into this baseline. Rejected optional architectural tricks
are not bug fixes. The existing deployed chain does not prove direct weight inheritance
from this historical96.83checkpoint; a new downstream chain will need explicit provenance.

Launch detached with status, logs and desktop completion notification. Do not poll
training. No manuscript edits; discuss measured outcomes first.
