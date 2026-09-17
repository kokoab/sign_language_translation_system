# Direct-translation resume verification — 2026-09-17

The first run completed epoch 1 and failed during epoch 2 with an Apple MPS out-of-memory exception in Adafactor. No final translation-quality results exist yet.

- Epoch 1: all 994 translation examples and 1,901 isolated examples covered in 497 steps; mean translation loss 2.902774, isolated loss 0.065100, elapsed 142.043 seconds.
- Train-only preflight passed: loss 3.875558 to 0.006077, with gradients reaching Stage 1, projection and text model. This verifies optimization wiring, not translation quality.
- The failure reported 7.31 GiB allocated and 22.32 GiB other GPU allocations against a 30.19 GiB limit. The safety cap remains enabled.

## Memory correction and validation

Fixed masked padding uses 256 visual tokens and 70 target tokens, bounding large-model graph shape variants without truncating admitted inputs. Unused MPS memory is released around optimizer updates. Checkpoint saving preserves shared tensor storage instead of separately copying aliased parameters.

Two focused real-network tests pass, including equality of translation loss with masked padding versus unpadded inputs. Four real joint optimizer updates on maximum-length training inputs passed from the saved epoch-1 checkpoint, including isolated auxiliary loss. Observed driver allocations at the recorded measurement points were at most 10.33 GB before cleanup and 8.78 GB after cleanup; these are not continuous peak-memory measurements. The four test updates were discarded. This is evidence to retry training, not a guarantee that the full run will finish.

The model, all input hashes, splits, learning rates, loss and fixed 20-epoch recipe remain unchanged. Original failure artifacts are preserved in `failed_attempt_01/`; `epoch_01_before_resume.pth` preserves the epoch-1 checkpoint. Resume restarts epoch 2 from that checkpoint and its optimizer state. Epoch-boundary reseeding uses seed + epoch; interrupted epoch-2 partial updates are discarded, so this is not a bitwise continuation of that partial epoch.

Code changes: `active/v17/direct_translation_v17.py`, `test/test_direct_translation_v17.py`, and this directory's `run_experiment.py`. `resume_provenance.json` records checkpoint, manifest and memory-check hashes. No Citizen test access or live-demo changes. Next action: detached resume with completion/failure notification; inspect the final report only after exit or a user-requested update.
