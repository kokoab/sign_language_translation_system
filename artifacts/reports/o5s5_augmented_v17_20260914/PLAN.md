# Bounded O5S5 augmentation experiment

User-authorized execution, 2026-09-14. Reuse the existing trainer, decoder and gates.

1. Freeze the combined manifest byte-for-byte and hash all 4,514 existing plus six O5S5 raw/replay inputs. Retain LG solely in validation.
2. Train a fresh seed17111 run from the original base for exactly 12 epochs, with unchanged 50/30/20 isolated/context/verified-background sampling. This paired seed isolates the data change; no checkpoint resume.
3. Evaluate all 12 checkpoints on 334 frozen development recordings and 318 LG positive windows; compare LG against original base and prior no-O5S5 seed. No full-narrative LG WER because annotation completeness is unproven.
4. Apply unchanged gates: connected WER <=151.478873%, I<=323, D<=11, familiar WER<=56.370656%, isolated accuracy within one point of own start, gap emissions<=15/16, runtime-inclusive median delay<=1s. Cached latency is not runtime evidence.
5. Only if an epoch passes measured accuracy/error gates, perform complete annotated-pool paced runtime; only after all gates pass, run confirmation17112. Failed gates stop further training/export.
6. Record all epochs, hashes, comparisons, limitations and decision; verify frozen inputs and focused tests.

Preflight: 14 focused training, selection and O5S5 tests passed. Existing frozen inputs unchanged; only pre-existing positive-only loader support differs in trainer.
