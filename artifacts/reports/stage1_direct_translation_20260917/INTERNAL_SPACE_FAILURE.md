# Internal storage failure after SSD launch

No epoch completed after epoch 7. The worker failed creating provenance.json.tmp with ENOSPC after loading the model, then could not write its failure/status files. The supervisor subsequently recorded failure. BART did not start.

SSD checkpoint storage worked in preparation; it does not relocate macOS swap or all internal runtime writes. Internal space was previously about 2.7 GiB. At this user-requested check it is about 16 GiB. Raised the internal free-space requirement from 512 MiB to 8 GiB; external checkpoint-space and mounted-volume guards remain. The role of swap is inferred from timing, not measured separately. This reserve cannot guarantee capacity against other concurrent disk consumers.

Failed attempt records are preserved in failed_attempt_03. All valid checkpoints remain. Retry resumes epoch 8 from the verified epoch-7 SSD copy and retains the approved evaluation/training recipe.
