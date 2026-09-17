# Epoch-13 recovery — 2026-09-18

Six epochs were successfully saved after the file-handle change (8 through 13). The SSD returned EIO while saving epoch14, including flush/fsync. No process remains from that attempt; BART did not start. This is a repeated storage-path failure; the precise hardware/driver cause is not established.

Epoch13 latest.pth was copied from SSD to internal epoch_13_before_internal_resume.pth. Source/copy SHA match, all ZIP CRCs pass, and mmap metadata confirms epoch13 with 13 history entries. Training loss at epoch13: translation0.411538, isolated0.004517; these are training losses, not validation accuracy. All valid source checkpoints remain intact.

Resume at epoch14 using the copied weights and optimizer. Both baseline and BART now write to internal artifacts/models rather than continuing SSD writes. About14GiB was free internally before the2.37GB copy. Keep8GiB startup headroom for model loading; once loaded, require4GiB on the checkpoint filesystem for the2.37GB atomic save plus reserve, and1GiB for internal reports. This distinguishes startup allocation from steady training; it does not eliminate storage checks.

Sources and failed records preserved in failed_attempt_05 (baseline) and failed_attempt_04 (comparison). internal_recovery.json links the verified source manifest and checkpoint; current code hashes updated, inputs and training recipe unchanged. Final translation evaluation remains pending. No promotion or Citizen test access. Resume/queue remain detached with exit notification and no polling.
