# SSD recovery — 2026-09-18

User authorized deleting only the corrupt temporary checkpoint and using the attached SSD. Removed latest.tmp.pth (2,063,536,128 bytes, invalid ZIP). All valid local checkpoints remain intact.

External SSD: /Volumes/secret, ExFAT, UUID 30DC5ABF-8AD5-3A96-A8E0-BF784EAAA236, approximately 283 GiB free at inspection. New model outputs go under /Volumes/secret/SLT_checkpoints_20260918/. The epoch-7 checkpoint was copied to stage1_direct_translation_20260917/epoch_07_before_ssd_resume.pth and SHA-256 matched against the verified local original. Its mmap metadata loads correctly from this filesystem; small checkpoint save/replacement/read also passed.

Recovery loads epoch-7 weights AND optimizer state, then restarts epoch 8 with the existing epoch seed. No input data, loss, architecture, learning rate or 20-epoch schedule changed. Original pre-recovery source and manifest are preserved in failed_attempt_02. ssd_recovery.json links the old checkpoint/manifest hashes; current manifests freeze the new recovery code and unchanged input hashes.

Added mount/path guards, an 8 GiB checkpoint-storage free-space minimum and a 512 MiB internal report-space minimum. Guards run before model work, each epoch and each checkpoint write. A disconnected SSD is an error, never a silent fallback to internal storage. Recovery record/checkpoint hashes are verified before loading. The English comparison also writes checkpoints to the SSD and queues on the recovered baseline process.

Validation: five focused CPU tests pass, including epoch-7 optimizer recovery, rejection of a wrong checkpoint hash, missing volume and insufficient space; existing queue and target handling remain covered. Compilation and git diff --check pass. No model-quality conclusion yet. Final action is detached baseline resume followed by the approved event-based comparison queue, with exit notifications and no polling. Keep the SSD attached while these jobs run.
