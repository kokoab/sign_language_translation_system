# Disk failure diagnosis — 2026-09-17

The baseline completed and saved epoch 7. Its epoch-8 checkpoint write failed; the same process subsequently reported errno 28 (No space left on device) when renaming its status file. The stale status and canonical state incorrectly continued to say training/running; both are now corrected. No training process is currently active.

The English comparison never reached trim evaluation or BART training: it found the failed prerequisite and stopped. Its exit handling worked, but queuing it without inspecting the already-failed baseline was an assistant oversight.

Verified latest.pth has valid ZIP structure, all stored entry CRCs pass, and torch.load(map_location=cpu, mmap=True) exposes epoch 7, seven history entries and 543 optimizer state entries. This is the usable recovery checkpoint. latest.tmp.pth is a corrupt incomplete ZIP, approximately 1.9 GiB. No files have been deleted. Original failure/status artifacts were copied into failed_attempt_02/ before correcting status.

Only about 3.0 GiB is currently free. An atomic checkpoint save must retain the existing approximately 2.2 GiB checkpoint while writing another; available space is insufficient for comfortable headroom. Proposed cleanup is only the corrupt latest.tmp.pth, after explicit permission required by AGENTS.md for checkpoint deletion. Keep epoch_01_before_resume.pth and verified latest.pth.

Recovery must start epoch 8 from the verified epoch-7 weights and optimizer, rather than using the current resume helper hard-coded to epoch 1. Add a free-space check before launch and each atomic checkpoint write; if insufficient, fail before expensive training or damaging the last valid checkpoint. Then requeue the approved comparison against the recovery process after preserving the failed queue records. No accuracy or mobile-readiness conclusion can be drawn from this failure.
