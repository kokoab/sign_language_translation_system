# SSD checkpoint writer diagnosis — 2026-09-18

The recovered run reached the save after epoch-8 computation, then failed renaming latest.tmp.pth on the ExFAT SSD. The file is now visible but truncated to 1,879,048,192 bytes and is not a valid ZIP. It is not a usable checkpoint. Epoch 7 remains the verified recovery point; BART has not started.

Two isolated CPU storage experiments were run with the same 2.4GB synthetic tensor and SSD, without model training:

- torch.save(payload, filename): failed with a PyTorch stream-writer unexpected-position exception.
- torch.save(payload, Python binary file handle), flush, fsync, close and rename: passed. Output 2,400,001,577 bytes; every ZIP entry CRC verified. Elapsed save/rename/verification 27.80 seconds.

This reproduces a write-path failure on this setup and establishes a tested alternative. It does not identify the underlying PyTorch/ExFAT driver defect or guarantee freedom from future storage failures. Updated only the shared checkpoint serialization path, also used by the pending BART run. Uses a distinct temporary filename so the corrupt prior output remains untouched. All valid checkpoints remain intact.

Evidence: ssd_rename_diagnosis.json, ssd_full_size_probe.json, ssd_filehandle_probe.json. Prior source/manifest/failure logs preserved in failed_attempt_04. Compilation and git diff --check pass. Data, optimizer, architecture and fixed 20-epoch recipe unchanged; hashes refreshed. Resume epoch8 from verified epoch7, then queue comparison with no polling.
