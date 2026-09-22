# Authorized BIO-preserving adaptation

One seed17621, original pretrained initialization, train only final attention block and
original BIO head. Freeze CNN, normalization, first3blocks and original decoder.
Use reviewed training intervals: B at first observed sample inside interval, I through
inclusive end, overlap/unknown masked (-1); never invent O in uncertified gaps.
Cross-entropy on known targets plus KL to original BIO distribution on all cached training
windows. This preserves a prior, not new human gap supervision. Keep two existing raw
observation-augmented variants and clean windows, one variant/example/epoch.
Cache frozen first3blocks for speed; labels/source clocks/500mslookahead unchanged.
Local59calibration chooses checkpoints by fullvideoWER with retained-sign/extra-word
constraints. Local30confirmation used once after selection. Expanded72 excluded.
Calibration and confirmation are familiar-signer development; confirmation previously
used in fixed-candidate diagnostics, so not a fresh test. Sixphrase/15gloss coverage only.
Maximum80epochs, calibration every4, stop after12epochs without eligible improvement;
LR halve after2 non-improving recognition evaluations. No arbitrary40epoch floor.
Keep selected checkpoint including frozen epoch0 fallback and best trained candidate.
No auto-promotion. Detached notification; never poll after launch.
