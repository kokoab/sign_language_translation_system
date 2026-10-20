# PyTorch sources for retained mobile models

The retained Core ML packages and English binary remain untouched. These are source checkpoint copies, not newly trained or converted models.

- `recognizer_primary.pth`: retained primary interval recognizer.
- `recognizer_refined.pth`: refined recognizer used for the adapted output. The mobile dual package uses the refined encoders/fusion plus the preserved original primary fusion head.
- `recognizer_auxiliary_head.pth`: auxiliary head included inside that same mobile package. It is not an additional standalone boundary model.
- `boundary_student.pth`: source of the retained word-boundary package.
- `english/`: five original `.pt` model state dictionaries, manifest and vocabulary. Together these produce the retained English `.bin`; they are not T5.
- `MobileCLIP2S0_original.safetensors`: original pretrained MobileCLIP2 checkpoint. This source contains image and text towers; the phone package exports only the image tower. Its native format is safetensors, not pth.

`.pt` and `.pth` are both conventional PyTorch checkpoint extensions. They are preserved rather than renamed. Checkpoints require their matching Python architecture definitions; they are not necessarily self-contained graphs that Netron can fully display. No export or inference was run for these copies.

`PYTORCH_SOURCES.json` records source paths, roles, sizes and verified SHA-256 hashes. Original sources and app loading are unchanged.
