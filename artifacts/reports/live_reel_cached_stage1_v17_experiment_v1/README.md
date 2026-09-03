# Exact-input cached Reel Stage-1 experiment

This separate live path removes repeated MobileCLIP work without dropping a modality,
view, or temporal position. `scripts/live_reel_cached_stage1_v17.py` hashes each RGB
hand crop and reuses its embedding only when the crop bytes are identical. The
accepted `scripts/live_reel_stage1_v17.py` command still uses its original verifier.

## Rejected shortcuts

Validation-only ablations were screened before implementation. Reducing the 16 hand
time points to 12 or 8 lost one correct Citizen validation prediction (363 to 362).
Removing the union view, retaining only the union view, or removing the union for
one-hand frames also lost one or more correct predictions. These approximations were
not implemented. No test or sealed split was accessed.

## Parity and timing

- A real Core ML benchmark over two growing windows returned bit-exact hand embeddings.
  Median image-encoding time changed from 784.07 ms to 563.86 ms (1.39x) with 35 cache
  hits and 40 misses. This benchmark reflects deliberately overlapping windows, not a
  guaranteed live speedup.
- A matched five-video local phrase replay produced identical final sequences in the
  baseline and cached lanes: `GOOD EASY`, `HELLO HOW YOU`, `MY NAME`, `HELP I`, and
  `GOOD READ`. Thus cached-vs-baseline exact count remained 2/5 and no token output
  changed; this test checks regression, not general accuracy.
- Across 14 verifier calls per lane, median full verification changed from 359.00 ms to
  304.15 ms and median hand encoding from 272.51 ms to 215.15 ms. The cached lane saw
  86 hits and 468 misses (15.52%). Ten verifier windows had exactly matching start/end
  times and all ten produced identical label, acceptance, and score decisions. Faster
  asynchronous completion produced one additional harmless proposal in one replay.

The next gate is a user webcam comparison because saved-video timing does not establish
display FPS, camera-frame drop rate, or subjective latency.

```bash
venv/bin/python scripts/live_reel_cached_stage1_v17.py
```
