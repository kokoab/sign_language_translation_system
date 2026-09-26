# Streaming segmental decoder + Apple Vision boundary student — result (2026-09-27)

User acceptance (set 2026-09-27): held-out 72 videos / 186 signs WER <= 25%, precision >= 90%,
recall >= 80%; sign end -> word shown < 0.5 s; vocabulary-only; MediaPipe only if as fast/accurate
as Apple Vision, otherwise distil.

## Held-out result (streaming, frozen config, Apple Vision only)

Config frozen in `frozen_stream_config.json` before this run; per-video output in
`final_test_streaming.json`.

| | WER | Correct | S / D / I | Precision | Recall |
|---|---:|---:|---|---:|---:|
| Start of day: frozen DGS + Reel, offline | 39.78% | 127/186 | 12 / 47 / 15 | 82% | 68% |
| Start of day: live TCN boundary option | 62.90% | — | — | — | — |
| **New pipeline, all** | **13.44%** | **173/186** | 4 / 9 / 12 | **91.5%** | **93.0%** |
| local60 (familiar signer_02) | 9.26% | 158/162 | 1 / 3 / 11 | 92.9% | 97.5% |
| asllrp12 (unseen JONATHAN) | 41.67% | 15/24 | 3 / 6 / 1 | 78.9% | 62.5% |

Latency, sign end (segment end) -> commit frame, 180 non-end-of-file commits: median 0.00 s
(identity is usually shown before the boundary closes), p90 0.35 s; <= 0.40 s for 97.2%,
<= 0.45 s for 99.4%, max 1.25 s (one word). 8 commits were forced by clip end and excluded.
Measured per-frame compute on this M4 (12 test videos, 928 frames, recognizer on MPS, per-frame
hand-crop cache): median 39 ms, p90 56 ms, p99 138 ms (Vision+crops 17.6, boundary 1.8, span
scoring 19.4, DP 0.1). Adding it, ~97% of words show within 0.5 s. Desktop timing only.

## Pipeline

1. Apple Vision per frame (20 Hz), hand crops encoded once per frame.
2. Boundary: `artifacts/models/av_boundary_student_v17_l6_a/model.pth` — 4-layer transformer on the
   live `boundary_features`, 300 ms lookahead, distilled from the pinned DGS pose segmenter
   (MediaPipe, 500 ms). MediaPipe itself measured 73 ms/frame vs Apple Vision 10.7 ms (same load),
   so it fails the user's speed rule and is only an offline teacher.
3. Candidate spans: BIO runs + B peaks, plus dense grid spans inside active regions (the upstream
   argmax decode never splits contiguous B/I, so un-paused adjacent signs merged).
4. Recognizer: `artifacts/models/span_recognizer_v17_local_a/best_model.pth` — reel_v2 with encoders
   unfrozen on 6,254 decoder-matched spans from 282 local train clips (forced alignment of the known
   transcript over DGS candidates + jitter), isolated KD replay. Isolated: Citizen 95.24,
   SemLex 89.67, local 96.31 (floors: >=90, SemLex keeps 89.16, <=1 pt loss).
5. Semi-Markov DP (sign/rest, unknown-label floor theta 0.5), same-gloss collapse within 1 s
   (reduplicated HOW/FRIEND/GOOD), fixed-lag commit (lag 1), early identity commit (open segment
   stable 3 steps at q >= 0.9), soft-boundary wait 2 frames.

## Evidence trail (tuning pool only unless marked)

- DGS lookahead, held-out, old gate: L10 39.78 / L8 40.32 / L6 47.85 / L4 54.84 %WER.
- Decoder alone (reel_v2): tune 51.33 -> 43.81%; any-span top-1 ceiling 159/226 -> recognizer bound.
- Recognizer A: tune 27.43% (offline DGS L8); + duplicate collapse 11.50% (L8) / 14.60% (L6).
- Held-out offline milestone (configs frozen first): DGS L6 15.05%, L8 12.37%.
- Streaming: premature commits of long signs and late splits of adjacent signs were the two
  failure modes; early identity commit fixed latency (78% -> 96-99% under 0.5 s frame latency).
- Rejected: recognizer B (+11,534 ASLLRP spans, local prefix crops) — tune 15.93% vs A 12.39%,
  ASLLRP JONATHAN validation 106/279 vs 100/279 with more insertions; stability-k commit (only
  added delay); student with L6 teacher target (tune offline 11.95% vs 10.62%).

## Limits (do not overstate)

- local60 is familiar signer_02 on the same six phrase templates used in recognizer training.
  Nothing here shows arbitrary sentences or unseen local signers.
- Unseen fluent ASLLRP signing stays weak: 15/24 held-out, ~76% WER on 222 JONATHAN validation
  videos (vocab-only). Out-of-vocabulary signing is not gated (user decision).
- Offline frame-accurate simulation of the live loop from video files; not yet wired into the app
  shell, not run on a live camera, not measured on iPhone.
- The held-out set was evaluated twice (offline milestone, final streaming), both after freezing.

## Artifacts

Code: `scripts/segmental_lab_v17.py`, `scripts/train_span_recognizer_v17.py`,
`scripts/train_av_boundary_v17.py`, `active/v17/av_boundary_v17.py`,
`scripts/prepare_youtube_distill_v17.py`, `scripts/download_youtube_asl_segments_v17.py`.
Caches (9.4 GB, regenerable): `artifacts/cache/segmental_decoder_v17/`. YouTube-ASL: 305 channel
segments (30 s, video only) in `data/local/youtube_asl_boundary_distill_v17/clips`; teacher/student
caches for ~110 of them (preparation paused).

## Wired into the app (2026-09-28)

Launch: `venv/bin/python scripts/app_shell_v17.py --segmental` (opt-in; default Reel unchanged).
Headless file run: `venv/bin/python scripts/live_segmental_v17.py --video <clip> --no-display --no-speech`.
Runtime: `active/v17/segmental_runtime_v17.py` (decoder functions shared with the lab), runner
`scripts/live_segmental_v17.py`, replay harness `scripts/replay_segmental_v17.py`,
tests `test/test_segmental_runtime_v17.py` (4 pass; app shell/integration 57 pass).
Live-only changes vs the simulation: bounded decode window (origin advances through rest,
never past an undecided segment), per-frame hand-crop embedding cache, masked-future flush of the
last 300 ms on finish (streaming boundary equals whole-clip boundary on 1,944/1,944 frames).

Held-out replay through the live code, same frozen config (`live_replay_test.json`):
WER 11.83%, 176/186, S/D/I 2/8/12, precision 92.6%, recall 94.6% (local60 8.64%, 159/162;
asllrp12 33.3%, 17/24). Sign end -> word shown incl. measured compute: median -0.09 s (words
usually appear before the sign ends), p90 0.38 s, 98.9% < 0.5 s, max 0.59 s. Frame compute on
M4: median 38 ms, p90 56 ms, p99 83 ms. Tuning replay: 11.06%, P 94.9, R 90.7.
Not yet run on a live camera by the user; not on iPhone.
