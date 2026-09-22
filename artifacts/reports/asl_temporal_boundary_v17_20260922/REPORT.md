# ASL temporal boundary experiment

Implemented the approved plan as a separate candidate. Default Reel is unchanged.

## Fixed contract

Continuous20Hz Apple Vision raw observations,64-hidden temporal convolution,4-frame future context. Two start/end outputs; no wrist activation, no motion trimming for completed intervals, no fabricated end at EOF. No physical background or inside classifier is trained from uncertified annotation gaps.

4,310deduplicated reviewed-timing events(3,518train/792validation) across1,121raw sequences. ASLLRP known/OOV timing supplies edges without lexical targets; current strict O5S5 positives remain positive-only. A parent-held subset of training selects epochs. Two seeds and skeleton/hand-relative-geometry arms share the same inputs.

## Paired whole-video results

The same12previously used ASLLRP development videos,24known reference signs. Unchanged Reel uses its actual asynchronous video scheduler. Oracle intervals are diagnostic and bypass proposal stability. Learned arms use the shared app runtime. All frames of each video are replayed; the last0.2s has no future context and remains unscored by the boundary head.

| Arm | Correct | S / D / I | Known WER | Retained baseline correct |
| --- | ---: | --- | ---: | ---: |
| Reel | 4/24 | 1 / 19 / 0 | 83.33% | 4/4 |
| Reviewed timing +100ms, no wrist trim | 12/24 | 1 / 11 / 0 | 50.00% | 3/4 |
| Seed17621, hand geometry=False | 5/24 | 0 / 19 / 0 | 79.17% | 2/4 |
| Seed17622, hand geometry=False | 6/24 | 0 / 18 / 0 | 75.00% | 2/4 |
| Seed17621, hand geometry=True | 7/24 | 1 / 16 / 0 | 70.83% | 3/4 |
| Seed17622, hand geometry=True | 8/24 | 0 / 16 / 0 | 66.67% | 3/4 |

Retention uses transcript-aligned reference positions; no real repeated-sign recording is present to establish temporal repeat accuracy. Boundary region match counts, guarded-gap commits and headless end-to-output timings are in learned_results.json. These are desktop replay timings, not sustained iPhone end-to-display performance.

## Decision and remaining gates

**No automatic promotion.** This reused tiny development set has no independently reviewed low-motion/hold/repeat/OOV stress coverage. Unit checks establish decoder mechanics, not recognition of real holds or repeats. Boundary losses do not establish sign accuracy.

With reviewed intervals+100ms the verifier identifies 16/24signs; even idealized timing leaves identity/commit errors. Contextual identity adaptation remains indicated for review, but the earlier combined-data failure prevents treating another generic mixed run as an established remedy. Use the saved per-interval proposal/verifier evidence to specify that next bounded intervention; no backbone swap or threshold sweep was launched.

All source roles, hashes, generic phrase gates and protected test sealing are preserved. No data acquired.

## Run

```sh
venv/bin/python scripts/app_shell_v17.py --boundary-checkpoint artifacts/models/asl_temporal_boundary_v17_20260922/seed_17621_geometry_1.pth
```

The command explicitly opts into a candidate; it is not a recommendation to replace default Reel. The representative checkpoint is named by the fixed seed/arm, not picked from these validation outcomes.
