# Live input and temporal adaptation experiment

Status: 12 training epochs complete; every epoch failed promotion. Raw adapted replay verification complete. Experimental runtime only, no checkpoint promoted.

## Question and controlled changes

Does running the sequence decoder during capture, using the same extraction/window contract for training and runtime, and adapting the existing temporal heads improve continuous recognition without losing familiar phrases?

Implemented `--sequence-preview` in the existing continuous Reel entry point. It bypasses Stage1 proposals and displays growing CTC hypotheses during capture. An irreversible prefix requires two fresh hypotheses agreeing in both labels and emission positions, with one feature window of lookahead. Reset discards stale asynchronous results. Finish retains the confirmed prefix and leaves the uncertain tail for review. Agreement is a stability rule, **not an accuracy guarantee**. The existing default remains unchanged; the experiment is opt-in.

The live observer is shared with the cache builder: source-time scheduling at nominal 20 Hz, 640px detection, 1280px retained images for crops, sparse body/face every 8 processed frames, elapsed 32/30s windows, identical frozen Stage1 and hand image encoding. New recordings retain source timestamps for each written low-resolution frame. This does not recover dropped frames or make old 15fps recordings exact replays.

## Data and training

All 1,647 existing real videos were re-extracted successfully in 3,386.7s (56.45min). Training:1,313 clips (390 local, 44 exact-variant ASLLRP, 879 OTHER-containing spans). Development:334 (97 local, 12 exact, 225 OTHER). Source/video hashes, original roles and the executable input-contract fingerprint are retained in `frozen_inputs.json` and `cache.json`. No new bulk data, protected test access, or old-cache replacement. The new contract deliberately omits source-specific zeroing of lip features because live capture has no dataset identity. All 334 development clips fit the 8-window context.

The bounded run uses 12 epochs, seed 17101, 2,048 balanced draws per epoch. Temporal primary/context/specialist weights are trainable; Stage1 and OTHER feature extraction remain frozen. Existing real matched sequences provide 60% sampling mass; 3,115 original training replay examples share 40% (44 exact, 390 local, 1,116 contextual, 90 STEM, 1,475 Citizen). Loss is gold CTC plus 0.5 decoded-teacher-sequence CTC on replay. No framewise distillation, new backbone, or added hold/endpoint augmentation. Endpoint augmentation remains an untested next experiment.

The differentiable OTHER branch starts strongly suppressed. Consequently epoch 0 restores the accepted composite's 603 original connected-target errors, rather than reproducing the repaired run-veto teacher's 470 errors. All improvement comparisons below use the actual deployed repair baseline, not this weaker initialization. `design.json` and `history.json` beside the checkpoints record all epochs and gates.

## Baseline online results

| Matched development group | Final sequence WER | Confirmed-prefix WER | Exact complete confirmed phrases | Nonempty confirmed output | Incorrect confirmed reference prefix |
|---|---:|---:|---:|---:|---:|
| Local,97 clips/259 tokens |19/259 =7.34% |139/259 =53.67% |26/97 =26.80% |64/97 =65.98% |9/97 =9.28% |
| Exact variant,12 clips/24 tokens |11/24 =45.83% |22/24 =91.67% |0/12 =0% |2/12 =16.67% |0/12 =0% |
| OTHER spans,225 clips/284 known tokens |478/284 =168.31% |294/284 =103.52% |34/225 =15.11% |173/225 =76.89% |129/225 =57.33% |

WER can exceed 100% because insertions count. It is not classification accuracy. The low wrong-prefix count on short exact-variant clips largely reflects missing output, not success. Prefix conflicts occurred on 9/97 local and 44/225 OTHER examples. Offline prefix timing is in source-window units and excludes extraction/scheduling latency. See `prefix_baseline.json`.

All 12 exact-variant raw paced replays completed. Final hypotheses match the matched cache 12/12; final WER 45.83%, exact phrases 25%, confirmed-prefix WER 91.67%, exact confirmed phrases 0%. Only 2/12 clips had an accepted sequence update before Finish. Most clips are too short to fill a 1.067s window, compute it, and obtain confirmation before EOF. See `online_replays.json` and `online_summary.json`.

Two longer local paced clips show the tradeoff directly:

- PLEASE HELP I: first PLEASE suggestion at 2.50s, PLEASE confirmed at 3.87s, HELP at 4.74s; Finish requested near 5.0s, correct complete prefix at 5.34s. Final review includes an extra I, which stays provisional. The initial no-hand window was rejected. These times are since capture start, not annotated sign completion.
- HELLO HOW YOU: first suggestion HELLO KNOW at 1.40s, revised to KNOW HOW at 2.63s, then incorrectly confirmed KNOW HOW YOU after Finish around 3.70s. Stability did not repair the substitution.

Hand image encoding accounts for most measured window-compute time in these runs (roughly 77–612ms per accepted window); CTC selector computation was roughly 13–20ms. Runs overlapped other development work, so these are observed desktop latencies, not isolated performance benchmarks or iPhone measurements.

## Adaptation result

The run completed in998s (16.63min, excluding cache extraction). The declared selection rule chose epoch8 as the best experimental checkpoint by aggregate matched development WER; **no epoch passed both gate groups**. Best and last checkpoints both reload with finite parameters. See `training_summary.json` and the full checkpoint-side history.

| Evaluation | Repaired baseline | Selected epoch8 | Outcome |
|---|---:|---:|---|
| Matched connected known-token WER |478/284 =168.31% |226/284 =79.58% |52.72% fewer errors |
| Matched local phrase WER |19/259 =7.34% |11/259 =4.25% |Improved |
| Matched exact-variant WER |11/24 =45.83% |12/24 =50.00% |Failed retention |
| Original connected known-token WER |470/284 =165.49% |223/284 =78.52% |Improved |
| Original local phrase WER |6/259 =2.32% |10/259 =3.86% |Failed retention |
| Original exact-variant WER |9/24 =37.50% |12/24 =50.00% |Failed retention |
| Original contextual WER |43/254 =16.93% |49/254 =19.29% |Failed retention |
| Citizen isolated validation accuracy |331/378 =87.57% |330/378 =87.30% |Within3-clip tolerance |
| STEM isolated validation accuracy |16/21 =76.19% |16/21 =76.19% |Retained |

These are repeated development evaluations, not new generalization/test scores. The matched and original rows use the same underlying development clips with different input extraction; they are paired views, not independent replication. Twelve exact-variant clips/24 reference signs are a small sample: one edit changes WER by4.17 percentage points. The current-baseline connected improvement gate is423 errors (90% of470), stronger than the previous v2 threshold542 (90% of603). Neither threshold rescues the phrase-retention failures.

Only one seed was run because none of its epochs qualified for promotion. The first invocation failed before training because the old evaluator expected absent contextual/isolated domains; the matched-domain adapter was corrected and the complete12-epoch retry succeeded. No data or checkpoint was selected from protected test errors.

## Adapted prefix evaluation

All334 matched development sequences were evaluated as growing prefixes. Local final exact-phrase accuracy improved from79/97 (81.44%) to86/97 (88.66%), but complete confirmed phrases reached only29/97 (29.90%), with134/259 edits (51.74% WER). Incorrect committed prefixes fell from9/97 to5/97 (5.15% of all clips, or7.94% of the63 clips with any commitment);6/97 clips had prefix conflicts.

Exact-variant final exact phrases fell from3/12 (25%) to2/12 (16.67%). Confirmed output was still present on only2/12, with0 complete exact phrases and91.67% WER. OTHER-span committed WER was220/284 (77.46%), with54 incorrect reference prefixes among225 clips (24.00%, or43.90% of the123 nonempty outputs) and26/225 conflict clips. These failures prevent promoting the commitment rule even where the final sequence improves. See `prefix_adapted.json`.

## Interpretation and next decision

The model has learned useful visual recognition; neither this experiment nor the earlier isolated results support 'it learned nothing'. Real connected timing and irreversible commitment remain weak. Matching inputs removes one uncontrolled difference but cannot alone teach boundaries, prevent repeat emissions, or guarantee the right lexical choice. More bulk isolated videos and another ensemble are not established remedies; the current system already combines primary/context/specialist evidence.

Keep this lane experimental. The next discussion should compare a faster bounded Stage1 Reel experience against continued sequence training using measured lock correctness, completion coverage, and time after sign completion. If continuing temporal adaptation, the missing targeted supervision is sign boundaries/holds/repetitions and verified transcripts on existing failed continuous recordings; the current run does not establish that more generic video volume is needed.

## Reproduce

Use the required Apple Vision environment from the repository root. The first command opens the opt-in sequence-preview experiment using the existing repaired checkpoint:

```sh
venv/bin/python scripts/live_reel_continuous_v17.py --sequence-preview --no-speech --naturalizer literal
```

To inspect the failed-promotion adapted candidate, add:

```sh
--stage2-live-checkpoint artifacts/models/stage2_v17_live_adapt_v1/seed_17101/best.pth
```

For a repeatable video check, add `--video data/raw_videos/PHRASES/PLEASE_HELP_ME/3e65d621.mp4 --realtime-video --finish-at-eof`. This is an existing development clip, not independent evidence. The12-clip exact-variant raw check is reproduced with `venv/bin/python artifacts/reports/stage2_v17_live_matched_v1/replay_online.py`, optionally adding `--checkpoint artifacts/models/stage2_v17_live_adapt_v1/seed_17101/best.pth`. Generated session histories include the model provenance and timing events.

Validation:56 focused unit/integration tests passed, covering live observer contract, model gradients/reload, prefix rules, reset/stale futures, context rollover, frame timestamps, continuous Reel and preservation behavior. Unit tests check implementation correctness; they do not establish recognition accuracy. `git diff --check` passed. The unrelated pre-existing worktree changes were preserved.

Adapted raw verification: all12 exact-variant paced video replays succeeded, with12/12 final hypotheses matching the corresponding matched-cache evaluation. Final WER50.00%, complete exact phrases2/12, confirmed-prefix WER91.67%. Accepted sequence updates arrived before Finish on1/12 clips. See `adapted_online_summary.json` and `adapted_online_replays.json`.
