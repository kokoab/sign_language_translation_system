# Stage 2 data-path root-cause audit

## Decision

**Keep the ASLLRP videos and source annotations. Keep the local phrases, but only through the existing signer-disjoint split. Replace the continuous preprocessing cache before replacing the data.**

The ASLLRP files are not low-resolution: all 1,160 audited Stage-2 clips are 1280×720 or 1280×960 at 29.97 fps. The continuous observer nevertheless detects landmarks at 640 pixels and 20 fps, while the isolated v17 extractor uses up to 1280 pixels. This discards one third of source frames and half the image-side resolution before boundary learning.

## Are the landmarks good enough?

For ASLLRP, yes for a workable baseline. Every one of the 1,160 ASLLRP Stage-2 clips produced accepted feature windows. In ASLLRP-other training windows, median hand-node presence is 99.9%, p10 is 92.5%, and median frames containing any detected hand is 100.0%. The adapted encoder recognizes complete held-out ASLLRP-other sign cores at **72.95%** and contiguous cores at **82.35%**. Those results reject a global landmark failure.

The frontend is still lossy enough to matter. On the same 11 ASLLRP phrase clips with fixed hand evidence, 1280/source-rate landmarks scored 8/22 edits (36.36% WER); 640/source-rate scored 10/22 (45.45%); 1280/20Hz scored 9/22 (40.91%); and the current 640/20Hz contract scored 10/22 (45.45%). Resolution and sampling contribute measurable errors, although neither explains every failure.

## The largest data-handling defect

The conservative manifest accepted 5,331/11,936 events. **6,020 events fail the four/six-observation rules.** Because the cache observes about 20 fps, a valid 0.20-second sign often has only four observations. Requiring six observations preferentially removes short signs and their boundaries. Across the earlier context objective, all 6,641 known windows ended before the annotated sign end, 2,865 contained less than half target frames, and 842 started after onset. This is a supervision-construction failure, not bad ASLLRP annotation.

Artificial slowing is not the repair. The reproducible probe expands 69 observed frames to 138, but adds zero camera observations and preserves matrix rank (69→69). Re-extraction at native timestamps is required.

## Local phrases

The project has 780 unique 640×480 phrase videos from three identified signers. The old Stage-2 manifest leaks all three signers into both train and validation, so its earlier 97-clip local score is optimistic. The corrected local split already exists: signers 01 and 03 train, signer 02 validates, with 287/200 labeled clips. On those 200 held-out-signer clips, the grounded causal model reaches **25.5% exact phrase accuracy and 37.04% WER**. This is useful evidence, but the labeled subset covers only 15 glosses and nine repeated phrase families. It can teach connected timing for those phrases; it cannot establish 100-gloss generalization.

## What failed in the last boundary experiment

Exact-core gloss recognition remained 74.78% overall and 81.22% on held-out ASLLRP-other, while online boundary F1 was 40.46% at ±200 ms and intentional repeats were 0/20. The model can identify many complete signs but receives weak temporal evidence and an unsuitable four-state boundary target. Better decoding reduced insertions; it did not restore missing evidence.

## Post-audit correction and experiment

The launch precheck found that source-rate, 1280-pixel ASLLRP caches and native 30 fps local caches already exist. The prior aligned grounded experiment already used them with the signer-disjoint local split, an 8-frame causal CTC window, locked-100+OTHER outputs, timed alignment, isolated replay, and the exact-core-adapted Stage 1 initialization. It reached 25.5% local exact/37.04% WER and 33.33% ASLLRP-contiguous exact/41.67% WER. Re-extracting or rerunning it would be duplicate work.

A matched 18-epoch ablation removed only standalone transition-as-blank clips. It completed in 106 seconds and regressed local exact to 16.0% and WER to 45.19%; ASLLRP contiguous stayed at 41.67% WER, NCSLGR worsened from 78% to 84% WER, and isolated exact rose slightly from 82.37% to 83.19%. It failed promotion. The replacement-data condition is now met: seek broader signer-disjoint connected recordings with exact locked-vocabulary transcripts rather than another CTC or boundary-state patch. Result: `artifacts/reports/native_ctc_no_blank_v17_20260921/`.

Review representative clips in [videos.html](videos.html). Machine-readable evidence: [inventory.json](inventory.json), [metrics.json](metrics.json), [diagnosis.json](diagnosis.json), and [verification.json](verification.json). No training or protected-test access occurred.
