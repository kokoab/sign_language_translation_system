# MY / HAVE regression experiments — 2026-10-11

**Outcome: the joint V2 prototype repairs the two diagnosed MY failures and retains HAVE recovery, with no changed hypotheses on the matched native regression checks. It remains experimental: no phone installation, production edits, retraining, or protected test access.**

The leading candidate is V2, rather than a wholesale model rollback. Its accuracy checks pass; model loading/warm-up and real-iPhone performance still need work before deployment. This does not claim every MY or HAVE error is solved. The exact rollback requested is also prepared and measured below.

The user prioritizes MY and explicitly accepts losing the situational HAVE→BIG recovery if a joint repair is not safe. The exact pre-contact-fix rollback is preserved in `fallback_before_contact/`; it retains the one-way ANGRY→HAVE handshape check and low-roll TIRED→HAVE rescue.

## What actually changed

1. **The HAVE omission repair changed non-HAVE recognition.** Its global ordering guard delayed early identity locking for every sign. That changes NAME's accepted interval, automatic spelling-mode timing, decoder selection, and resets. With the recognizer held fixed, take03's first NAME mode moves from 5.267–9.267 seconds to 4.767–8.767 seconds. The later reviewed MY is decoded as LIKE. Disabling that guard reverses this result. The earlier model-only explanation was incomplete.
2. **Word endpoints had a separate timestamp bug.** Same-word merges extended `endFrame` without extending `endSeconds`. NAME mode replayed an already-consumed tail, producing a duplicate NAME. Updating both endpoints removes that duplicate without losing the recovered MY.
3. **Early recognition and late closure corrupted duplicate tracking.** The first ordered-output prototype added HELP in one development sentence. An earlier PLEASE closure replaced the later early-locked HELP as `last`. The next HELP fragment then looked like a new word. V2 keeps `last` as the word with the latest start time. Focused checks verify this and the corrected real sentence.
4. **A model weakness remains independent of those bugs.** On the first exact phone input, the current recognizer ranks LIKE .862 and MY .014. The older LocalA model ranks MY .791 on the same span. Across 3,959 matched isolated validation inputs, LocalA has 41 fewer correct predictions overall than the current model (3,743 versus 3,784). Rolling back the decoder alone does not fix this example. Conversely, the old model still fails the later reviewed MY and introduces other wrong words. No single older checkpoint is a clean replacement.

## Native replay matrix

Same current model unless the row explicitly names another model. HAVE entries are output counts, not general accuracy. Take11 still has the pre-existing first HAVE→ANGRY error.

| Experiment | Exact first MY | Take03 MY around 9.2 s | HAVE counts: take11 / 09:55 / 10:06 |
|---|---|---|---|
| Installed control | LIKE | LIKE | 10 / 9 / 9 |
| Rollback, no HAVE cues | LIKE | MY | 4 / 8 / 1 |
| Rollback, bent-vs-claw cue only | LIKE | MY | 4 / 8 / 1 |
| Rollback, finger-roll cue only | LIKE | MY | 5 / 8 / 4 |
| Rollback, both existing cues | LIKE | MY | 5 / 8 / 4 |
| Keep contact recovery; separate identity from publication | LIKE | MY | 10 / 9 / 9 |
| Above + endpoint timestamp correction | LIKE | MY | 10 / 9 / 9 |
| Fragment model + rollback/both cues | LIKE | LIKE | 7 / 9 / 7 |
| Fragment model + ordered output | LIKE | LIKE | 10 / 9 / 9 |
| LocalA model + rollback/both cues | MY, wrong word order | LIKE | 7 / 9 / 7 |
| LocalA model + ordered output | MY | LIKE | 11 / 9 / 9 |
| Joint prototype V2, described below | MY | MY | 10 / 9 / 9 |

The rollback and first ordering prototype reintroduced duplicate NAME; the endpoint correction removes it. The basic rollback also returns NAME-before-LIKE on the exact first trace. These are measured tradeoffs, not hypothetical objections to the fallback.

## Geometry and old-model experiments

- Enabling ordinary early MY recognition does not fix the first trace. The selected MY candidate lasts only one decoder tick, has probability .649, and covers three frames before LIKE takes over. It fails the existing stability/minimum-length requirements. I did not remove those requirements to force a success.
- A 576-combination sweep of finger straightness, middle-finger extension, thumb–middle-finger pinch, pinch change, and MY rank found no safe direct LIKE→MY rule that rescues the first miss. Every matching rule changes at least three genuine LIKE validation clips.
- Adding wrist movement and fingertip droop expands this to **14,400 combinations**. There is still no matching rule with zero false MYs; the best harms one genuine LIKE.
- Using the older recognizer's MY vote alone harms three genuine LIKE clips. The old recognizer is therefore not a universal override.
- The intersection of a conservative static-palm check and an old-model MY vote changes none of the **3,959 cached whole-clip validation classifications**. This is only an isolated-clip screen; it cannot establish streaming safety. The subsequent streaming experiments are reported separately below.

These are development sweeps, not held-out accuracy or proof that no other geometric rule could work. The 59 local MY/LIKE videos below are part of the existing validation collection, not a new independent set.

## Joint prototype

The prototype preserves HAVE contact recovery and the existing bent/claw and roll cues. It separates identity locking from chronological publication, updates both endpoint representations, and preserves the latest recognized word during earlier closures.

For the remaining model-sensitive case, it consults LocalA **only** when the current model's top word is LIKE, MY is in its top three, the candidate has at least six frames, and the visible hand geometry passes all of these checks:

- median minimum finger straightness ≥ .98; lower-quartile middle-finger straightness ≥ .98;
- lower-decile thumb–middle-tip distance ≥ .8 hand lengths; 10th-to-90th-percentile pinch range ≤ .5;
- wrist travel ≤ .75 hand lengths; 10th-to-90th-percentile fingertip-droop range ≤ .5.

All sufficiently visible hands must pass. Only if LocalA also ranks MY first are the current LIKE/MY scores swapped. This is a heuristic decision, not a newly calibrated probability. It uses existing models, with no new training.

V1 fixes the exact first MY but inherits the HELP duplicate from the ordering prototype. V2 adds the chronological `last` correction. The phone replays and first-mode timings are unchanged between V1 and V2; V2 restores PLEASE HELP I in the failing sentence.

## Validation

- Six original decoder arms: all **103 pair-validation hypotheses identical**, including 36 TIRED and 25 ANGRY controls; no false HAVEs in that set.
- Python ordering/rollback comparison: all **378 Citizen validation hypotheses identical** across installed control, rollback/both cues, ordered V1, and ordered V1 with endpoint correction. All 100 classes are represented.
- Python development sentences, 89 clips / 226 reference words: installed 29 errors (12.83%); rollback 34 (15.04%); initial ordered prototype 30 (13.27%). Its one new HELP insertion is the state bug corrected in V2.
- Native full-video MY/LIKE check: **59 videos, 22 MY + 37 LIKE; zero changed hypotheses** between endpoint-corrected ordering control and joint V1. Existing errors remain: exact sequences are 15/22 MY and 23/37 LIKE in both. This is not a claim of 100% accuracy.
- Exact first phone trace: Python/Core ML and Swift joint V2 both produce HELLO MY NAME with identical frame boundaries and a maximum score difference below 5e-16.
- V2 native replay: first trace HELLO MY NAME; both other complete word-mode traces unchanged; all three HAVE hypotheses unchanged; whole-engine phone videos/mode events unchanged from joint V1; PLEASE HELP I restored.
- 23 existing runtime tests pass. Added focused checks cover early identity preservation, chronological HAVE/BIG publication, final flush exactly once, reset, consistent frame/timestamp endpoints, and avoiding the HELP duplicate.

**Final native paired check: the installed-source decoder and joint V2 have identical hypotheses on every one of the 570 cached-input cases**, covering all 100 classes. The same Float32 observations/embeddings and timestamps feed both; recognition uses the deployed FP16 Core ML models.

| Pool | Cases | Installed source | Joint V2 |
|---|---:|---:|---:|
| HAVE / ANGRY / TIRED / HAPPY validation | 103 | 58 exact; 82 contain target | Identical per case |
| Citizen validation, all 100 classes | 378 | 302 exact; 332 contain target | Identical per case |
| Development sentences | 89 / 226 reference words | 29 errors, 12.83% WER | Identical per case |
| Separate full-video local MY/LIKE replay | 59 | Endpoint-corrected ordering control | V2 identical per case |

Native V2 without the old-vote guard also matches those 570 outputs. Thus the old-vote addition repairs the diagnosed exact phone miss without changing this regression pool. V2's state correction changes exactly one case relative to the initial native ordered prototype: PLEASE HELP HELP I becomes PLEASE HELP I.

The separate Python/Torch joint V1 sweep completed all three pools: pair/Citizen hypotheses unchanged, 30 development errors including the known HELP duplicate. This is preserved as a failed intermediate result, not presented as V2's result. Native FP16 and Python/Torch differ by one Citizen exact sequence; backend figures are kept separate.

All 59 MY/LIKE videos were rerun after the V2 state correction, again with no changes. Exported dimensions were checked against decoded first frames for all 570 cases: zero mismatches. `verify_results.py` checks completeness, paired hypotheses, the two MY cases, HAVE controls, endpoint/ordering regression, geometry sweep, and unchanged production hashes.

The pair and Citizen pools overlap: **570 is a count of replay cases, not independent clips.** Neither these pools nor the 59 videos are a new held-out test.

## Limits and deployment

Only the first contiguous phone trace and the other complete word-mode excerpts are exact recorded phone inputs. Full-video replays use real presentation timestamps but regenerate Vision/crops and can sample different frames from the live phone. The phone trace omits NAME-mode frames, so replaying the whole trace as one word-mode run would be invalid.

The user deliberately signed LIKE somewhere in the recordings but cannot locate it. Other MY/LIKE output counts are not error labels, and those predictions were not used as training targets.

The dual-model prototype has an extra model residency and first-use cost. Desktop runs were concurrent and are not a phone latency benchmark. In the native MY/LIKE run its maximum frame cost was about 4.2 seconds, consistent with a cold model/load/compile event; this does not isolate its cause. The lazy fallback needs deliberate loading/warm-up and real-phone latency/memory measurement before deployment. No claim of mobile readiness is made. The prototype's `commit_seconds` still records identity-lock time; delayed publication needs its own logged timestamp before using it as a display-latency measurement.

## Reproduction and files

- `prepare.py`, `run_native.py`: six model-fixed decoder arms and full-engine replay.
- `run_models.py`, `run_ends.py`: older-model crossovers and endpoint repair.
- `sweep_geometry.py`, `sweep_geometry_motion.py`: geometry screens using cached validation scores and landmarks.
- `hybrid/`, `v2/`, `runtime_hybrid*.py`: experimental joint implementations; the Python experiment requires the old-vote callback supplied by its evaluator. Do not copy it into production without integration.
- `evaluate.py`, `evaluate_fast.py`, `evaluate_hybrid.py`: development-only Python replay. Faster evaluation memoizes identical boundary/model inputs; no post-rule scores are shared across variants.
- `export_validation_traces.py`, `validation/`, `validation_v2_probe`: native replay of existing cached observations. Embeddings remain Float32, frame times are preserved, and isolated clips receive the same 20 repeated final observations as the Python evaluation.
- `check_ordered.py`: focused causal checks.
- `fallback_before_contact/`: exact preserved pre-contact decoder sources and reviewable rollback patches; both existing HAVE cues retained.
- `source_manifest.json`: hashes of production sources before this work. They must remain unchanged.

All outputs stay under this experiment directory. Original recordings, datasets, checkpoints, app sources, and the installed app are preserved. Historical model/source evidence remains in `../my_like_regression_v17_20261011/rollback_review/`.

The superseded native V1 dual-model run was stopped after V2 replaced it; partial files are explicitly marked incomplete. One V2 launch had a mistyped input directory and failed before replay; the completed corrected launch is `native_validation_hybrid_v2_run.log`. Early Python harness setup errors/restarts are retained in logs; only complete result files contribute to the final claims.

`git diff --check` reports only the pre-existing whitespace at `scripts/app_shell_v17.py:632`; this experiment does not modify that file.
